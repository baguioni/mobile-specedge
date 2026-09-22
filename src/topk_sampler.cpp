#include "topk_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

#include "ggml.h"

namespace specedge {

namespace {

struct TopKLogprobCtx {
    int32_t k = 0;
};

const char* SamplerName(const llama_sampler* /*smpl*/) {
    return "specedge-topk-logprob";
}

// Host fallback. Never runs when a backend sampler is active, but the
// interface requires it, and if it ever does run it must agree with
// BackendApply: softmax over the full candidate set first, truncate after.
void SamplerApply(llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* sctx = static_cast<TopKLogprobCtx*>(smpl->ctx);
    const size_t k = std::min<size_t>(static_cast<size_t>(sctx->k), cur_p->size);
    if (k == 0 || cur_p->size == 0) {
        return;
    }

    float max_logit = cur_p->data[0].logit;
    for (size_t i = 1; i < cur_p->size; ++i) {
        max_logit = std::max(max_logit, cur_p->data[i].logit);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float e = std::exp(cur_p->data[i].logit - max_logit);
        cur_p->data[i].p = e;
        sum += e;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= sum;
    }

    // Softmax is monotonic, so ranking by p and by logit agree.
    std::partial_sort(
        cur_p->data, cur_p->data + k, cur_p->data + cur_p->size,
        [](const llama_token_data& a, const llama_token_data& b) { return a.logit > b.logit; });
    cur_p->size = k;
    cur_p->sorted = true;
}

// No probe is exposed to user code (llama_sampler_backend_support is
// internal), so this asserts support rather than testing for it. Every op
// BackendApply emits handles vocabulary-width rows on the backends this
// project runs on: CUDA, CPU, and -- since llama.cpp ce8caa6e6 -- Hexagon,
// whose TOP_K kernel takes single rows up to 262144 wide and whose EXP,
// SUM_ROWS, LOG, SUB and GET_ROWS kernels have no width cap at all. A
// backend missing one fails loudly at graph build rather than silently
// producing wrong tokens.
bool BackendInit(
    llama_sampler* /*smpl*/,
    ggml_backend_buffer_type_t /*buft*/,
    uint32_t /*n_outputs_max_per_seq*/) {
    return true;
}

// How many rows to fold a `n`-column vocabulary row into before running
// vocabulary-wide elementwise work over it. Returns the smallest power of two
// that divides `n` and brings the row under kMaxCols; 1 (no fold) when `n` has
// no such factor, which still works everywhere except Hexagon.
//
// kMaxCols is set from Hexagon's VTCM budget, the tightest of the backends
// here: a binary op stages 2 source rows + 2 destination rows per thread
// (double-buffered), so it needs 16 * cols bytes per thread out of 8 MB. At
// 32768 columns that is 2 MB across 4 threads and 4 MB across 8, both
// comfortable. Other backends do not care about the shape.
int64_t fold_rows(int64_t n) {
    constexpr int64_t kMaxCols = 32768;
    for (int64_t rows = 1; rows <= n; rows *= 2) {
        if (n % rows == 0 && n / rows <= kMaxCols) {
            return rows;
        }
    }
    return 1;
}

void BackendApply(
    llama_sampler* smpl,
    ggml_context* ctx,
    ggml_cgraph* gf,
    llama_sampler_data* data) {
    auto* sctx = static_cast<TopKLogprobCtx*>(smpl->ctx);
    const int32_t k = sctx->k;

    // data->logits arrives as a view of one output row's full-vocab logits.
    const int64_t n = ggml_nelements(data->logits);
    ggml_tensor* logits = ggml_reshape_1d(ctx, data->logits, n);  // [n]
    ggml_tensor* lrows = ggml_reshape_2d(ctx, logits, 1, n);      // [1, n]

    // Select first, normalize second. This is *not* the same order as the
    // Python reference's log_softmax(...).topk(...), but it gives the same
    // answer: softmax is monotonic, so the k largest probabilities sit on
    // the k largest logits, and the normalizer below is still taken over
    // the whole vocabulary. What it buys is that no op in this graph is
    // wider than the backend can take -- in particular ggml_soft_max, which
    // Hexagon rejects above 131072 columns (SOFTMAX_MAX_ROW_SIZE in
    // ggml-hexagon.cpp) while every Qwen3 vocabulary is wider than that:
    // 151936 for 0.6B, 248320 for the 3.5 hybrid. A soft_max here falls
    // back to the CPU and splits the graph, which drags the full-vocab
    // logits across the bus -- the exact copy this sampler exists to avoid.
    ggml_tensor* top_k = ggml_top_k(ctx, logits, k);              // [k] i32
    ggml_set_name(top_k, "logprob_topk_indices");

    ggml_tensor* top_logits = ggml_get_rows(ctx, lrows, top_k);   // [1, k]
    ggml_set_name(top_logits, "logprob_topk_logits");

    // log Z = log sum_v exp(logit_v - m) + m, with m the largest logit.
    // m is taken over the k survivors rather than the vocabulary: the
    // global maximum is in the top-k set by construction, so this is exact
    // and costs a k-wide reduction instead of a second full-vocab pass.
    // ggml_top_k gives no ordering guarantee, hence the reduction rather
    // than reading slot 0.
    ggml_tensor* top_flat = ggml_reshape_1d(ctx, top_logits, k);       // [k]
    ggml_tensor* top_rows = ggml_reshape_2d(ctx, top_flat, 1, k);      // [1, k]
    ggml_tensor* max_idx = ggml_top_k(ctx, top_flat, 1);               // [1] i32
    ggml_tensor* max_logit = ggml_get_rows(ctx, top_rows, max_idx);    // [1, 1]
    ggml_set_name(max_logit, "logprob_topk_max");

    // The elementwise and reduce ops below run over the whole vocabulary, so
    // their shape decides whether they fit on the device. Hexagon's kernels
    // stage a *whole row* in VTCM -- for a binary op, two source rows and a
    // destination row, double-buffered, times the DSP's thread count. At
    // 151936 columns that is 2.4 MB per thread against 8 MB of VTCM, so the
    // op is rejected at run time with VTCM-TOO-SMALL and the graph splits
    // back to the CPU. Folding the row into `n_rows` shorter ones keeps the
    // arithmetic identical (these ops are elementwise, and sum_rows over the
    // folded shape is just a partial sum that the second sum_rows finishes)
    // while bringing each staged row down to tens of KB.
    const int64_t n_rows = fold_rows(n);
    ggml_tensor* folded = ggml_reshape_2d(ctx, logits, n / n_rows, n_rows);
    ggml_tensor* shifted = ggml_sub(ctx, folded, max_logit);
    ggml_tensor* partial = ggml_sum_rows(ctx, ggml_exp(ctx, shifted));  // [1, n_rows]
    ggml_tensor* summed =
        ggml_sum_rows(ctx, ggml_reshape_2d(ctx, partial, n_rows, 1));   // [1, 1]
    ggml_tensor* log_z = ggml_add(ctx, ggml_log(ctx, summed), max_logit);
    ggml_set_name(log_z, "logprob_topk_logz");

    // Same gather pattern as llama.cpp's stock top-k sampler: a prior
    // sampler in the chain may already have narrowed the candidate set, in
    // which case top_k indexes *into that set* and has to be mapped back to
    // vocabulary ids. With no prior filter the indices are the ids.
    if (data->candidates) {
        ggml_tensor* candidate_rows =
            ggml_reshape_2d(ctx, data->candidates, 1, data->candidates->ne[0]);
        data->candidates = ggml_get_rows(ctx, candidate_rows, top_k);
    } else {
        data->candidates = top_k;
    }
    ggml_set_name(data->candidates, "logprob_topk_candidates");

    // Full-vocabulary probabilities of the k survivors. The caller takes
    // std::log of these (graph_engine.cpp), so this stays probabilities
    // rather than log-probabilities: exp-then-log round trips exactly as
    // well as the softmax-then-log it replaces, and keeps the readback
    // path unchanged.
    data->probs = ggml_exp(ctx, ggml_sub(ctx, top_logits, log_z));     // [1, k]
    ggml_set_name(data->probs, "logprob_topk_probs");

    data->logits = top_logits;

    GGML_UNUSED(gf);
}

void SamplerFree(llama_sampler* smpl) {
    delete static_cast<TopKLogprobCtx*>(smpl->ctx);
}

llama_sampler_i& Iface();

llama_sampler* SamplerClone(const llama_sampler* smpl) {
    const auto* sctx = static_cast<const TopKLogprobCtx*>(smpl->ctx);
    return make_topk_logprob_sampler(sctx->k);
}

// Configuration is immutable, so there is no mutable state to carry across.
void CopyState(const llama_sampler* /*src*/, llama_sampler* /*dst*/) {}

llama_sampler_i g_iface = {
    /* .name              = */ SamplerName,
    /* .accept            = */ nullptr,
    /* .apply             = */ SamplerApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ SamplerClone,
    /* .free              = */ SamplerFree,
    /* .backend_init      = */ BackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ BackendApply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ CopyState,
};

llama_sampler_i& Iface() {
    return g_iface;
}

} // namespace

llama_sampler* make_topk_logprob_sampler(int32_t k) {
    if (k <= 0) {
        return nullptr;
    }
    auto* sctx = new TopKLogprobCtx{k};
    llama_sampler* inner = llama_sampler_init(&Iface(), sctx);

    // llama_context_params::samplers rejects anything that is not a chain
    // ("the backend samplers must be of type llama_sampler_chain" -- it
    // probes with llama_sampler_chain_get(s, -1)), so hand back a one-entry
    // chain. The chain owns `inner` and frees it.
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, inner);
    return chain;
}

} // namespace specedge
