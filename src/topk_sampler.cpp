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
// internal), so this asserts support rather than testing for it. Both ops
// have CUDA kernels that handle vocabulary-width rows: soft_max loops over
// columns with a shared-memory fallback, and top_k drops to a CUB radix
// select above 1024 columns. A backend without them fails loudly at graph
// build rather than silently producing wrong tokens.
bool BackendInit(
    llama_sampler* /*smpl*/,
    ggml_backend_buffer_type_t /*buft*/,
    uint32_t /*n_outputs_max_per_seq*/) {
    return true;
}

void BackendApply(
    llama_sampler* smpl,
    ggml_context* ctx,
    ggml_cgraph* gf,
    llama_sampler_data* data) {
    auto* sctx = static_cast<TopKLogprobCtx*>(smpl->ctx);

    // data->logits arrives as a view of one output row's full-vocab logits.
    ggml_tensor* logits = ggml_reshape_1d(ctx, data->logits, ggml_nelements(data->logits));

    // The whole point: normalize across the full vocabulary *before*
    // selecting, so the k survivors carry comparable probabilities.
    ggml_tensor* probs = ggml_soft_max(ctx, logits);
    ggml_set_name(probs, "logprob_topk_softmax");

    ggml_tensor* top_k = ggml_top_k(ctx, probs, sctx->k);
    ggml_set_name(top_k, "logprob_topk_indices");

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

    ggml_tensor* prob_rows = ggml_reshape_2d(ctx, probs, 1, probs->ne[0]);
    data->probs = ggml_get_rows(ctx, prob_rows, top_k);
    ggml_set_name(data->probs, "logprob_topk_probs");

    ggml_tensor* logit_rows = ggml_reshape_2d(ctx, logits, 1, logits->ne[0]);
    data->logits = ggml_get_rows(ctx, logit_rows, top_k);
    ggml_set_name(data->logits, "logprob_topk_logits");

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
