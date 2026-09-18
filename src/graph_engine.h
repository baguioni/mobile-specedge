#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"

#include "host_topk.h"

namespace specedge {

// C++ port of graph.py's engine role, on top of llama.cpp's KV cache.
//
// Two modes, selected by Config::max_seqs:
//  - Linear (max_seqs == 1): drives a single llama.cpp sequence (seq_id 0)
//    for chain drafting -- prefill()/forward()/gather()/backfill().
//  - Tree (max_seqs > 1): drives one llama.cpp sequence per root-to-leaf
//    draft branch, the way llama.cpp's examples/speculative/speculative.cpp
//    does. The context is created with kv_unified = true so all sequences
//    share one cell pool: a cell can carry several seq tags, seq_cp() is a
//    tag-only copy (no KV data moves), and the tree attention mask is
//    *derived* by llama.cpp from (seq tags, pos) -- no mask is ever
//    injected. Branch forks are seq_cp(), levels are decoded in one batch
//    via forward_batch_topk(), and acceptance is collapse_to_seq(): keep the
//    winning branch's cells, retag them onto the canonical seq 0. Cells
//    never move -- unlike graph.py's gather, which permutes torch cache
//    rows.
//
// Differences from the Python version, by design:
//  - No torch::Tensor: llama.cpp's own token/position types (llama_token,
//    llama_pos, both int32_t) are used directly instead of pulling in
//    LibTorch for what are, in the Python file, just flattened index lists.
//  - No HF tokenizer object: the vocab-parity check takes a plain expected
//    vocab size, and forward-log decoding uses llama.cpp's own
//    llama_detokenize instead of an external tokenizer.
//  - No cache_seq_indices addressing: the Python engine writes KV by tree
//    slot; llama.cpp cells are addressed by (seq_id, pos). The caller owns
//    the slot -> (seq, pos) mapping (see tree.h) and passes seq/pos here.
class LlamaCppEngine {
public:
    struct Config {
        std::string model_path;
        int32_t max_len = 0;
        int32_t max_n_beams = 1;
        // Number of llama.cpp sequences (= concurrent draft branches).
        // 1 selects linear mode; > 1 selects tree mode (kv_unified cache).
        // llama.cpp caps this at LLAMA_MAX_SEQ (256).
        int32_t max_seqs = 1;
        int32_t n_gpu_layers = 0;
        int32_t main_gpu = 0;
        // Comma-separated ggml backend device names (e.g. "HTP0", or
        // "HTP0,HTP1"), resolved via ggml_backend_dev_by_name() into
        // llama_model_params::devices. Empty leaves llama.cpp's default of
        // considering every registered device, in which case main_gpu is an
        // index into whatever order they happened to register in -- with
        // both GGML_OPENCL and GGML_HEXAGON compiled in, that is NOT
        // reliably "the NPU" at index 0. Set this to pin a specific one.
        std::string device;
        // Pin the whole model to a single GPU (main_gpu). true selects
        // llama.cpp's LLAMA_SPLIT_MODE_NONE; false restores the library
        // default (LLAMA_SPLIT_MODE_LAYER), which spreads layers across
        // every visible CUDA device. Only meaningful with a GPU build and
        // n_gpu_layers != 0.
        bool single_gpu = true;
        // Comma-separated floats, one per entry in `device` in the same
        // order (e.g. "1,0" for a 2-device list): the fraction of layers/rows
        // llama.cpp offloads to each device (llama_model_params::tensor_split).
        // Only takes effect with single_gpu = false (LLAMA_SPLIT_MODE_NONE
        // discards every device but main_gpu before tensor_split is ever
        // read -- see llama_prepare_model_devices in llama.cpp). Empty
        // leaves llama.cpp's default: split by free memory across every
        // device in the list. Use this to pin all real layers to one device
        // (e.g. "1,0") while still keeping a second device in the scheduler
        // purely as an op-level fallback target (e.g. for topk_sampler.cpp's
        // SOFT_MAX/TOP_K/GET_ROWS chain on a backend with no TOP_K kernel).
        std::string tensor_split;
        // Required in tree mode, and must equal the draft's branch width.
        // A backend top-k sampler (topk_sampler.h) is attached to every
        // sequence, which is what makes forward_batch_topk() work and what
        // stops llama.cpp copying raw logits to the host. Left at 0 in
        // linear mode, which has no sampler and returns full logits.
        int32_t draft_top_k = 0;
        // Tree mode only. Score the draft on the host instead of on the
        // backend: no sampler is attached, llama.cpp hands back each row's
        // raw logits, and HostTopK (host_topk.h) does the full-vocabulary
        // log-softmax + top-k across n_threads host threads. For a backend
        // that cannot run the sampler's ops -- Hexagon has no TOP_K, and a
        // sampler split between NPU and CPU costs two graph splits per
        // sequence. draft_top_k still sets k.
        bool host_topk = false;
        std::string role = "unknown";
        // Optional external vocab size (e.g. an HF tokenizer's vocab_size)
        // to sanity-check against the GGUF's own vocab. Skipped if unset.
        std::optional<int32_t> expected_vocab_size;
        std::optional<uint32_t> n_threads;
        std::optional<uint32_t> n_threads_batch;
        // Unset leaves llama.cpp's own default (LLAMA_FLASH_ATTN_TYPE_AUTO).
        // true/false force LLAMA_FLASH_ATTN_TYPE_ENABLED/DISABLED. Matters
        // most on the Hexagon backend, whose HMX flash-attention kernel
        // wins on both prefill and decode-at-depth (see
        // mobile-benchmarks/hexagon-npu-benchmark.md) -- the opposite of
        // CPU/Adreno, where FA hurts decode.
        std::optional<bool> flash_attn;
    };

    explicit LlamaCppEngine(Config config);
    ~LlamaCppEngine();

    LlamaCppEngine(const LlamaCppEngine&) = delete;
    LlamaCppEngine& operator=(const LlamaCppEngine&) = delete;

    void close();

    // Prefills all but the last prompt token (the last token is deferred to
    // the first forward()/forward_batch_topk(), matching the Python
    // implementation). batch_idx must be 0. Both modes; commits to seq 0.
    void prefill(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        int32_t batch_idx);

    // Linear mode only: decodes a single token and returns logits over the
    // vocab (size n_vocab()).
    std::vector<float> forward(
        llama_token input_id,
        llama_pos position_id,
        int32_t cache_batch_index,
        int32_t cache_seq_index);

    // Per-row top-k result of forward_batch_topk(). ids[r*k + j] and
    // logprobs[r*k + j] are the j-th best continuation of input row r and
    // its log-probability under the *full* vocabulary distribution -- the
    // same quantity log_softmax(logits).topk(k) yields, so scores stay
    // comparable across rows.
    struct TopKRows {
        int32_t k = 0;
        int32_t n_rows = 0;
        std::vector<llama_token> ids;
        std::vector<float> logprobs;
    };

    // Tree mode only: decodes one batch of draft nodes -- typically all the
    // candidates of one tree level -- each row under its own sequence. Rows
    // may share a position as long as their seq_ids differ; llama.cpp
    // derives the tree mask from (seq tags, pos). slot_indices carries the
    // rows' tree slots for the forward log only.
    //
    // The softmax and top-k are graph nodes, so they run on the backend and
    // only k ids + k probabilities per row come back rather than a full
    // n_vocab row -- for a 151,936-entry vocabulary at 32 beams, ~19 MB per
    // level replaced by ~4 KB.
    TopKRows forward_batch_topk(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& seq_ids,
        const std::vector<int32_t>& slot_indices);

    // Wall-clock split of the most recent forward_batch_topk() call, in ms.
    // The four spans cover the whole call bar argument checks and batch
    // setup, so they are what a caller's own span around it decomposes into.
    struct ForwardTiming {
        // llama_decode(): graph build, scheduling and -- on a synchronous
        // backend -- all of the compute, sampler subgraph included.
        double decode_ms = 0.0;
        // llama_synchronize(): whatever llama_decode() left queued on an
        // asynchronous backend. Near zero on a synchronous one.
        double sync_ms = 0.0;
        // The llama_get_sampled_* reads over every row -- or, with
        // Config::host_topk, reading the raw logits plus the host
        // log-softmax + top-k.
        double readback_ms = 0.0;
        // Writing the graph-engine.log record, detokenization included.
        double log_ms = 0.0;
    };
    const ForwardTiming& last_forward_timing() const { return last_forward_timing_; }

    // Tree mode only: forks a draft branch. Under kv_unified this is a
    // tag-only copy -- every cell of src_seq (the root..fork path) gains
    // dst_seq's tag, no KV data is copied. Call before the fork's first
    // sibling is decoded so the copied path is exactly root..parent.
    void seq_cp(int32_t src_seq_id, int32_t dst_seq_id);

    // Tree mode only: drops seq_id's tag from cells at positions [p0, p1)
    // (p1 == -1 means "to the end"). Only this seq's tag is removed -- a
    // cell another sequence still tags survives untouched, and the
    // sequence copied *from* is unaffected. Public alongside seq_cp so a
    // caller that forks past seq_cp's "before the first sibling is
    // decoded" precondition (see above) can trim a copy back down to a
    // clean prefix afterward -- see ProactiveDraft::ChooseBet's
    // already-decoded-leaf path.
    void seq_rm(int32_t seq_id, int32_t p0, int32_t p1);

    // Tree mode only: end-of-round acceptance of a non-proactive round.
    // path_tokens/path_positions run from the round's seed (the last
    // committed token, decoded as the tree root) through the deepest
    // accepted node, which sits on keep_seq. Afterwards seq 0 holds exactly
    // prompt + path and every draft branch is gone.
    //
    // Plain-attention models: collapse_to_seq(keep_seq, tip - 1), then the
    // tip re-decoded solo on seq 0. Recurrent/hybrid models (recurrent()):
    // a recurrent state cannot be rewound past what the draft decoded on top
    // of it, so every branch is dropped and the path is re-decoded in one
    // batch onto a hidden sequence that only ever holds committed state,
    // which seq 0 then shares.
    void accept_path(
        int32_t keep_seq,
        const std::vector<llama_token>& path_tokens,
        const std::vector<llama_pos>& path_positions);

    // Tree mode only: end-of-round acceptance. Keeps only seq_id's cells at
    // positions [0, last_pos], drops every other branch (a cell whose tag
    // set empties is freed), and retags the survivors onto the canonical
    // seq 0 -- the seq_rm/seq_keep/seq_cp/seq_keep pattern of
    // examples/speculative/speculative.cpp. Cells never move. Not available
    // on recurrent models, which cannot truncate by position.
    void collapse_to_seq(int32_t seq_id, llama_pos last_pos);

    // Tree mode only: end-of-round acceptance when a proactive subtree
    // survives. Frees every cell not tagged by one of keep_seq_ids and
    // leaves the survivors on the sequences they are already on.
    //
    // Two differences from collapse_to_seq, both required by the subtree:
    // it keeps many branches rather than one, and it does not truncate by
    // position -- the speculative nodes live *past* the committed prefix
    // and are exactly what is being kept. Retagging onto seq 0 is likewise
    // skipped, since the branches must stay distinguishable; the caller's
    // slot -> seq map keeps naming them, and seq 0 simply tags nothing
    // until the next non-proactive round.
    //
    // new_seq_len is the caller's post-collapse sequence length (the
    // committed prefix), used to keep seq_len() meaningful.
    void collapse_to_seqs(const std::vector<int32_t>& keep_seq_ids, int32_t new_seq_len);

    // Tree mode only: decodes one token without returning logits. Used to
    // backfill the deepest accepted node when it was a never-decoded
    // CANDIDATE leaf (the tree analogue of linear backfill()).
    void decode_token(llama_token token, llama_pos pos, int32_t seq_id);

    // Linear mode only. Prefix-truncation gather: keeps the first
    // dest_indices.size() cache cells. Only src == dest == arange(n) is
    // supported; tree acceptance goes through collapse_to_seq() instead.
    void gather(
        const std::vector<int32_t>& src_indices,
        const std::vector<int32_t>& dest_indices);
    // Boolean-mask overload, mirroring the Python bool-tensor src_indices path.
    void gather(
        const std::vector<bool>& src_mask,
        const std::vector<int32_t>& dest_indices);

    void reset();

    int32_t n_vocab() const { return n_vocab_; }
    int32_t max_len() const { return max_len_; }
    int32_t seq_len() const { return seq_len_; }
    int32_t max_seqs() const { return max_seqs_; }
    bool tree_mode() const { return tree_mode_; }
    // The model keeps recurrent state (Mamba, Gated DeltaNet, ...), alone or
    // alongside attention layers as in Qwen3.5. Such a sequence can only move
    // forward: partial seq_rm is unsupported, so acceptance goes through
    // accept_path()'s re-decode and proactive drafting is unavailable.
    bool recurrent() const { return recurrent_; }
    // The attached sampler's k. Callers index forward_batch_topk()'s rows
    // against this, so it must match their branch width.
    int32_t draft_top_k() const { return draft_top_k_; }
    // Exposed so callers can tokenize/detokenize prompts themselves, the
    // same way graph.py leaves tokenization to an external HF tokenizer.
    const llama_vocab* vocab() const { return vocab_; }
    // Exposed so callers can apply the GGUF's built-in chat template to raw
    // dataset prompts (see src/script/client.cpp), mirroring vocab()'s
    // "text handling is the caller's job" rationale.
    const llama_model* model() const { return model_; }

private:
    void load_model(const Config& config);
    void check_vocab_parity(std::optional<int32_t> expected_vocab_size) const;
    void batch_set(int32_t i, llama_token token, llama_pos pos, int32_t seq_id, bool want_logits);
    void decode(int32_t n_tokens);
    std::vector<float> read_logits(int32_t n_rows) const;
    void backfill(int32_t have, int32_t need);
    void log_forward(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& logits) const;
    void log_forward_topk(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const TopKRows& top) const;
    std::string detokenize(const std::vector<llama_token>& tokens) const;

    void log_info(const char* fmt, ...) const;
    void log_warn(const char* fmt, ...) const;

    void require_mode(bool tree, const char* method) const;

    std::string role_;
    int32_t max_len_ = 0;
    int32_t max_n_beams_ = 1;
    int32_t max_seqs_ = 1;
    // Config::draft_top_k, and one backend sampler per sequence when it is
    // set. llama_context_params::samplers holds borrowed pointers, so these
    // must outlive ctx_ -- they are freed after llama_free() in close().
    int32_t draft_top_k_ = 0;
    std::vector<llama_sampler*> samplers_;
    // Set instead of samplers_ when Config::host_topk is.
    std::unique_ptr<HostTopK> host_topk_;
    bool tree_mode_ = false;
    bool recurrent_ = false;
    // Tree mode on a recurrent model: the llama.cpp sequence holding only
    // committed state (== max_seqs_, one past every caller-visible seq id).
    // -1 otherwise.
    int32_t committed_seq_ = -1;

    int32_t seq_len_ = 0;
    // (position, predicted_token) for the bonus token the draft model
    // expects next; used to backfill a KV-cache gap of exactly one cell.
    std::optional<std::pair<int32_t, llama_token>> predicted_;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_memory_t memory_ = nullptr;
    llama_batch batch_{};
    bool batch_allocated_ = false;
    int32_t n_batch_ = 0;
    int32_t n_vocab_ = 0;
    bool closed_ = false;

    ForwardTiming last_forward_timing_;

    std::shared_ptr<std::ofstream> forward_log_;
};

} // namespace specedge
