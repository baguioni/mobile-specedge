#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"

namespace specedge {

// C++ port of graph.py's LlamaCppEngine. Drives a single llama.cpp sequence
// (seq_id 0) for linear (non-tree) speculative-decoding drafting.
//
// Differences from the Python version, by design:
//  - No torch::Tensor: llama.cpp's own token/position types (llama_token,
//    llama_pos, both int32_t) are used directly instead of pulling in
//    LibTorch for what are, in the Python file, just flattened index lists.
//  - No HF tokenizer object: the vocab-parity check takes a plain expected
//    vocab size, and forward-log decoding uses llama.cpp's own
//    llama_detokenize instead of an external tokenizer.
//  - forward() takes/returns scalars instead of [1, num_beams] tensors,
//    since num_beams == batch_size == 1 is enforced either way.
class LlamaCppEngine {
public:
    struct Config {
        std::string model_path;
        int32_t max_len = 0;
        int32_t max_n_beams = 1;
        int32_t n_gpu_layers = 0;
        int32_t main_gpu = 0;
        std::string role = "unknown";
        // Optional external vocab size (e.g. an HF tokenizer's vocab_size)
        // to sanity-check against the GGUF's own vocab. Skipped if unset.
        std::optional<int32_t> expected_vocab_size;
        std::optional<uint32_t> n_threads;
        std::optional<uint32_t> n_threads_batch;
    };

    explicit LlamaCppEngine(Config config);
    ~LlamaCppEngine();

    LlamaCppEngine(const LlamaCppEngine&) = delete;
    LlamaCppEngine& operator=(const LlamaCppEngine&) = delete;

    void close();

    // Prefills all but the last prompt token (the last token is deferred to
    // the first forward(), matching the Python implementation). batch_idx
    // must be 0.
    void prefill(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        int32_t batch_idx);

    // Decodes a single token and returns logits over the vocab
    // (size n_vocab()).
    std::vector<float> forward(
        llama_token input_id,
        llama_pos position_id,
        int32_t cache_batch_index,
        int32_t cache_seq_index);

    // Prefix-truncation gather: keeps the first dest_indices.size() cache
    // cells. Only src == dest == arange(n) is supported (no tree drafting).
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
    void seq_rm(int32_t seq_id, int32_t p0, int32_t p1);
    void backfill(int32_t have, int32_t need);
    void log_forward(
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& logits) const;
    std::string detokenize(const std::vector<llama_token>& tokens) const;

    void log_info(const char* fmt, ...) const;
    void log_warn(const char* fmt, ...) const;

    std::string role_;
    int32_t max_len_ = 0;
    int32_t max_n_beams_ = 1;

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

    std::shared_ptr<std::ofstream> forward_log_;
};

} // namespace specedge
