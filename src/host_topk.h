#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"

namespace specedge {

// Full-vocabulary log-softmax + top-k over rows of raw logits, on the host:
// the host-side counterpart of the backend sampler in topk_sampler.h, for a
// backend that cannot run that sampler itself. Hexagon is the case in point:
// it has no TOP_K kernel, and a sampler split between the NPU and the CPU
// costs two graph splits per sequence (514 per decode with 256 sequences on
// llama.cpp 5202104b5).
//
// Work is divided by vocabulary chunk rather than by row, so a 1-row decode
// uses every thread too: thread t scans columns [t*C, (t+1)*C) of every row,
// keeping that chunk's max, its sum of exp(x - max), and its top-k. The
// calling thread then merges the per-chunk partials of each row. Chunks are
// fixed and ties break on the lower token id, so the result does not depend
// on thread timing.
//
// The workers are persistent: this runs once per draft level, and spawning
// threads that often would cost more than the scan.
class HostTopK {
public:
    // n_threads counts the calling thread, which does chunk 0 itself.
    HostTopK(int32_t k, int32_t n_vocab, int32_t n_threads);
    ~HostTopK();

    HostTopK(const HostTopK&) = delete;
    HostTopK& operator=(const HostTopK&) = delete;

    // rows[r] points at n_vocab logits. Writes, for each row, the k best
    // token ids and their full-vocabulary log-probabilities (logit - log Z)
    // to ids[r*k + j] / logprobs[r*k + j], best first.
    void Run(const std::vector<const float*>& rows, llama_token* ids, float* logprobs);

private:
    using Scored = std::pair<float, llama_token>;  // (logit, token id)

    // One (row, chunk) of the current job.
    struct Partial {
        float max = 0.0f;
        double sum = 0.0;          // sum of exp(x - max) over the chunk
        std::vector<Scored> top;   // min-heap on logit, at most k entries
    };

    void Work(int32_t t);        // chunk t of every row of the current job
    void WorkerLoop(int32_t t);

    int32_t k_;
    int32_t n_vocab_;
    int32_t n_threads_;

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    uint64_t generation_ = 0;  // bumped once per Run(); workers wait for a change
    int32_t pending_ = 0;      // workers still busy with the current job
    bool stop_ = false;

    const std::vector<const float*>* rows_ = nullptr;
    std::vector<Partial> partials_;  // [row * n_threads + chunk]
};

} // namespace specedge
