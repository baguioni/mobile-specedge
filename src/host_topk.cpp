#include "host_topk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace specedge {

namespace {

// Min-heap order on logit, so the heap's front is the weakest survivor. On
// equal logits the higher id counts as weaker, which keeps the lower id.
bool HeapLess(const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
}

} // namespace

HostTopK::HostTopK(int32_t k, int32_t n_vocab, int32_t n_threads)
    : k_(k), n_vocab_(n_vocab), n_threads_(std::max<int32_t>(1, n_threads)) {
    if (k_ <= 0 || k_ > n_vocab_) {
        throw std::invalid_argument(
            "HostTopK: k must be in [1, n_vocab], got k=" + std::to_string(k_) +
            " n_vocab=" + std::to_string(n_vocab_));
    }
    workers_.reserve(static_cast<size_t>(n_threads_ - 1));
    for (int32_t t = 1; t < n_threads_; ++t) {
        workers_.emplace_back(&HostTopK::WorkerLoop, this, t);
    }
}

HostTopK::~HostTopK() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_start_.notify_all();
    for (std::thread& w : workers_) {
        w.join();
    }
}

void HostTopK::WorkerLoop(int32_t t) {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_start_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) {
                return;
            }
            seen = generation_;
        }
        Work(t);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (--pending_ == 0) {
                cv_done_.notify_one();
            }
        }
    }
}

void HostTopK::Work(int32_t t) {
    const int32_t chunk = (n_vocab_ + n_threads_ - 1) / n_threads_;
    const int32_t begin = std::min(n_vocab_, t * chunk);
    const int32_t end = std::min(n_vocab_, begin + chunk);
    const std::vector<const float*>& rows = *rows_;

    for (size_t r = 0; r < rows.size(); ++r) {
        Partial& p = partials_[r * static_cast<size_t>(n_threads_) + static_cast<size_t>(t)];
        p.top.clear();
        p.max = -std::numeric_limits<float>::infinity();
        p.sum = 0.0;
        const float* x = rows[r];

        // Pass 1: chunk max and chunk top-k.
        for (int32_t i = begin; i < end; ++i) {
            const float v = x[i];
            p.max = std::max(p.max, v);
            if (static_cast<int32_t>(p.top.size()) < k_) {
                p.top.emplace_back(v, static_cast<llama_token>(i));
                std::push_heap(p.top.begin(), p.top.end(), HeapLess);
            } else if (v > p.top.front().first) {
                std::pop_heap(p.top.begin(), p.top.end(), HeapLess);
                p.top.back() = {v, static_cast<llama_token>(i)};
                std::push_heap(p.top.begin(), p.top.end(), HeapLess);
            }
        }

        // Pass 2: sum of exp relative to the chunk max. The chunk is still
        // in cache from pass 1.
        double sum = 0.0;
        for (int32_t i = begin; i < end; ++i) {
            sum += std::exp(x[i] - p.max);
        }
        p.sum = sum;
    }
}

void HostTopK::Run(const std::vector<const float*>& rows, llama_token* ids, float* logprobs) {
    const size_t n_rows = rows.size();
    if (n_rows == 0) {
        return;
    }
    const size_t n_partials = n_rows * static_cast<size_t>(n_threads_);
    if (partials_.size() < n_partials) {
        partials_.resize(n_partials);
        for (Partial& p : partials_) {
            p.top.reserve(static_cast<size_t>(k_));
        }
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        rows_ = &rows;
        pending_ = n_threads_ - 1;
        ++generation_;
    }
    cv_start_.notify_all();
    Work(0);
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [&] { return pending_ == 0; });
        rows_ = nullptr;
    }

    // Merge each row's chunks: log Z from the chunk (max, sum) pairs, then
    // the best k of the chunks' k-best lists.
    std::vector<Scored> merged;
    merged.reserve(static_cast<size_t>(k_) * static_cast<size_t>(n_threads_));
    for (size_t r = 0; r < n_rows; ++r) {
        const Partial* parts = &partials_[r * static_cast<size_t>(n_threads_)];

        float row_max = -std::numeric_limits<float>::infinity();
        for (int32_t t = 0; t < n_threads_; ++t) {
            row_max = std::max(row_max, parts[t].max);
        }
        double z = 0.0;
        merged.clear();
        for (int32_t t = 0; t < n_threads_; ++t) {
            if (parts[t].sum > 0.0) {
                z += parts[t].sum * std::exp(static_cast<double>(parts[t].max) - row_max);
            }
            merged.insert(merged.end(), parts[t].top.begin(), parts[t].top.end());
        }
        const double log_z = static_cast<double>(row_max) + std::log(z);

        std::partial_sort(
            merged.begin(), merged.begin() + k_, merged.end(),
            [](const Scored& a, const Scored& b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
        for (int32_t j = 0; j < k_; ++j) {
            const size_t f = r * static_cast<size_t>(k_) + static_cast<size_t>(j);
            ids[f] = merged[static_cast<size_t>(j)].second;
            logprobs[f] = static_cast<float>(static_cast<double>(merged[static_cast<size_t>(j)].first) - log_z);
        }
    }
}

} // namespace specedge
