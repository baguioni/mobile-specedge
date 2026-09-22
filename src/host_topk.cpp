#include "host_topk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define SPECEDGE_HOST_TOPK_NEON 1
#endif

namespace specedge {

namespace {

// Min-heap order on logit, so the heap's front is the weakest survivor. On
// equal logits the higher id counts as weaker, which keeps the lower id.
bool HeapLess(const std::pair<float, llama_token>& a, const std::pair<float, llama_token>& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
}

#if SPECEDGE_HOST_TOPK_NEON
// Four-lane expf, lifted verbatim from llama.cpp's ggml_v_expf
// (ggml/src/ggml-cpu/vec.h), which is what ggml_vec_soft_max_f32 uses on
// aarch64. Copied rather than included because vec.h is an internal ggml-cpu
// header that drags in ggml-impl.h and the SIMD mappings; this function is
// self-contained NEON intrinsics. Accurate to about 1 ULP, which is far
// inside what a log-normalizer needs.
inline float32x4_t VExpF32(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b =
        vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
        u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
        return vfmaq_f32(k, j, k);
    }
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// Every term is exp(x - chunk_max), so all of them are in (0, 1] and a f32
// accumulator would still drift over a 25k-element chunk. Fold the vector
// accumulator into the double one this often to keep that bounded.
constexpr int32_t kExpFlushBlocks = 4096;
#endif

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
        //
        // The heap update is the awkward part: it is scalar, and it sits in
        // the same loop as the max. But once the heap is full it almost never
        // fires -- a later element has to beat the k-th best of everything
        // before it. So run the max four lanes at a time and test the same
        // four against the heap's weakest survivor; only when a lane wins do
        // we drop into the scalar path, for that group of four, exactly as
        // written below. Running the whole group keeps the semantics
        // identical including ties, since a hit on one lane can raise the
        // threshold for the next.
        auto scalar_step = [&](int32_t i) {
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
        };

        int32_t i = begin;
#if SPECEDGE_HOST_TOPK_NEON
        // Fill the heap first so there is a threshold to test against.
        const int32_t warm = std::min(end, begin + k_);
        for (; i < warm; ++i) {
            scalar_step(i);
        }
        if (static_cast<int32_t>(p.top.size()) == k_) {
            float32x4_t vmax = vdupq_n_f32(p.max);
            float32x4_t vthr = vdupq_n_f32(p.top.front().first);
            for (; i + 3 < end; i += 4) {
                const float32x4_t v = vld1q_f32(x + i);
                vmax = vmaxq_f32(vmax, v);
                if (vpaddd_u64(vreinterpretq_u64_u32(vcgtq_f32(v, vthr)))) {
                    // p.max is folded in below, so the scalar path's own max
                    // update here is redundant but harmless.
                    for (int32_t j = 0; j < 4; ++j) {
                        scalar_step(i + j);
                    }
                    vthr = vdupq_n_f32(p.top.front().first);
                }
            }
            p.max = std::max(p.max, vmaxvq_f32(vmax));
        }
#endif
        for (; i < end; ++i) {
            scalar_step(i);
        }

        // Pass 2: sum of exp relative to the chunk max. The chunk is still
        // in cache from pass 1, so this is bound by the exp, not the loads --
        // hence the four-lane version.
        double sum = 0.0;
        i = begin;
#if SPECEDGE_HOST_TOPK_NEON
        const float32x4_t vmax_e = vdupq_n_f32(p.max);
        float32x4_t vacc = vdupq_n_f32(0.0f);
        int32_t blocks = 0;
        for (; i + 3 < end; i += 4) {
            vacc = vaddq_f32(vacc, VExpF32(vsubq_f32(vld1q_f32(x + i), vmax_e)));
            if (++blocks == kExpFlushBlocks) {
                sum += static_cast<double>(vaddvq_f32(vacc));
                vacc = vdupq_n_f32(0.0f);
                blocks = 0;
            }
        }
        sum += static_cast<double>(vaddvq_f32(vacc));
#endif
        for (; i < end; ++i) {
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
