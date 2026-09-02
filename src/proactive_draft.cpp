#include "proactive_draft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace specedge {

namespace {

using SteadyClock = std::chrono::steady_clock;

double MillisSince(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

// Restores the tree's bounds however Draft() leaves -- return or throw.
// The scratch nodes past `end` are harmless on their own; `end` and
// `prefix_len` pointing into them is not, and this runs concurrently with
// the RPC whose result the caller still has to process.
class BoundsGuard {
public:
    BoundsGuard(Tree& tree, int32_t prefix_len, int32_t end)
        : tree_(tree), prefix_len_(prefix_len), end_(end) {}
    ~BoundsGuard() {
        tree_.set_end(end_);
        tree_.set_prefix_len(prefix_len_);
    }

    BoundsGuard(const BoundsGuard&) = delete;
    BoundsGuard& operator=(const BoundsGuard&) = delete;

private:
    Tree& tree_;
    int32_t prefix_len_;
    int32_t end_;
};

// proactive.py's decay, and deliberately not the 0.9 the main draft uses.
const float kDecayFactor = std::log(0.95f);

} // namespace

ProactiveDraft::ProactiveDraft(
    LlamaCppEngine& engine, Tree& tree, SeqAllocator alloc_seq, Config config)
    : engine_(engine), tree_(tree), alloc_seq_(std::move(alloc_seq)), config_(config) {
    if (!engine_.tree_mode()) {
        throw std::invalid_argument("ProactiveDraft requires an engine in tree mode");
    }
    if (config_.max_n_beams < 1 || config_.max_beam_len < 1 ||
        config_.max_branch_width < 1 || config_.max_budget < 1) {
        throw std::invalid_argument(
            "ProactiveDraft: proactive max_n_beams, max_beam_len, max_branch_width "
            "and max_budget must all be >= 1");
    }
    // Rows of forward_batch_topk() are strided by the engine's own k and
    // sorted best-first, so a narrower proactive width is just a prefix of
    // each row. A wider one has nowhere to come from.
    if (config_.max_branch_width > engine_.draft_top_k()) {
        throw std::invalid_argument(
            "ProactiveDraft: proactive_max_branch_width=" +
            std::to_string(config_.max_branch_width) +
            " exceeds the engine's sampler width draft_top_k=" +
            std::to_string(engine_.draft_top_k()) +
            ". Lower it, or raise max_branch_width to match.");
    }
}

std::vector<int32_t> ProactiveDraft::FrontierLeaves() const {
    const int32_t end = tree_.end();
    const int32_t prefix = tree_.prefix_len();

    std::vector<uint8_t> is_parent(static_cast<size_t>(end), 0);
    for (int32_t i = 0; i < end; ++i) {
        const int32_t p = tree_.parents()[i];
        if (p >= 0 && p < end) {
            is_parent[static_cast<size_t>(p)] = 1;
        }
    }

    std::vector<int32_t> leaves;
    for (int32_t i = prefix; i < end; ++i) {
        if (!is_parent[static_cast<size_t>(i)]) {
            leaves.push_back(i);
        }
    }

    if (static_cast<int32_t>(leaves.size()) > config_.max_n_beams) {
        std::stable_sort(leaves.begin(), leaves.end(), [this](int32_t a, int32_t b) {
            return tree_.logprobs()[a] > tree_.logprobs()[b];
        });
        leaves.resize(static_cast<size_t>(config_.max_n_beams));
        std::sort(leaves.begin(), leaves.end());
    }
    return leaves;
}

std::optional<ProactiveDraft::Bet> ProactiveDraft::ChooseBet() {
    const std::vector<int32_t> leaves = FrontierLeaves();
    if (leaves.empty()) {
        return std::nullopt;
    }

    const int32_t n = static_cast<int32_t>(leaves.size());
    std::vector<llama_token> in_tokens(n);
    std::vector<llama_pos> in_pos(n);
    std::vector<int32_t> in_seqs(n);
    for (int32_t b = 0; b < n; ++b) {
        in_tokens[b] = tree_.tokens()[leaves[b]];
        in_pos[b] = tree_.positions()[leaves[b]];
        in_seqs[b] = tree_.seq_ids()[leaves[b]];
    }

    const LlamaCppEngine::TopKRows top =
        engine_.forward_batch_topk(in_tokens, in_pos, in_seqs, leaves);
    const int32_t stride = top.k;

    // These leaves now have real KV cells. Say so: llama.cpp would append a
    // duplicate cell if anything decoded them again at the same (seq, pos),
    // and SpecExecClient's acceptance backfill decides exactly that from
    // this status.
    for (int32_t b = 0; b < n; ++b) {
        tree_.set_status(leaves[b], Tree::kProcessed);
    }

    // argmax over beam_score[i] + decay + logprob[i][j]. beam_score is
    // constant within a leaf, so only each leaf's rank-0 token can win.
    int32_t best_b = -1;
    float best_score = 0.0f;
    for (int32_t b = 0; b < n; ++b) {
        const float score = tree_.logprobs()[leaves[b]] + kDecayFactor +
                            top.logprobs[static_cast<size_t>(b) * stride];
        if (best_b < 0 || score > best_score) {
            best_score = score;
            best_b = b;
        }
    }

    Bet bet;
    bet.leaf_slot = leaves[best_b];
    bet.token = top.ids[static_cast<size_t>(best_b) * stride];
    return bet;
}

std::optional<ProactiveDraft::Result> ProactiveDraft::Draft() {
    const SteadyClock::time_point started = SteadyClock::now();
    const int32_t prev_end = tree_.end();
    const int32_t prev_prefix_len = tree_.prefix_len();
    BoundsGuard guard(tree_, prev_prefix_len, prev_end);

    // The subtree needs its root plus at least one level to be worth
    // anything, and both live in scratch past `end`.
    if (tree_.max_len() - prev_end < 2) {
        last_elapsed_ms_ = MillisSince(started);
        return std::nullopt;
    }

    std::optional<Bet> bet = ChooseBet();
    if (!bet.has_value()) {
        last_elapsed_ms_ = MillisSince(started);
        return std::nullopt;
    }

    // Treat the scratch region as its own tree: the budget and beam
    // bookkeeping below all measure [prefix_len, end), and proactive.py
    // rebases prefix_len for exactly this reason.
    tree_.set_prefix_len(prev_end);

    // The bonus token needs its own branch off the leaf being bet on. If
    // there is no sequence left for it there is no bet to place.
    int32_t root_seq;
    try {
        root_seq = alloc_seq_();
    } catch (const std::exception&) {
        last_elapsed_ms_ = MillisSince(started);
        return std::nullopt;
    }
    engine_.seq_cp(tree_.seq_ids()[bet->leaf_slot], root_seq);
    tree_.add(
        {bet->token},
        {static_cast<llama_pos>(tree_.positions()[bet->leaf_slot] + 1)},
        {bet->leaf_slot},
        {0.0f},
        {root_seq},
        Tree::kPostCandidate);

    const int32_t stride = engine_.draft_top_k();
    const int32_t width = config_.max_branch_width;

    for (int32_t level = 0; level < config_.max_beam_len; ++level) {
        // ---- _process_candidates over the scratch region ----
        std::vector<int32_t> candidates;
        for (int32_t i = tree_.prefix_len(); i < tree_.end(); ++i) {
            if (tree_.status()[i] == Tree::kPostCandidate) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            break;
        }
        if (static_cast<int32_t>(candidates.size()) > config_.max_n_beams) {
            std::stable_sort(
                candidates.begin(), candidates.end(),
                [this](int32_t a, int32_t b) {
                    return tree_.logprobs()[a] > tree_.logprobs()[b];
                });
            candidates.resize(static_cast<size_t>(config_.max_n_beams));
            std::sort(candidates.begin(), candidates.end());
        }

        const int32_t n_beams = static_cast<int32_t>(candidates.size());
        std::vector<llama_token> in_tokens(n_beams);
        std::vector<llama_pos> in_pos(n_beams);
        std::vector<int32_t> in_seqs(n_beams);
        for (int32_t b = 0; b < n_beams; ++b) {
            const int32_t c = candidates[b];
            in_tokens[b] = tree_.tokens()[c];
            in_pos[b] = tree_.positions()[c];
            in_seqs[b] = tree_.seq_ids()[c];
        }

        const LlamaCppEngine::TopKRows top =
            engine_.forward_batch_topk(in_tokens, in_pos, in_seqs, candidates);

        std::vector<float> beam_scores(n_beams);
        std::vector<llama_pos> beam_positions(n_beams);
        for (int32_t b = 0; b < n_beams; ++b) {
            tree_.set_status(candidates[b], Tree::kPostProcessed);
            beam_scores[b] = tree_.logprobs()[candidates[b]];
            beam_positions[b] = tree_.positions()[candidates[b]];
        }

        // ---- _get_next_beams ----
        const int32_t n_flat = n_beams * width;
        std::vector<llama_token> flat_ids(n_flat);
        std::vector<float> flat_scores(n_flat);
        for (int32_t b = 0; b < n_beams; ++b) {
            for (int32_t k = 0; k < width; ++k) {
                const size_t src = static_cast<size_t>(b) * stride + k;
                const int32_t f = b * width + k;
                flat_ids[f] = top.ids[src];
                flat_scores[f] = beam_scores[b] + kDecayFactor + top.logprobs[src];
            }
        }

        const int32_t n_existing = tree_.end() - tree_.prefix_len();
        const int32_t joint_size = n_existing + n_flat;

        std::vector<int32_t> keep;
        if (joint_size > config_.max_budget) {
            std::vector<float> joint;
            joint.reserve(static_cast<size_t>(joint_size));
            for (int32_t i = tree_.prefix_len(); i < tree_.end(); ++i) {
                joint.push_back(tree_.logprobs()[i]);
            }
            joint.insert(joint.end(), flat_scores.begin(), flat_scores.end());

            const int32_t k = std::min<int32_t>(config_.max_budget, joint_size);
            std::nth_element(
                joint.begin(), joint.begin() + (k - 1), joint.end(), std::greater<float>());
            const float min_joint_prob = joint[static_cast<size_t>(k - 1)];

            for (int32_t f = 0; f < n_flat; ++f) {
                if (flat_scores[f] >= min_joint_prob) {
                    keep.push_back(f);
                }
            }
        } else {
            keep.resize(static_cast<size_t>(n_flat));
            std::iota(keep.begin(), keep.end(), 0);
        }

        // Scratch has to fit in what the tree has left. Unlike the main
        // draft, overflowing here is not an error -- this is speculative
        // work and stopping early just means a shallower subtree.
        const int32_t room = tree_.max_len() - tree_.end();
        if (static_cast<int32_t>(keep.size()) > room) {
            std::stable_sort(keep.begin(), keep.end(), [&flat_scores](int32_t a, int32_t b) {
                return flat_scores[a] > flat_scores[b];
            });
            keep.resize(static_cast<size_t>(std::max(0, room)));
            std::sort(keep.begin(), keep.end());
        }
        if (keep.empty()) {
            break;
        }

        // ---- branch forks + add ----
        std::vector<llama_token> child_ids;
        std::vector<llama_pos> child_pos;
        std::vector<int32_t> child_parents;
        std::vector<float> child_logprobs;
        std::vector<int32_t> child_seqs;
        std::vector<uint8_t> inherited(static_cast<size_t>(n_beams), 0);
        bool out_of_seqs = false;
        for (int32_t f : keep) {
            const int32_t b = f / width;
            const int32_t parent = candidates[b];
            int32_t seq;
            if (!inherited[b]) {
                seq = tree_.seq_ids()[parent];
                inherited[b] = 1;
            } else {
                // Running out of sequences is not fatal here either: keep
                // the branches already forked and stop widening.
                try {
                    seq = alloc_seq_();
                } catch (const std::exception&) {
                    out_of_seqs = true;
                    break;
                }
                engine_.seq_cp(tree_.seq_ids()[parent], seq);
            }
            child_ids.push_back(flat_ids[f]);
            child_pos.push_back(beam_positions[b] + 1);
            child_parents.push_back(parent);
            child_logprobs.push_back(flat_scores[f]);
            child_seqs.push_back(seq);
        }

        if (!child_ids.empty()) {
            tree_.add(
                child_ids, child_pos, child_parents, child_logprobs, child_seqs,
                Tree::kPostCandidate);
        }
        if (out_of_seqs || child_ids.empty()) {
            break;
        }
    }

    const int32_t pro_begin = tree_.prefix_len();
    const int32_t pro_end = tree_.end();

    // The scratch bounds unwind when `guard` goes out of scope on the way
    // out of this function. The nodes stay where they are; whether they are
    // ever read again is the caller's call, made once the target's answer
    // says whether the bet was right. Nothing below reads end/prefix_len.

    // A root with nothing under it is a bet with no prize: next round would
    // rebuild the same single token for free. Report no bet so the caller
    // takes the ordinary acceptance path.
    if (pro_end <= pro_begin + 1 || tree_.status()[pro_begin] != Tree::kPostProcessed) {
        last_elapsed_ms_ = MillisSince(started);
        return std::nullopt;
    }

    Result result;
    result.leaf_slot = bet->leaf_slot;
    result.bonus_token = bet->token;
    result.begin = pro_begin;
    result.end = pro_end;
    result.root_seq_id = tree_.seq_ids()[pro_begin];
    result.seq_ids.reserve(static_cast<size_t>(pro_end - pro_begin));
    for (int32_t i = pro_begin; i < pro_end; ++i) {
        result.seq_ids.push_back(tree_.seq_ids()[i]);
    }
    std::sort(result.seq_ids.begin(), result.seq_ids.end());
    result.seq_ids.erase(
        std::unique(result.seq_ids.begin(), result.seq_ids.end()), result.seq_ids.end());

    last_elapsed_ms_ = MillisSince(started);
    return result;
}

} // namespace specedge
