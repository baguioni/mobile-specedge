#include "spec_exec_client.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace specedge {

namespace {

using SteadyClock = std::chrono::steady_clock;

// Wall-clock milliseconds since `start`. llama_decode() and the Validate
// RPC both run synchronously here, so a plain steady_clock span is the
// faithful stand-in for util.Timing's CUDA-sync / event timing on the
// Python side -- there is no async work to miss.
double MillisSince(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

// ISO-8601 local time, millisecond precision, numeric UTC offset -- the
// exact shape log.py's _formatTime produces
// (datetime.isoformat(timespec="milliseconds")), e.g.
// 2026-08-26T14:23:01.123+09:00. Same helper as graph_engine.cpp's.
std::string Iso8601Now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);

    std::tm local_tm{};
    localtime_r(&t, &local_tm);
    std::tm utc_tm{};
    gmtime_r(&t, &utc_tm);

    // Offset = local wall-clock minus UTC wall-clock, both reinterpreted as
    // UTC so the subtraction ignores each side's own DST.
    std::time_t local_as_utc = timegm(&local_tm);
    std::time_t utc_as_utc = timegm(&utc_tm);
    long offset_seconds = static_cast<long>(local_as_utc - utc_as_utc);

    char sign = offset_seconds >= 0 ? '+' : '-';
    long abs_offset = std::labs(offset_seconds);
    int offset_h = static_cast<int>(abs_offset / 3600);
    int offset_m = static_cast<int>((abs_offset % 3600) / 60);

    char buf[64];
    std::snprintf(
        buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03d%c%02d:%02d",
        local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday,
        local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec,
        static_cast<int>(ms.count()), sign, offset_h, offset_m);
    return std::string(buf);
}

// Result files live under ./log (created on first use), one per client
// index: log/client_<idx>.jsonl -- the client_*.jsonl name
// metric/specedge.py globs for. The stream is truncated the first time
// this process opens a given path and shared (append) by every client
// afterwards, matching log.py's ResultHandler opening "w" once while the
// QueueListener appends each record.
std::mutex g_result_log_mutex;
std::unordered_map<std::string, std::shared_ptr<std::ofstream>> g_result_logs;

std::shared_ptr<std::ofstream> GetResultLog(int32_t client_idx) {
    namespace fs = std::filesystem;
    fs::path log_dir = "log";
    fs::create_directories(log_dir);
    fs::path path = log_dir / ("client_" + std::to_string(client_idx) + ".jsonl");

    std::lock_guard<std::mutex> lock(g_result_log_mutex);
    auto it = g_result_logs.find(path.string());
    if (it != g_result_logs.end()) {
        return it->second;
    }
    auto stream = std::make_shared<std::ofstream>(path, std::ios::out | std::ios::trunc);
    if (!stream->is_open()) {
        throw std::runtime_error("SpecExecClient: could not open result log " + path.string());
    }
    g_result_logs.emplace(path.string(), stream);
    return stream;
}

// A token id produced by a well-behaved target model is always a valid
// vocab entry; this is only defensive against a misbehaving/test server
// returning an id that token_to_piece has no piece data for, which
// llama.cpp reports as an uncaught std::out_of_range. A raw-id dump beats
// a crash after an otherwise-successful generation.
std::string Detokenize(const llama_vocab* vocab, const std::vector<llama_token>& tokens) {
    try {
        std::vector<char> buf(tokens.size() * 8 + 16);
        int32_t n = llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
            static_cast<int32_t>(buf.size()), /*remove_special=*/false, /*unparse_special=*/true);
        if (n < 0) {
            buf.resize(static_cast<size_t>(-n));
            n = llama_detokenize(
                vocab, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
                static_cast<int32_t>(buf.size()), false, true);
        }
        return std::string(buf.data(), n > 0 ? static_cast<size_t>(n) : 0);
    } catch (const std::exception& e) {
        std::string fallback = "<detokenize failed: " + std::string(e.what()) + "; token ids:";
        for (llama_token tok : tokens) {
            fallback += " " + std::to_string(tok);
        }
        return fallback + ">";
    }
}

} // namespace

SpecExecClient::ProactiveType SpecExecClient::ParseProactiveType(const std::string& name) {
    if (name == "disabled") return ProactiveType::kDisabled;
    if (name == "excluded") return ProactiveType::kExcluded;
    if (name == "included") return ProactiveType::kIncluded;
    throw std::invalid_argument(
        "proactive.type must be one of disabled, excluded, included (got '" + name + "')");
}

SpecExecClient::SpecExecClient(
    LlamaCppEngine& engine,
    GrpcClient& validator,
    std::vector<llama_token> prompt_tokens,
    std::string prompt_text,
    Config config)
    : engine_(engine),
      validator_(validator),
      prompt_text_(std::move(prompt_text)),
      config_(config),
      tree_(std::move(prompt_tokens), engine.max_len()),
      result_log_(GetResultLog(config.client_idx)) {
    if (!engine_.tree_mode()) {
        throw std::invalid_argument(
            "SpecExecClient requires an engine in tree mode "
            "(LlamaCppEngine::Config::max_seqs > 1).");
    }
    if (config_.max_n_beams < 1 || config_.max_beam_len < 1 ||
        config_.max_branch_width < 1 || config_.max_budget < 1) {
        throw std::invalid_argument(
            "SpecExecClient: max_n_beams, max_beam_len, max_branch_width and "
            "max_budget must all be >= 1.");
    }

    seq_in_use_.assign(static_cast<size_t>(engine_.max_seqs()), 0);
    RebuildSeqPool();

    if (config_.proactive_type != ProactiveType::kDisabled) {
        if (config_.proactive_type == ProactiveType::kIncluded &&
            config_.proactive.max_beam_len >= config_.max_beam_len) {
            throw std::invalid_argument(
                "SpecExecClient: proactive.type=included charges the subtree's depth "
                "against max_beam_len, so proactive.max_beam_len (" +
                std::to_string(config_.proactive.max_beam_len) +
                ") must be less than max_beam_len (" +
                std::to_string(config_.max_beam_len) + ").");
        }
        proactive_ = std::make_unique<ProactiveDraft>(
            engine_, tree_, [this] { return AllocSeq(); }, config_.proactive);
    }

    engine_.reset();
}

void SpecExecClient::RebuildSeqPool() {
    std::fill(seq_in_use_.begin(), seq_in_use_.end(), uint8_t{0});
    for (int32_t i = 0; i < tree_.end(); ++i) {
        const int32_t s = tree_.seq_ids()[i];
        if (s > 0 && s < static_cast<int32_t>(seq_in_use_.size())) {
            seq_in_use_[static_cast<size_t>(s)] = 1;
        }
    }
    seq_in_use_[0] = 1;  // canonical committed sequence, never forked onto
    seq_cursor_ = 1;
}

int32_t SpecExecClient::AllocSeq() {
    const int32_t n = static_cast<int32_t>(seq_in_use_.size());
    for (int32_t i = 0; i < n; ++i) {
        const int32_t s = (seq_cursor_ + i) % n;
        if (s != 0 && !seq_in_use_[static_cast<size_t>(s)]) {
            seq_in_use_[static_cast<size_t>(s)] = 1;
            seq_cursor_ = (s + 1) % n;
            return s;
        }
    }
    throw std::runtime_error(
        "SpecExecClient: ran out of llama.cpp sequences mid-round "
        "(max_seqs=" + std::to_string(engine_.max_seqs()) + "). Raise the "
        "engine's max_seqs or lower max_n_beams/max_branch_width/"
        "max_beam_len (and their proactive counterparts). Seq ids are never "
        "recycled within a round: a pruned branch's cells may still back the "
        "accepted prefix.");
}

std::vector<llama_token> SpecExecClient::Generate(int32_t req_idx, GenerateTrace* trace) {
    std::fprintf(stderr, "[SpecExecClient] Generating sequence req_idx=%d\n", req_idx);

    int32_t step_idx = 0;
    int32_t num_original_tokens = tree_.prefix_len();

    std::vector<llama_token> warmup_tokens = RunCycle(req_idx, step_idx, /*prefill=*/true);

    step_idx = 1;
    bool eog_flag = false;
    const llama_vocab* vocab = engine_.vocab();

    // Between rounds tree_.prefix_len() == tree_.end() and already counts
    // prompt + every accepted token + bonus, so it plays the role of
    // specexec.py's separately-accumulated self._prefix_tokens.
    while (tree_.prefix_len() - num_original_tokens
               < config_.max_new_tokens + static_cast<int32_t>(warmup_tokens.size())
           && !eog_flag) {
        std::fprintf(
            stderr, "[SpecExecClient] Speculative decoding req_idx=%d step_idx=%d\n",
            req_idx, step_idx);

        std::vector<llama_token> fresh_tokens = RunCycle(req_idx, step_idx, /*prefill=*/false);

        for (llama_token tok : fresh_tokens) {
            if (llama_vocab_is_eog(vocab, tok)) {
                eog_flag = true;
                break;
            }
        }

        step_idx += 1;
    }

    std::fprintf(
        stderr, "[SpecExecClient] %s req_idx=%d\n",
        eog_flag ? "EOS token found." : "Max new tokens reached.", req_idx);

    std::vector<llama_token> generated(
        tree_.tokens().begin(), tree_.tokens().begin() + tree_.prefix_len());

    // Where the stop landed. The scan starts at num_original_tokens, not 0:
    // a ChatML prompt ends its user turn with <|im_end|>, which is the EOS
    // token and therefore an EOG token, so a scan from 0 finds the prompt's
    // own terminator every time and never reaches the completion.
    int32_t eog_index = -1;
    for (size_t i = static_cast<size_t>(num_original_tokens); i < generated.size(); ++i) {
        if (llama_vocab_is_eog(vocab, generated[i])) {
            eog_index = static_cast<int32_t>(i);
            break;
        }
    }

    if (trace != nullptr) {
        trace->tokens = generated;
        trace->stopped_on_eog = eog_flag;
        trace->eog_index = eog_index;
    }

    // Trim the reported sequence at the first end-of-generation token, same
    // as specexec.py's generate() slicing fresh_tokens at eos_idx --
    // tree_/engine_ state itself is left untouched since no further round
    // will read past it.
    if (eog_flag && eog_index >= 0) {
        generated.resize(static_cast<size_t>(eog_index) + 1);
    }

    std::fprintf(
        stderr, "[SpecExecClient] Generated sequence:\n%s\n",
        Detokenize(vocab, generated).c_str());

    return generated;
}

std::vector<llama_token> SpecExecClient::RunCycle(int32_t req_idx, int32_t step_idx, bool prefill) {
    // Draft phase: specexec.py wraps _grow_tree in util.Timing("sync").
    DraftStats draft_stats;
    SteadyClock::time_point draft_start = SteadyClock::now();
    GrowTree(prefill, draft_stats);
    double draft_end_to_end_ms = MillisSince(draft_start);

    // Target phase: wraps _validate_tree in util.Timing("sync").
    TargetStats stats;
    SteadyClock::time_point target_start = SteadyClock::now();
    std::vector<llama_token> fresh_tokens = ValidateTree(req_idx, prefill, stats);
    double target_end_to_end_ms = MillisSince(target_start);

    LogCycle(
        req_idx, step_idx, draft_stats, draft_end_to_end_ms, stats, target_end_to_end_ms);

    return fresh_tokens;
}

void SpecExecClient::GrowTree(bool prefill, DraftStats& stats) {
    stats.forward_ms.clear();
    stats.fork_ms.clear();
    stats.n_beams.clear();

    // Mirrors _grow_tree's "no candidates -> skip growth" guard.
    bool has_candidate = false;
    for (int32_t i = 0; i < tree_.end(); ++i) {
        if (tree_.status()[i] == Tree::kCandidate) {
            has_candidate = true;
            break;
        }
    }
    int32_t max_beam_len = has_candidate ? config_.max_beam_len : 0;

    // `included` mode pays for the spliced subtree out of this round's
    // depth budget, so a hit shows up as a shorter draft rather than a
    // deeper tree. `excluded` leaves the budget alone and banks the depth.
    if (config_.proactive_type == ProactiveType::kIncluded && prev_proactive_hit_) {
        max_beam_len = std::max(0, max_beam_len - config_.proactive.max_beam_len);
    }

    const float kDecayFactor = std::log(0.9f);
    const int32_t width = config_.max_branch_width;

    // The engine's sampler decides how many children each beam gets, and
    // flat_ids/flat_scores below are indexed b * width + k against what it
    // returns, so its k has to be exactly the branch width.
    if (engine_.draft_top_k() != width) {
        throw std::runtime_error(
            "SpecExecClient: engine draft_top_k=" + std::to_string(engine_.draft_top_k()) +
            " does not match max_branch_width=" + std::to_string(width) +
            ". Configure the engine with draft_top_k = max_branch_width.");
    }

    for (int32_t cnt = 0; cnt < max_beam_len; ++cnt) {
        // ---- _process_candidates ----
        std::vector<int32_t> candidates;
        for (int32_t i = 0; i < tree_.end(); ++i) {
            if (tree_.status()[i] == Tree::kCandidate) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            break;  // defensive; python only checks before the loop
        }

        if (static_cast<int32_t>(candidates.size()) > config_.max_n_beams) {
            // Top max_n_beams by cumulative logprob, restored to slot order
            // (python: topk(sorted=False) then .sort()). Stable sort keeps
            // the tie-break deterministic where torch.topk's is not.
            std::stable_sort(
                candidates.begin(), candidates.end(),
                [this](int32_t a, int32_t b) { return tree_.logprobs()[a] > tree_.logprobs()[b]; });
            candidates.resize(static_cast<size_t>(config_.max_n_beams));
            std::sort(candidates.begin(), candidates.end());
        }

        if (prefill) {
            // Python prefills slots [0, candidates.min()) -- everything
            // before the seed. LlamaCppEngine::prefill defers its own last
            // input token, so pass the slice through the seed inclusive.
            const int32_t cand_min = candidates.front();
            std::vector<llama_token> ids(
                tree_.tokens().begin(), tree_.tokens().begin() + cand_min + 1);
            std::vector<llama_pos> pos(
                tree_.positions().begin(), tree_.positions().begin() + cand_min + 1);
            engine_.prefill(ids, pos, /*batch_idx=*/0);
            prefill = false;
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

        // One llama_decode for the whole tree level: rows may share a
        // position because each lives on its own sequence. The softmax and
        // top-k are graph nodes inside this call, so their cost lands in
        // this span and only width ids + log-probs per row come back
        // instead of a full n_vocab row.
        SteadyClock::time_point forward_start = SteadyClock::now();
        LlamaCppEngine::TopKRows top =
            engine_.forward_batch_topk(in_tokens, in_pos, in_seqs, candidates);
        stats.forward_ms.push_back(MillisSince(forward_start));
        stats.n_beams.push_back(n_beams);

        std::vector<float> beam_scores(n_beams);
        std::vector<llama_pos> beam_positions(n_beams);
        for (int32_t b = 0; b < n_beams; ++b) {
            tree_.set_status(candidates[b], Tree::kProcessed);
            beam_scores[b] = tree_.logprobs()[candidates[b]];
            beam_positions[b] = tree_.positions()[candidates[b]];
        }

        // ---- _get_next_beams ----
        // The engine already normalized over the full vocabulary and kept
        // the best `width` per row, in (row, rank) order -- the same layout
        // as flat_ids/flat_scores -- so a child's score is just its parent's
        // plus the decay plus its log-prob.
        const int32_t n_flat = n_beams * width;
        std::vector<llama_token> flat_ids(n_flat);
        std::vector<float> flat_scores(n_flat);
        for (int32_t b = 0; b < n_beams; ++b) {
            for (int32_t k = 0; k < width; ++k) {
                const int32_t f = b * width + k;
                flat_ids[f] = top.ids[static_cast<size_t>(f)];
                flat_scores[f] =
                    beam_scores[b] + kDecayFactor + top.logprobs[static_cast<size_t>(f)];
            }
        }

        const int32_t n_existing = tree_.end() - tree_.prefix_len();
        const int32_t joint_size = n_existing + n_flat;

        std::vector<int32_t> keep;  // flat indices, ascending
        if (joint_size > config_.max_budget ||
            joint_size + n_existing > tree_.max_len()) {
            // joint_probs = concat(tree.logprobs[prefix:end), flat_scores)
            std::vector<float> joint;
            joint.reserve(static_cast<size_t>(joint_size));
            for (int32_t i = tree_.prefix_len(); i < tree_.end(); ++i) {
                joint.push_back(tree_.logprobs()[i]);
            }
            joint.insert(joint.end(), flat_scores.begin(), flat_scores.end());

            // python: joint.topk(k=max_budget).values.min(). k is clamped
            // here for the max_len-triggered case where joint can be
            // smaller than the budget (python would raise on that topk).
            const int32_t k = std::min<int32_t>(config_.max_budget, joint_size);
            std::nth_element(joint.begin(), joint.begin() + (k - 1), joint.end(),
                             std::greater<float>());
            const float min_joint_prob = joint[static_cast<size_t>(k - 1)];

            for (int32_t f = 0; f < n_flat; ++f) {
                if (flat_scores[f] >= min_joint_prob) {
                    keep.push_back(f);
                }
            }
            if (static_cast<int32_t>(keep.size()) + tree_.end() > tree_.max_len()) {
                // python parity: NotImplementedError("Implement trim budget")
                throw std::runtime_error(
                    "SpecExecClient: budget trim inside _get_next_beams is "
                    "not implemented (tree would exceed max_len)");
            }
        } else {
            keep.resize(static_cast<size_t>(n_flat));
            std::iota(keep.begin(), keep.end(), 0);
        }

        if (keep.empty()) {
            std::fprintf(stderr, "[SpecExecClient] No more beams to grow\n");
            stats.fork_ms.push_back(0.0);  // level ends before any fork
            break;
        }

        // Budget early stop -- python checks before tree.add.
        if (n_existing >= config_.max_budget) {
            std::vector<float> kept_scores;
            kept_scores.reserve(keep.size());
            for (int32_t f : keep) {
                kept_scores.push_back(flat_scores[f]);
            }
            if (!CheckNewTokenInBudget(kept_scores)) {
                std::fprintf(stderr, "[SpecExecClient] Max budget reached. early stopping\n");
                stats.fork_ms.push_back(0.0);  // level ends before any fork
                break;
            }
        }

        // ---- branch forks + tree.add ----
        // Slot -> seq assignment (see tree.h): per parent, the first (best)
        // kept child inherits the parent's sequence; every sibling forks a
        // fresh one with a tag-only seq_cp of the root..parent path. The
        // forks happen before any child is decoded, so the copied path is
        // exactly root..parent.
        SteadyClock::time_point fork_start = SteadyClock::now();
        std::vector<llama_token> child_ids;
        std::vector<llama_pos> child_pos;
        std::vector<int32_t> child_parents;
        std::vector<float> child_logprobs;
        std::vector<int32_t> child_seqs;
        std::vector<uint8_t> inherited(static_cast<size_t>(n_beams), 0);
        for (int32_t f : keep) {
            const int32_t b = f / width;
            const int32_t parent = candidates[b];
            int32_t seq;
            if (!inherited[b]) {
                seq = tree_.seq_ids()[parent];
                inherited[b] = 1;
            } else {
                seq = AllocSeq();
                engine_.seq_cp(tree_.seq_ids()[parent], seq);
            }
            child_ids.push_back(flat_ids[f]);
            child_pos.push_back(beam_positions[b] + 1);
            child_parents.push_back(parent);
            child_logprobs.push_back(flat_scores[f]);
            child_seqs.push_back(seq);
        }
        tree_.add(child_ids, child_pos, child_parents, child_logprobs, child_seqs,
                  Tree::kCandidate);
        stats.fork_ms.push_back(MillisSince(fork_start));
    }

    if (tree_.end() - tree_.prefix_len() >= config_.max_budget) {
        TrimByBudget();
    }
}

bool SpecExecClient::CheckNewTokenInBudget(const std::vector<float>& new_scores) const {
    // Lowest logprob still inside the top-max_budget of the existing draft
    // nodes; the caller guarantees end - prefix >= max_budget.
    std::vector<float> existing(
        tree_.logprobs().begin() + tree_.prefix_len(),
        tree_.logprobs().begin() + tree_.end());
    const int32_t k =
        std::min<int32_t>(config_.max_budget, static_cast<int32_t>(existing.size()));
    std::nth_element(existing.begin(), existing.begin() + (k - 1), existing.end(),
                     std::greater<float>());
    const float lowest_tree_logprob = existing[static_cast<size_t>(k - 1)];

    const float best_new_logprob = *std::max_element(new_scores.begin(), new_scores.end());
    return best_new_logprob >= lowest_tree_logprob;
}

void SpecExecClient::TrimByBudget() {
    const int32_t prefix = tree_.prefix_len();
    const int32_t n_draft = tree_.end() - prefix;
    const int32_t budget = config_.max_budget;

    // python trims whenever n_draft >= budget; at exactly the budget the
    // gather would be a (topk-ordered) permutation of the full draft set,
    // so it is skipped here as a no-op.
    if (n_draft <= budget) {
        return;
    }
    std::fprintf(stderr, "[SpecExecClient] Trimming tree\n");

    std::vector<int32_t> src(static_cast<size_t>(n_draft));
    std::iota(src.begin(), src.end(), prefix);
    std::stable_sort(src.begin(), src.end(), [this](int32_t a, int32_t b) {
        return tree_.logprobs()[a] > tree_.logprobs()[b];
    });
    src.resize(static_cast<size_t>(budget));
    // Ascending keeps the surviving nodes' relative slot order, so parents
    // stay at lower slots than children (cumulative logprobs decay with
    // depth, so the kept set is closed under ancestors). tree.py's
    // topk(sorted=False) order is arbitrary; this is the deterministic,
    // order-preserving choice.
    std::sort(src.begin(), src.end());

    std::vector<int32_t> dest(static_cast<size_t>(budget));
    std::iota(dest.begin(), dest.end(), prefix);

    tree_.gather(src, dest);
    // Engine untouched, unlike python's engine.gather: llama.cpp cells are
    // (seq, pos)-addressed, so slot compaction does not concern them, and
    // the pruned branches' cells are dropped wholesale by ValidateTree's
    // collapse_to_seq.
}

std::vector<llama_token> SpecExecClient::ValidateTree(
    int32_t req_idx, bool prefill, TargetStats& stats) {
    // Preprocess: assemble the Validate request buffers. Matches the
    // preprocess_t window in specexec.py's _validate_tree.
    SteadyClock::time_point preprocess_start = SteadyClock::now();

    const int32_t end = tree_.end();
    const int32_t prefix = tree_.prefix_len();

    // Target rows: every draft node. status >= PROCESSED covers both
    // decoded nodes and the last level's never-decoded CANDIDATE leaves;
    // the prefix is masked off.
    //
    // The kPost* pair is excluded on purpose. Those mark a proactive
    // subtree, which belongs to the round *after* this one and must not be
    // validated with this round's tree. This scan stops at `end` and the
    // subtree lives past it, so today the bound already excludes them; the
    // status test says so explicitly rather than leaving the wire contract
    // resting on a coincidence of layout.
    std::vector<uint8_t> target_mask(static_cast<size_t>(end), 0);
    std::vector<int32_t> target_indices;
    for (int32_t i = prefix; i < end; ++i) {
        if (tree_.status()[i] >= Tree::kProcessed &&
            tree_.status()[i] < Tree::kPostCandidate) {
            target_mask[i] = 1;
            target_indices.push_back(i);
        }
    }
    if (target_indices.empty()) {
        throw std::runtime_error(
            "SpecExecClient: no draft nodes to validate; the tree failed to grow.");
    }
    std::vector<int32_t> target_parents;
    target_parents.reserve(target_indices.size());
    for (int32_t t : target_indices) {
        target_parents.push_back(tree_.parents()[t]);
    }

    // Input rows: targets plus their parents. Resolves to the seed (the
    // last committed token, whose KV the server has never computed) plus
    // every draft node, in ascending slot order -- the contiguous layout
    // the server's _reorder_kv_cache offsets assume.
    std::vector<uint8_t> input_mask = target_mask;
    for (int32_t p : target_parents) {
        input_mask[p] = 1;
    }
    std::vector<int32_t> input_slots;
    for (int32_t i = 0; i < end; ++i) {
        if (input_mask[i]) {
            input_slots.push_back(i);
        }
    }

    std::vector<llama_token> input_ids;
    std::vector<llama_pos> position_ids;
    input_ids.reserve(input_slots.size());
    position_ids.reserve(input_slots.size());
    for (int32_t s : input_slots) {
        input_ids.push_back(tree_.tokens()[s]);
        position_ids.push_back(tree_.positions()[s]);
    }
    const std::vector<int32_t>& cache_seq_indices = input_slots;
    std::vector<float> attention_mask = tree_.attention_mask_rows(input_slots);

    stats.preprocess_ms = MillisSince(preprocess_start);

    // Wait: the round-trip, and the window the proactive draft runs in --
    // specexec.py's wait_t, which overlaps the two via asyncio.
    //
    // The split of labour is what makes this safe without locks: the worker
    // owns the RPC and touches only the request buffers built above (which
    // outlive it -- the future is joined before this scope ends, on the
    // throwing path too), while the engine and the tree stay on this
    // thread. llama_context is not thread-safe and is never handed over.
    SteadyClock::time_point wait_start = SteadyClock::now();

    auto do_validate = [&]() {
        return validator_.Validate(
            config_.client_idx, req_idx, input_ids, position_ids, cache_seq_indices,
            attention_mask, target_parents, prefill,
            prefill ? std::optional<std::string>(prompt_text_) : std::nullopt);
    };

    GrpcClient::ValidateResult result;
    std::optional<ProactiveDraft::Result> bet;

    if (proactive_ == nullptr) {
        // Nothing to overlap with, so no thread hop to pay for: this is the
        // pre-proactive path, unchanged.
        result = do_validate();
    } else {
        std::future<GrpcClient::ValidateResult> rpc =
            std::async(std::launch::async, do_validate);

        // Bet on the bonus token and pre-grow next round's tree while the
        // target thinks. A losing bet costs only this window; a winning one
        // is spliced in below instead of being discarded.
        // A bet is an optimization and is never allowed to cost correctness:
        // on any failure, drop it and take the ordinary acceptance path.
        // That path is safe even after a partial draft -- it keeps one
        // sequence and frees every other, which is exactly what a
        // half-built subtree's branches are. ProactiveDraft restores the
        // tree's bounds on the way out however it exits.
        try {
            bet = proactive_->Draft();
        } catch (const std::exception& e) {
            std::fprintf(
                stderr, "[SpecExecClient] proactive draft failed, skipping it: %s\n",
                e.what());
            bet.reset();
        }
        stats.proactive_ms = proactive_->last_elapsed_ms();
        stats.proactive_ran = bet.has_value();
        if (bet.has_value()) {
            stats.proactive_nodes = bet->end - bet->begin;
        }

        result = rpc.get();
    }
    stats.wait_ms = MillisSince(wait_start);

    if (result.selection.size() < input_slots.size()) {
        throw std::runtime_error(
            "SpecExecClient: Validate returned " + std::to_string(result.selection.size()) +
            " selections for " + std::to_string(input_slots.size()) + " input rows.");
    }

    // Postprocess: acceptance + tree/KV fix-up. Matches postprocess_t.
    SteadyClock::time_point postprocess_start = SteadyClock::now();

    // interim_t: the target's per-slot next-token choice. The fill value 1
    // is never read -- every parent of a target is an input slot.
    std::vector<int64_t> interim(static_cast<size_t>(end), 1);
    for (size_t j = 0; j < input_slots.size(); ++j) {
        interim[input_slots[j]] = result.selection[j];
    }

    // accept_flags: a draft node is accepted when the target's choice at
    // its parent equals the drafted token. This must reproduce the server's
    // _reorder_kv_cache walk exactly, or draft and target caches diverge.
    std::vector<uint8_t> accept_mask(static_cast<size_t>(end), 0);
    for (int32_t i = 0; i < prefix; ++i) {
        accept_mask[i] = 1;
    }
    for (int32_t t : target_indices) {
        if (static_cast<int64_t>(tree_.tokens()[t]) == interim[tree_.parents()[t]]) {
            accept_mask[t] = 1;
        }
    }

    // Deepest fully-accepted row: a row scores its ancestor-set size when
    // every visible node is accepted, else 0; first maximum wins (torch
    // argmax parity). The seed row is always fully accepted, so the result
    // is never empty-handed.
    int32_t best_row = 0;
    int64_t best_val = -1;
    for (size_t r = 0; r < input_slots.size(); ++r) {
        const int32_t slot = input_slots[r];
        int64_t row_sum = 0;
        bool all_accepted = true;
        for (int32_t c = 0; c < end; ++c) {
            if (tree_.amask_at(slot, c)) {
                row_sum += 1;
                if (!accept_mask[c]) {
                    all_accepted = false;
                }
            }
        }
        const int64_t val = all_accepted ? row_sum : 0;
        if (val > best_val) {
            best_val = val;
            best_row = static_cast<int32_t>(r);
        }
    }
    const int32_t best_slot = input_slots[best_row];

    // Ascending slots along one root-to-leaf path are ascending depth
    // (adds are chronological and gather preserves relative order), so the
    // last fresh slot is the deepest accepted node.
    std::vector<int32_t> fresh_slots;
    std::vector<llama_token> fresh_tokens;
    for (int32_t c = prefix; c < end; ++c) {
        if (tree_.amask_at(best_slot, c)) {
            fresh_slots.push_back(c);
            fresh_tokens.push_back(tree_.tokens()[c]);
        }
    }
    const int32_t last_slot = fresh_slots.empty() ? prefix - 1 : fresh_slots.back();
    const llama_token extra_token = static_cast<llama_token>(interim[last_slot]);

    std::fprintf(
        stderr, "[SpecExecClient] Num of accepted tokens: %zu\n", fresh_slots.size() + 1);

    // The bet is won only if the target agreed on both halves of it: the
    // leaf it hung the subtree off must be the deepest accepted node, and
    // the token it guessed must be the bonus token that actually came back.
    const bool proactive_hit = bet.has_value() && bet->leaf_slot == last_slot &&
                               bet->bonus_token == extra_token;
    stats.prev_proactive_hit = prev_proactive_hit_;
    stats.proactive_hit = proactive_hit;
    prev_proactive_hit_ = proactive_hit;

    std::vector<uint8_t> seq_mask(static_cast<size_t>(end), 0);
    for (int32_t c = 0; c < end; ++c) {
        seq_mask[c] = tree_.amask_at(best_slot, c);
    }

    if (proactive_hit) {
        // ---- hit: keep the subtree ----
        // Every branch of it tags the whole root..bonus path, so keeping
        // their sequences keeps the committed prefix too; everything the
        // target rejected is on some other sequence and is freed. No
        // truncation and no retag onto seq 0 -- the branches have to stay
        // distinguishable for next round's drafting.
        //
        // No backfill either, unlike the miss path below: the proactive
        // draft decoded the winning leaf when it scored it, and decoded the
        // bonus token when it grew the subtree's first level.
        const int32_t committed_len = static_cast<int32_t>(fresh_tokens.size()) + prefix + 1;
        engine_.collapse_to_seqs(bet->seq_ids, committed_len);
        tree_.reorder_by_sequence_proactive(
            seq_mask, bet->begin, bet->end, bet->root_seq_id);
    } else {
        // ---- miss: the llama.cpp replacement for python's
        // engine.gather(seq_indices, arange). The accepted path is
        // addressed by its (seq, pos) coordinates from the slot map: keep
        // the deepest accepted node's primary sequence (it tags the whole
        // root..node path, whether the node inherited it or was forked onto
        // it), drop everything else -- a losing bet's branches included --
        // and retag onto seq 0. Cells never move.
        const int32_t keep_seq = tree_.seq_ids()[last_slot];
        const llama_pos last_pos = tree_.positions()[last_slot];
        // kCandidate here means "never decoded". A leaf the proactive draft
        // scored is kProcessed by now precisely so this does not decode it
        // a second time -- llama.cpp would append a duplicate cell at the
        // same (seq, pos) rather than overwrite.
        const bool tip_undecoded = tree_.status()[last_slot] == Tree::kCandidate;
        const llama_token tip_token = tree_.tokens()[last_slot];

        engine_.collapse_to_seq(keep_seq, tip_undecoded ? last_pos - 1 : last_pos);
        if (tip_undecoded) {
            // The deepest accepted node was a frontier CANDIDATE the draft
            // engine never decoded; close the gap so next round's seed
            // attends to a complete prefix (the tree analogue of linear
            // backfill()).
            engine_.decode_token(tip_token, last_pos, /*seq_id=*/0);
        }

        tree_.reorder_by_sequence(seq_mask);

        tree_.add(
            {extra_token},
            {static_cast<llama_pos>(tree_.positions()[tree_.end() - 1] + 1)},
            {tree_.end() - 1},
            {0.0f},
            {0},
            Tree::kCandidate);
        tree_.set_prefix_len(tree_.end());
        for (int32_t i = 0; i < tree_.prefix_len() - 1; ++i) {
            tree_.set_status(i, Tree::kPrompt);
        }
    }

    // Whatever survived now defines which sequence ids are still spoken
    // for; the rest go back on the free list.
    RebuildSeqPool();

    fresh_tokens.push_back(extra_token);

    stats.postprocess_ms = MillisSince(postprocess_start);
    stats.prefill_cnt = result.prefill;
    stats.num_accepted_tokens = static_cast<int32_t>(fresh_tokens.size());

    return fresh_tokens;
}

void SpecExecClient::LogCycle(
    int32_t req_idx,
    int32_t step_idx,
    const DraftStats& draft_stats,
    double draft_end_to_end_ms,
    const TargetStats& stats,
    double target_end_to_end_ms) {
    // Schema is specexec.py's _cycle result-logger record, plus
    // target.proactive_ms / proactive_ran / proactive_nodes -- the three
    // things that decide whether the bet is worth its window and that the
    // Python logs have no equivalent of.
    nlohmann::json entry;
    entry["timestamp"] = Iso8601Now();
    entry["client_idx"] = config_.client_idx;
    entry["req_idx"] = req_idx;
    entry["step_idx"] = step_idx;
    entry["draft"]["forward"] = draft_stats.forward_ms;
    entry["draft"]["fork"] = draft_stats.fork_ms;
    entry["draft"]["n_beams"] = draft_stats.n_beams;
    entry["draft"]["end_to_end"] = draft_end_to_end_ms;
    // What draft.end_to_end holds that the two spans above do not: the
    // candidate scan, the joint-budget nth_element, and TrimByBudget. All
    // are O(max_budget), so a residual anywhere near the spans means the
    // cost is somewhere this breakdown does not yet look.
    const auto sum = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    };
    entry["draft"]["residual"] =
        draft_end_to_end_ms - sum(draft_stats.forward_ms) - sum(draft_stats.fork_ms);
    entry["target"]["client_preprocess"] = stats.preprocess_ms;
    entry["target"]["client_wait"] = stats.wait_ms;
    entry["target"]["client_postprocess"] = stats.postprocess_ms;
    entry["target"]["end_to_end"] = target_end_to_end_ms;
    entry["target"]["prefill"] = stats.prefill_cnt;
    // specexec.py's names for the two booleans, so metric/ can read either
    // implementation's logs with one parser.
    entry["target"]["proactive"] = stats.proactive_hit;
    entry["target"]["prev_proactive"] = stats.prev_proactive_hit;
    entry["target"]["proactive_ran"] = stats.proactive_ran;
    entry["target"]["proactive_ms"] = stats.proactive_ms;
    entry["target"]["proactive_nodes"] = stats.proactive_nodes;
    entry["num_accepted_tokens"] = stats.num_accepted_tokens;

    std::lock_guard<std::mutex> lock(g_result_log_mutex);
    *result_log_ << entry.dump(-1, ' ', false,
                               nlohmann::json::error_handler_t::replace)
                 << "\n";
    result_log_->flush();
}

} // namespace specedge
