#include "spec_client.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace specedge {

namespace {

int32_t Argmax(const std::vector<float>& logits) {
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<int32_t> Arange(int32_t n) {
    std::vector<int32_t> out(n);
    std::iota(out.begin(), out.end(), 0);
    return out;
}

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
// this process opens a given path and shared (append) by every SpecClient
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
        throw std::runtime_error("SpecClient: could not open result log " + path.string());
    }
    g_result_logs.emplace(path.string(), stream);
    return stream;
}

// A token id produced by a well-behaved target model is always a valid
// vocab entry; this is only defensive against a misbehaving/test server
// (e.g. parity_test/mock_server.py's placeholder `selection = input_ids * 2`
// response) returning an id that token_to_piece has no piece data for,
// which llama.cpp reports as an uncaught std::out_of_range. A raw-id dump
// beats a crash after an otherwise-successful generation.
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

SpecClient::SpecClient(
    LlamaCppEngine& engine,
    GrpcClient& validator,
    std::vector<llama_token> prompt_tokens,
    std::string prompt_text,
    Config config)
    : engine_(engine),
      validator_(validator),
      prompt_text_(std::move(prompt_text)),
      config_(config),
      chain_(prompt_tokens, engine.max_len()),
      result_log_(GetResultLog(config.client_idx)) {
    // Chain's own constructor already rejects an empty prompt.
    engine_.reset();
    confirmed_len_ = chain_.prefix_len();
}

std::vector<llama_token> SpecClient::Generate(int32_t req_idx) {
    std::fprintf(stderr, "[SpecClient] Generating sequence req_idx=%d\n", req_idx);

    int32_t step_idx = 0;
    int32_t num_original_tokens = chain_.prefix_len();

    std::vector<llama_token> warmup_tokens = RunCycle(req_idx, step_idx, /*prefill=*/true);

    step_idx = 1;
    bool eog_flag = false;
    const llama_vocab* vocab = engine_.vocab();

    // chain_.end() already accumulates prompt + every accepted generation +
    // bonus token round over round, so it plays the role that
    // linearspecexec.py's separately-accumulated self._prefix_tokens does
    // -- no parallel accumulator is needed here.
    while (chain_.end() - num_original_tokens < config_.max_new_tokens + static_cast<int32_t>(warmup_tokens.size())
           && !eog_flag) {
        std::fprintf(
            stderr, "[SpecClient] Speculative decoding req_idx=%d step_idx=%d\n", req_idx, step_idx);

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
        stderr, "[SpecClient] %s req_idx=%d\n",
        eog_flag ? "EOS token found." : "Max new tokens reached.", req_idx);

    std::vector<llama_token> generated(chain_.tokens().begin(), chain_.tokens().begin() + chain_.end());
    // Trim the reported sequence at the first end-of-generation token, same
    // as linearspecexec.py's generate() slicing fresh_tokens at eos_idx --
    // chain_/engine_ state itself is left untouched since no further round
    // will read past it.
    if (eog_flag) {
        for (size_t i = 0; i < generated.size(); ++i) {
            if (llama_vocab_is_eog(vocab, generated[i])) {
                generated.resize(i + 1);
                break;
            }
        }
    }

    std::fprintf(
        stderr, "[SpecClient] Generated sequence:\n%s\n",
        Detokenize(vocab, generated).c_str());

    return generated;
}

std::vector<llama_token> SpecClient::RunCycle(int32_t req_idx, int32_t step_idx, bool prefill) {
    // Draft phase: linearspecexec.py wraps _grow_chain in util.Timing("sync").
    std::vector<double> draft_forward_ms;
    SteadyClock::time_point draft_start = SteadyClock::now();
    int32_t draft_len = GrowChain(prefill, draft_forward_ms);
    double draft_end_to_end_ms = MillisSince(draft_start);

    // Target phase: wraps _validate_chain in util.Timing("sync").
    TargetStats stats;
    SteadyClock::time_point target_start = SteadyClock::now();
    std::vector<llama_token> fresh_tokens = ValidateChain(req_idx, prefill, draft_len, stats);
    double target_end_to_end_ms = MillisSince(target_start);

    LogCycle(
        req_idx, step_idx, draft_forward_ms, draft_end_to_end_ms, stats, target_end_to_end_ms);

    return fresh_tokens;
}

int32_t SpecClient::GrowChain(bool prefill, std::vector<double>& forward_ms) {
    if (prefill) {
        std::vector<llama_token> prefill_ids(
            chain_.tokens().begin(), chain_.tokens().begin() + confirmed_len_);
        std::vector<llama_pos> prefill_pos(
            chain_.positions().begin(), chain_.positions().begin() + confirmed_len_);
        // engine_.prefill() defers the last prompt token to the first
        // forward() call itself (see graph_engine.h), so the full confirmed
        // prefix is passed here, not confirmed_len_ - 1.
        engine_.prefill(prefill_ids, prefill_pos, /*batch_idx=*/0);
    }

    int32_t chain_len = std::max(
        0, std::min(config_.chain_len, chain_.max_len() - confirmed_len_ - 1));
    int32_t cur_pos = confirmed_len_ - 1;

    // One entry per drafted token, mirroring linearspecexec.py's
    // _grow_chain appending t.elapsed each iteration; stays [] when
    // chain_len clamps to 0 near max_len.
    forward_ms.clear();
    forward_ms.reserve(static_cast<size_t>(chain_len));

    for (int32_t i = 0; i < chain_len; ++i) {
        llama_token input_id = chain_.tokens()[cur_pos];
        SteadyClock::time_point forward_start = SteadyClock::now();
        std::vector<float> logits = engine_.forward(
            input_id, static_cast<llama_pos>(cur_pos), /*cache_batch_index=*/0,
            /*cache_seq_index=*/cur_pos);
        forward_ms.push_back(MillisSince(forward_start));

        llama_token next_token = static_cast<llama_token>(Argmax(logits));
        cur_pos += 1;
        chain_.add(next_token, static_cast<llama_pos>(cur_pos));
    }

    return chain_len;
}

std::vector<llama_token> SpecClient::ValidateChain(
    int32_t req_idx, bool prefill, int32_t draft_len, TargetStats& stats) {
    int32_t seed_pos = confirmed_len_ - 1;
    int32_t input_count = draft_len + 1; // seed token + every drafted token

    // Preprocess: assemble the Validate request buffers. Matches the
    // preprocess_t window in specexec.py's _validate_tree.
    SteadyClock::time_point preprocess_start = SteadyClock::now();

    std::vector<llama_token> input_ids(
        chain_.tokens().begin() + seed_pos, chain_.tokens().begin() + seed_pos + input_count);
    std::vector<llama_pos> position_ids(
        chain_.positions().begin() + seed_pos, chain_.positions().begin() + seed_pos + input_count);
    std::vector<int32_t> cache_seq_indices(input_count);
    for (int32_t i = 0; i < input_count; ++i) {
        cache_seq_indices[i] = seed_pos + i;
    }
    std::vector<float> attention_mask = chain_.attention_mask(seed_pos, input_count);

    // Row i's prediction verifies the draft token at row i+1 -- a chain has
    // no branching, so "parent" is just "the previous row" (see
    // linearspecexec.py's _validate_chain for the same derivation).
    std::vector<int32_t> parent_indices(cache_seq_indices.begin(), cache_seq_indices.end() - 1);

    stats.preprocess_ms = MillisSince(preprocess_start);

    // Wait: blocked on the target. specexec.py's wait_t window also overlaps
    // proactive drafting via asyncio; this client has neither, so this is
    // purely the round-trip.
    SteadyClock::time_point wait_start = SteadyClock::now();
    GrpcClient::ValidateResult result = validator_.Validate(
        config_.client_idx, req_idx, input_ids, position_ids, cache_seq_indices, attention_mask,
        parent_indices, prefill, prefill ? std::optional<std::string>(prompt_text_) : std::nullopt);
    stats.wait_ms = MillisSince(wait_start);

    // Postprocess: acceptance + chain/KV fix-up. Matches postprocess_t.
    SteadyClock::time_point postprocess_start = SteadyClock::now();

    // selection[i] is the target's prediction for the token following input
    // row i, i.e. for the draft token at chain position seed_pos + 1 + i.
    int32_t n_accept = draft_len;
    for (int32_t i = 0; i < draft_len; ++i) {
        llama_token draft_token = chain_.tokens()[seed_pos + 1 + i];
        if (static_cast<int64_t>(draft_token) != result.selection[i]) {
            n_accept = i;
            break;
        }
    }
    llama_token bonus_token = static_cast<llama_token>(result.selection[n_accept]);

    std::vector<llama_token> fresh_tokens(
        chain_.tokens().begin() + seed_pos + 1, chain_.tokens().begin() + seed_pos + 1 + n_accept);
    fresh_tokens.push_back(bonus_token);

    std::fprintf(stderr, "[SpecClient] Num of accepted tokens: %d\n", n_accept + 1);

    int32_t n_keep = confirmed_len_ + n_accept;
    chain_.truncate(n_keep);
    chain_.add(bonus_token, static_cast<llama_pos>(n_keep));
    confirmed_len_ = n_keep + 1;

    // Drop the rejected trailing draft from the engine's KV cache too. This
    // is always a shrink (n_keep <= the chain_len tokens GrowChain just
    // forwarded), so it's a plain prefix truncation -- never the
    // backfill()/predicted_ path, which exists in LlamaCppEngine for the
    // tree-based algorithm's proactive-reuse case that this client doesn't
    // implement. The bonus token itself is forwarded fresh at the start of
    // next round's GrowChain, exactly like every other drafted token.
    std::vector<int32_t> keep = Arange(n_keep);
    engine_.gather(keep, keep);

    stats.postprocess_ms = MillisSince(postprocess_start);

    stats.prefill_cnt = result.prefill;
    stats.num_accepted_tokens = n_accept + 1;

    return fresh_tokens;
}

void SpecClient::LogCycle(
    int32_t req_idx,
    int32_t step_idx,
    const std::vector<double>& draft_forward_ms,
    double draft_end_to_end_ms,
    const TargetStats& stats,
    double target_end_to_end_ms) {
    // Schema is specexec.py's _cycle result-logger record minus
    // target.proactive / target.prev_proactive (no proactive draft here).
    nlohmann::json entry;
    entry["timestamp"] = Iso8601Now();
    entry["client_idx"] = config_.client_idx;
    entry["req_idx"] = req_idx;
    entry["step_idx"] = step_idx;
    entry["draft"]["forward"] = draft_forward_ms;
    entry["draft"]["end_to_end"] = draft_end_to_end_ms;
    entry["target"]["client_preprocess"] = stats.preprocess_ms;
    entry["target"]["client_wait"] = stats.wait_ms;
    entry["target"]["client_postprocess"] = stats.postprocess_ms;
    entry["target"]["end_to_end"] = target_end_to_end_ms;
    entry["target"]["prefill"] = stats.prefill_cnt;
    entry["num_accepted_tokens"] = stats.num_accepted_tokens;

    std::lock_guard<std::mutex> lock(g_result_log_mutex);
    *result_log_ << entry.dump() << "\n";
    result_log_->flush();
}

} // namespace specedge
