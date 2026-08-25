#include "spec_client.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <numeric>
#include <optional>

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
      chain_(prompt_tokens, engine.max_len()) {
    // Chain's own constructor already rejects an empty prompt.
    engine_.reset();
    confirmed_len_ = chain_.prefix_len();
}

std::vector<llama_token> SpecClient::Generate(int32_t req_idx) {
    std::fprintf(stderr, "[SpecClient] Generating sequence req_idx=%d\n", req_idx);

    int32_t step_idx = 0;
    int32_t num_original_tokens = chain_.prefix_len();

    int32_t draft_len = GrowChain(/*prefill=*/true);
    std::vector<llama_token> warmup_tokens = ValidateChain(req_idx, /*prefill=*/true, draft_len);

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

        int32_t round_draft_len = GrowChain(/*prefill=*/false);
        std::vector<llama_token> fresh_tokens = ValidateChain(req_idx, /*prefill=*/false, round_draft_len);

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

int32_t SpecClient::GrowChain(bool prefill) {
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

    for (int32_t i = 0; i < chain_len; ++i) {
        llama_token input_id = chain_.tokens()[cur_pos];
        std::vector<float> logits = engine_.forward(
            input_id, static_cast<llama_pos>(cur_pos), /*cache_batch_index=*/0,
            /*cache_seq_index=*/cur_pos);

        llama_token next_token = static_cast<llama_token>(Argmax(logits));
        cur_pos += 1;
        chain_.add(next_token, static_cast<llama_pos>(cur_pos));
    }

    return chain_len;
}

std::vector<llama_token> SpecClient::ValidateChain(
    int32_t req_idx, bool prefill, int32_t draft_len) {
    int32_t seed_pos = confirmed_len_ - 1;
    int32_t input_count = draft_len + 1; // seed token + every drafted token

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

    GrpcClient::ValidateResult result = validator_.Validate(
        config_.client_idx, req_idx, input_ids, position_ids, cache_seq_indices, attention_mask,
        parent_indices, prefill, prefill ? std::optional<std::string>(prompt_text_) : std::nullopt);

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

    return fresh_tokens;
}

} // namespace specedge
