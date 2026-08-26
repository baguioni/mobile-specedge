#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "chain.h"
#include "graph_engine.h"
#include "grpc_client.h"

namespace specedge {

// C++ port of linearspecexec.py's LinearSpecExecClient: drives chain
// (non-tree) speculative decoding by drafting `chain_len` tokens locally
// through LlamaCppEngine, verifying the whole chain against the target in
// one GrpcClient::Validate call, and keeping the longest accepted prefix
// plus one bonus token.
//
// Differences from the Python version, by design:
//  - No torch::Tensor / Tree: token storage is Chain (see chain.h), which
//    is already the linear-collapsed port of tree.py's Tree. Chain grows by
//    push_back and shrinks by truncate() instead of the Python file's
//    pre-allocated buffer with random-access overwrite; a rejected trailing
//    draft is dropped via Chain::truncate() before the next round appends
//    fresh tokens, rather than being silently overwritten in place.
//  - No HF tokenizer / prompt string encode-decode: the caller supplies an
//    already-tokenized prompt (mirrors main.cpp's tokenize() convention,
//    and graph_engine.h's "no HF tokenizer object" design note). The raw
//    prompt string is still threaded through to Validate()'s `prefix` field
//    for the server-side prefill request, exactly as linearspecexec.py does.
//  - EOS uses llama_vocab_is_eog() instead of a single eos_token_id
//    equality check: several GGUF chat models (e.g. Llama 3) have more than
//    one end-of-generation token, and is_eog() is llama.cpp's own way of
//    covering all of them.
//  - No async/await: Validate() blocks synchronously, matching GrpcClient
//    and LlamaCppEngine (see grpc_client.h's design note); there is no
//    Python-style overlap between kicking off the request and doing local
//    work while it's in flight.
//  - No proactive draft: SpecExecProactiveDraft has no port here. See
//    linearspecexec.py's own header comment for why the tree-based
//    proactive mechanism doesn't have an obvious linear analogue.
class SpecClient {
public:
    struct Config {
        int32_t chain_len = 0;    // tokens drafted (and verified) per round
        int32_t max_new_tokens = 0;
        int32_t client_idx = 0;
    };

    // engine and validator are owned by the caller and must outlive this
    // client, matching how graph.py's engine and the gRPC controller are
    // constructed once and threaded through the Python client.
    SpecClient(
        LlamaCppEngine& engine,
        GrpcClient& validator,
        std::vector<llama_token> prompt_tokens,
        std::string prompt_text,
        Config config);

    // Generates a sequence up to max_new_tokens (or an end-of-generation
    // token, whichever comes first) and returns it, prompt included.
    std::vector<llama_token> Generate(int32_t req_idx);

private:
    // Per-round target-phase timings (milliseconds) and counters, gathered
    // by ValidateChain and forwarded to the result log. Mirrors the `stats`
    // dict specexec.py's _validate_tree returns, minus the proactive-draft
    // booleans, which have no analogue in this linear client.
    struct TargetStats {
        double preprocess_ms = 0.0;   // building the Validate request buffers
        double wait_ms = 0.0;         // blocked on the Validate RPC
        double postprocess_ms = 0.0;  // acceptance check + chain / KV fix-up
        int32_t prefill_cnt = 0;      // ValidateResponse.prefill for this batch
        int32_t num_accepted_tokens = 0;
    };

    // Port of specexec.py's _cycle: one draft + verify round. Times the
    // draft and target phases end to end, appends one JSONL result record,
    // and returns the tokens committed this round.
    std::vector<llama_token> RunCycle(int32_t req_idx, int32_t step_idx, bool prefill);

    // Drafts up to config_.chain_len tokens greedily, appending each to
    // chain_. Runs engine_.prefill() first when prefill is true. Appends
    // one wall-clock forward time (ms) per drafted token to forward_ms.
    // Returns the number of tokens actually drafted (may be less than
    // config_.chain_len near max_len).
    int32_t GrowChain(bool prefill, std::vector<double>& forward_ms);

    // Sends the just-drafted chain (config_.chain_len tokens, or fewer if
    // GrowChain hit max_len) to the target for verification, accepts the
    // longest matching prefix, appends the bonus token, and truncates both
    // chain_ and the engine's KV cache back to the accepted boundary.
    // Fills stats with the phase timings and counters. Returns the newly
    // committed tokens (accepted draft tokens + bonus).
    std::vector<llama_token> ValidateChain(
        int32_t req_idx, bool prefill, int32_t draft_len, TargetStats& stats);

    // Appends one result record -- the 11 fields portable from specexec.py's
    // _cycle result-logger call -- to log/client_<client_idx>.jsonl.
    void LogCycle(
        int32_t req_idx,
        int32_t step_idx,
        const std::vector<double>& draft_forward_ms,
        double draft_end_to_end_ms,
        const TargetStats& stats,
        double target_end_to_end_ms);

    LlamaCppEngine& engine_;
    GrpcClient& validator_;
    std::string prompt_text_;
    Config config_;

    Chain chain_;
    // Length of the validated sequence so far (prompt + accepted
    // generations). Always equal to chain_.end() between rounds; diverges
    // from it only while GrowChain is actively drafting.
    int32_t confirmed_len_ = 0;

    // Shared, process-wide result sink: truncated the first time this
    // process opens it, appended to thereafter. script/client.cpp builds
    // one SpecClient per request, so every round of every request in a run
    // writes through the same handle -- the lifecycle log.py's
    // ResultHandler + QueueListener give the Python client's jsonl.
    std::shared_ptr<std::ofstream> result_log_;
};

} // namespace specedge
