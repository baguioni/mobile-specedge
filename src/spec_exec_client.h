#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "graph_engine.h"
#include "grpc_client.h"
#include "tree.h"

namespace specedge {

// C++ port of specexec.py's SpecExecClient: tree-based speculative decoding.
// Each round drafts a token tree locally through LlamaCppEngine (tree mode),
// verifies every draft node against the target in one GrpcClient::Validate
// call, accepts the deepest fully-matching root-to-leaf path, and appends
// one bonus token from the target.
//
// Differences from the Python version, by design:
//  - No torch::Tensor / device work: Tree (tree.h) plus plain vectors and
//    per-beam CPU loops for the log-softmax/top-k drafting math.
//  - No proactive draft: SpecExecProactiveDraft has no port here (same
//    stance as the former linear client; see linearspecexec.py's note).
//  - Engine cache management is (seq, pos)-based, not slot-based. Python's
//    engine.gather physically permutes torch cache rows; llama.cpp cells
//    are immovable, so:
//      * a branch fork is LlamaCppEngine::seq_cp (tag-only) at child-add
//        time -- the first child inherits the parent's sequence, every
//        sibling forks a fresh one (see Tree's slot -> (seq set, pos) note);
//      * TrimByBudget touches only the Tree's slots (the wire contract
//        needs draft slots contiguous); pruned branches' KV cells are
//        dropped wholesale by the end-of-round collapse;
//      * acceptance is collapse_to_seq (keep the winning branch, retag it
//        onto seq 0) plus a one-token backfill when the deepest accepted
//        node is a never-decoded CANDIDATE leaf, instead of a gather.
//  - EOS uses llama_vocab_is_eog() instead of a single eos_token_id check,
//    same as the former linear client.
//  - No async/await: Validate() blocks synchronously.
class SpecExecClient {
public:
    struct Config {
        int32_t max_n_beams = 0;      // candidates expanded per draft step
        int32_t max_beam_len = 0;     // draft steps (max added depth) per round
        int32_t max_branch_width = 0; // children sampled per expanded candidate
        int32_t max_budget = 0;       // draft-node budget per round
        int32_t max_new_tokens = 0;
        int32_t client_idx = 0;
    };

    // engine must be configured in tree mode (Config::max_seqs > 1). engine
    // and validator are owned by the caller and must outlive this client.
    SpecExecClient(
        LlamaCppEngine& engine,
        GrpcClient& validator,
        std::vector<llama_token> prompt_tokens,
        std::string prompt_text,
        Config config);

    // Generates a sequence up to max_new_tokens (or an end-of-generation
    // token, whichever comes first) and returns it, prompt included.
    std::vector<llama_token> Generate(int32_t req_idx);

private:
    // Per-round target-phase timings (milliseconds) and counters, mirroring
    // the `stats` dict specexec.py's _validate_tree returns minus the
    // proactive-draft booleans.
    struct TargetStats {
        double preprocess_ms = 0.0;   // building the Validate request buffers
        double wait_ms = 0.0;         // blocked on the Validate RPC
        double postprocess_ms = 0.0;  // acceptance + tree / KV fix-up
        int32_t prefill_cnt = 0;      // ValidateResponse.prefill for this batch
        int32_t num_accepted_tokens = 0;
    };

    // Per-round draft-phase timings (milliseconds), one entry per tree
    // level. forward is the engine call; the other three break down the
    // per-level CPU work _get_next_beams and the fork loop do around it.
    // All four sum to nearly all of draft.end_to_end -- the remainder is
    // the joint-budget selection and Tree bookkeeping (tree_.add,
    // TrimByBudget), which touch O(max_budget) floats rather than the
    // O(n_beams * n_vocab) the timed spans do.
    struct DraftStats {
        std::vector<double> forward_ms;  // engine_.forward_batch
        std::vector<double> softmax_ms;  // row_max + sum_exp log-softmax denominator
        std::vector<double> topk_ms;     // TopKIndices over the full vocab
        std::vector<double> fork_ms;     // seq_cp branch forks + tree_.add
        std::vector<int32_t> n_beams;    // rows expanded at this level
    };

    // Port of specexec.py's _cycle: one draft + verify round.
    std::vector<llama_token> RunCycle(int32_t req_idx, int32_t step_idx, bool prefill);

    // Port of _grow_tree + _process_candidates + _get_next_beams: up to
    // max_beam_len batched draft steps. Runs engine_.prefill() first when
    // prefill is true. Appends one entry per draft step to each of stats'
    // vectors.
    void GrowTree(bool prefill, DraftStats& stats);

    // Port of _validate_tree: ships the seed + every draft node to the
    // target, walks acceptance from the returned selection exactly the way
    // the server's _reorder_kv_cache does, collapses the engine onto the
    // accepted branch, reorders the tree, and appends the bonus token.
    // Returns the newly committed tokens (accepted draft tokens + bonus).
    std::vector<llama_token> ValidateTree(int32_t req_idx, bool prefill, TargetStats& stats);

    // Port of _trim_by_budget: compacts the top max_budget draft nodes (by
    // cumulative logprob) into contiguous slots. Tree-only -- see the class
    // comment for why the engine is untouched.
    void TrimByBudget();

    // Port of _check_new_token_in_budget.
    bool CheckNewTokenInBudget(const std::vector<float>& new_scores) const;

    // Hands out fresh llama.cpp seq ids for branch forks. Never recycles
    // mid-round (a pruned branch's cells may still be needed if its prefix
    // is accepted); reset to 1 after every collapse.
    int32_t AllocSeq();

    // Appends one result record -- the same 11 fields the former linear
    // client ported from specexec.py's _cycle -- to
    // log/client_<client_idx>.jsonl.
    void LogCycle(
        int32_t req_idx,
        int32_t step_idx,
        const DraftStats& draft_stats,
        double draft_end_to_end_ms,
        const TargetStats& stats,
        double target_end_to_end_ms);

    LlamaCppEngine& engine_;
    GrpcClient& validator_;
    std::string prompt_text_;
    Config config_;

    Tree tree_;

    // Next fresh seq id for AllocSeq. Seq 0 is the canonical committed
    // sequence; forks start at 1.
    int32_t next_seq_ = 1;

    // Shared, process-wide result sink (same lifecycle as the former linear
    // client's): truncated the first time this process opens it, appended
    // to thereafter.
    std::shared_ptr<std::ofstream> result_log_;
};

} // namespace specedge
