#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "graph_engine.h"
#include "grpc_client.h"
#include "proactive_draft.h"
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
//  - No asyncio: the Validate RPC is moved to a std::async worker so the
//    proactive draft can run against the engine while it is in flight. The
//    worker touches nothing but its own copies of the request buffers --
//    the engine and the tree stay with this thread.
class SpecExecClient {
public:
    // Mirrors config.py's proactive_type. `included` charges the subtree's
    // depth against the next round's draft budget (a hit shortens the
    // draft); `excluded` leaves the budget alone (a hit deepens the tree).
    enum class ProactiveType { kDisabled, kExcluded, kIncluded };

    static ProactiveType ParseProactiveType(const std::string& name);

    struct Config {
        int32_t max_n_beams = 0;      // candidates expanded per draft step
        int32_t max_beam_len = 0;     // draft steps (max added depth) per round
        int32_t max_branch_width = 0; // children sampled per expanded candidate
        int32_t max_budget = 0;       // draft-node budget per round
        int32_t max_new_tokens = 0;
        int32_t client_idx = 0;

        ProactiveType proactive_type = ProactiveType::kDisabled;
        ProactiveDraft::Config proactive;
    };

    // engine must be configured in tree mode (Config::max_seqs > 1). engine
    // and validator are owned by the caller and must outlive this client.
    SpecExecClient(
        LlamaCppEngine& engine,
        GrpcClient& validator,
        std::vector<llama_token> prompt_tokens,
        std::string prompt_text,
        Config config);

    // What Generate() decided, for callers comparing two runs rather than
    // timing one. Kept separate from the return value because the two want
    // different things: the caller printing a completion wants the sequence
    // trimmed at the stop token, while a caller diffing against another
    // implementation needs the untrimmed sequence plus where the stop
    // actually landed -- trimming first hides whether two runs stopped at
    // the same place or merely reported the same way.
    struct GenerateTrace {
        std::vector<llama_token> tokens;  // untrimmed: prompt + accepted + bonus
        bool stopped_on_eog = false;      // false => the max_new_tokens budget
        int32_t eog_index = -1;           // index into tokens, -1 if none
    };

    // Generates a sequence up to max_new_tokens (or an end-of-generation
    // token, whichever comes first) and returns it, prompt included.
    // trace, when non-null, receives the pre-trim record described above.
    std::vector<llama_token> Generate(int32_t req_idx, GenerateTrace* trace = nullptr);

private:
    // Per-round target-phase timings (milliseconds) and counters, mirroring
    // the `stats` dict specexec.py's _validate_tree returns minus the
    // proactive-draft booleans.
    struct TargetStats {
        double preprocess_ms = 0.0;   // building the Validate request buffers
        double wait_ms = 0.0;         // RPC in flight, proactive draft included
        double postprocess_ms = 0.0;  // acceptance + tree / KV fix-up
        int32_t prefill_cnt = 0;      // ValidateResponse.prefill for this batch
        int32_t num_accepted_tokens = 0;

        // Proactive draft, all zero/false when it is disabled.
        double proactive_ms = 0.0;    // spent inside ProactiveDraft::Draft()
        bool proactive_ran = false;   // a bet was placed this round
        bool proactive_hit = false;   // ...and the target agreed with it
        bool prev_proactive_hit = false;  // the previous round's outcome
        int32_t proactive_nodes = 0;  // subtree size, spliced or discarded
    };

    // Per-round draft-phase timings (milliseconds), one entry per tree
    // level. Both spans plus the residual account for nearly all of
    // draft.end_to_end; the remainder is the joint-budget selection and
    // Tree bookkeeping (tree_.add, TrimByBudget), all O(max_budget).
    //
    // There is no softmax or top-k span because there is no host softmax or
    // top-k: both are graph nodes inside the engine call, so their cost is
    // part of forward_ms. Logs written before that change also carry
    // draft.softmax / draft.topk.
    struct DraftStats {
        std::vector<double> forward_ms;  // engine_.forward_batch_topk
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

    // Hands out llama.cpp seq ids for branch forks, from the set no live
    // tree node is using. Never recycles mid-round (a pruned branch's cells
    // may still be needed if its prefix is accepted).
    //
    // A free list rather than a counter reset to 1, because a spliced
    // proactive subtree carries its branches -- and their sequence ids --
    // into the next round; those ids are still live and must not be handed
    // out again. RebuildSeqPool() re-derives the free set from the tree
    // after each acceptance.
    int32_t AllocSeq();
    void RebuildSeqPool();

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

    // Null unless proactive_type != kDisabled.
    std::unique_ptr<ProactiveDraft> proactive_;
    // Whether the previous round's bet was spliced in. Drives `included`
    // mode's draft-budget reduction, and is logged so a run can be scored
    // on hit rate.
    bool prev_proactive_hit_ = false;

    // Free list backing AllocSeq: seq_in_use_[s] covers both live tree
    // nodes and ids already handed out this round. Slot 0 is reserved for
    // the canonical committed sequence and never handed out.
    std::vector<uint8_t> seq_in_use_;
    int32_t seq_cursor_ = 1;

    // Shared, process-wide result sink (same lifecycle as the former linear
    // client's): truncated the first time this process opens it, appended
    // to thereafter.
    std::shared_ptr<std::ofstream> result_log_;
};

} // namespace specedge
