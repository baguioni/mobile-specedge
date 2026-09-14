#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "llama.h"

#include "graph_engine.h"
#include "tree.h"

namespace specedge {

// C++ port of proactive.py's SpecExecProactiveDraft.
//
// Run while the Validate RPC is in flight, this bets on what the target is
// about to return and pre-grows next round's tree from that bet. The bet is
// a single (leaf, token) pair: the frontier leaf most likely to be the last
// accepted node, and the token most likely to follow it -- i.e. the bonus
// token the target will hand back. If the bet is right, the subtree grown
// past it is spliced in as next round's starting tree
// (Tree::reorder_by_sequence_proactive) instead of being thrown away.
//
// The subtree is written to tree slots [end, ...) as scratch and `end` is
// restored before Draft() returns, so a losing bet costs the caller nothing
// but the time it ran in. Result::begin/end name the scratch range for the
// winning case.
//
// Differences from the Python version, by design:
//
//  - The bonus-token search asks the engine for one token per leaf, not
//    1024. The score is beam_score[i] + decay + logprob[i][j] and
//    beam_score is constant within a leaf, so the maximum always lands on
//    rank 0 of some leaf: k = 1 and k = 1024 select the identical pair.
//    That also removes the reference's need for a second, wider sampler --
//    the engine's existing top-k width is enough.
//
//  - Scoring a frontier leaf decodes it, and that cell is real. torch lets
//    proactive.py rewrite a slot it has already written; llama.cpp would
//    append a *second* cell at the same (seq, pos) for a leaf that already
//    has one, which llama_decode rejects for the whole batch with -1
//    ("invalid input batch", positions must satisfy Y = X + 1) -- losing
//    every bet, not just the offending leaf. FrontierLeaves() is therefore
//    shape-only again, at parity with _get_leaves_nodes: childless is
//    sufficient. ChooseBet is what handles an already-decoded (kProcessed)
//    childless leaf -- it forks a scratch sequence from the leaf's
//    *parent* and decodes the leaf's own token onto that scratch sequence
//    instead of its real one, exactly as it would for a never-decoded
//    leaf. The parent's seq is not simply "history minus this leaf's
//    cell": a node's first child inherits the parent's seq id outright
//    rather than forking, so that child's descendants keep extending the
//    same seq past the parent's position, and a *later* sibling's fork
//    would copy those cells too -- at the sibling's own position,
//    reproducing the identical conflict one hop removed. The fork is
//    therefore followed by a trim: seq_rm(scratch, leaf's pos, -1) removes
//    everything from the leaf's position onward on the copy only, leaving
//    exactly the root..parent chain. Both calls are cheap: seq_cp is a
//    tag-only copy and seq_rm only drops this seq's tag, under kv_unified
//    (see graph_engine.h) -- no KV data moves and no other sequence is
//    touched. The real leaf keeps its original seq id and status
//    untouched throughout, so a later bet win still forks the winning
//    branch from the leaf's true KV history. Every leaf this scores --
//    forked or not -- is marked kProcessed on its real slot afterward,
//    which is both true and what stops SpecExecClient's acceptance
//    backfill from decoding it twice.
//
//  - Branch forks are seq_cp, as in SpecExecClient::GrowTree; the caller
//    supplies the allocator so seq ids stay unique across both.
//
//  - _get_leaves_nodes returns absolute tree slots on both paths. The
//    reference's top-k branch returns indices relative to prefix_len while
//    its fallback returns absolute slots; only the fallback is correct.
class ProactiveDraft {
public:
    struct Config {
        int32_t max_n_beams = 0;      // leaves scored for the bet, and beams per level
        int32_t max_beam_len = 0;     // levels grown past the bonus token
        int32_t max_branch_width = 0; // children sampled per expanded beam
        int32_t max_budget = 0;       // node budget for the subtree
    };

    // Hands out llama.cpp sequence ids for branch forks. Shared with the
    // owning client so a subtree that survives keeps ids the next round's
    // drafting will not reuse.
    using SeqAllocator = std::function<int32_t()>;

    ProactiveDraft(
        LlamaCppEngine& engine, Tree& tree, SeqAllocator alloc_seq, Config config);

    // What the bet was and where the subtree landed.
    struct Result {
        int32_t leaf_slot = -1;      // the leaf bet on
        llama_token bonus_token = 0; // the token bet on following it
        int32_t begin = -1;          // scratch slot of the subtree root
        int32_t end = -1;            // one past its last node
        // Every branch of the subtree, for the engine-side keep set.
        std::vector<int32_t> seq_ids;
        // The subtree root's own sequence: it tags the whole root..bonus
        // path, so it addresses the committed prefix after a hit.
        int32_t root_seq_id = 0;
    };

    // Grows the subtree. Returns nullopt when there is nothing to bet on --
    // no frontier leaves, no room left in the tree, or the subtree failed
    // to grow past its root, in which case a hit would save nothing.
    // Leaves tree.end()/tree.prefix_len() as it found them either way.
    std::optional<Result> Draft();

    // Milliseconds spent inside the most recent Draft(), whatever it
    // returned; the caller logs this against the RPC wait it ran inside.
    double last_elapsed_ms() const { return last_elapsed_ms_; }

private:
    // Port of _get_best_bonus_token_candidate. Decodes the frontier leaves
    // (marking them kProcessed) and returns the best (leaf slot, token,
    // that leaf's top-1 log-probability) triple.
    struct Bet {
        int32_t leaf_slot = -1;
        llama_token token = 0;
    };
    std::optional<Bet> ChooseBet();

    // Port of _get_leaves_nodes: frontier nodes of the *current* draft
    // tree, capped at max_n_beams by cumulative logprob. Absolute slots.
    //
    // Frontier means childless, full stop -- shape only, matching the
    // Python reference. A childless node that is already kProcessed (its
    // children were pruned by an earlier budget/top-k pass) is still
    // forwarded; ChooseBet scores it via a scratch-sequence fork instead of
    // excluding it (see the note above the class).
    std::vector<int32_t> FrontierLeaves() const;

    LlamaCppEngine& engine_;
    Tree& tree_;
    SeqAllocator alloc_seq_;
    Config config_;
    double last_elapsed_ms_ = 0.0;
};

} // namespace specedge
