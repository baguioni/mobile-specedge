// Tier-2 sandbox for the ProactiveDraft frontier-leaf fix (see
// proactive_draft.cpp's FrontierLeaves/ChooseBet and
// src/test/frontier_leaves_test.cpp for the Tier-1, no-engine version).
//
// Tier 1 only proved the bookkeeping: that a childless node already marked
// kProcessed is *included* in the frontier set. It used a hand copy of the
// selection loop and no real engine, so it could not prove llama.cpp's real
// KV cache actually tolerates ChooseBet's fix -- forking a scratch sequence
// from such a leaf's parent and decoding the leaf's own token onto it,
// instead of the append-only conflict ("invalid input batch", llama_decode
// returning -1) that a naive re-decode of an already-processed leaf would
// hit.
//
// This test proves that, against a real LlamaCppEngine and a real GGUF
// model: it hand-grows a small tree mode draft tree to reproduce the exact
// shape the fix targets (a childless node marked kProcessed with no
// children -- as if an earlier budget/top-k trim pruned them away), gives
// that node an artificially dominant cumulative logprob so ChooseBet's
// argmax is forced to bet on it deterministically, then calls
// ProactiveDraft::Draft() for real and checks it does not throw, does not
// silently swallow the failure as "no bet", and lands on exactly that leaf.
//
// host_topk=true is used deliberately: it routes forward_batch_topk()
// through host-side log-softmax + top-k over raw logits instead of the
// backend custom sampler (topk_sampler.h), which this host build has never
// exercised before. That keeps the test's signal on the KV-cache mechanism
// under test rather than on whether an unrelated backend op is wired up on
// this machine.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

#include "graph_engine.h"
#include "proactive_draft.h"
#include "tree.h"

using specedge::LlamaCppEngine;
using specedge::ProactiveDraft;
using specedge::Tree;

namespace {

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok:   %s\n", what);
    }
}

std::vector<llama_token> Tokenize(const llama_vocab* vocab, const std::string& text) {
    int32_t n_max = static_cast<int32_t>(text.size()) + 16;
    std::vector<llama_token> tokens(static_cast<size_t>(n_max));
    int32_t n = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n_max,
        /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
            static_cast<int32_t>(tokens.size()), /*add_special=*/true, /*parse_special=*/true);
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

#ifndef PROACTIVE_DRAFT_TEST_MODEL_PATH
#error "PROACTIVE_DRAFT_TEST_MODEL_PATH must be defined by the build"
#endif

} // namespace

int main() {
    LlamaCppEngine::Config cfg;
    cfg.model_path = PROACTIVE_DRAFT_TEST_MODEL_PATH;
    cfg.max_len = 64;
    cfg.max_n_beams = 4;
    cfg.max_seqs = 16;   // seq 0 (root) + headroom for forks below.
    cfg.n_gpu_layers = 0; // CPU only: fastest, most portable on this host.
    cfg.draft_top_k = 4;  // == the branch width used below.
    cfg.host_topk = true; // bypass the backend sampler; not under test here.
    cfg.role = "proactive_draft_test";

    LlamaCppEngine engine(cfg);
    check(engine.tree_mode(), "engine constructed in tree mode (max_seqs > 1)");

    const llama_vocab* vocab = engine.vocab();
    std::vector<llama_token> prompt = Tokenize(vocab, "The capital of France is");
    check(prompt.size() >= 2, "prompt tokenizes to at least 2 tokens");

    Tree tree(prompt, cfg.max_len);
    // Tree ctor seeds [0, prefix_len) causally; the last slot (prefix_len-1)
    // is the single seed kCandidate, on seq 0 (the root/committed sequence).
    const int32_t seed = tree.prefix_len() - 1;

    // ---- prefill: everything up to and including the seed ----
    {
        std::vector<llama_token> ids(tree.tokens().begin(), tree.tokens().begin() + seed + 1);
        std::vector<llama_pos> pos(
            tree.positions().begin(), tree.positions().begin() + seed + 1);
        engine.prefill(ids, pos, /*batch_idx=*/0);
    }

    int32_t next_seq = 1; // 0 is the root/committed sequence.
    auto alloc_seq = [&]() -> int32_t {
        if (next_seq >= engine.max_seqs()) {
            throw std::runtime_error("test: out of llama.cpp sequences");
        }
        return next_seq++;
    };

    // ---- level 1: decode the seed, branch 4 ways: A, B, C, D ----
    LlamaCppEngine::TopKRows top1 = engine.forward_batch_topk(
        {tree.tokens()[seed]}, {tree.positions()[seed]}, {tree.seq_ids()[seed]}, {seed});
    check(top1.k == cfg.draft_top_k, "level-1 forward_batch_topk returns draft_top_k rows");
    tree.set_status(seed, Tree::kProcessed);

    const int32_t level1_start = tree.end();
    {
        std::vector<llama_token> ids(4);
        std::vector<llama_pos> pos(4, tree.positions()[seed] + 1);
        std::vector<int32_t> parents(4, seed);
        std::vector<float> logprobs(4);
        std::vector<int32_t> seqs(4);
        for (int32_t k = 0; k < 4; ++k) {
            ids[static_cast<size_t>(k)] = top1.ids[static_cast<size_t>(k)];
            // A gets an ordinary (small, real) cumulative logprob. B, C, D
            // get artificially dominant ones -- log-probabilities are never
            // positive, so these guarantee ChooseBet's
            // leaf_logprob + decay + token_logprob argmax picks one of
            // B/C/D over anything grown from A, regardless of what the real
            // model actually scores each branch. This is what makes the
            // "does the fix's leaf get bet on" outcome deterministic
            // instead of a coin flip on real model probabilities.
            if (k == 0) {
                logprobs[static_cast<size_t>(k)] = -0.1f; // A
                seqs[static_cast<size_t>(k)] = tree.seq_ids()[seed]; // inherits seq 0
            } else {
                logprobs[static_cast<size_t>(k)] = 10.0f * static_cast<float>(k); // B,C,D
                int32_t s = alloc_seq();
                engine.seq_cp(tree.seq_ids()[seed], s);
                seqs[static_cast<size_t>(k)] = s;
            }
        }
        tree.add(ids, pos, parents, logprobs, seqs);
    }
    const int32_t slot_a = level1_start + 0;
    const int32_t slot_b = level1_start + 1;
    const int32_t slot_c = level1_start + 2;
    const int32_t slot_d = level1_start + 3; // highest logprob: 30.0

    // ---- level 2: decode all four, but only give A real children ----
    // Reproduces the exact shape ChooseBet's fix targets: B, C and D get
    // decoded (so they own a real KV cell and go kProcessed) but an
    // (simulated) budget/top-k trim drops every one of their candidate
    // children, leaving them childless AND kProcessed. A gets to keep 2
    // children, becoming an ordinary internal node with ordinary
    // childless-kCandidate leaves under it -- the control case.
    std::vector<llama_token> in_tokens = {
        tree.tokens()[slot_a], tree.tokens()[slot_b], tree.tokens()[slot_c],
        tree.tokens()[slot_d]};
    std::vector<llama_pos> in_pos = {
        tree.positions()[slot_a], tree.positions()[slot_b], tree.positions()[slot_c],
        tree.positions()[slot_d]};
    std::vector<int32_t> in_seqs = {
        tree.seq_ids()[slot_a], tree.seq_ids()[slot_b], tree.seq_ids()[slot_c],
        tree.seq_ids()[slot_d]};
    LlamaCppEngine::TopKRows top2 =
        engine.forward_batch_topk(in_tokens, in_pos, in_seqs, {slot_a, slot_b, slot_c, slot_d});
    check(top2.n_rows == 4, "level-2 forward_batch_topk decodes all four beams");
    tree.set_status(slot_a, Tree::kProcessed);
    tree.set_status(slot_b, Tree::kProcessed);
    tree.set_status(slot_c, Tree::kProcessed);
    tree.set_status(slot_d, Tree::kProcessed);
    // B, C, D: no tree.add() call follows -- their children are the ones
    // the simulated trim drops. Only A's top 2 continuations get added.

    const int32_t a_children_start = tree.end();
    {
        const int32_t stride = top2.k;
        std::vector<llama_token> ids = {
            top2.ids[static_cast<size_t>(0 * stride + 0)],
            top2.ids[static_cast<size_t>(0 * stride + 1)]};
        std::vector<llama_pos> pos(2, tree.positions()[slot_a] + 1);
        std::vector<int32_t> parents(2, slot_a);
        std::vector<float> logprobs = {
            -0.1f + top2.logprobs[static_cast<size_t>(0 * stride + 0)],
            -0.1f + top2.logprobs[static_cast<size_t>(0 * stride + 1)]};
        std::vector<int32_t> seqs = {tree.seq_ids()[slot_a], tree.seq_ids()[slot_a]};
        // Both inherit A's seq for simplicity; only one is ever decoded
        // again by anything below, so this does not need a second fork.
        tree.add(ids, pos, parents, logprobs, seqs);
    }
    check(
        tree.end() == a_children_start + 2,
        "tree grown to expected size: seed + 4 (A,B,C,D) + 2 (A's children)");

    // Sanity check on the shape before handing off to ProactiveDraft: B, C,
    // D must be childless. (parents() scan mirrors FrontierLeaves' own
    // is_parent pass.)
    {
        std::vector<uint8_t> is_parent(static_cast<size_t>(tree.end()), 0);
        for (int32_t i = 0; i < tree.end(); ++i) {
            int32_t p = tree.parents()[i];
            if (p >= 0 && p < tree.end()) is_parent[static_cast<size_t>(p)] = 1;
        }
        check(!is_parent[static_cast<size_t>(slot_d)], "slot D is childless before ProactiveDraft::Draft()");
        check(tree.status()[slot_d] == Tree::kProcessed, "slot D is kProcessed before ProactiveDraft::Draft()");
    }

    // ---- the actual thing under test ----
    ProactiveDraft::Config pd_cfg;
    pd_cfg.max_n_beams = 10;    // generous: keep every leaf in scope, so the
                                // artificial logprobs alone decide the bet.
    pd_cfg.max_beam_len = 2;
    pd_cfg.max_branch_width = 4; // must be <= engine.draft_top_k()
    pd_cfg.max_budget = 8;

    ProactiveDraft proactive(engine, tree, alloc_seq, pd_cfg);

    bool threw = false;
    std::string thrown_what;
    std::optional<ProactiveDraft::Result> result;
    try {
        result = proactive.Draft();
    } catch (const std::exception& e) {
        threw = true;
        thrown_what = e.what();
    }

    check(!threw, threw ? ("Draft() threw: " + thrown_what).c_str()
                         : "Draft() did not throw (no llama_decode -1 / KV-cache conflict)");
    check(result.has_value(), "Draft() returned a bet (not nullopt)");

    if (result.has_value()) {
        check(
            result->leaf_slot == slot_d,
            "the bet landed on slot D -- the childless+kProcessed leaf the fix targets");
        check(result->end > result->begin + 1, "a subtree was actually grown past the bet root");
        std::fprintf(
            stderr, "    bet: leaf_slot=%d bonus_token=%d subtree=[%d,%d) seq_ids=%zu\n",
            result->leaf_slot, result->bonus_token, result->begin, result->end,
            result->seq_ids.size());
    }

    engine.close();

    if (failures == 0) {
        std::fprintf(stderr, "\nAll checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d check(s) failed.\n", failures);
    return 1;
}
