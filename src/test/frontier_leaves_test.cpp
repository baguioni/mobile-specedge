// Tier-1 sandbox for the ProactiveDraft frontier-leaf fix (see
// proactive_draft.cpp's FrontierLeaves): a real specedge::Tree, no
// llama.cpp engine, no GGUF model. Confirms the *bookkeeping* is right --
// that a childless node already marked kProcessed (its children pruned by
// an earlier budget/top-k pass) is now included in the frontier set,
// instead of silently dropped.
//
// This does NOT prove llama.cpp's KV cache actually accepts decoding such
// a leaf onto a scratch-forked sequence without the append-only conflict
// the fix exists to dodge -- that needs a real LlamaCppEngine with a real
// model loaded (ChooseBet's scratch-fork path), which is what a Tier-2
// sandbox would cover. This test only checks the shape decision.
//
// The selection loop below is a hand copy of FrontierLeaves()'s logic,
// not a call to the real (private) method: ProactiveDraft's constructor
// requires a live LlamaCppEngine (it queries tree_mode()/draft_top_k()),
// which this sandbox deliberately avoids. Keep this in sync by hand if
// FrontierLeaves()'s shape check ever changes.
//
// Links standalone, like tree_splice_test: Tree touches llama.h only for
// llama_token/llama_pos typedefs, so no libllama, no gRPC.
//   clang++ -std=c++17 -Isrc -I<llama include dir> \
//       src/test/frontier_leaves_test.cpp src/tree.cpp -o frontier_leaves_test

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tree.h"

using specedge::Tree;

namespace {

std::vector<int32_t> FrontierLeaves(const Tree& tree, int32_t max_n_beams) {
    const int32_t end = tree.end();
    const int32_t prefix = tree.prefix_len();

    std::vector<uint8_t> is_parent(static_cast<size_t>(end), 0);
    for (int32_t i = 0; i < end; ++i) {
        const int32_t p = tree.parents()[i];
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

    if (static_cast<int32_t>(leaves.size()) > max_n_beams) {
        std::stable_sort(leaves.begin(), leaves.end(), [&tree](int32_t a, int32_t b) {
            return tree.logprobs()[a] > tree.logprobs()[b];
        });
        leaves.resize(static_cast<size_t>(max_n_beams));
        std::sort(leaves.begin(), leaves.end());
    }
    return leaves;
}

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok:   %s\n", what);
    }
}

bool contains(const std::vector<int32_t>& v, int32_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

} // namespace

int main() {
    // Prefix "abc" (3 dummy tokens) -> prefix_len == 3, seed candidate at
    // slot 2, end() == 3.
    Tree tree({1, 2, 3}, /*max_len=*/32);

    // Round 1: grow two children off the seed (slot 2) -- slots 3, 4.
    tree.add({10, 11}, {3, 3}, {2, 2}, {-0.1f, -0.5f}, {0, 1});
    // end() == 5. Slot 2 now has children -- no longer a leaf.

    // GrowTree processes both, then decides slot 3 is worth expanding
    // further and slot 4 is not: slot 4's would-be children get pruned by
    // the budget/top-k trim before they're ever added. Slot 4 stays
    // childless *and* kProcessed -- exactly the case the fix targets.
    tree.set_status(3, Tree::kProcessed);
    tree.set_status(4, Tree::kProcessed);
    tree.add({20}, {4}, {3}, {-0.2f}, {0});
    // end() == 6.
    //   slot 3: has a child (slot 5) -> not a leaf.
    //   slot 4: childless, kProcessed -> the case under test.
    //   slot 5: childless, kCandidate -> the ordinary case.

    const std::vector<int32_t> leaves = FrontierLeaves(tree, /*max_n_beams=*/10);

    check(contains(leaves, 4), "childless + kProcessed leaf (slot 4) is included");
    check(contains(leaves, 5), "childless + kCandidate leaf (slot 5) is included");
    check(!contains(leaves, 2), "non-leaf (slot 2, has children) is excluded");
    check(!contains(leaves, 3), "non-leaf (slot 3, has children) is excluded");
    check(leaves.size() == 2, "frontier is exactly {4, 5}, nothing else");

    // Cap test: budget for one leaf only. Slot 5's logprob (-0.2) beats
    // slot 4's (-0.5), so slot 5 should win -- and slot 4's kProcessed
    // status must not have kept it out of consideration up front.
    const std::vector<int32_t> capped = FrontierLeaves(tree, /*max_n_beams=*/1);
    check(capped.size() == 1, "cap to max_n_beams=1 keeps exactly one leaf");
    check(contains(capped, 5), "cap keeps the higher-logprob leaf (slot 5, -0.2 > -0.5)");

    if (failures == 0) {
        std::fprintf(stderr, "\nAll checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d check(s) failed.\n", failures);
    return 1;
}
