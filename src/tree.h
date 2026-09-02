#pragma once

#include <cstdint>
#include <vector>

#include "llama.h"

namespace specedge {

// C++ port of tree.py's Tree class: the client-side draft tree for
// tree-based (SpecExec) speculative decoding.
//
// A node lives at a "slot" -- its index into the flat per-node arrays. Slots
// are the tree's (and the remote target's) addressing scheme: the Validate
// RPC's cache_seq_indices / parent_indices and the attention-mask columns are
// all slot indices, and the server writes each forwarded row's KV into
// exactly that slot of its per-client cache.
//
// Differences from tree.py, by design:
//  - No torch: flat std::vectors preallocated to max_len, mirroring the
//    Python file's preallocated buffers.
//  - amask is uint8_t 0/1 (max_len x max_len, row-major) instead of a
//    model-dtype tensor; attention_mask_rows() expands the requested rows to
//    float for the RPC wire format.
//  - seq_ids: one extra per-slot array tree.py does not have -- the
//    llama.cpp bridge, see below.
//
// Slot -> (seq set, pos) mapping
// ------------------------------
// The local draft engine is llama.cpp, whose KV cells are addressed by
// (seq_id, pos), never by slot, and whose cells cannot be moved or
// overwritten in place. Two of the coordinates are stored per slot:
//   - positions()[slot] : the cell's pos (prompt length + tree depth).
//   - seq_ids()[slot]   : the node's *primary* sequence -- the llama_seq_id
//                         its token was (or will be) decoded under. One seq
//                         per root-to-leaf branch: the first child of a
//                         branching node inherits the parent's seq, every
//                         sibling gets a fresh seq forked with
//                         LlamaCppEngine::seq_cp (a tag-only copy of the
//                         whole root..parent path under kv_unified).
// The full seq *set* tagging a node's KV cell is implicit: it is the set of
// primary seqs of every node in the slot's subtree, because each fork
// seq_cp's the entire root..parent path. It never needs materializing --
// forwarding a node uses its primary seq, and the post-acceptance collapse
// only needs the primary seq of the deepest accepted node.
//
// gather() moves seq_ids in lockstep with the other per-slot arrays, so the
// mapping survives slot permutation (budget trims). reorder_by_sequence()
// resets it to seq 0, matching the engine-side collapse of the accepted path
// onto the canonical sequence at the end of every round.
// reorder_by_sequence_proactive() is the exception: a hit keeps several
// branches alive, so it preserves the surviving nodes' seq ids instead.
class Tree {
public:
    // Mirrors tree.py's status constants; ordering matters. The Validate row
    // selection uses `status >= kProcessed && status < kPostCandidate`: the
    // kPost* pair marks nodes the proactive draft speculated past the bonus
    // token, which belong to *next* round's tree and must never be shipped
    // to the target with this round's.
    enum Status : int32_t {
        kPrompt = 0,
        kGenerated = 5,
        kProcessed = 10,
        kCandidate = 15,
        kPostCandidate = 20,
        kPostProcessed = 25,
    };

    // Mirrors Tree._initialize_data: seeds slots [0, prefix_len) as a causal
    // chain (positions arange, parents i-1, statuses PROMPT except the last,
    // which is the first CANDIDATE). Throws std::invalid_argument if
    // prefix_tokens is empty or longer than max_len.
    Tree(std::vector<llama_token> prefix_tokens, int32_t max_len);

    const std::vector<llama_token>& tokens() const { return tokens_; }
    const std::vector<llama_pos>& positions() const { return positions_; }
    const std::vector<int32_t>& parents() const { return parents_; }
    const std::vector<int32_t>& status() const { return status_; }
    const std::vector<float>& logprobs() const { return logprobs_; }
    const std::vector<int32_t>& seq_ids() const { return seq_ids_; }

    int32_t prefix_len() const { return prefix_len_; }
    int32_t end() const { return end_; }
    int32_t max_len() const { return max_len_; }

    void set_status(int32_t slot, int32_t status);
    // Mirrors the Python client assigning tree.prefix_len directly after
    // appending the bonus token.
    void set_prefix_len(int32_t prefix_len);

    // Mirrors the Python proactive draft assigning tree.end directly to
    // unwind its scratch growth. Slots [end, max_len) hold whatever the
    // proactive draft last wrote there; ProactiveDraft is the only caller,
    // and it keeps the bounds of that live scratch region itself. Must not
    // be used to grow `end` over slots that were never written by add().
    void set_end(int32_t end);

    // Mirrors Tree.add: appends the nodes at slots [end, end + add_size).
    // parent_indices must reference existing slots (< end); each new node's
    // amask row is its parent's row plus its own diagonal bit -- the
    // amask_draft/eye construction in tree.py. Throws std::invalid_argument
    // on inconsistent sizes or capacity overflow.
    void add(
        const std::vector<llama_token>& token_ids,
        const std::vector<llama_pos>& token_positions,
        const std::vector<int32_t>& parent_indices,
        const std::vector<float>& logprobs,
        const std::vector<int32_t>& seq_ids,
        int32_t status = kCandidate);

    // Mirrors Tree.gather: moves the nodes at src slots to dest slots
    // (reading every source before writing, like torch fancy assignment),
    // remaps parents through the permutation, permutes amask rows then
    // columns, and shrinks end to max(dest) + 1, zeroing everything beyond.
    void gather(
        const std::vector<int32_t>& src_indices,
        const std::vector<int32_t>& dest_indices);

    // Mirrors Tree.reorder_by_sequence: compacts the accepted path (the set
    // slots of seq_mask, which must cover the whole prefix) to the front,
    // relinearizes positions/parents, promotes everything to prefix, and
    // resets amask to causal. seq_ids reset to 0: after the engine-side
    // collapse every committed cell lives on the canonical seq 0.
    // seq_mask must have end() entries.
    void reorder_by_sequence(const std::vector<uint8_t>& seq_mask);

    // Mirrors specexec.py's _reorder_by_sequence_proactive: the acceptance
    // path taken when the proactive draft's bet was right. Compacts the
    // accepted path to the front exactly as reorder_by_sequence does, then
    // moves the speculative subtree at slots [pro_begin, pro_end) in behind
    // it, remapping its parents through the same permutation and promoting
    // kPost* back to their live equivalents.
    //
    // Unlike reorder_by_sequence this preserves seq ids: the subtree's
    // branches are still live llama.cpp sequences, and the committed prefix
    // is retagged onto `prefix_seq_id` (any sequence tagging the whole
    // root..bonus path -- the subtree root's own is the natural choice).
    //
    // The speculative nodes keep their positions unchanged, which is only
    // sound because the accepted path's positions are already dense and
    // equal to their new slots; that is checked, not assumed.
    //
    // seq_mask must have end() entries and cover the whole prefix, and
    // pro_begin must be the subtree root -- a node whose parent is inside
    // the accepted set and at or beyond the old prefix.
    void reorder_by_sequence_proactive(
        const std::vector<uint8_t>& seq_mask,
        int32_t pro_begin,
        int32_t pro_end,
        int32_t prefix_seq_id);

    uint8_t amask_at(int32_t row, int32_t col) const;

    // Expands amask rows `row_slots` to floats (row-major,
    // row_slots.size() x max_len) -- the shape Validate's attention_mask
    // wire field carries.
    std::vector<float> attention_mask_rows(const std::vector<int32_t>& row_slots) const;

private:
    void init_attention_mask();

    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    std::vector<int32_t> parents_;
    std::vector<int32_t> status_;
    std::vector<float> logprobs_;
    std::vector<int32_t> seq_ids_;
    std::vector<uint8_t> amask_;  // max_len * max_len, row-major, 0/1

    int32_t prefix_len_ = 0;
    int32_t end_ = 0;
    int32_t max_len_ = 0;
};

} // namespace specedge
