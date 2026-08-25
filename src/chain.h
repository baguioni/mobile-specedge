#pragma once

#include <cstdint>
#include <vector>

#include "llama.h"

namespace specedge {

// C++ port of tree.py's Tree class, collapsed to the linear (non-tree) case
// enforced by max_n_beams == 1 && max_branch_width == 1 (see
// specexec-port.md, sections 1 and 3): one active beam, one child per
// drafted step, so the speculation "tree" is always a straight chain.
//
// Differences from Tree, by design:
//  - No `parents` array: parent(i) is always i-1 by construction (every
//    node has exactly one child), so it's never stored, just implied by
//    index.
//  - No `status` enum: it existed only to distinguish beam roles
//    (candidate/processed/pruned) during branch bookkeeping; with one beam
//    there's nothing left to distinguish.
//  - No stored `amask` matrix: Tree eagerly maintains an O(max_len^2)
//    attention-mask buffer so siblings can diverge from a shared parent
//    row. A chain never branches, so the causal mask is derived on demand
//    instead (see attention_mask()).
//  - No `logprobs`: they existed to compare/prune across sibling beams;
//    with a single path there's nothing to compare against.
//  - gather()/reorder_by_sequence() collapse into a single truncate():
//    rejection in the linear case always drops a contiguous suffix, never
//    a scattered subset, so no index-remapping (Tree::gather's
//    interim_indices dance) is needed.
class Chain {
public:
    // Mirrors Tree::_initialize_data: seeds positions [0, prefix_len) with
    // arange(prefix_len). Throws std::invalid_argument if prefix_tokens is
    // empty or longer than max_len.
    Chain(std::vector<llama_token> prefix_tokens, int32_t max_len);

    // Valid over [0, end()). Mirror Tree::tokens()/positions().
    const std::vector<llama_token>& tokens() const { return tokens_; }
    const std::vector<llama_pos>& positions() const { return positions_; }

    int32_t prefix_len() const { return prefix_len_; }
    int32_t end() const { return end_; }
    int32_t max_len() const { return max_len_; }

    // Appends one drafted token. Mirrors the add_size == 1 case of
    // Tree::add, minus the parent_indices/status/logprobs/amask bookkeeping
    // that only matters for branching. `position` must equal
    // positions().back() + 1 -- the linear invariant is exactly that no
    // caller can add a sibling instead of extending the one chain, unlike a
    // tree client. Throws std::invalid_argument if violated, or
    // std::runtime_error if the chain is already at max_len.
    void add(llama_token token_id, llama_pos position);

    // Mirrors Tree::gather's degenerate identity-mapped case (see
    // specexec-port.md section 6): drops everything past new_end. Since
    // rejection always cuts a contiguous suffix in the linear case, this is
    // a plain resize, never Tree::gather's index-permutation. new_end must
    // stay within [prefix_len, end()] -- rejection can shrink back to at
    // most the last accepted prefix, never below it. Throws
    // std::invalid_argument otherwise.
    void truncate(int32_t new_end);

    // Replaces reading Tree.amask directly: builds `count` rows of a causal
    // mask, each row max_len() wide, for chain positions
    // [from_index, from_index + count). Row k has 1s in columns
    // [0, from_index + k] and 0s elsewhere -- the same restriction Tree
    // stored eagerly per node, computed on demand here since there's no
    // per-node divergence to encode. Returned flattened row-major, ready
    // for GrpcClient::Validate's attention_mask argument. Throws
    // std::invalid_argument if the range falls outside [0, end()).
    std::vector<float> attention_mask(int32_t from_index, int32_t count) const;

private:
    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    int32_t prefix_len_ = 0;
    int32_t end_ = 0;
    int32_t max_len_ = 0;
};

} // namespace specedge
