#include "tree.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace specedge {

Tree::Tree(std::vector<llama_token> prefix_tokens, int32_t max_len) : max_len_(max_len) {
    prefix_len_ = static_cast<int32_t>(prefix_tokens.size());

    if (prefix_len_ == 0) {
        throw std::invalid_argument("Tree: prefix_tokens must be non-empty");
    }
    if (prefix_len_ > max_len_) {
        throw std::invalid_argument("Tree: prefix_len exceeds max_len");
    }

    tokens_.assign(static_cast<size_t>(max_len_), 0);
    positions_.assign(static_cast<size_t>(max_len_), 0);
    parents_.assign(static_cast<size_t>(max_len_), 0);
    status_.assign(static_cast<size_t>(max_len_), kPrompt);
    logprobs_.assign(static_cast<size_t>(max_len_), 0.0f);
    seq_ids_.assign(static_cast<size_t>(max_len_), 0);
    amask_.assign(static_cast<size_t>(max_len_) * static_cast<size_t>(max_len_), 0);

    // _initialize_data
    std::copy(prefix_tokens.begin(), prefix_tokens.end(), tokens_.begin());
    for (int32_t i = 0; i < prefix_len_; ++i) {
        positions_[i] = static_cast<llama_pos>(i);
    }
    for (int32_t i = 1; i < prefix_len_; ++i) {
        parents_[i] = i - 1;
    }
    status_[prefix_len_ - 1] = kCandidate;

    end_ = prefix_len_;
    init_attention_mask();
}

void Tree::init_attention_mask() {
    // _init_attention_mask: zero everything, causal tril over the prefix.
    std::fill(amask_.begin(), amask_.end(), uint8_t{0});
    for (int32_t r = 0; r < prefix_len_; ++r) {
        std::fill_n(amask_.begin() + static_cast<size_t>(r) * max_len_, r + 1, uint8_t{1});
    }
}

void Tree::set_status(int32_t slot, int32_t status) {
    if (slot < 0 || slot >= max_len_) {
        throw std::invalid_argument("Tree::set_status: slot out of range");
    }
    status_[slot] = status;
}

void Tree::set_prefix_len(int32_t prefix_len) {
    if (prefix_len < 1 || prefix_len > end_) {
        throw std::invalid_argument("Tree::set_prefix_len: prefix_len out of [1, end]");
    }
    prefix_len_ = prefix_len;
}

void Tree::add(
    const std::vector<llama_token>& token_ids,
    const std::vector<llama_pos>& token_positions,
    const std::vector<int32_t>& parent_indices,
    const std::vector<float>& logprobs,
    const std::vector<int32_t>& seq_ids,
    int32_t status) {
    const int32_t add_size = static_cast<int32_t>(token_ids.size());

    if (add_size > max_len_ - end_) {
        throw std::invalid_argument(
            "Tree::add: required addition size " + std::to_string(add_size) +
            " exceeds remaining tree capacity of " + std::to_string(max_len_ - end_));
    }
    if (token_positions.size() != token_ids.size() ||
        parent_indices.size() != token_ids.size() ||
        logprobs.size() != token_ids.size() ||
        seq_ids.size() != token_ids.size()) {
        throw std::invalid_argument("Tree::add: inconsistent sizes of input vectors");
    }

    for (int32_t k = 0; k < add_size; ++k) {
        const int32_t slot = end_ + k;
        const int32_t parent = parent_indices[k];
        // tree.py reads amask[parent] before writing the new rows, so parents
        // must already exist -- new nodes never parent each other.
        if (parent < 0 || parent >= end_) {
            throw std::invalid_argument(
                "Tree::add: parent slot " + std::to_string(parent) +
                " must reference an existing node (< " + std::to_string(end_) + ")");
        }

        tokens_[slot] = token_ids[k];
        positions_[slot] = token_positions[k];
        parents_[slot] = parent;
        status_[slot] = status;
        logprobs_[slot] = logprobs[k];
        seq_ids_[slot] = seq_ids[k];

        // amask row = parent's row + self bit. The parent's row is zero at
        // every column >= its own add-time end, so copying the full row is
        // exactly tree.py's amask_draft (parent rows over [0, end)) plus the
        // appended eye.
        std::memcpy(
            amask_.data() + static_cast<size_t>(slot) * max_len_,
            amask_.data() + static_cast<size_t>(parent) * max_len_,
            static_cast<size_t>(max_len_));
        amask_[static_cast<size_t>(slot) * max_len_ + slot] = 1;
    }

    end_ += add_size;
}

void Tree::gather(
    const std::vector<int32_t>& src_indices, const std::vector<int32_t>& dest_indices) {
    if (src_indices.empty()) {
        throw std::invalid_argument("Tree::gather: no source indices provided");
    }
    if (src_indices.size() != dest_indices.size()) {
        throw std::invalid_argument("Tree::gather: src/dest size mismatch");
    }
    for (size_t i = 0; i < src_indices.size(); ++i) {
        if (src_indices[i] < 0 || src_indices[i] >= end_) {
            throw std::invalid_argument("Tree::gather: src index out of [0, end)");
        }
        if (dest_indices[i] < 0 || dest_indices[i] >= max_len_) {
            throw std::invalid_argument("Tree::gather: dest index out of [0, max_len)");
        }
    }

    const size_t n = src_indices.size();

    // interim_indices: identity permutation with src slots remapped to dest.
    std::vector<int32_t> interim(static_cast<size_t>(end_));
    std::iota(interim.begin(), interim.end(), 0);
    for (size_t i = 0; i < n; ++i) {
        interim[src_indices[i]] = dest_indices[i];
    }

    // parents[src] = interim[parents[src]] -- read all before writing, like
    // the torch fancy assignment.
    std::vector<int32_t> remapped_parents(n);
    for (size_t i = 0; i < n; ++i) {
        remapped_parents[i] = interim[parents_[src_indices[i]]];
    }
    for (size_t i = 0; i < n; ++i) {
        parents_[src_indices[i]] = remapped_parents[i];
    }

    // data[dest] = data[src]: snapshot the source slices first so aliasing
    // src/dest slots cannot corrupt the move.
    std::vector<llama_token> t_tokens(n);
    std::vector<llama_pos> t_positions(n);
    std::vector<int32_t> t_parents(n), t_status(n), t_seq(n);
    std::vector<float> t_logprobs(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t s = src_indices[i];
        t_tokens[i] = tokens_[s];
        t_positions[i] = positions_[s];
        t_parents[i] = parents_[s];
        t_status[i] = status_[s];
        t_logprobs[i] = logprobs_[s];
        t_seq[i] = seq_ids_[s];
    }
    for (size_t i = 0; i < n; ++i) {
        const int32_t d = dest_indices[i];
        tokens_[d] = t_tokens[i];
        positions_[d] = t_positions[i];
        parents_[d] = t_parents[i];
        status_[d] = t_status[i];
        logprobs_[d] = t_logprobs[i];
        seq_ids_[d] = t_seq[i];
    }

    // amask rows first, then columns (tree.py's order: the column
    // permutation runs on the already row-permuted matrix).
    std::vector<uint8_t> row_tmp(n * static_cast<size_t>(max_len_));
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(
            row_tmp.data() + i * max_len_,
            amask_.data() + static_cast<size_t>(src_indices[i]) * max_len_,
            static_cast<size_t>(max_len_));
    }
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(
            amask_.data() + static_cast<size_t>(dest_indices[i]) * max_len_,
            row_tmp.data() + i * max_len_,
            static_cast<size_t>(max_len_));
    }
    std::vector<uint8_t> col_tmp(n);
    for (int32_t r = 0; r < max_len_; ++r) {
        uint8_t* row = amask_.data() + static_cast<size_t>(r) * max_len_;
        for (size_t i = 0; i < n; ++i) {
            col_tmp[i] = row[src_indices[i]];
        }
        for (size_t i = 0; i < n; ++i) {
            row[dest_indices[i]] = col_tmp[i];
        }
    }

    end_ = *std::max_element(dest_indices.begin(), dest_indices.end()) + 1;

    // amask[end:, :] = 0 and amask[:, end:] = 0.
    for (int32_t r = end_; r < max_len_; ++r) {
        std::memset(amask_.data() + static_cast<size_t>(r) * max_len_, 0, static_cast<size_t>(max_len_));
    }
    for (int32_t r = 0; r < end_; ++r) {
        std::memset(
            amask_.data() + static_cast<size_t>(r) * max_len_ + end_, 0,
            static_cast<size_t>(max_len_ - end_));
    }
}

void Tree::reorder_by_sequence(const std::vector<uint8_t>& seq_mask) {
    if (static_cast<int32_t>(seq_mask.size()) != end_) {
        throw std::invalid_argument("Tree::reorder_by_sequence: seq_mask must have end() entries");
    }

    std::vector<int32_t> seq_indices;
    for (int32_t i = 0; i < end_; ++i) {
        if (seq_mask[i]) {
            seq_indices.push_back(i);
        }
    }

    // The accepted set is an amask row: it always contains the full causal
    // prefix. reorder_by_sequence in tree.py silently assumes this; check it.
    for (int32_t i = 0; i < prefix_len_; ++i) {
        if (!seq_mask[i]) {
            throw std::invalid_argument(
                "Tree::reorder_by_sequence: seq_mask must cover the whole prefix");
        }
    }

    const int32_t new_len = static_cast<int32_t>(seq_indices.size());

    // if torch.any(seq_mask[prefix_len:]): compact accepted draft nodes to
    // slots [prefix_len, new_len). src is ascending and dest[i] <= src[i],
    // so the in-place forward loop never overwrites an unread source.
    if (new_len > prefix_len_) {
        for (int32_t i = prefix_len_; i < new_len; ++i) {
            const int32_t s = seq_indices[i];
            tokens_[i] = tokens_[s];
            positions_[i] = static_cast<llama_pos>(i);
            parents_[i] = i - 1;
            status_[i] = kGenerated;
            seq_ids_[i] = 0;
        }
    }

    end_ = new_len;
    prefix_len_ = new_len;
    for (int32_t i = 0; i < prefix_len_; ++i) {
        status_[i] = kPrompt;
        seq_ids_[i] = 0;
    }

    // logprobs[end:] and _data[:, end:] zeroed.
    for (int32_t i = end_; i < max_len_; ++i) {
        tokens_[i] = 0;
        positions_[i] = 0;
        parents_[i] = 0;
        status_[i] = kPrompt;
        logprobs_[i] = 0.0f;
        seq_ids_[i] = 0;
    }

    init_attention_mask();
}

uint8_t Tree::amask_at(int32_t row, int32_t col) const {
    if (row < 0 || row >= max_len_ || col < 0 || col >= max_len_) {
        throw std::invalid_argument("Tree::amask_at: index out of range");
    }
    return amask_[static_cast<size_t>(row) * max_len_ + col];
}

std::vector<float> Tree::attention_mask_rows(const std::vector<int32_t>& row_slots) const {
    std::vector<float> out(row_slots.size() * static_cast<size_t>(max_len_), 0.0f);
    for (size_t i = 0; i < row_slots.size(); ++i) {
        const int32_t r = row_slots[i];
        if (r < 0 || r >= end_) {
            throw std::invalid_argument("Tree::attention_mask_rows: row slot out of [0, end)");
        }
        const uint8_t* row = amask_.data() + static_cast<size_t>(r) * max_len_;
        float* dst = out.data() + i * static_cast<size_t>(max_len_);
        for (int32_t c = 0; c < max_len_; ++c) {
            dst[c] = static_cast<float>(row[c]);
        }
    }
    return out;
}

} // namespace specedge
