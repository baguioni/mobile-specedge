#include "chain.h"

#include <algorithm>
#include <stdexcept>

namespace specedge {

Chain::Chain(std::vector<llama_token> prefix_tokens, int32_t max_len)
    : max_len_(max_len) {
    prefix_len_ = static_cast<int32_t>(prefix_tokens.size());

    if (prefix_len_ == 0) {
        throw std::invalid_argument("Chain: prefix_tokens must be non-empty");
    }
    if (prefix_len_ > max_len_) {
        throw std::invalid_argument("Chain: prefix_len exceeds max_len");
    }

    tokens_.reserve(max_len_);
    positions_.reserve(max_len_);

    tokens_ = std::move(prefix_tokens);
    positions_.resize(prefix_len_);
    for (int32_t i = 0; i < prefix_len_; ++i) {
        positions_[i] = static_cast<llama_pos>(i);
    }

    end_ = prefix_len_;
}

void Chain::add(llama_token token_id, llama_pos position) {
    if (end_ >= max_len_) {
        throw std::runtime_error("Chain::add: chain is already at max_len capacity");
    }
    if (position != positions_[end_ - 1] + 1) {
        throw std::invalid_argument(
            "Chain::add: position must be exactly one past the last token; "
            "a linear chain never branches");
    }

    tokens_.push_back(token_id);
    positions_.push_back(position);
    ++end_;
}

void Chain::truncate(int32_t new_end) {
    if (new_end < prefix_len_ || new_end > end_) {
        throw std::invalid_argument(
            "Chain::truncate: new_end must stay within [prefix_len, end]");
    }

    tokens_.resize(new_end);
    positions_.resize(new_end);
    end_ = new_end;
}

std::vector<float> Chain::attention_mask(int32_t from_index, int32_t count) const {
    if (from_index < 0 || count < 0 || from_index + count > end_) {
        throw std::invalid_argument("Chain::attention_mask: range out of bounds");
    }

    std::vector<float> mask(static_cast<size_t>(count) * static_cast<size_t>(max_len_), 0.0f);
    for (int32_t k = 0; k < count; ++k) {
        int32_t row_index = from_index + k;
        int32_t causal_len = std::min(row_index + 1, max_len_);
        std::fill_n(mask.begin() + static_cast<size_t>(k) * max_len_, causal_len, 1.0f);
    }
    return mask;
}

} // namespace specedge
