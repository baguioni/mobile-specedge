#include "oracle_validator.h"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace specedge {

OracleValidator::OracleValidator(
    std::vector<llama_token> reference,
    llama_token past_end_token,
    double sim_rtt_ms)
    : reference_(std::move(reference)),
      past_end_token_(past_end_token),
      sim_rtt_ms_(sim_rtt_ms) {
    if (reference_.empty()) {
        throw std::invalid_argument("OracleValidator: empty reference sequence");
    }
}

Validator::Result OracleValidator::Validate(
    int32_t /*client_idx*/,
    int32_t /*req_idx*/,
    const std::vector<llama_token>& input_ids,
    const std::vector<llama_pos>& position_ids,
    const std::vector<int32_t>& /*cache_seq_indices*/,
    const std::vector<float>& /*attention_mask*/,
    const std::vector<int32_t>& /*parent_indices*/,
    bool prefill,
    std::optional<std::string> /*prefix*/) {
    if (position_ids.size() != input_ids.size()) {
        throw std::invalid_argument("OracleValidator: inconsistent input sizes");
    }

    if (sim_rtt_ms_ > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sim_rtt_ms_));
    }

    Result result;
    result.selection.reserve(position_ids.size());
    for (llama_pos pos : position_ids) {
        const int64_t next = static_cast<int64_t>(pos) + 1;
        const bool in_range = next >= 0 && next < static_cast<int64_t>(reference_.size());
        result.selection.push_back(
            in_range ? reference_[static_cast<size_t>(next)] : past_end_token_);
    }
    // Alone on the target, the batch carries exactly this request's prefill
    // on its first round and nothing after -- what the server would report.
    result.prefill = prefill ? 1 : 0;
    return result;
}

} // namespace specedge
