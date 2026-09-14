#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"

#include "validator.h"

namespace specedge {

// A target stand-in that answers from a completion an earlier run recorded
// (trace.jsonl's prompt_tokens + output_tokens), so the draft side can be
// re-run on the phone alone: no server, no network, and every draft backend
// judged against the same continuation.
//
// The answer for an input row at position p is reference[p + 1] -- the token
// the target committed after position p. A draft node is accepted when its
// token equals the target's choice at its parent, i.e. reference[pos], so a
// path is accepted iff it spells the reference, and the bonus token is the
// reference token after the deepest such node. Rows off the reference path
// get the same lookup, but they only ever judge nodes whose ancestors have
// already failed, so their answer cannot change the outcome.
//
// Consequences:
//  - Replaying the run that produced the trace, with a deterministic draft,
//    reproduces its per-round num_accepted_tokens exactly -- whatever the
//    target's temperature was, because the reference *is* what the target
//    chose. That is the tool's self-check.
//  - Replaying a different draft backend against the same trace measures
//    that backend on the same text. A live run cannot: the target is not
//    bit-stable across tree shapes, so live runs of two backends drift onto
//    different continuations.
class OracleValidator : public Validator {
public:
    // reference is the prompt tokens followed by the recorded output tokens.
    // past_end_token answers a row whose p + 1 falls past the reference --
    // a replayed tree reaching deeper than the recorded run did on its final
    // round. Pass the vocab's EOS so generation stops there.
    //
    // sim_rtt_ms > 0 sleeps inside Validate() to stand in for the network
    // round trip: the window a proactive draft runs in, and the idle gap
    // that lets the draft device's clocks drop between rounds.
    OracleValidator(
        std::vector<llama_token> reference,
        llama_token past_end_token,
        double sim_rtt_ms = 0.0);

    Result Validate(
        int32_t client_idx,
        int32_t req_idx,
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& attention_mask,
        const std::vector<int32_t>& parent_indices,
        bool prefill = false,
        std::optional<std::string> prefix = std::nullopt) override;

private:
    std::vector<llama_token> reference_;
    llama_token past_end_token_;
    double sim_rtt_ms_;
};

} // namespace specedge
