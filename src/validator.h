#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"

namespace specedge {

// What SpecExecClient needs from a target: one call that takes a round's
// draft tree and returns the target's next-token choice for every input row.
// GrpcClient is the real one (the remote SpecEdgeService); OracleValidator
// answers from a recorded completion instead, so the draft side can be
// replayed on the phone with no server (see oracle_validator.h).
//
// SpecExecClient calls Validate() from a std::async worker when a proactive
// draft is enabled, so an implementation must not touch the engine or the
// tree -- only the request buffers it is handed.
class Validator {
public:
    struct Result {
        // selection[j] is the target's next token after input row j. Wire
        // dtype is fixed by the server contract (torch.long on the Python
        // side), independent of whatever width input_ids etc. use.
        std::vector<int64_t> selection;
        // Number of prefill requests the server bundled into the batch that
        // served this call (ValidateResponse.prefill, i.e. grpc.py's
        // returned `prefill_cnt`). 0 on a pure decode round; kept as the
        // count rather than a bool so it matches specexec.py's
        // target.prefill result field.
        int32_t prefill = 0;
    };

    virtual ~Validator() = default;

    // Argument contract as GrpcClient::Validate documents it.
    virtual Result Validate(
        int32_t client_idx,
        int32_t req_idx,
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& attention_mask,
        const std::vector<int32_t>& parent_indices,
        bool prefill = false,
        std::optional<std::string> prefix = std::nullopt) = 0;
};

} // namespace specedge
