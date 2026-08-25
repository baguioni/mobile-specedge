#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "llama.h"
#include "specedge_grpc/specedge.grpc.pb.h"

namespace specedge {

// C++ port of grpc.py's GrpcClientController: talks to the SpecEdgeService
// Validate RPC that drives remote speculative-decoding validation.
//
// Differences from the Python version, by design (mirrors graph_engine.h):
//  - No torch::Tensor: request/response fields are plain std::vector<T>,
//    serialized to/from the proto's raw `bytes` fields the same way the
//    Python side packs/unpacks numpy buffers.
//  - Wire widths are fixed by the server contract, independent of the local
//    types llama.cpp uses: the server decodes input_ids, position_ids,
//    cache_seq_indices, and parent_indices as torch.long (int64_t), even
//    though llama.cpp's own llama_token/llama_pos are int32_t locally, and
//    attention_mask as a float tensor (the model's dtype), never an integer
//    type. Validate() upcasts the int32_t fields to int64_t and expects
//    attention_mask already as float before serializing, so its wire bytes
//    match what the server (and the Python client) actually send/expect.
//  - No device parameter: results land in host memory (std::vector); this
//    project doesn't route tensors through a GPU tensor library.
//  - Synchronous instead of grpc.aio: nothing else in this project runs an
//    async event loop (LlamaCppEngine is fully synchronous too), so
//    Validate() blocks on the underlying grpc::Channel like every other
//    call site here.
class GrpcClient {
public:
    struct ValidateResult {
        // Wire dtype is fixed by the server contract (torch.long on the
        // Python side), independent of whatever width input_ids etc. use.
        std::vector<int64_t> selection;
        bool prefill = false;
    };

    explicit GrpcClient(const std::string& host);

    // Mirrors GrpcClientController.client_idx in grpc.py: stored on the
    // client but not read by Validate() itself (Validate takes its own
    // client_idx argument), left here only for parity with callers that
    // may inspect/set it.
    int32_t client_idx = 0;

    // Throws std::invalid_argument if prefill is true and prefix is unset,
    // matching the Python ValueError. Throws std::runtime_error if the RPC
    // itself fails.
    //
    // attention_mask must already be in the server's configured model dtype
    // as far as bit width goes; this sends it as 32-bit float (fp32). If a
    // deployment configures the server with SPECEDGE_DTYPE=fp16 or bf16,
    // this needs to encode 16-bit elements instead - the wire format has no
    // dtype tag, so client and server widths must be arranged to match out
    // of band, same as on the Python side.
    ValidateResult Validate(
        int32_t client_idx,
        int32_t req_idx,
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& attention_mask,
        const std::vector<int32_t>& parent_indices,
        bool prefill = false,
        std::optional<std::string> prefix = std::nullopt);

private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<SpecEdgeService::Stub> stub_;
};

} // namespace specedge
