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
//    though llama.cpp's own llama_token/llama_pos are int32_t locally.
//    Validate() upcasts those int32_t fields to int64_t before serializing.
//    attention_mask is different: the server decodes it as its configured
//    model dtype (fp16/bf16/fp32) and the proto carries no dtype tag, so
//    the caller passes fp32 values and Validate() narrows them to
//    attention_mask_dtype (see below), which must be set to match the
//    server's base.dtype the same way the Python client's mask tensor is.
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
        // Number of prefill requests the server bundled into the batch that
        // served this call (ValidateResponse.prefill, i.e. grpc.py's
        // returned `prefill_cnt`). 0 on a pure decode round; kept as the
        // count rather than a bool so it matches specexec.py's
        // target.prefill result field.
        int32_t prefill = 0;
    };

    explicit GrpcClient(const std::string& host);

    // Mirrors GrpcClientController.client_idx in grpc.py: stored on the
    // client but not read by Validate() itself (Validate takes its own
    // client_idx argument), left here only for parity with callers that
    // may inspect/set it.
    int32_t client_idx = 0;

    // Wire encoding for the attention_mask bytes. The proto has no dtype
    // tag, so this must match the server's configured model dtype
    // (base.dtype / SPECEDGE_DTYPE on the server) out of band, exactly as
    // the Python client relies on its mask tensor already being in
    // config.dtype. Only attention_mask is affected -- the index fields go
    // out as torch.long (int64) regardless. Defaults to fp16, the server's
    // usual setting; set kFP32 for an fp32 server (e.g. mobile.example.yaml).
    enum class MaskDType { kFP32, kFP16, kBF16 };
    MaskDType attention_mask_dtype = MaskDType::kFP16;

    // Maps "fp32" / "fp16" / "bf16" (the server's base.dtype spelling) to
    // the matching MaskDType. Throws std::invalid_argument on anything else.
    static MaskDType ParseMaskDType(const std::string& name);

    // Throws std::invalid_argument if prefill is true and prefix is unset,
    // matching the Python ValueError. Throws std::runtime_error if the RPC
    // itself fails.
    //
    // attention_mask is passed as fp32 values regardless of the wire dtype;
    // Validate() narrows them to attention_mask_dtype (default fp16) before
    // serializing. That field must match the server's configured model
    // dtype - the wire format has no dtype tag, so the widths are arranged
    // to match out of band, same as on the Python side.
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
