#include "grpc_client.h"

#include <cstring>
#include <stdexcept>

namespace specedge {

namespace {

template <typename T>
std::string Encode(const std::vector<T>& values) {
    return std::string(
        reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}

// The server decodes index fields as torch.long (int64_t) regardless of the
// narrower width llama.cpp uses locally (llama_token/llama_pos/int32_t), so
// every index field must be widened before it hits the wire.
template <typename WireT, typename SrcT>
std::string EncodeWiden(const std::vector<SrcT>& values) {
    std::vector<WireT> widened(values.begin(), values.end());
    return Encode(widened);
}

template <typename T>
std::vector<T> Decode(const std::string& bytes) {
    if (bytes.size() % sizeof(T) != 0) {
        throw std::runtime_error(
            "GrpcClient::Decode: byte buffer size is not a multiple of the element size");
    }
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

// attention_mask always arrives as fp32 values; the server decodes it as
// its configured model dtype, so narrow to that width here (the proto has
// no dtype tag). ggml_fp32_to_{fp16,bf16}_row give the same rounding as the
// torch cast the Python client's mask tensor went through.
std::string EncodeMask(
    const std::vector<float>& mask, GrpcClient::MaskDType dtype) {
    switch (dtype) {
        case GrpcClient::MaskDType::kFP16: {
            std::vector<ggml_fp16_t> half(mask.size());
            if (!mask.empty()) {
                ggml_fp32_to_fp16_row(
                    mask.data(), half.data(), static_cast<int64_t>(mask.size()));
            }
            return Encode(half);
        }
        case GrpcClient::MaskDType::kBF16: {
            std::vector<ggml_bf16_t> bf(mask.size());
            if (!mask.empty()) {
                ggml_fp32_to_bf16_row(
                    mask.data(), bf.data(), static_cast<int64_t>(mask.size()));
            }
            return Encode(bf);
        }
        case GrpcClient::MaskDType::kFP32:
            break;
    }
    return Encode(mask);
}

} // namespace

GrpcClient::GrpcClient(const std::string& host)
    : channel_(grpc::CreateChannel(host, grpc::InsecureChannelCredentials())),
      stub_(SpecEdgeService::NewStub(channel_)) {}

// Mirrors util.convert_dtype on the Python side.
GrpcClient::MaskDType GrpcClient::ParseMaskDType(const std::string& name) {
    if (name == "fp32") return MaskDType::kFP32;
    if (name == "fp16") return MaskDType::kFP16;
    if (name == "bf16") return MaskDType::kBF16;
    throw std::invalid_argument(
        "GrpcClient::ParseMaskDType: expected fp32, fp16, or bf16, got '" + name + "'");
}

GrpcClient::ValidateResult GrpcClient::Validate(
    int32_t client_idx,
    int32_t req_idx,
    const std::vector<llama_token>& input_ids,
    const std::vector<llama_pos>& position_ids,
    const std::vector<int32_t>& cache_seq_indices,
    const std::vector<float>& attention_mask,
    const std::vector<int32_t>& parent_indices,
    bool prefill,
    std::optional<std::string> prefix) {
    if (prefill && !prefix.has_value()) {
        throw std::invalid_argument("Prefix must be provided for prefill requests.");
    }

    ValidateRequest request;
    request.set_client_idx(client_idx);
    request.set_req_idx(req_idx);
    request.set_input_ids(EncodeWiden<int64_t>(input_ids));
    request.set_position_ids(EncodeWiden<int64_t>(position_ids));
    request.set_cache_seq_indices(EncodeWiden<int64_t>(cache_seq_indices));
    request.set_parent_indices(EncodeWiden<int64_t>(parent_indices));
    request.set_attention_mask(EncodeMask(attention_mask, attention_mask_dtype));
    request.set_prefill(prefill);
    if (prefix.has_value()) {
        request.set_prefix(*prefix);
    }

    grpc::ClientContext context;
    ValidateResponse response;
    grpc::Status status = stub_->Validate(&context, request, &response);
    if (!status.ok()) {
        throw std::runtime_error("Validate RPC failed: " + status.error_message());
    }

    ValidateResult result;
    result.selection = Decode<int64_t>(response.selection());
    result.prefill = response.prefill();
    return result;
}

} // namespace specedge
