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

} // namespace

GrpcClient::GrpcClient(const std::string& host)
    : channel_(grpc::CreateChannel(host, grpc::InsecureChannelCredentials())),
      stub_(SpecEdgeService::NewStub(channel_)) {}

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
    request.set_attention_mask(Encode(attention_mask));
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
