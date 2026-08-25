// Scratch parity-test executable: sends a small, known Validate request
// through the C++ GrpcClient port and prints what comes back, so it can be
// compared against run_python_client.py sending the "same" logical request
// through the Python GrpcClientController — both against mock_server.py.
#include <iostream>

#include "grpc_client.h"

int main() {
    specedge::GrpcClient client("localhost:50555");

    std::vector<llama_token> input_ids = {101, 102, 103};
    std::vector<llama_pos> position_ids = {0, 1, 2};
    std::vector<int32_t> cache_seq_indices = {0, 1, 2};
    std::vector<int32_t> parent_indices = {0, 1, 2};
    std::vector<float> attention_mask(24, 1.0f); // (1,3,8) matching python test's mask shape

    std::cout << "--- C++ client: sending request ---\n";
    try {
        auto result = client.Validate(
            /*client_idx=*/1,
            /*req_idx=*/1,
            input_ids,
            position_ids,
            cache_seq_indices,
            attention_mask,
            parent_indices,
            /*prefill=*/false);

        std::cout << "--- C++ client: response selection=[";
        for (size_t i = 0; i < result.selection.size(); ++i) {
            std::cout << result.selection[i] << (i + 1 < result.selection.size() ? "," : "");
        }
        std::cout << "] prefill=" << result.prefill << " ---\n";
    } catch (const std::exception& e) {
        std::cout << "--- C++ client: EXCEPTION: " << e.what() << " ---\n";
    }

    return 0;
}
