// Minimal CLI for SpecClient (spec_client.h): tokenizes a
// prompt, runs it through chain speculative decoding against a running
// SpecEdgeService target, and prints the result. Mirrors main.cpp's role
// for GraphEngine -- proves the client links and round-trips against a real
// server, not a unit test suite.
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"
#include "graph_engine.h"
#include "grpc_client.h"
#include "spec_client.h"

namespace {

struct Args {
    std::string model_path = "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf";
    std::string host = "localhost:50555";
    std::string prompt = "The capital of France is";
    int32_t max_len = 256;
    int32_t chain_len = 4;
    int32_t max_new_tokens = 32;
    int32_t client_idx = 0;
    int32_t req_idx = 0;
    int32_t n_gpu_layers = 0;
    std::optional<uint32_t> n_threads;
    std::optional<uint32_t> n_threads_batch;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model <path>            GGUF draft model path\n"
        "  --host <host:port>        SpecEdgeService target address (default: %s)\n"
        "  --prompt <text>           Prompt to complete\n"
        "  --max-len <n>             Context / max_len (default: 256)\n"
        "  --chain-len <n>           Tokens drafted per round (default: 4)\n"
        "  --max-new-tokens <n>      Tokens to generate (default: 32)\n"
        "  --n-gpu-layers <n>        Layers to offload to GPU, -1 for all (default: 0)\n"
        "  --n-threads <n>           Decode thread count\n"
        "  --n-threads-batch <n>     Batch thread count\n"
        "  -h, --help                Show this message\n",
        argv0, "localhost:50555");
}

bool parse_args(int argc, char** argv, Args& args) {
    auto next_value = [&](int& i) -> std::optional<std::string> {
        if (i + 1 >= argc) return std::nullopt;
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::optional<std::string> value;

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--model") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--model needs a value\n"); return false; }
            args.model_path = *value;
        } else if (arg == "--host") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--host needs a value\n"); return false; }
            args.host = *value;
        } else if (arg == "--prompt") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--prompt needs a value\n"); return false; }
            args.prompt = *value;
        } else if (arg == "--max-len") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--max-len needs a value\n"); return false; }
            args.max_len = std::stoi(*value);
        } else if (arg == "--chain-len") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--chain-len needs a value\n"); return false; }
            args.chain_len = std::stoi(*value);
        } else if (arg == "--max-new-tokens") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--max-new-tokens needs a value\n"); return false; }
            args.max_new_tokens = std::stoi(*value);
        } else if (arg == "--n-gpu-layers") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-gpu-layers needs a value\n"); return false; }
            args.n_gpu_layers = std::stoi(*value);
        } else if (arg == "--n-threads") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-threads needs a value\n"); return false; }
            args.n_threads = static_cast<uint32_t>(std::stoul(*value));
        } else if (arg == "--n-threads-batch") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-threads-batch needs a value\n"); return false; }
            args.n_threads_batch = static_cast<uint32_t>(std::stoul(*value));
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_bos) {
    int32_t n_needed = -llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, true);
    std::vector<llama_token> tokens(n_needed);
    int32_t n = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), add_bos, true);
    tokens.resize(n);
    return tokens;
}

// See spec_client.cpp's Detokenize() for why this needs a fallback:
// a misbehaving/test target can hand back a token id with no piece data,
// which llama.cpp reports as an uncaught std::out_of_range.
std::string detokenize(const llama_vocab* vocab, const std::vector<llama_token>& tokens) {
    try {
        std::vector<char> buf(tokens.size() * 8 + 16);
        int32_t n = llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
            static_cast<int32_t>(buf.size()), false, true);
        if (n < 0) {
            buf.resize(static_cast<size_t>(-n));
            n = llama_detokenize(
                vocab, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
                static_cast<int32_t>(buf.size()), false, true);
        }
        return std::string(buf.data(), n > 0 ? static_cast<size_t>(n) : 0);
    } catch (const std::exception& e) {
        std::string fallback = "<detokenize failed: " + std::string(e.what()) + "; token ids:";
        for (llama_token tok : tokens) {
            fallback += " " + std::to_string(tok);
        }
        return fallback + ">";
    }
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    specedge::LlamaCppEngine::Config engine_config;
    engine_config.model_path = args.model_path;
    engine_config.max_len = args.max_len;
    engine_config.max_n_beams = 1;
    engine_config.n_gpu_layers = args.n_gpu_layers;
    engine_config.n_threads = args.n_threads;
    engine_config.n_threads_batch = args.n_threads_batch;
    engine_config.role = "linear_client";

    specedge::LlamaCppEngine engine(engine_config);
    specedge::GrpcClient validator(args.host);

    std::vector<llama_token> prompt_tokens = tokenize(engine.vocab(), args.prompt, /*add_bos=*/true);
    if (static_cast<int32_t>(prompt_tokens.size()) + args.max_new_tokens > args.max_len) {
        std::fprintf(stderr,
            "Warning: %zu prompt tokens + %d generated tokens exceeds --max-len=%d; "
            "decode will fail once the context fills up.\n",
            prompt_tokens.size(), args.max_new_tokens, args.max_len);
    }

    specedge::SpecClient::Config client_config;
    client_config.chain_len = args.chain_len;
    client_config.max_new_tokens = args.max_new_tokens;
    client_config.client_idx = args.client_idx;

    specedge::SpecClient client(engine, validator, prompt_tokens, args.prompt, client_config);

    std::vector<llama_token> generated = client.Generate(args.req_idx);

    std::printf("Prompt: %s\n", args.prompt.c_str());
    std::printf("Completion: %s\n", detokenize(engine.vocab(), generated).c_str());
    return 0;
}
