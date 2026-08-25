// Smoke test / minimal CLI for GraphEngine (graph_engine.h): prefills a
// prompt, greedily decodes n_generate tokens through forward()/gather(), and
// prints the result. Not a unit test suite -- just enough to prove the CMake
// + FetchContent build links against real llama.cpp and that
// prefill/forward/gather/reset round-trip correctly.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"
#include "graph_engine.h"

namespace {

struct Args {
    std::string model_path = "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf";
    std::string prompt = "The capital of France is";
    int32_t max_len = 256;
    int32_t n_generate = 8;
    int32_t n_gpu_layers = 0;
    std::optional<uint32_t> n_threads;
    std::optional<uint32_t> n_threads_batch;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model <path>            GGUF model path (default: %s)\n"
        "  --prompt <text>           Prompt to complete (default: \"%s\")\n"
        "  --max-len <n>             Context / max_len passed to GraphEngine (default: 256)\n"
        "  --n-generate <n>          Number of tokens to greedily decode (default: 8)\n"
        "  --n-gpu-layers <n>        Layers to offload to GPU, -1 for all (default: 0)\n"
        "  --n-threads <n>           Decode thread count (default: llama.cpp's own default)\n"
        "  --n-threads-batch <n>     Batch thread count (default: same as --n-threads)\n"
        "  -h, --help                Show this message\n",
        argv0, "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf", "The capital of France is");
}

// Returns false (after printing usage) on --help or a parse error.
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
        } else if (arg == "--prompt") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--prompt needs a value\n"); return false; }
            args.prompt = *value;
        } else if (arg == "--max-len") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--max-len needs a value\n"); return false; }
            args.max_len = std::stoi(*value);
        } else if (arg == "--n-generate") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-generate needs a value\n"); return false; }
            args.n_generate = std::stoi(*value);
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

std::string detokenize(const llama_vocab* vocab, const std::vector<llama_token>& tokens) {
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
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    specedge::LlamaCppEngine::Config config;
    config.model_path = args.model_path;
    config.max_len = args.max_len;
    config.max_n_beams = 1;
    config.n_gpu_layers = args.n_gpu_layers;
    config.n_threads = args.n_threads;
    config.n_threads_batch = args.n_threads_batch;
    config.role = "smoke_test";

    specedge::LlamaCppEngine engine(config);
    const llama_vocab* vocab = engine.vocab();

    std::vector<llama_token> prompt_tokens = tokenize(vocab, args.prompt, /*add_bos=*/true);
    if (static_cast<int32_t>(prompt_tokens.size()) + args.n_generate > args.max_len) {
        std::fprintf(stderr,
            "Warning: %zu prompt tokens + %d generated tokens exceeds --max-len=%d; "
            "decode will fail once the context fills up.\n",
            prompt_tokens.size(), args.n_generate, args.max_len);
    }

    std::vector<llama_pos> positions(prompt_tokens.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<llama_pos>(i);
    }

    engine.prefill(prompt_tokens, positions, /*batch_idx=*/0);

    // The last prompt token is deferred to the first forward() call.
    llama_token next_token = prompt_tokens.back();
    llama_pos next_pos = static_cast<llama_pos>(prompt_tokens.size() - 1);

    std::vector<llama_token> generated;
    for (int32_t step = 0; step < args.n_generate; ++step) {
        std::vector<float> logits = engine.forward(next_token, next_pos, /*cache_batch_index=*/0,
                                                     /*cache_seq_index=*/next_pos);
        llama_token argmax = static_cast<llama_token>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        generated.push_back(argmax);

        next_pos = next_pos + 1;
        next_token = argmax;
    }

    // Exercise gather() as a prefix-truncation no-op, then reset() so the
    // engine is left ready for a fresh sequence.
    std::vector<int32_t> keep(engine.seq_len());
    for (int32_t i = 0; i < engine.seq_len(); ++i) {
        keep[i] = i;
    }
    engine.gather(keep, keep);
    engine.reset();

    std::fprintf(stderr, "n_vocab=%d, prompt_tokens=%zu\n", engine.n_vocab(), prompt_tokens.size());
    std::printf("Prompt: %s\n", args.prompt.c_str());
    std::printf("Completion: %s\n", detokenize(vocab, generated).c_str());
    return 0;
}
