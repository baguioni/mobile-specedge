// Config-driven batch client. C++ port of specedge/src/script/client.py.
//
// Where client_main.cpp runs a single --prompt one-shot, this reads a YAML
// config holding every client-side option (draft model, target host,
// max_len, decoding params, dataset selection) and then walks the entire
// dataset: it builds the request-index list the same way client.py does
// (offset slice -> stride subsample -> seeded shuffle) and feeds each prompt
// through tree-based (SpecExec) speculative decoding against a running
// SpecEdgeService target, one request at a time.
//
// Differences from client.py, by design:
//  - Config comes from a YAML file parsed here, not from environment
//    variables populated by an external launcher (config.h's env-var path is
//    left untouched for the other executables). Every option lives directly
//    under a single `client:` key.
//  - Datasets are always read from ./data (the project layout), so there is
//    no configurable data directory.
//  - No stub.Sync() handshake before the loop: GrpcClient exposes only
//    Validate(); the first Validate() of each request carries prefill=true,
//    which is what the server keys off.
//  - Shuffle uses std::mt19937 seeded with client_idx, so the visiting
//    order is deterministic per client but not bit-identical to CPython's
//    random.shuffle().
//  - reasoning: llama_chat_apply_template() has no thinking toggle, so the
//    flag is accepted but only gates whether the chat template is applied
//    at all (parity with client.py applying a template for specbench).
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "llama.h"

#include "graph_engine.h"
#include "grpc_client.h"
#include "spec_exec_client.h"

namespace {

// Every field is read from a single `client:` mapping in the YAML.
struct ClientConfig {
    // model / engine
    std::string draft_model;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu = 0;
    std::optional<uint32_t> n_threads;
    std::optional<uint32_t> n_threads_batch;

    // target + decoding (SpecExec drafting parameters, see
    // spec_exec_client.h)
    std::string host = "localhost:50555";
    int32_t max_len = 2048;
    int32_t max_n_beams = 4;
    int32_t max_beam_len = 8;
    int32_t max_branch_width = 2;
    int32_t max_budget = 16;
    int32_t max_seqs = 0;  // llama.cpp sequences; 0 = derive from the above
    int32_t max_new_tokens = 64;
    int32_t client_idx = 0;

    // dataset selection (mirrors client.py's config fields); the dataset
    // file is always looked up under ./data.
    std::string dataset = "mtbench";
    bool reasoning = false;
    int32_t max_request_num = -1;  // -1 -> whole dataset
    int32_t req_offset = 0;
    int32_t sample_req_cnt = 1;    // take every Nth prompt after the offset
};

// Datasets always live here, relative to the working directory.
constexpr const char* kDataDir = "data";

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [--config <path>]\n"
        "  --config, -c <path>   YAML client config (default: config/client.example.yaml)\n"
        "  -h, --help            Show this message\n",
        argv0);
}

template <typename T>
T node_or(const YAML::Node& n, const T& fallback) {
    return (n && !n.IsNull()) ? n.as<T>() : fallback;
}

ClientConfig load_config(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    ClientConfig c;

    const YAML::Node cl = root["client"];
    if (!cl || !cl.IsMap()) {
        throw std::runtime_error("config: top-level `client:` mapping is required");
    }

    c.draft_model = node_or<std::string>(cl["draft_model"], c.draft_model);
    c.n_gpu_layers = node_or<int32_t>(cl["n_gpu_layers"], c.n_gpu_layers);
    c.main_gpu = node_or<int32_t>(cl["main_gpu"], c.main_gpu);
    if (cl["n_threads"] && !cl["n_threads"].IsNull()) {
        c.n_threads = cl["n_threads"].as<uint32_t>();
    }
    if (cl["n_threads_batch"] && !cl["n_threads_batch"].IsNull()) {
        c.n_threads_batch = cl["n_threads_batch"].as<uint32_t>();
    }

    c.host = node_or<std::string>(cl["host"], c.host);
    c.max_len = node_or<int32_t>(cl["max_len"], c.max_len);
    c.max_n_beams = node_or<int32_t>(cl["max_n_beams"], c.max_n_beams);
    c.max_beam_len = node_or<int32_t>(cl["max_beam_len"], c.max_beam_len);
    c.max_branch_width = node_or<int32_t>(cl["max_branch_width"], c.max_branch_width);
    c.max_budget = node_or<int32_t>(cl["max_budget"], c.max_budget);
    c.max_seqs = node_or<int32_t>(cl["max_seqs"], c.max_seqs);
    c.max_new_tokens = node_or<int32_t>(cl["max_new_tokens"], c.max_new_tokens);
    c.client_idx = node_or<int32_t>(cl["client_idx"], c.client_idx);

    c.dataset = node_or<std::string>(cl["dataset"], c.dataset);
    c.reasoning = node_or<bool>(cl["reasoning"], c.reasoning);
    c.max_request_num = node_or<int32_t>(cl["max_request_num"], c.max_request_num);
    c.req_offset = node_or<int32_t>(cl["req_offset"], c.req_offset);
    c.sample_req_cnt = node_or<int32_t>(cl["sample_req_cnt"], c.sample_req_cnt);

    if (c.draft_model.empty()) {
        throw std::runtime_error("config: model.draft_model is required");
    }
    if (c.max_n_beams < 1 || c.max_beam_len < 1 || c.max_branch_width < 1 ||
        c.max_budget < 1) {
        throw std::runtime_error(
            "config: max_n_beams, max_beam_len, max_branch_width and "
            "max_budget must all be >= 1");
    }
    if (c.sample_req_cnt < 1) {
        throw std::runtime_error("config: dataset.sample_req_cnt must be >= 1");
    }
    if (c.max_request_num < -1) {
        throw std::runtime_error("config: dataset.max_request_num must be -1 or >= 0");
    }
    return c;
}

// Upper bound on llama.cpp sequences a round can fork: every draft step can
// split each expanded beam into (width - 1) fresh branches, plus the
// canonical seq 0. Clamped to llama.cpp's LLAMA_MAX_SEQ.
int32_t derive_max_seqs(const ClientConfig& cfg) {
    const int64_t forks = 1 +
        static_cast<int64_t>(cfg.max_beam_len) * cfg.max_n_beams *
            (cfg.max_branch_width - 1);
    return static_cast<int32_t>(std::clamp<int64_t>(forks, 2, 256));
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

// Same fallback rationale as client_main.cpp: a misbehaving/test target can
// return a token id with no piece data, which llama.cpp raises as an
// uncaught std::out_of_range.
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

// Wraps a single user turn with the GGUF's built-in chat template, the way
// util.load_dataset() calls tokenizer.apply_chat_template() for specbench.
std::string apply_chat_template(const llama_model* model, const std::string& user_msg) {
    const char* tmpl = llama_model_chat_template(model, /*name=*/nullptr);
    if (tmpl == nullptr) {
        return user_msg;  // model ships no template; use the raw turn.
    }
    llama_chat_message msg{"user", user_msg.c_str()};
    std::vector<char> buf(user_msg.size() + 2048);
    int32_t n = llama_chat_apply_template(
        tmpl, &msg, 1, /*add_ass=*/true, buf.data(), static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(n));
        n = llama_chat_apply_template(
            tmpl, &msg, 1, true, buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (n < 0) {
        throw std::runtime_error("llama_chat_apply_template failed for the draft model");
    }
    return std::string(buf.data(), static_cast<size_t>(n));
}

std::string to_lower(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

// Port of util.load_dataset(): "<name>_prompts", specbench is a JSONL of
// chat turns (templated), every other set is a JSON array of [id, text]
// pairs whose text is taken verbatim.
std::vector<std::string> load_dataset(
    const std::string& name, const llama_model* model, bool reasoning) {
    const std::string stem = to_lower(name) + "_prompts";
    const std::string data_dir = kDataDir;
    std::vector<std::string> prompts;

    // client.py applies a chat template for specbench regardless of the
    // reasoning flag (that flag only feeds enable_thinking); accepted and
    // ignored here since llama_chat_apply_template() has no equivalent.
    (void)reasoning;

    if (stem == "specbench_prompts") {
        const std::string path = data_dir + "/" + stem + ".jsonl";
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("Missing dataset file: " + path);
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) {
                continue;
            }
            const nlohmann::json row = nlohmann::json::parse(line);
            const std::string turn0 = row.at("turns").at(0).get<std::string>();
            prompts.push_back(apply_chat_template(model, turn0));
        }
    } else {
        const std::string path = data_dir + "/" + stem + ".json";
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("Missing dataset file: " + path);
        }
        const nlohmann::json rows = nlohmann::json::parse(f);
        for (const auto& row : rows) {
            prompts.push_back(row.at(1).get<std::string>());
        }
    }

    return prompts;
}

// Port of client.py's req_indices construction:
//   req_indices = list(range(len(dataset)))
//   req_indices = req_indices[req_offset : max_req_num][::sample_req_cnt]
//   random.seed(client_idx); random.shuffle(req_indices)
std::vector<int32_t> build_request_indices(int32_t dataset_len, const ClientConfig& cfg) {
    const int32_t max_req_num =
        (cfg.max_request_num == -1) ? dataset_len : cfg.max_request_num;
    const int32_t start = std::clamp(cfg.req_offset, 0, dataset_len);
    const int32_t stop = std::clamp(max_req_num, 0, dataset_len);

    std::vector<int32_t> req_indices;
    for (int32_t i = start; i < stop; i += cfg.sample_req_cnt) {
        req_indices.push_back(i);
    }

    std::mt19937 rng(static_cast<uint32_t>(cfg.client_idx));
    std::shuffle(req_indices.begin(), req_indices.end(), rng);
    return req_indices;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "config/client.example.yaml";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        const ClientConfig cfg = load_config(config_path);

        specedge::LlamaCppEngine::Config engine_config;
        engine_config.model_path = cfg.draft_model;
        engine_config.max_len = cfg.max_len;
        engine_config.max_n_beams = cfg.max_n_beams;
        engine_config.max_seqs = cfg.max_seqs > 0 ? cfg.max_seqs : derive_max_seqs(cfg);
        engine_config.n_gpu_layers = cfg.n_gpu_layers;
        engine_config.main_gpu = cfg.main_gpu;
        engine_config.n_threads = cfg.n_threads;
        engine_config.n_threads_batch = cfg.n_threads_batch;
        engine_config.role = "tree_client";

        specedge::LlamaCppEngine engine(engine_config);
        specedge::GrpcClient validator(cfg.host);
        validator.client_idx = cfg.client_idx;

        const std::vector<std::string> dataset =
            load_dataset(cfg.dataset, engine.model(), cfg.reasoning);
        if (dataset.empty()) {
            throw std::runtime_error("dataset '" + cfg.dataset + "' has no prompts");
        }

        const std::vector<int32_t> req_indices =
            build_request_indices(static_cast<int32_t>(dataset.size()), cfg);

        std::fprintf(stderr,
            "Loaded dataset '%s' (%zu prompts); running %zu requests against %s\n",
            cfg.dataset.c_str(), dataset.size(), req_indices.size(), cfg.host.c_str());

        for (size_t k = 0; k < req_indices.size(); ++k) {
            const int32_t req_idx = req_indices[k];
            const std::string& prompt = dataset[static_cast<size_t>(req_idx)];
            std::fprintf(stderr, "Request %zu/%zu, req_idx: %d\n",
                         k + 1, req_indices.size(), req_idx);

            std::vector<llama_token> prompt_tokens =
                tokenize(engine.vocab(), prompt, /*add_bos=*/true);
            if (static_cast<int32_t>(prompt_tokens.size()) + cfg.max_new_tokens > cfg.max_len) {
                std::fprintf(stderr,
                    "  Skipping req_idx=%d: %zu prompt + %d new tokens exceeds max_len=%d\n",
                    req_idx, prompt_tokens.size(), cfg.max_new_tokens, cfg.max_len);
                continue;
            }

            specedge::SpecExecClient::Config client_config;
            client_config.max_n_beams = cfg.max_n_beams;
            client_config.max_beam_len = cfg.max_beam_len;
            client_config.max_branch_width = cfg.max_branch_width;
            client_config.max_budget = cfg.max_budget;
            client_config.max_new_tokens = cfg.max_new_tokens;
            client_config.client_idx = cfg.client_idx;

            specedge::SpecExecClient client(
                engine, validator, prompt_tokens, prompt, client_config);
            const std::vector<llama_token> generated = client.Generate(req_idx);

            std::vector<llama_token> completion = generated;
            if (generated.size() >= prompt_tokens.size()) {
                completion.assign(
                    generated.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
                    generated.end());
            }

            std::printf("=== req_idx=%d (%zu/%zu) ===\n", req_idx, k + 1, req_indices.size());
            std::printf("Prompt: %s\n", prompt.c_str());
            std::printf("Completion: %s\n\n",
                        detokenize(engine.vocab(), completion).c_str());
            std::fflush(stdout);
        }

        return 0;
    } catch (const std::exception& e) {
        // Model load, YAML parse, the gRPC channel (target unreachable), or
        // decoding all surface as plain exceptions -- report and exit
        // cleanly rather than unwinding past main() into an abort.
        std::fprintf(stderr, "client: fatal error: %s\n", e.what());
        return 1;
    }
}
