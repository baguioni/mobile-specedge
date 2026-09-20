// Config-driven batch client. C++ port of specedge/src/script/client.py.
//
// Reads a YAML config holding every client-side option (draft model, target host,
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
//    which is what the server keys off. The contract does define Sync and
//    Done (specedge.proto), and Sync is how the Python client hands the
//    server its exp_name / result_path so a persistent server re-points its
//    own result logger at the run's folder. Since this client never calls
//    it, `result_path` / `exp_name` below place *this process's* files only
//    -- the server's server.jsonl still has to be copied in by hand before
//    src/metric/mobile.py can read the pair.
//  - Shuffle uses std::mt19937 seeded with client_idx, so the visiting
//    order is deterministic per client but not bit-identical to CPython's
//    random.shuffle().
//
// Replay mode (--replay <trace.jsonl>) swaps the target for an
// OracleValidator: each request of an earlier run is re-drafted from its
// recorded prompt_tokens and judged against its recorded output_tokens, so
// no server is contacted and the config's host / dataset fields go unused.
// Engine, tree and proactive settings still come from the YAML and outputs
// land in the same files, so draft_breakdown.py and compare_runs.py read a
// replay like a live run. mobile.py does not: there is no server.jsonl.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "chat.h"
#include "llama.h"

#include "graph_engine.h"
#include "grpc_client.h"
#include "oracle_validator.h"
#include "spec_exec_client.h"

namespace {

// Every field is read from a single `client:` mapping in the YAML.
struct ClientConfig {
    // model / engine
    std::string draft_model;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu = 0;
    // true (default) pins the draft model to a single GPU (main_gpu);
    // false lets llama.cpp split its layers across all visible GPUs.
    bool single_gpu = true;
    // See LlamaCppEngine::Config::device. Needed to pin the Hexagon NPU
    // ("HTP0") specifically when the binary also has GGML_OPENCL compiled
    // in -- otherwise main_gpu is an index into an unspecified device order.
    std::string device;
    // See LlamaCppEngine::Config::tensor_split.
    std::string tensor_split;
    std::optional<uint32_t> n_threads;
    std::optional<uint32_t> n_threads_batch;
    // See LlamaCppEngine::Config::flash_attn.
    std::optional<bool> flash_attn;
    // Where the draft's log-softmax + top-k runs: "backend" (a sampler in
    // the decode graph, topk_sampler.h) or "host" (HostTopK over the raw
    // logits, host_topk.h). See LlamaCppEngine::Config::host_topk.
    //
    // Default is "host": the backend path's ggml_top_k has no ordering
    // guarantee (ggml-cpu deliberately swaps its first two output slots --
    // see ggml_compute_forward_top_k_f32 -- and OpenCL/Hexagon have no
    // TOP_K kernel at all, so they fall back to that same CPU op), while
    // ProactiveDraft::ChooseBet reads row index 0 assuming it is the best
    // candidate. On the backend path that assumption does not hold, so
    // ChooseBet can bet on the wrong token. host_topk's HostTopK::Run()
    // genuinely sorts its output, and benchmarked faster on-device at the
    // repo's configured n_rows=32/draft_top_k=16 besides.
    std::string draft_scoring = "host";

    // target + decoding (SpecExec drafting parameters, see
    // spec_exec_client.h)
    std::string host = "localhost:50555";
    // attention_mask wire dtype; must equal the server's base.dtype since
    // the proto carries no dtype tag. One of fp32, fp16, bf16.
    std::string dtype = "fp16";
    int32_t max_len = 2048;
    int32_t max_n_beams = 4;
    int32_t max_beam_len = 8;
    int32_t max_branch_width = 2;
    int32_t max_budget = 16;
    int32_t max_seqs = 0;  // llama.cpp sequences; 0 = derive from the above
    int32_t max_new_tokens = 64;
    int32_t client_idx = 0;

    // Where a run's outputs go, mirroring the Python side's base.result_path
    // / base.exp_name (config.py: "result_path/exp_name/process_name"). When
    // both are set every output of this process -- client_<idx>.jsonl,
    // trace.{txt,jsonl} and graph-engine.log -- is written to
    // <result_path>/<exp_name>/ instead of ./log, so one run's files stay
    // together and a server's server.jsonl can be dropped in beside them for
    // src/metric/mobile.py. Leaving either empty keeps the ./log default.
    std::string result_path;
    std::string exp_name;

    // Proactive draft (see proactive_draft.h). type is one of disabled,
    // excluded, included; the rest are ignored when it is disabled.
    std::string proactive_type = "disabled";
    int32_t proactive_max_n_beams = 32;
    int32_t proactive_max_beam_len = 2;
    int32_t proactive_max_branch_width = 16;
    int32_t proactive_max_budget = 32;

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
        "Usage: %s [--config <path>] [--exp-name <name>]\n"
        "          [--replay <trace.jsonl> [--replay-limit <n>] [--sim-rtt-ms <ms>]]\n"
        "  --config, -c <path>   YAML client config (default: config/client.example.yaml)\n"
        "  --exp-name <name>     override client.exp_name, i.e. where outputs go\n"
        "  --replay <path>       draft-only replay of a recorded run's trace.jsonl: no\n"
        "                        target server, the recorded completions judge each\n"
        "                        round (see src/oracle_validator.h)\n"
        "  --replay-limit <n>    replay only the trace's first n requests\n"
        "  --sim-rtt-ms <ms>     replay: sleep this long per round in place of the\n"
        "                        target round trip (default 0)\n"
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
    c.single_gpu = node_or<bool>(cl["single_gpu"], c.single_gpu);
    c.device = node_or<std::string>(cl["device"], c.device);
    c.tensor_split = node_or<std::string>(cl["tensor_split"], c.tensor_split);
    if (cl["n_threads"] && !cl["n_threads"].IsNull()) {
        c.n_threads = cl["n_threads"].as<uint32_t>();
    }
    if (cl["n_threads_batch"] && !cl["n_threads_batch"].IsNull()) {
        c.n_threads_batch = cl["n_threads_batch"].as<uint32_t>();
    }
    if (cl["flash_attn"] && !cl["flash_attn"].IsNull()) {
        c.flash_attn = cl["flash_attn"].as<bool>();
    }
    c.draft_scoring = node_or<std::string>(cl["draft_scoring"], c.draft_scoring);
    if (c.draft_scoring != "backend" && c.draft_scoring != "host") {
        throw std::runtime_error("config: client.draft_scoring must be backend or host");
    }

    c.host = node_or<std::string>(cl["host"], c.host);
    c.dtype = node_or<std::string>(cl["dtype"], c.dtype);
    c.max_len = node_or<int32_t>(cl["max_len"], c.max_len);
    c.max_n_beams = node_or<int32_t>(cl["max_n_beams"], c.max_n_beams);
    c.max_beam_len = node_or<int32_t>(cl["max_beam_len"], c.max_beam_len);
    c.max_branch_width = node_or<int32_t>(cl["max_branch_width"], c.max_branch_width);
    c.max_budget = node_or<int32_t>(cl["max_budget"], c.max_budget);
    c.max_seqs = node_or<int32_t>(cl["max_seqs"], c.max_seqs);
    c.max_new_tokens = node_or<int32_t>(cl["max_new_tokens"], c.max_new_tokens);
    c.client_idx = node_or<int32_t>(cl["client_idx"], c.client_idx);
    c.result_path = node_or<std::string>(cl["result_path"], c.result_path);
    c.exp_name = node_or<std::string>(cl["exp_name"], c.exp_name);

    // Nested `proactive:` block, matching specedge.example.yaml's shape.
    const YAML::Node pro = cl["proactive"];
    if (pro && pro.IsMap()) {
        c.proactive_type = node_or<std::string>(pro["type"], c.proactive_type);
        c.proactive_max_n_beams =
            node_or<int32_t>(pro["max_n_beams"], c.proactive_max_n_beams);
        c.proactive_max_beam_len =
            node_or<int32_t>(pro["max_beam_len"], c.proactive_max_beam_len);
        c.proactive_max_branch_width =
            node_or<int32_t>(pro["max_branch_width"], c.proactive_max_branch_width);
        c.proactive_max_budget =
            node_or<int32_t>(pro["max_budget"], c.proactive_max_budget);
    }

    c.dataset = node_or<std::string>(cl["dataset"], c.dataset);
    c.reasoning = node_or<bool>(cl["reasoning"], c.reasoning);
    c.max_request_num = node_or<int32_t>(cl["max_request_num"], c.max_request_num);
    c.req_offset = node_or<int32_t>(cl["req_offset"], c.req_offset);
    c.sample_req_cnt = node_or<int32_t>(cl["sample_req_cnt"], c.sample_req_cnt);

    if (c.draft_model.empty()) {
        throw std::runtime_error("config: model.draft_model is required");
    }
    if (c.dtype != "fp32" && c.dtype != "fp16" && c.dtype != "bf16") {
        throw std::runtime_error(
            "config: client.dtype must be one of fp32, fp16, bf16");
    }
    if (c.max_n_beams < 1 || c.max_beam_len < 1 || c.max_branch_width < 1 ||
        c.max_budget < 1) {
        throw std::runtime_error(
            "config: max_n_beams, max_beam_len, max_branch_width and "
            "max_budget must all be >= 1");
    }
    // Throws on an unknown name; do it here so a typo fails at config load
    // rather than after the model is on the GPU.
    specedge::SpecExecClient::ParseProactiveType(c.proactive_type);
    if (c.proactive_type != "disabled") {
        if (c.proactive_max_n_beams < 1 || c.proactive_max_beam_len < 1 ||
            c.proactive_max_branch_width < 1 || c.proactive_max_budget < 1) {
            throw std::runtime_error(
                "config: proactive max_n_beams, max_beam_len, max_branch_width and "
                "max_budget must all be >= 1");
        }
        if (c.proactive_max_branch_width > c.max_branch_width) {
            throw std::runtime_error(
                "config: proactive.max_branch_width must be <= max_branch_width "
                "(the draft engine's backend sampler is built once, at the wider "
                "of the two, and a narrower proactive width reads as a prefix of "
                "each row)");
        }
    }
    if (c.sample_req_cnt < 1) {
        throw std::runtime_error("config: dataset.sample_req_cnt must be >= 1");
    }
    if (c.max_request_num < -1) {
        throw std::runtime_error("config: dataset.max_request_num must be -1 or >= 0");
    }
    return c;
}

// Every output of a run goes in one directory: <result_path>/<exp_name> when
// the config names both, matching config.py's "result_path/exp_name" layout,
// otherwise the repo-relative ./log the client has always used. Requiring
// both is deliberate -- half of the pair would silently write a run into
// ./<exp_name> or <result_path>/, neither of which is what the Python
// launcher produces, and the metric scripts read a whole directory.
std::string resolve_log_dir(const ClientConfig& cfg) {
    if (cfg.result_path.empty() || cfg.exp_name.empty()) {
        return "log";
    }
    return (std::filesystem::path(cfg.result_path) / cfg.exp_name).string();
}

// Upper bound on llama.cpp sequences a round can fork: every draft step can
// split each expanded beam into (width - 1) fresh branches, plus the
// canonical seq 0. The proactive draft forks from the same pool inside the
// same round -- and its branches stay live into the next one when the bet
// wins -- so its own worst case is added on top. Clamped to llama.cpp's
// LLAMA_MAX_SEQ.
int32_t derive_max_seqs(const ClientConfig& cfg) {
    int64_t forks = 1 +
        static_cast<int64_t>(cfg.max_beam_len) * cfg.max_n_beams *
            (cfg.max_branch_width - 1);
    if (cfg.proactive_type != "disabled") {
        forks += 1 +  // the bonus token's own fork off the leaf it bets on
            static_cast<int64_t>(cfg.proactive_max_beam_len) * cfg.proactive_max_n_beams *
                (cfg.proactive_max_branch_width - 1);
    }
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

// A misbehaving/test target can return a token id with no piece data, which
// llama.cpp raises as an uncaught std::out_of_range.
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

// Per-request generation trace, the C++ half of specedge/src/gen_trace.py.
//
// Aggregate metrics cannot answer "are these two systems decoding the same
// text?" -- two runs can agree on mean accept rate and throughput while
// starting from different prompts and producing different tokens. This
// records, per request, the exact prompt that was tokenized, the tokens
// generated from it, and why generation stopped.
//
//   log/trace.jsonl  one record per request, streamed as each finishes, so
//                    an interrupted run still leaves a usable trace.
//   log/trace.txt    the same records sorted by req_idx, written at Close().
//                    The two runs visit requests in different orders
//                    (std::mt19937 here, CPython's random.shuffle there),
//                    so sorting is what lets them line up. Diff directly:
//
//                        diff serverA/trace.txt log/trace.txt
//
// The field list, the 13-column key gutter and the ", " list separator are
// a contract with gen_trace.py: change one side and every record diffs on
// formatting alone. nlohmann's dump() escapes strings the way Python's
// json.dumps(ensure_ascii=False) does, which is why both sides route every
// value through it -- a newline inside a prompt must not break the
// one-field-per-line alignment that makes the diff readable.
class TraceWriter {
public:
    explicit TraceWriter(const std::string& log_dir) {
        std::filesystem::create_directories(log_dir);
        txt_path_ = log_dir + "/trace.txt";
        jsonl_.open(log_dir + "/trace.jsonl", std::ios::out | std::ios::trunc);
        if (!jsonl_.is_open()) {
            throw std::runtime_error("could not open " + log_dir + "/trace.jsonl");
        }
    }

    ~TraceWriter() { Close(); }

    void Add(
        int32_t req_idx,
        const std::string& prompt_text,
        const std::string& output_text,
        const std::vector<llama_token>& prompt_tokens,
        const std::vector<llama_token>& output_tokens,
        const std::string& stop_reason) {
        nlohmann::ordered_json record;
        record["req_idx"] = req_idx;
        record["stop_reason"] = stop_reason;
        record["prompt_len"] = prompt_tokens.size();
        record["n_generated"] = output_tokens.size();
        record["prompt_text"] = prompt_text;
        record["output_text"] = output_text;
        record["prompt_tokens"] = prompt_tokens;
        record["output_tokens"] = output_tokens;

        records_[req_idx] = record;
        // A draft/target mismatch can decode to invalid UTF-8 (mangled
        // multi-byte sequences); nlohmann's default dump() throws on that
        // and would abort the whole run over one bad request. Replace
        // instead, matching spec_exec_client.cpp's per-round result log.
        jsonl_ << record.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
        jsonl_.flush();
    }

    void Close() {
        if (!jsonl_.is_open()) {
            return;
        }
        jsonl_.close();

        std::ofstream txt(txt_path_, std::ios::out | std::ios::trunc);
        for (const auto& [req_idx, record] : records_) {  // std::map: sorted
            txt << "==== req_idx=" << req_idx << " ====\n";
            for (const char* field : kFields) {
                txt << Pad(field) << " " << Render(record[field]) << "\n";
            }
        }
    }

private:
    // Fixed order, shared with gen_trace.py's _FIELDS.
    static constexpr const char* kFields[] = {
        "stop_reason", "prompt_len",    "n_generated",   "prompt_text",
        "output_text", "prompt_tokens", "output_tokens",
    };

    static std::string Pad(const std::string& key) {
        return key.size() >= 13 ? key : key + std::string(13 - key.size(), ' ');
    }

    // json.dumps() puts ", " between list elements; dump() puts ",".
    // Same invalid-UTF-8-output rationale as TraceWriter::Add()'s dump():
    // replace rather than throw.
    static std::string Render(const nlohmann::ordered_json& value) {
        if (!value.is_array()) {
            return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        }
        std::string out = "[";
        for (size_t i = 0; i < value.size(); ++i) {
            out += (i ? ", " : "") +
                   value[i].dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        }
        return out + "]";
    }

    std::string txt_path_;
    std::ofstream jsonl_;
    std::map<int32_t, nlohmann::ordered_json> records_;
};

// Wraps a single user turn with the GGUF's built-in chat template, the way
// util.load_dataset() calls tokenizer.apply_chat_template() for specbench.
//
// Goes through common_chat_templates_apply() (llama.cpp's Jinja engine)
// rather than the plain C llama_chat_apply_template() API: the latter is a
// hand-rolled reimplementation of a handful of known templates and has no
// concept of `enable_thinking`, so a model's own reasoning toggle was
// silently dropped no matter what `reasoning` was set to. Running the
// model's actual chat_template through Jinja means `enable_thinking` does
// whatever that template defines it to do -- the same mechanism
// tokenizer.apply_chat_template(enable_thinking=...) uses on the Python
// side -- for any model, not just ones special-cased here.
std::string apply_chat_template(
    const common_chat_templates* tmpls, const std::string& user_msg, bool reasoning) {
    common_chat_msg msg;
    msg.role = "user";
    msg.content = user_msg;

    common_chat_templates_inputs inputs;
    inputs.messages.push_back(msg);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = reasoning;

    return common_chat_templates_apply(tmpls, inputs).prompt;
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

    if (stem == "specbench_prompts") {
        // Parses the model's own chat_template once (Jinja), then reused
        // for every prompt below -- see apply_chat_template().
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        if (reasoning && !common_chat_templates_support_enable_thinking(tmpls.get())) {
            std::fprintf(
                stderr,
                "[client] warning: the draft model's chat template has no "
                "enable_thinking toggle; `reasoning: true` has no effect for "
                "this model\n");
        }

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
            prompts.push_back(apply_chat_template(tmpls.get(), turn0, reasoning));
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

specedge::SpecExecClient::Config make_client_config(
    const ClientConfig& cfg, const std::string& log_dir) {
    specedge::SpecExecClient::Config client_config;
    client_config.max_n_beams = cfg.max_n_beams;
    client_config.max_beam_len = cfg.max_beam_len;
    client_config.max_branch_width = cfg.max_branch_width;
    client_config.max_budget = cfg.max_budget;
    client_config.max_new_tokens = cfg.max_new_tokens;
    client_config.client_idx = cfg.client_idx;
    client_config.log_dir = log_dir;
    client_config.proactive_type =
        specedge::SpecExecClient::ParseProactiveType(cfg.proactive_type);
    client_config.proactive.max_n_beams = cfg.proactive_max_n_beams;
    client_config.proactive.max_beam_len = cfg.proactive_max_beam_len;
    client_config.proactive.max_branch_width = cfg.proactive_max_branch_width;
    client_config.proactive.max_budget = cfg.proactive_max_budget;
    return client_config;
}

// One request of a recorded run, as TraceWriter wrote it.
struct ReplayRequest {
    int32_t req_idx = 0;
    std::string prompt_text;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> output_tokens;
};

// Reads a trace.jsonl in file order, which is the order the recorded run
// visited its requests -- so a replay meets them in the same sequence and
// the draft device goes through the same load history. Requests the
// recorded run skipped, or that produced nothing, have no reference and are
// dropped. limit < 0 keeps every request.
std::vector<ReplayRequest> load_replay_trace(const std::string& path, int32_t limit) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("could not open replay trace " + path);
    }
    std::vector<ReplayRequest> requests;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        const nlohmann::json row = nlohmann::json::parse(line);
        if (row.value("stop_reason", std::string()) == "skipped_max_len") {
            continue;
        }
        ReplayRequest r;
        r.req_idx = row.at("req_idx").get<int32_t>();
        r.prompt_text = row.value("prompt_text", std::string());
        r.prompt_tokens = row.at("prompt_tokens").get<std::vector<llama_token>>();
        r.output_tokens = row.at("output_tokens").get<std::vector<llama_token>>();
        if (r.prompt_tokens.empty() || r.output_tokens.empty()) {
            continue;
        }
        requests.push_back(std::move(r));
        if (limit >= 0 && static_cast<int32_t>(requests.size()) >= limit) {
            break;
        }
    }
    if (requests.empty()) {
        throw std::runtime_error("replay trace " + path + " has no usable requests");
    }
    return requests;
}

struct ReplayOptions {
    std::string trace_path;
    int32_t limit = -1;
    double sim_rtt_ms = 0.0;
};

// The --replay loop: the live loop's shape, with each request's recorded
// completion standing in for the target (see oracle_validator.h).
int run_replay(
    const ClientConfig& cfg,
    specedge::LlamaCppEngine& engine,
    const std::string& log_dir,
    const ReplayOptions& opts) {
    const std::vector<ReplayRequest> requests = load_replay_trace(opts.trace_path, opts.limit);
    const llama_vocab* vocab = engine.vocab();

    TraceWriter trace(log_dir);

    std::fprintf(stderr,
        "Replaying %zu requests from %s against their recorded completions "
        "(no target server, sim_rtt_ms=%.1f)\n",
        requests.size(), opts.trace_path.c_str(), opts.sim_rtt_ms);

    int32_t n_past_end = 0;
    for (size_t k = 0; k < requests.size(); ++k) {
        const ReplayRequest& req = requests[k];
        std::fprintf(stderr, "Request %zu/%zu, req_idx: %d\n",
                     k + 1, requests.size(), req.req_idx);

        if (static_cast<int32_t>(req.prompt_tokens.size()) + cfg.max_new_tokens > cfg.max_len) {
            std::fprintf(stderr,
                "  Skipping req_idx=%d: %zu prompt + %d new tokens exceeds max_len=%d\n",
                req.req_idx, req.prompt_tokens.size(), cfg.max_new_tokens, cfg.max_len);
            trace.Add(req.req_idx, req.prompt_text, "", req.prompt_tokens, {}, "skipped_max_len");
            continue;
        }

        std::vector<llama_token> reference = req.prompt_tokens;
        reference.insert(reference.end(), req.output_tokens.begin(), req.output_tokens.end());
        specedge::OracleValidator oracle(reference, llama_vocab_eos(vocab), opts.sim_rtt_ms);

        specedge::SpecExecClient client(
            engine, oracle, req.prompt_tokens, req.prompt_text, make_client_config(cfg, log_dir));
        specedge::SpecExecClient::GenerateTrace gen_trace;
        client.Generate(req.req_idx, &gen_trace);

        // The oracle only ever accepts reference tokens, so everything
        // committed must be the reference; anything else means its position
        // mapping is wrong and every number from this replay with it. Past
        // the reference's end it answers EOS, so a longer sequence means the
        // final round outran the recording.
        const size_t n_common = std::min(gen_trace.tokens.size(), reference.size());
        for (size_t i = 0; i < n_common; ++i) {
            if (gen_trace.tokens[i] != reference[i]) {
                throw std::runtime_error(
                    "replay of req_idx=" + std::to_string(req.req_idx) +
                    " diverged from its reference at token " + std::to_string(i));
            }
        }
        const bool past_end = gen_trace.tokens.size() > reference.size();
        n_past_end += past_end ? 1 : 0;

        const std::vector<llama_token> traced_output(
            gen_trace.tokens.begin() + static_cast<std::ptrdiff_t>(req.prompt_tokens.size()),
            gen_trace.tokens.end());
        trace.Add(
            req.req_idx, req.prompt_text, detokenize(vocab, traced_output),
            req.prompt_tokens, traced_output,
            past_end ? "reference_end" : gen_trace.stopped_on_eog ? "eos" : "max_new_tokens");

        std::printf("=== req_idx=%d (%zu/%zu) replayed: %zu tokens ===\n",
                    req.req_idx, k + 1, requests.size(), traced_output.size());
        std::fflush(stdout);
    }

    trace.Close();

    if (n_past_end > 0) {
        std::fprintf(stderr,
            "note: %d request(s) ran past the end of their recorded completion on the "
            "final round (trace stop_reason \"reference_end\"); that round's acceptance "
            "is capped by the recording, not the draft\n",
            n_past_end);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "config/client.example.yaml";
    std::string exp_name_override;
    ReplayOptions replay;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--exp-name" && i + 1 < argc) {
            exp_name_override = argv[++i];
        } else if (arg == "--replay" && i + 1 < argc) {
            replay.trace_path = argv[++i];
        } else if (arg == "--replay-limit" && i + 1 < argc) {
            replay.limit = std::stoi(argv[++i]);
        } else if (arg == "--sim-rtt-ms" && i + 1 < argc) {
            replay.sim_rtt_ms = std::stod(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        ClientConfig loaded = load_config(config_path);
        if (!exp_name_override.empty()) {
            loaded.exp_name = exp_name_override;
        }
        const ClientConfig& cfg = loaded;

        // <result_path>/<exp_name> when both are set, else ./log. Exported
        // before the engine is built because LlamaCppEngine resolves
        // graph-engine.log's directory from these two env vars in its
        // constructor -- the same handoff batch_server.py makes on the
        // Python side (os.environ["SPECEDGE_RESULT_PATH"] = result_path).
        const std::string log_dir = resolve_log_dir(cfg);
        if (!cfg.result_path.empty() && !cfg.exp_name.empty()) {
            ::setenv("SPECEDGE_RESULT_PATH", cfg.result_path.c_str(), /*overwrite=*/1);
            ::setenv("SPECEDGE_EXP_NAME", cfg.exp_name.c_str(), /*overwrite=*/1);
        }
        std::fprintf(stderr, "Writing results to %s/\n", log_dir.c_str());

        specedge::LlamaCppEngine::Config engine_config;
        engine_config.model_path = cfg.draft_model;
        engine_config.max_len = cfg.max_len;
        engine_config.max_n_beams = cfg.max_n_beams;
        engine_config.max_seqs = cfg.max_seqs > 0 ? cfg.max_seqs : derive_max_seqs(cfg);
        engine_config.n_gpu_layers = cfg.n_gpu_layers;
        engine_config.main_gpu = cfg.main_gpu;
        engine_config.single_gpu = cfg.single_gpu;
        engine_config.device = cfg.device;
        engine_config.tensor_split = cfg.tensor_split;
        engine_config.n_threads = cfg.n_threads;
        engine_config.n_threads_batch = cfg.n_threads_batch;
        engine_config.flash_attn = cfg.flash_attn;
        // Scores the draft tree on the device; the sampler's k is the branch
        // width because it decides each beam's children. See topk_sampler.h.
        engine_config.draft_top_k = cfg.max_branch_width;
        engine_config.host_topk = cfg.draft_scoring == "host";
        engine_config.role = "tree_client";

        specedge::LlamaCppEngine engine(engine_config);

        if (!replay.trace_path.empty()) {
            return run_replay(cfg, engine, log_dir, replay);
        }

        specedge::GrpcClient validator(cfg.host);
        validator.client_idx = cfg.client_idx;
        validator.attention_mask_dtype =
            specedge::GrpcClient::ParseMaskDType(cfg.dtype);

        const std::vector<std::string> dataset =
            load_dataset(cfg.dataset, engine.model(), cfg.reasoning);
        if (dataset.empty()) {
            throw std::runtime_error("dataset '" + cfg.dataset + "' has no prompts");
        }

        const std::vector<int32_t> req_indices =
            build_request_indices(static_cast<int32_t>(dataset.size()), cfg);

        TraceWriter trace(log_dir);

        std::fprintf(stderr,
            "Loaded dataset '%s' (%zu prompts); running %zu requests against %s\n",
            cfg.dataset.c_str(), dataset.size(), req_indices.size(), cfg.host.c_str());

        // Hand the server this run's identity before the first Validate, the
        // way client.py does. A persistent server re-points its own result
        // logger at <result_path>/<exp_name> and drops the previous run's KV
        // state; skip it and two back-to-back experiments both land in
        // whatever folder the server was started with.
        validator.Sync(cfg.client_idx, cfg.exp_name, cfg.result_path);

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
                trace.Add(
                    req_idx, prompt, "", prompt_tokens, {}, "skipped_max_len");
                continue;
            }

            specedge::SpecExecClient client(
                engine, validator, prompt_tokens, prompt, make_client_config(cfg, log_dir));
            specedge::SpecExecClient::GenerateTrace gen_trace;
            const std::vector<llama_token> generated = client.Generate(req_idx, &gen_trace);

            std::vector<llama_token> completion = generated;
            if (generated.size() >= prompt_tokens.size()) {
                completion.assign(
                    generated.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
                    generated.end());
            }

            // Traced from the untrimmed sequence, so a length difference
            // against the Python run reads as a real difference in what was
            // decoded rather than a difference in what each side reports.
            const std::vector<llama_token> traced_output(
                gen_trace.tokens.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
                gen_trace.tokens.end());
            trace.Add(
                req_idx, prompt, detokenize(engine.vocab(), traced_output),
                prompt_tokens, traced_output,
                gen_trace.stopped_on_eog ? "eos" : "max_new_tokens");

            std::printf("=== req_idx=%d (%zu/%zu) ===\n", req_idx, k + 1, req_indices.size());
            std::printf("Prompt: %s\n", prompt.c_str());
            std::printf("Completion: %s\n\n",
                        detokenize(engine.vocab(), completion).c_str());
            std::fflush(stdout);
        }

        trace.Close();

        // Best-effort, like client.py's: every result is already on disk by
        // now, so a server that has gone away is worth a warning and nothing
        // more. shutdown stays false -- tripping the server's teardown is the
        // sweep orchestrator's job, not a single run's.
        if (!validator.Done(cfg.client_idx)) {
            std::fprintf(stderr, "warning: Done notification failed\n");
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
