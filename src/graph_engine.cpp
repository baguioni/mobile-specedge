#include "graph_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace specedge {

namespace {

// graph-engine.log is meant to be shared across every engine instance in the
// process (e.g. draft + target), the same way graph.py's single
// "graph_engine" logging.Logger is shared: the file is truncated once, and
// every instance after that appends through the same handle.
std::mutex g_forward_log_mutex;
std::unordered_map<std::string, std::shared_ptr<std::ofstream>> g_forward_logs;

std::filesystem::path forward_log_path() {
    namespace fs = std::filesystem;
    const char* result_path = std::getenv("SPECEDGE_RESULT_PATH");
    const char* exp_name = std::getenv("SPECEDGE_EXP_NAME");
    fs::path log_dir = (result_path && exp_name) ? fs::path(result_path) / exp_name : fs::path(".");
    fs::create_directories(log_dir);
    return log_dir / "graph-engine.log";
}

std::shared_ptr<std::ofstream> get_forward_log_stream() {
    std::filesystem::path path = forward_log_path();
    std::lock_guard<std::mutex> lock(g_forward_log_mutex);
    auto it = g_forward_logs.find(path.string());
    if (it != g_forward_logs.end()) {
        return it->second;
    }
    auto stream = std::make_shared<std::ofstream>(path, std::ios::out | std::ios::trunc);
    g_forward_logs.emplace(path.string(), stream);
    return stream;
}

std::string iso8601_now_local() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);

    std::tm local_tm{};
    localtime_r(&t, &local_tm);
    std::tm utc_tm{};
    gmtime_r(&t, &utc_tm);

    // Offset = local wall-clock time minus UTC wall-clock time, both
    // reinterpreted as UTC so the subtraction ignores each side's own DST.
    std::time_t local_as_utc = timegm(&local_tm);
    std::time_t utc_as_utc = timegm(&utc_tm);
    long offset_seconds = static_cast<long>(local_as_utc - utc_as_utc);

    char sign = offset_seconds >= 0 ? '+' : '-';
    long abs_offset = std::labs(offset_seconds);
    int offset_h = static_cast<int>(abs_offset / 3600);
    int offset_m = static_cast<int>((abs_offset % 3600) / 60);

    char buf[64];
    std::snprintf(
        buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03d%c%02d:%02d",
        local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday,
        local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec,
        static_cast<int>(ms.count()), sign, offset_h, offset_m);
    return std::string(buf);
}

void log_line(const char* level, const std::string& role, const char* fmt, va_list args) {
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    std::fprintf(stderr, "[%s] LlamaCppEngine(role=%s): %s\n", level, role.c_str(), buf);
}

} // namespace

LlamaCppEngine::LlamaCppEngine(Config config)
    : role_(config.role), max_len_(config.max_len), max_n_beams_(config.max_n_beams) {
    if (max_n_beams_ != 1) {
        throw std::invalid_argument(
            "LlamaCppEngine supports linear drafting only (max_n_beams must "
            "be 1, got " + std::to_string(max_n_beams_) + "). llama.cpp has "
            "no API for injecting a custom tree attention mask.");
    }

    forward_log_ = get_forward_log_stream();
    load_model(config);
}

LlamaCppEngine::~LlamaCppEngine() {
    close();
}

void LlamaCppEngine::load_model(const Config& config) {
    namespace fs = std::filesystem;
    if (!fs::exists(config.model_path)) {
        throw std::invalid_argument("GGUF model path does not exist: " + config.model_path);
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.main_gpu = config.main_gpu;

    log_info("Loading GGUF model: %s", config.model_path.c_str());
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (model_ == nullptr) {
        throw std::runtime_error("Failed to load GGUF model: " + config.model_path);
    }

    vocab_ = llama_model_get_vocab(model_);
    if (vocab_ == nullptr) {
        throw std::runtime_error("Failed to read vocab from GGUF model: " + config.model_path);
    }

    n_vocab_ = llama_vocab_n_tokens(vocab_);
    check_vocab_parity(config.expected_vocab_size);

    // This engine drives a single llama.cpp sequence (seq_id 0), so only
    // one parallel sequence slot is ever needed.
    constexpr int32_t n_seq_max = 1;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(max_len_);
    ctx_params.n_batch = static_cast<uint32_t>(max_len_);
    ctx_params.n_ubatch = std::min<uint32_t>(ctx_params.n_batch, 512);
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.kv_unified = false;

    if (config.n_threads.has_value()) {
        ctx_params.n_threads = static_cast<int32_t>(*config.n_threads);
    }
    if (config.n_threads_batch.has_value()) {
        ctx_params.n_threads_batch = static_cast<int32_t>(*config.n_threads_batch);
    } else if (config.n_threads.has_value()) {
        ctx_params.n_threads_batch = static_cast<int32_t>(*config.n_threads);
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("Failed to create llama_context");
    }

    memory_ = llama_get_memory(ctx_);
    if (memory_ == nullptr) {
        throw std::runtime_error(
            "llama_get_memory returned NULL; the model was likely built "
            "without a KV cache (embedding-only model?).");
    }

    uint32_t n_ctx_seq = llama_n_ctx_seq(ctx_);
    if (n_ctx_seq < static_cast<uint32_t>(max_len_)) {
        throw std::runtime_error(
            "llama.cpp allocated " + std::to_string(n_ctx_seq) +
            " context cells per sequence but max_len is " +
            std::to_string(max_len_) +
            ". Lower max_len or raise the context budget.");
    }

    // One token belongs to exactly one sequence in linear mode.
    batch_ = llama_batch_init(static_cast<int32_t>(ctx_params.n_batch), 0, 1);
    batch_allocated_ = true;
    n_batch_ = static_cast<int32_t>(ctx_params.n_batch);

    log_info(
        "llama.cpp context ready: n_ctx=%u n_ctx_seq=%u n_batch=%d "
        "n_seq_max=%d n_vocab=%d",
        ctx_params.n_ctx, n_ctx_seq, n_batch_, n_seq_max, n_vocab_);
}

void LlamaCppEngine::check_vocab_parity(std::optional<int32_t> expected_vocab_size) const {
    if (!expected_vocab_size.has_value()) {
        return;
    }
    if (*expected_vocab_size != n_vocab_) {
        log_warn(
            "Vocab size mismatch: GGUF has %d tokens, expected %d. Padding "
            "differences are benign, but a large gap means the GGUF does "
            "not match the draft model and acceptance rate will collapse.",
            n_vocab_, *expected_vocab_size);
    }
}

void LlamaCppEngine::close() {
    if (closed_) {
        return;
    }
    closed_ = true;

    if (batch_allocated_) {
        llama_batch_free(batch_);
        batch_allocated_ = false;
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

void LlamaCppEngine::log_info(const char* fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    log_line("INFO", role_, fmt, args);
    va_end(args);
}

void LlamaCppEngine::log_warn(const char* fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    log_line("WARN", role_, fmt, args);
    va_end(args);
}

std::string LlamaCppEngine::detokenize(const std::vector<llama_token>& tokens) const {
    if (tokens.empty()) {
        return "";
    }
    std::vector<char> buf(tokens.size() * 8 + 16);
    int32_t n = llama_detokenize(
        vocab_, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
        static_cast<int32_t>(buf.size()), /*remove_special=*/false,
        /*unparse_special=*/true);
    if (n < 0) {
        buf.resize(static_cast<size_t>(-n));
        n = llama_detokenize(
            vocab_, tokens.data(), static_cast<int32_t>(tokens.size()), buf.data(),
            static_cast<int32_t>(buf.size()), false, true);
    }
    return std::string(buf.data(), n > 0 ? static_cast<size_t>(n) : 0);
}

void LlamaCppEngine::log_forward(
    const std::vector<llama_token>& input_ids,
    const std::vector<llama_pos>& position_ids,
    const std::vector<int32_t>& cache_seq_indices,
    const std::vector<float>& logits) const {
    if (!forward_log_) {
        return;
    }

    int32_t argmax_id = static_cast<int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    nlohmann::json entry;
    entry["engine"] = "LlamaCppEngine";
    entry["role"] = role_;
    entry["timestamp"] = iso8601_now_local();
    entry["graph_replay"] = false; // kept so the log schema matches GraphEngine's.
    entry["num_beams"] = input_ids.size();
    entry["input_ids"] = input_ids;
    entry["position_ids"] = position_ids;
    entry["cache_seq_indices"] = cache_seq_indices;
    entry["logits_shape"] = std::vector<int64_t>{1, 1, n_vocab_};
    entry["logits_dtype"] = "float32";
    entry["argmax_token_ids"] = std::vector<int32_t>{argmax_id};
    entry["input_string"] = detokenize(input_ids);
    entry["argmax_string"] = detokenize({static_cast<llama_token>(argmax_id)});

    std::lock_guard<std::mutex> lock(g_forward_log_mutex);
    *forward_log_ << entry.dump() << "\n";
    forward_log_->flush();
}

void LlamaCppEngine::batch_set(
    int32_t i, llama_token token, llama_pos pos, int32_t seq_id, bool want_logits) {
    batch_.token[i] = token;
    batch_.pos[i] = pos;
    batch_.n_seq_id[i] = 1;
    batch_.seq_id[i][0] = seq_id;
    batch_.logits[i] = want_logits ? 1 : 0;
}

void LlamaCppEngine::decode(int32_t n_tokens) {
    batch_.n_tokens = n_tokens;
    int32_t rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        // rc > 0 means "no KV slot could be found for the batch", which in
        // practice means n_ctx is too small for the live sequence.
        throw std::runtime_error(
            "llama_decode failed with code " + std::to_string(rc) + " for " +
            std::to_string(n_tokens) + " tokens (seq_len=" +
            std::to_string(seq_len_) + ").");
    }
}

std::vector<float> LlamaCppEngine::read_logits(int32_t n_rows) const {
    float* ptr = llama_get_logits(ctx_);
    if (ptr == nullptr) {
        throw std::runtime_error("llama_get_logits returned NULL after decode.");
    }
    std::vector<float> out(static_cast<size_t>(n_rows) * static_cast<size_t>(n_vocab_));
    std::memcpy(out.data(), ptr, out.size() * sizeof(float));
    return out;
}

void LlamaCppEngine::seq_rm(int32_t seq_id, int32_t p0, int32_t p1) {
    // Drop cells of seq_id in [p0, p1); p1 < 0 means "to end".
    llama_memory_seq_rm(memory_, seq_id, p0, p1);
}

void LlamaCppEngine::prefill(
    const std::vector<llama_token>& input_ids,
    const std::vector<llama_pos>& position_ids,
    int32_t batch_idx) {
    if (batch_idx != 0) {
        throw std::invalid_argument(
            "LlamaCppEngine drives a single llama.cpp sequence; got batch_idx=" +
            std::to_string(batch_idx) + ".");
    }
    if (input_ids.size() != position_ids.size()) {
        throw std::invalid_argument(
            "prefill got " + std::to_string(input_ids.size()) + " tokens but " +
            std::to_string(position_ids.size()) + " positions.");
    }
    if (static_cast<int32_t>(input_ids.size()) > max_len_) {
        throw std::invalid_argument(
            "Prompt of " + std::to_string(input_ids.size()) +
            " tokens exceeds max_len=" + std::to_string(max_len_) + ".");
    }

    // A new request may land on the slot a finished one just vacated.
    seq_rm(0, -1, -1);
    seq_len_ = 0;
    predicted_.reset();

    // The last prompt token belongs to the first forward(), not to prefill.
    // n_tokens is 0 when the prompt is a single token, which correctly
    // commits nothing.
    int32_t n_tokens = static_cast<int32_t>(input_ids.size()) - 1;

    for (int32_t start = 0; start < n_tokens; start += n_batch_) {
        int32_t stop = std::min(start + n_batch_, n_tokens);
        for (int32_t idx = start; idx < stop; ++idx) {
            int32_t i = idx - start;
            // Request logits on the last token of every chunk. The values
            // are discarded (GraphEngine.prefill drops its outputs too),
            // but llama.cpp's own batch decode path always marks the final
            // token of a decode, so a chunk with zero requested outputs is
            // off the tested path.
            batch_set(i, input_ids[idx], position_ids[idx], /*seq_id=*/0,
                      /*want_logits=*/idx == stop - 1);
        }
        decode(stop - start);
    }

    seq_len_ = n_tokens;
}

std::vector<float> LlamaCppEngine::forward(
    llama_token input_id, llama_pos position_id, int32_t cache_batch_index,
    int32_t cache_seq_index) {
    if (cache_batch_index != 0) {
        throw std::invalid_argument(
            "cache_batch_indices maps to seq_id " + std::to_string(cache_batch_index) +
            ". This engine only ever drives a single llama.cpp sequence (seq_id 0).");
    }
    // Invariant: in a chain the tree slot is the absolute position.
    if (cache_seq_index != position_id) {
        throw std::invalid_argument(
            "cache_seq_indices=" + std::to_string(cache_seq_index) +
            " != position_ids=" + std::to_string(position_id) +
            ". Tree drafting is not supported by LlamaCppEngine.");
    }
    // llama.cpp appends; it cannot overwrite a cell in place. A gap or an
    // overlap here means the KV cache and the tree have diverged.
    if (position_id != seq_len_) {
        throw std::invalid_argument(
            "Token at position " + std::to_string(position_id) +
            " does not append to a sequence of length " + std::to_string(seq_len_) +
            ". KV cache and BatchTree are out of sync.");
    }

    batch_set(0, input_id, position_id, /*seq_id=*/0, /*want_logits=*/true);
    decode(1);

    std::vector<float> logits = read_logits(1);

    // Remember what the draft model expects next. In linear mode
    // _get_next_beams takes topk(k=max_branch_width=1) over
    // log_softmax(logits), i.e. the argmax, so this is exactly the token
    // the caller is about to append to the tree. backfill() uses it to
    // reconstruct the trailing drafted node.
    int32_t pred_token = static_cast<int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    seq_len_ += 1;
    predicted_ = std::make_pair(seq_len_, static_cast<llama_token>(pred_token));

    log_forward({input_id}, {position_id}, {cache_seq_index}, logits);

    return logits;
}

void LlamaCppEngine::backfill(int32_t have, int32_t need) {
    int32_t missing = need - have;

    if (missing != 1) {
        throw std::invalid_argument(
            "Accepted prefix of " + std::to_string(need) + " cells exceeds the " +
            std::to_string(have) + " cells in the KV cache by " +
            std::to_string(missing) +
            ". Only the single trailing drafted token can be reconstructed.");
    }
    if (!predicted_.has_value() || predicted_->first != have) {
        throw std::invalid_argument(
            "Need the token at position " + std::to_string(have) +
            " to close a gap in the KV cache, but the cached draft "
            "prediction is not available. Cannot reconstruct it.");
    }

    batch_set(0, predicted_->second, have, /*seq_id=*/0, /*want_logits=*/true);
    decode(1);
    log_info("Backfilled trailing draft token %d at position %d", predicted_->second, have);
}

void LlamaCppEngine::gather(
    const std::vector<int32_t>& src_indices, const std::vector<int32_t>& dest_indices) {
    if (src_indices.empty()) {
        // Nothing was accepted thus drop the sequence.
        seq_rm(0, -1, -1);
        seq_len_ = 0;
        predicted_.reset();
        return;
    }

    if (src_indices.size() != dest_indices.size()) {
        throw std::invalid_argument(
            "gather size mismatch: " + std::to_string(src_indices.size()) + " src vs " +
            std::to_string(dest_indices.size()) + " dest.");
    }

    int32_t n_keep = static_cast<int32_t>(dest_indices.size());
    for (int32_t i = 0; i < n_keep; ++i) {
        if (src_indices[i] != dest_indices[i] || dest_indices[i] != i) {
            throw std::invalid_argument(
                "LlamaCppEngine::gather only supports prefix truncation "
                "(src == dest == arange(n)); this indicates tree drafting.");
        }
    }

    if (n_keep > seq_len_) {
        backfill(seq_len_, n_keep);
    } else if (n_keep < seq_len_) {
        seq_rm(0, n_keep, -1);
    }

    seq_len_ = n_keep;
    // The next token is the bonus token, which the caller supplies thus stale.
    predicted_.reset();
}

void LlamaCppEngine::gather(
    const std::vector<bool>& src_mask, const std::vector<int32_t>& dest_indices) {
    std::vector<int32_t> src_indices;
    src_indices.reserve(src_mask.size());
    for (int32_t i = 0; i < static_cast<int32_t>(src_mask.size()); ++i) {
        if (src_mask[i]) {
            src_indices.push_back(i);
        }
    }
    gather(src_indices, dest_indices);
}

void LlamaCppEngine::reset() {
    llama_memory_clear(memory_, true);
    seq_len_ = 0;
    predicted_.reset();
}

} // namespace specedge
