#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

class SpecEdgeClientConfig {
public:
    enum class DType { kBF16, kFP16, kFP32 };

    static const std::string& result_path() { return field().result_path_; }
    static const std::string& exp_name() { return field().exp_name_; }
    static const std::string& process_name() { return field().process_name_; }
    static int seed() { return field().seed_; }
    static int optimization() { return field().optimization_; }
    static int max_len() { return field().max_len_; }

    static const std::string& draft_model() { return field().draft_model_; }
    static const std::string& device() { return field().device_; }
    static DType dtype() { return field().dtype_; }
    static bool reasoning() { return field().reasoning_; }

    static const std::string& dataset() { return field().dataset_; }

    static int max_n_beams() { return field().max_n_beams_; }
    static int max_beam_len() { return field().max_beam_len_; }
    static int max_branch_width() { return field().max_branch_width_; }
    static int max_budget() { return field().max_budget_; }

    static const std::string& proactive_type() { return field().proactive_type_; }
    static int proactive_max_n_beams() { return field().proactive_max_n_beams_; }
    static int proactive_max_beam_len() { return field().proactive_max_beam_len_; }
    static int proactive_max_branch_width() { return field().proactive_max_branch_width_; }
    static int proactive_max_budget() { return field().proactive_max_budget_; }

    static int max_new_tokens() { return field().max_new_tokens_; }
    static int max_request_num() { return field().max_request_num_; }
    static int req_offset() { return field().req_offset_; }
    static int sample_req_cnt() { return field().sample_req_cnt_; }

    static const std::string& host() { return field().host_; }
    static int client_idx() { return field().client_idx_; }

    // Mirrors `_ConfigMeta.reset()`.
    static void reset() { instance().initialized_ = false; }

private:
    SpecEdgeClientConfig() = default;

    static SpecEdgeClientConfig& instance() {
        static SpecEdgeClientConfig inst;
        return inst;
    }

    // Every accessor routes through here so initialization stays lazy,
    // exactly like `_ConfigMeta.__getattr__` triggering `_initialize()` on
    // first attribute access.
    static SpecEdgeClientConfig& field() {
        auto& inst = instance();
        if (!inst.initialized_) {
            inst.initialize();
        }
        return inst;
    }

    // Mirrors `_ConfigMeta._from_env`: raises (throws) if the variable is
    // unset or literally the string "null".
    static std::string from_env(const char* key) {
        const char* value = std::getenv(key);
        if (value == nullptr || std::string(value) == "null") {
            throw std::runtime_error(
                std::string("Environment variable '") + key + "' is not set");
        }
        return std::string(value);
    }

    static int from_env_int(const char* key) { return std::stoi(from_env(key)); }

    // Mirrors `util.convert_dtype`.
    static DType convert_dtype(const std::string& dtype) {
        if (dtype == "bf16") return DType::kBF16;
        if (dtype == "fp16") return DType::kFP16;
        if (dtype == "fp32") return DType::kFP32;
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

    void initialize() {
        // experiment configuration
        result_path_ = from_env("SPECEDGE_RESULT_PATH");
        exp_name_ = from_env("SPECEDGE_EXP_NAME");
        process_name_ = from_env("SPECEDGE_PROCESS_NAME");
        seed_ = from_env_int("SPECEDGE_SEED");
        optimization_ = from_env_int("SPECEDGE_OPTIMIZATION");
        max_len_ = from_env_int("SPECEDGE_MAX_LEN");

        // model configuration
        draft_model_ = from_env("SPECEDGE_DRAFT_MODEL");
        device_ = from_env("SPECEDGE_DEVICE");
        dtype_ = convert_dtype(from_env("SPECEDGE_DTYPE"));
        reasoning_ = from_env("SPECEDGE_REASONING") == "True";

        // dataset configuration
        dataset_ = from_env("SPECEDGE_DATASET");

        // SpecExec configuration
        max_n_beams_ = from_env_int("SPECEDGE_MAX_N_BEAMS");
        max_beam_len_ = from_env_int("SPECEDGE_MAX_BEAM_LEN");
        max_branch_width_ = from_env_int("SPECEDGE_MAX_BRANCH_WIDTH");
        max_budget_ = from_env_int("SPECEDGE_MAX_BUDGET");

        // proactive draft configuration
        proactive_type_ = from_env("SPECEDGE_PROACTIVE_TYPE");
        proactive_max_n_beams_ = from_env_int("SPECEDGE_PROACTIVE_MAX_N_BEAMS");
        proactive_max_beam_len_ = from_env_int("SPECEDGE_PROACTIVE_MAX_BEAM_LEN");
        proactive_max_branch_width_ = from_env_int("SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH");
        proactive_max_budget_ = from_env_int("SPECEDGE_PROACTIVE_MAX_BUDGET");

        // token generation configuration
        max_new_tokens_ = from_env_int("SPECEDGE_MAX_NEW_TOKENS");
        max_request_num_ = from_env_int("SPECEDGE_MAX_REQUEST_NUM");
        req_offset_ = from_env_int("SPECEDGE_REQ_OFFSET");
        sample_req_cnt_ = from_env_int("SPECEDGE_SAMPLE_REQ_CNT");

        // server configuration
        host_ = from_env("SPECEDGE_HOST");
        client_idx_ = from_env_int("SPECEDGE_CLIENT_IDX");

        initialized_ = true;
    }

    bool initialized_ = false;

    std::string result_path_;
    std::string exp_name_;
    std::string process_name_;
    int seed_ = 0;
    int optimization_ = 0;
    int max_len_ = 0;

    std::string draft_model_;
    std::string device_;
    DType dtype_ = DType::kFP32;
    bool reasoning_ = false;

    std::string dataset_;

    int max_n_beams_ = 0;
    int max_beam_len_ = 0;
    int max_branch_width_ = 0;
    int max_budget_ = 0;

    std::string proactive_type_;
    int proactive_max_n_beams_ = 0;
    int proactive_max_beam_len_ = 0;
    int proactive_max_branch_width_ = 0;
    int proactive_max_budget_ = 0;

    int max_new_tokens_ = 0;
    int max_request_num_ = 0;
    int req_offset_ = 0;
    int sample_req_cnt_ = 0;

    std::string host_;
    int client_idx_ = 0;
};
