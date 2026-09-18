// Local smoke test / minimal CLI for GraphEngine (graph_engine.h), no target
// server needed. Two sections, selected by --mode:
//  - linear: prefills a prompt, greedily decodes n_generate tokens through
//    forward()/gather(), and prints the result.
//  - tree: the path the real client takes. Runs SpecExecClient (tree drafting,
//    branch forks, end-of-round acceptance) against an OracleValidator that
//    judges every round against a fixed reference continuation, so rounds
//    accept partially and the engine has to discard rejected draft tokens.
//    That discard is what breaks on hybrid models (Qwen3.5's Gated DeltaNet
//    layers keep a recurrent state, not a per-position KV cache).
// Not a unit test suite -- just enough to prove the build links against real
// llama.cpp and that a given GGUF runs through both paths on this machine.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"
#include "graph_engine.h"
#include "oracle_validator.h"
#include "spec_exec_client.h"

namespace {

constexpr const char* kDefaultModel = "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf";
constexpr const char* kDefaultPrompt = "The capital of France is";
// Reference continuation for the tree section. Plain factual text, so the
// draft agrees with it often but not always -- partial acceptance is the
// point.
constexpr const char* kDefaultReference =
    " Paris. It is known for the Eiffel Tower, the Louvre Museum, and the many "
    "cafes along the river Seine. Paris is also the largest city in France, with "
    "a population of more than two million people in the city itself and about "
    "twelve million in the wider metropolitan area.";

struct Args {
    std::string model_path = kDefaultModel;
    std::string prompt = kDefaultPrompt;
    std::string reference = kDefaultReference;
    std::string mode = "all";
    int32_t max_len = 256;
    int32_t n_generate = 8;
    int32_t n_gpu_layers = 0;
    std::string device;
    std::optional<uint32_t> n_threads;
    std::optional<uint32_t> n_threads_batch;
    std::optional<bool> flash_attn;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model <path>            GGUF model path (default: %s)\n"
        "  --prompt <text>           Prompt to complete (default: \"%s\")\n"
        "  --reference <text>        Tree mode: continuation the oracle judges against\n"
        "  --mode <linear|tree|all>  Which section(s) to run (default: all)\n"
        "  --max-len <n>             Context / max_len passed to GraphEngine (default: 256)\n"
        "  --n-generate <n>          Number of tokens to greedily decode (default: 8)\n"
        "  --n-gpu-layers <n>        Layers to offload to GPU, -1 for all (default: 0)\n"
        "  --device <name[,name..]>  Pin ggml backend device(s), e.g. HTP0 (default: all registered)\n"
        "  --n-threads <n>           Decode thread count (default: llama.cpp's own default)\n"
        "  --n-threads-batch <n>     Batch thread count (default: same as --n-threads)\n"
        "  --flash-attn <0|1>        Force flash-attn off/on (default: llama.cpp's auto)\n"
        "  -h, --help                Show this message\n",
        argv0, kDefaultModel, kDefaultPrompt);
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
        } else if (arg == "--reference") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--reference needs a value\n"); return false; }
            args.reference = *value;
        } else if (arg == "--mode") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--mode needs a value\n"); return false; }
            args.mode = *value;
            if (args.mode != "linear" && args.mode != "tree" && args.mode != "all") {
                std::fprintf(stderr, "--mode must be linear, tree or all\n");
                return false;
            }
        } else if (arg == "--max-len") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--max-len needs a value\n"); return false; }
            args.max_len = std::stoi(*value);
        } else if (arg == "--n-generate") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-generate needs a value\n"); return false; }
            args.n_generate = std::stoi(*value);
        } else if (arg == "--n-gpu-layers") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-gpu-layers needs a value\n"); return false; }
            args.n_gpu_layers = std::stoi(*value);
        } else if (arg == "--device") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--device needs a value\n"); return false; }
            args.device = *value;
        } else if (arg == "--n-threads") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-threads needs a value\n"); return false; }
            args.n_threads = static_cast<uint32_t>(std::stoul(*value));
        } else if (arg == "--n-threads-batch") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--n-threads-batch needs a value\n"); return false; }
            args.n_threads_batch = static_cast<uint32_t>(std::stoul(*value));
        } else if (arg == "--flash-attn") {
            if (!(value = next_value(i))) { std::fprintf(stderr, "--flash-attn needs a value\n"); return false; }
            args.flash_attn = (std::stoi(*value) != 0);
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

specedge::LlamaCppEngine::Config base_engine_config(const Args& args, const char* role) {
    specedge::LlamaCppEngine::Config config;
    config.model_path = args.model_path;
    config.max_len = args.max_len;
    config.max_n_beams = 1;
    config.n_gpu_layers = args.n_gpu_layers;
    config.device = args.device;
    config.n_threads = args.n_threads;
    config.n_threads_batch = args.n_threads_batch;
    config.flash_attn = args.flash_attn;
    config.role = role;
    return config;
}

// Counts rounds so the tree section can report mean tokens committed per round.
class CountingValidator : public specedge::Validator {
public:
    explicit CountingValidator(specedge::Validator& inner) : inner_(inner) {}

    Result Validate(
        int32_t client_idx,
        int32_t req_idx,
        const std::vector<llama_token>& input_ids,
        const std::vector<llama_pos>& position_ids,
        const std::vector<int32_t>& cache_seq_indices,
        const std::vector<float>& attention_mask,
        const std::vector<int32_t>& parent_indices,
        bool prefill,
        std::optional<std::string> prefix) override {
        ++rounds;
        return inner_.Validate(
            client_idx, req_idx, input_ids, position_ids, cache_seq_indices,
            attention_mask, parent_indices, prefill, std::move(prefix));
    }

    int32_t rounds = 0;

private:
    specedge::Validator& inner_;
};

// Drives SpecExecClient exactly as src/script/client.cpp does, with the
// target replaced by an oracle over prompt + reference. Returns false on any
// failure.
bool run_tree(const Args& args) {
    constexpr int32_t kBeams = 4;
    constexpr int32_t kBeamLen = 3;
    constexpr int32_t kWidth = 4;
    constexpr int32_t kBudget = 16;

    specedge::LlamaCppEngine::Config config = base_engine_config(args, "local_test_tree");
    config.max_n_beams = kBeams;
    // Same bound as client.cpp's derive_max_seqs with proactive disabled.
    config.max_seqs = 1 + kBeamLen * kBeams * (kWidth - 1);
    config.draft_top_k = kWidth;

    specedge::LlamaCppEngine engine(config);
    const llama_vocab* vocab = engine.vocab();

    std::vector<llama_token> prompt_tokens = tokenize(vocab, args.prompt, /*add_bos=*/true);
    std::vector<llama_token> continuation = tokenize(vocab, args.reference, /*add_bos=*/false);
    std::vector<llama_token> reference = prompt_tokens;
    reference.insert(reference.end(), continuation.begin(), continuation.end());

    // Stop a full draft round short of the reference's end, so every round
    // is judged against real text rather than the oracle's past-end EOS.
    const int32_t max_new_tokens = static_cast<int32_t>(continuation.size()) - (kBeamLen + 2);
    if (max_new_tokens < 1) {
        std::fprintf(stderr, "[tree] FAIL: --reference is too short\n");
        return false;
    }

    specedge::OracleValidator oracle(reference, llama_vocab_eos(vocab));
    CountingValidator validator(oracle);

    specedge::SpecExecClient::Config client_config;
    client_config.max_n_beams = kBeams;
    client_config.max_beam_len = kBeamLen;
    client_config.max_branch_width = kWidth;
    client_config.max_budget = kBudget;
    client_config.max_new_tokens = max_new_tokens;
    client_config.log_dir = "log/local_test";

    specedge::SpecExecClient client(engine, validator, prompt_tokens, args.prompt, client_config);
    specedge::SpecExecClient::GenerateTrace trace;
    client.Generate(/*req_idx=*/0, &trace);

    // The oracle only ever accepts reference tokens, so anything else in the
    // committed sequence means acceptance itself went wrong.
    const size_t n_check = std::min(trace.tokens.size(), reference.size());
    for (size_t i = 0; i < n_check; ++i) {
        if (trace.tokens[i] != reference[i]) {
            std::fprintf(stderr,
                "[tree] FAIL: committed token %zu is %d, reference has %d\n",
                i, trace.tokens[i], reference[i]);
            return false;
        }
    }

    const int32_t n_new = static_cast<int32_t>(trace.tokens.size() - prompt_tokens.size());
    std::printf("Tree: %d rounds, %d tokens committed, %.2f per round (max %d)\n",
                validator.rounds, n_new,
                validator.rounds > 0 ? static_cast<double>(n_new) / validator.rounds : 0.0,
                kBeamLen + 1);
    return true;
}

// Exactness check for end-of-round acceptance. Drives the tree engine by
// hand through rounds that each leave rejected draft tokens behind -- on the
// winning branch, on forks, and past the accepted tip -- and after every
// accept_path() compares the next seed's prediction against a linear engine
// that only ever decoded the committed tokens. A committed state polluted by
// rejected tokens shows up as a different argmax or a large log-prob gap.
bool run_commit_check(const Args& args) {
    constexpr int32_t kWidth = 4;
    constexpr int32_t kMaxSeqs = 8;
    // The two sides build the same state through different kernels (batched
    // vs one token at a time, unified vs single-sequence KV), so they agree
    // closely but not bit-exactly: Qwen3-0.6B Q4_0 on the unchanged
    // attention-only path drifts by up to ~0.11, and near-ties can swap the
    // top-1. So the linear argmax only has to be among the tree's top-k,
    // with a log-prob this close.
    constexpr float kMaxLogprobGap = 0.25f;

    specedge::LlamaCppEngine::Config tree_config = base_engine_config(args, "local_test_commit");
    tree_config.max_n_beams = kWidth;
    tree_config.max_seqs = kMaxSeqs;
    tree_config.draft_top_k = kWidth;
    specedge::LlamaCppEngine tree(tree_config);
    specedge::LlamaCppEngine linear(base_engine_config(args, "local_test_commit_ref"));
    const llama_vocab* vocab = tree.vocab();
    const int32_t n_vocab = tree.n_vocab();

    std::vector<llama_token> ref = tokenize(vocab, args.prompt, /*add_bos=*/true);
    const int32_t prompt_len = static_cast<int32_t>(ref.size());
    std::vector<llama_token> continuation = tokenize(vocab, args.reference, /*add_bos=*/false);
    ref.insert(ref.end(), continuation.begin(), continuation.end());
    const int32_t n_ref = static_cast<int32_t>(ref.size());

    std::vector<llama_pos> prompt_pos(static_cast<size_t>(prompt_len));
    for (int32_t i = 0; i < prompt_len; ++i) prompt_pos[i] = i;
    std::vector<llama_token> prompt(ref.begin(), ref.begin() + prompt_len);
    tree.prefill(prompt, prompt_pos, 0);
    linear.prefill(prompt, prompt_pos, 0);

    // Linear reference: (argmax, its log-prob) after ref[0..q], walked
    // forward one token at a time and never rewound.
    int32_t linear_next = prompt_len - 1;
    auto linear_top1 = [&](int32_t q) {
        std::vector<float> logits;
        while (linear_next <= q) {
            logits = linear.forward(ref[linear_next], linear_next, 0, linear_next);
            ++linear_next;
        }
        const int32_t best = static_cast<int32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        double sum = 0.0;
        for (float l : logits) sum += std::exp(static_cast<double>(l - logits[best]));
        return std::make_pair(static_cast<llama_token>(best), static_cast<float>(-std::log(sum)));
    };
    auto other = [&](llama_token t) { return static_cast<llama_token>((t + 1) % n_vocab); };
    auto decode_level = [&](const std::vector<llama_token>& toks,
                            const std::vector<llama_pos>& pos,
                            const std::vector<int32_t>& seqs) {
        std::vector<int32_t> slots(toks.size(), 0);
        tree.forward_batch_topk(toks, pos, seqs, slots);
    };

    int32_t rounds = 0;
    int32_t passed = 0;
    int32_t top1_same = 0;
    float max_gap = 0.0f;
    for (int32_t q = prompt_len - 1; q + 3 < n_ref; ++rounds) {
        // The seed's own row is the prediction under test: it attends to
        // exactly the committed state.
        std::vector<int32_t> slot0{0};
        specedge::LlamaCppEngine::TopKRows top =
            tree.forward_batch_topk({ref[q]}, {q}, {0}, slot0);
        int32_t best = 0;
        int32_t found = -1;
        const auto [want_id, want_lp] = linear_top1(q);
        for (int32_t j = 0; j < top.k; ++j) {
            if (top.logprobs[j] > top.logprobs[best]) best = j;
            if (top.ids[j] == want_id) found = j;
        }
        top1_same += top.ids[best] == want_id ? 1 : 0;
        const float gap = found >= 0 ? std::fabs(top.logprobs[found] - want_lp) : INFINITY;
        max_gap = std::max(max_gap, gap);
        if (gap <= kMaxLogprobGap) {
            ++passed;
        } else {
            std::fprintf(stderr,
                "[commit] round %d (pos %d): linear argmax %d (%.4f) %s in tree top-%d\n",
                rounds, q, want_id, want_lp,
                found >= 0 ? "too far off" : "missing", top.k);
        }

        // Grow a small tree off the seed and accept part of it. Seq 0 holds
        // the seed. `other(...)` tokens are always rejected.
        int32_t keep_seq = 0;
        int32_t n_accept = 0;
        switch (rounds % 4) {
            case 0:  // accept nothing past the seed; seq 0 ran one deeper
                decode_level({ref[q + 1]}, {q + 1}, {0});
                break;
            case 1:  // accept one; winner seq 0 ran one deeper, a fork lost
                tree.seq_cp(0, 1);
                decode_level({ref[q + 1], other(ref[q + 1])}, {q + 1, q + 1}, {0, 1});
                decode_level({ref[q + 2]}, {q + 2}, {0});
                n_accept = 1;
                break;
            case 2:  // accept two on a forked branch; seq 0 went elsewhere
                tree.seq_cp(0, 1);
                decode_level({other(ref[q + 1]), ref[q + 1]}, {q + 1, q + 1}, {0, 1});
                tree.seq_cp(1, 2);
                decode_level({ref[q + 2], other(ref[q + 2])}, {q + 2, q + 2}, {1, 2});
                keep_seq = 1;
                n_accept = 2;
                break;
            default:  // accept two; the tip is a never-decoded leaf
                decode_level({ref[q + 1]}, {q + 1}, {0});
                n_accept = 2;
                break;
        }

        std::vector<llama_token> path_tokens;
        std::vector<llama_pos> path_positions;
        for (int32_t d = 0; d <= n_accept; ++d) {
            path_tokens.push_back(ref[q + d]);
            path_positions.push_back(q + d);
        }
        tree.accept_path(keep_seq, path_tokens, path_positions);
        q += n_accept + 1;
    }

    std::printf("Commit check: %d/%d rounds pass, max |d logprob| = %.4f, same top-1 in %d/%d\n",
                passed, rounds, max_gap, top1_same, rounds);
    return rounds > 0 && passed == rounds;
}

bool run_linear(const Args& args) {
    specedge::LlamaCppEngine engine(base_engine_config(args, "local_test"));
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
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    bool ok = true;
    try {
        if (args.mode != "tree") {
            ok = run_linear(args) && ok;
        }
        if (args.mode != "linear") {
            ok = run_commit_check(args) && ok;
            ok = run_tree(args) && ok;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return ok ? 0 : 1;
}
