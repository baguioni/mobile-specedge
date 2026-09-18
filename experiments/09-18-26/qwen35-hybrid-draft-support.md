# Qwen3.5 hybrid draft support in mobile-specedge

**Date:** 2026-09-18 · **Model:** `Qwen3.5-0.8B-Q4_0.gguf` (Qwen/Qwen3.5-0.8B, quantized by Unsloth) · **Device:** Galaxy S25 Ultra (SM-S938N), CPU backend

The C++ client could load Qwen3.5 but crashed on the second speculative round. This change makes tree drafting work on hybrid (recurrent + attention) draft models. Attention-only models such as Qwen3 keep the same behavior as before.

## Symptom

On the phone (CPU build) the first round ran, then every run died with:

```
[SpecExecClient] Num of accepted tokens: 2
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 34
 - the tokens for sequence 0 in the input batch have a starting position of Y = 34
 for M-RoPE, it is required that the position satisfies: X < Y
client: fatal error: llama_decode failed with code -1 for 1 tokens (seq_len=34).
```

The same failure reproduces on the host, where it hits on round 2 with `X = 6, Y = 6`.

## Root cause

Qwen3.5 interleaves three Gated DeltaNet layers with each gated full-attention layer. A DeltaNet layer keeps one recurrent state per sequence instead of per-position KV cells.

The tree drafter works like this:

1. Each draft branch is a llama.cpp sequence. The first child of a node inherits its parent's sequence, so seq 0 itself gets draft tokens past the seed.
2. Each tree level is one `llama_decode`, with one token per sequence.
3. At the end of the round, `collapse_to_seq()` removes rejected tokens with `seq_rm(keep_seq, last_pos, -1)`, and `decode_token()` redoes the tip.

Step 3 relies on positional truncation, which a recurrent state can't do. In the pinned llama.cpp (`5202104b5`):

- `llama_memory_recurrent::seq_rm` can partially roll back only through the `n_rs_seq` snapshots. If the rollback is deeper than `n_rs_seq`, or a rollback is already pending, it returns `false`. The engine ignored that return value, so the sequence position went stale and the next decode failed the M-RoPE check.
- The snapshots are written inside a single ubatch. `build_recurrent_attn` writes only the last `min(n_seq_tokens, n_rs_seq + 1)` states of the current decode. With one token per sequence per decode, any rollback of 2 or more tokens would restore a stale snapshot. So raising `n_rs_seq` can't fix tree drafting. It also multiplies the recurrent-state buffer by `1 + n_rs_seq`: 1849.5 MiB at 32 sequences with `n_rs_seq = 2`, versus about 616 MiB without snapshots.

## Design

Tree mode on a recurrent or hybrid model gets one extra, hidden llama.cpp sequence, `committed_seq_ = max_seqs`. It only ever holds accepted state.

- **Prefill** decodes the prompt, minus its deferred last token, onto the committed sequence. It then runs `seq_cp(committed, 0)`.
- **Drafting** is unchanged. Seq 0 shares the committed recurrent cell until its first decode. At that point llama.cpp's `find_slot` copies the cell, so drafting never writes into the committed state. Attention KV cells are tag-shared as before.
- **Acceptance** (`accept_path`):
  1. `seq_rm(s, -1, -1)` for every draft sequence, seq 0 included. Full removal is always supported.
  2. Decode the round's seed plus the accepted tokens onto the committed sequence in one batch. That is at most `max_beam_len + 1` tokens.
  3. `seq_cp(committed, 0)` so the next round starts from it.

The cost is re-decoding the accepted path once per round. For the 0.8B draft on the phone's CPU, draft time was about 41–46 ms per round, including this re-decode.

Attention-only models take the old path inside `accept_path`: `collapse_to_seq(keep_seq, tip - 1)`, then a solo re-decode of the tip.

## Changes by file

### `src/graph_engine.h` / `src/graph_engine.cpp`

- `load_model`:
  - Sets `recurrent_ = llama_model_is_recurrent() || llama_model_is_hybrid()`.
  - In tree mode on such a model, reserves `committed_seq_ = max_seqs_` and creates the context with `n_seq_max = max_seqs_ + 1`. If `max_seqs` is 256 (llama.cpp's cap), it is lowered to 255 with a log line.
  - The existing per-sequence backend samplers also cover the extra sequence.
  - The "context ready" log line now prints `recurrent=0|1`.
- New `accept_path(keep_seq, path_tokens, path_positions)`: the end-of-round acceptance described above. It validates that the path is non-empty, has consecutive positions and fits in `n_batch`.
- `prefill`: on a recurrent tree-mode engine, decodes onto `committed_seq_` and then shares it to seq 0.
- `collapse_to_seq`: throws `std::logic_error` on a recurrent model.
- `gather` (linear mode): throws on a recurrent model when asked to drop decoded tokens.
- New accessor `recurrent()`.
- Removed `Config::n_rs_seq` and the `ctx_params.n_rs_seq` wiring. This was an earlier uncommitted attempt at the same bug.

### `src/spec_exec_client.cpp` / `.h`

- On a proactive miss, `ValidateTree` builds the path from the seed (`prefix - 1`) plus `fresh_slots` and calls `engine_.accept_path(...)`. Before, it called `collapse_to_seq` followed by `decode_token`.
- The constructor throws `std::invalid_argument` when proactive drafting is enabled with a recurrent draft model. `ProactiveDraft::ChooseBet` trims forked sequences by position, and a proactive hit keeps subtree sequences instead of a committed one. Neither works on a state that can only move forward.
- Updated the acceptance comments.

### `src/script/client.cpp`

- Removed `derive_n_rs_seq()` and `engine_config.n_rs_seq`.

### `src/local_test.cpp` (smoke test)

New `--mode linear|tree|all` flag (default `all`) and a `--reference <text>` flag. The process exits non-zero if any section fails.

- **linear**: the original greedy decode.
- **commit check** (new, `run_commit_check`):
  - Drives the tree engine by hand through rounds that leave rejected draft tokens behind: on the winning branch, on a losing fork, on a forked winner, and past a never-decoded tip.
  - After every `accept_path`, it compares the next prediction with a linear engine that only decoded the committed tokens.
  - Pass rule: the linear argmax must be in the tree's top-k with a log-prob within 0.25. The two engines use different kernels. On the unchanged attention-only path, Qwen3-0.6B drifts by up to about 0.11 and swaps the top-1 at near-ties.
- **tree** (new, `run_tree`):
  - Runs the real `SpecExecClient` against an in-process `OracleValidator` over prompt + reference text.
  - Checks that committed tokens match the reference, and reports rounds and tokens per round.

### Docs and configs

- `README.md`:
  - Models section: note on hybrid drafts.
  - `local_test` section: rewritten for the new sections and flags.
  - Proactive section: note that it is unavailable on hybrid drafts.
  - Repository-layout row for `local_test`.
- `CMakeLists.txt`: updated the `local_test` comment.
- New `config/client.cpu-qwen35.yaml` and `config/client.hexagon-qwen35.yaml`:
  - Qwen3.5 draft, `host: 202.122.49.242:17599`.
  - `max_n_beams 4`, `max_beam_len 2`, `max_branch_width 4`, `max_budget 16`, `max_seqs 32`.
  - Proactive disabled.

## Verification

Host (macOS, CPU, pinned llama.cpp `5202104b5`, `build_host_5202/`):

| Run | Result |
|---|---|
| Qwen3.5, tree mode, before the fix | crash on round 2 (`X = 6, Y = 6`) |
| Qwen3.5, `local_test` after the fix | linear ` Paris.`; commit check 25/25 pass, max drift 0.044; tree run 21 rounds, 2.67 tokens per round |
| Qwen3.5, `--flash-attn 1` | commit check 25/25; tree run 2.67 tokens per round |
| Qwen3-0.6B regression | commit check 26/26, max drift 0.106. Per-round acceptance sequence identical to the pre-change run |
| Negative control: one corrupted committed token | commit check drops to 6/25; acceptance drops to 1.96 per round |
| `proactive_draft_test` (Qwen3) | all checks pass |

Phone (Android arm64 CPU build via `build/build_project_android_5202.sh`, pushed as `client-cpu-q35` / `local_test-cpu-q35`):

| Run | Result |
|---|---|
| `local_test-cpu-q35` | commit check 25/25 pass, max drift 0.064; tree run 2.67 tokens per round |
| `client-cpu-q35` with `client.cpu-qwen35.yaml`, live against the Qwen3.5 target at `202.122.49.242:17599` | 2 requests, 55 rounds, no errors; 2.44 tokens per round; draft 41–46 ms per round; server wait 244–259 ms per round; fluent completions |

Reproduce on the host:

```sh
cmake --build build_host_5202 --target local_test
./build_host_5202/local_test --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf --flash-attn 0
```

## Known limitations

- **Proactive drafting** is refused for hybrid drafts. Supporting it would mean redoing `ChooseBet`'s scratch-fork trim and the hit-path splice so they never rewind a recurrent sequence.
- **Memory:** the recurrent-state buffer scales with `max_seqs + 1`, at about 19 MB per sequence for the 0.8B model. Set `max_seqs` explicitly rather than letting it derive, since it can derive up to 256.
- **Hexagon NPU is untested after this fix.** The earlier NPU run hung in `dspqueue` (`0xc` timeout waiting for `DSPQUEUE_SIGNAL_RESP_PACKET`). That run also had the 1849.5 MiB recurrent buffer on HTP0 from the old `n_rs_seq` setting, so it needs a retest.
- **`sample_req_cnt` is a stride, not a count.** `240` over the 480 specbench prompts runs 2 requests. Use `2` for 240 requests.
