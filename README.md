# mobile-specedge

Edge/client side of **SpecEdge** speculative decoding, in C++.

A small **draft model** runs locally through [llama.cpp](https://github.com/ggml-org/llama.cpp).
Each round it drafts a *tree* of candidate tokens and ships the whole tree to a
remote **target** (`SpecEdgeService`, a gRPC server that runs the large model) in
a single `Validate` RPC. The target returns which root-to-leaf path it accepts
plus one bonus token; the client commits that path and starts the next round.
This is a C++ port of the Python `specedge` project's client, intended to run on
a phone / edge device against a server-class target.

The **target server is not in this repository** — you need a running
`SpecEdgeService` from the Python `specedge` project (or a mock) to do anything
beyond the local smoke test.

---

## Repository layout

| Path | What it is |
|------|------------|
| `specedge.proto` | gRPC contract: `SpecEdgeService.Validate` / `.Sync`. |
| `src/specedge_grpc/` | **Pre-generated** protobuf/gRPC C++ stubs, checked in. Regenerate only with a matching `protoc` / `grpc_cpp_plugin` (see below). |
| `src/graph_engine.{h,cpp}` | `LlamaCppEngine` — wraps llama.cpp's KV cache. Linear mode (1 sequence) and tree mode (`kv_unified`, one sequence per draft branch). |
| `src/grpc_client.{h,cpp}` | `GrpcClient` — synchronous client for the `Validate` RPC. |
| `src/tree.{h,cpp}` | `Tree` — the client-side draft tree (slots, positions, parents, attention mask). |
| `src/spec_exec_client.{h,cpp}` | `SpecExecClient` — the draft + verify round loop (port of `specexec.py`). |
| `src/graph_engine_main.cpp` | → `graph_engine` binary. Local-only smoke test. |
| `src/client_main.cpp` | → `tree_client` binary. One `--prompt` against a live target. |
| `src/script/client.cpp` | → `client` binary. Config-driven batch run over a dataset. |
| `src/parity_test.cpp` | → `parity_test` binary. Fixed `Validate` request, for Python/C++ wire parity checks. |
| `src/config.h` | Env-var config reader kept for parity with the Python launcher. **Not used by any current binary.** |
| `src/metric/mobile.py` | Post-run latency/throughput analysis of the JSONL result logs. |
| `config/client.example.yaml` | Example config for the `client` binary. |
| `data/` | Prompt datasets: `mtbench`, `c4`, `oasst`, `wikitext` (JSON `[id, text]` arrays), `specbench` (JSONL chat turns). |

---

## Building

### Prerequisites

- **CMake ≥ 3.14** and a **C++17** compiler.
- **Network access on the first configure** — CMake `FetchContent` downloads and
  builds, pinned by tag:
  - llama.cpp `b10615`
  - nlohmann/json `v3.11.3`
  - yaml-cpp `0.8.0`
- **gRPC and Protobuf installed on the system**, discoverable via
  `find_package(Protobuf CONFIG)` and `find_package(gRPC CONFIG)`. These are
  *not* fetched (gRPC from source is a heavy build). Install via your package
  manager or a vcpkg/conan toolchain, e.g.:
  - macOS: `brew install grpc protobuf`
  - Debian/Ubuntu: `apt install libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev`
- A **GGUF draft model** (see [Models](#models)).
- *For a CUDA build:* the **NVIDIA CUDA Toolkit** (`nvcc`) and a matching
  driver — see [CUDA build](#cuda-build).

> The checked-in stubs in `src/specedge_grpc/` must be ABI-compatible with the
> gRPC/Protobuf you link against. If `find_package` picks up a very different
> version, regenerate them (below).

### Configure and build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binaries are written to `build/`: `graph_engine`, `tree_client`, `client`,
`parity_test`.

### GPU / accelerator offload

llama.cpp backend options pass straight through at configure time, e.g.
`-DGGML_METAL=ON` (Apple), `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`. At run time use
`--n-gpu-layers` (CLI) or `n_gpu_layers` (YAML): `-1` offloads all layers, `0` is
CPU-only.

### CUDA build

Builds the bundled llama.cpp (`b10615`) with its CUDA backend so the draft
model runs on an NVIDIA GPU.

**Prerequisites**

- NVIDIA GPU with a driver new enough for your CUDA Toolkit.
- **CUDA Toolkit** (`nvcc`) — 12.x recommended. Check with `nvcc --version`;
  `nvidia-smi` should list the GPU.
- CMake must be able to find CUDA. If `nvcc` is not on `PATH`, either add it
  (`export PATH=/usr/local/cuda/bin:$PATH`) or pass
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- gRPC / Protobuf as for any build (`find_package` must still resolve them).

**Configure and build**

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

- `-DGGML_CUDA=ON` is forwarded to the fetched llama.cpp; no other flag is
  needed to link the CUDA backend into the `graph_engine_lib` target.
- `-DCMAKE_CUDA_ARCHITECTURES=native` targets the build host's GPU. For a
  portable binary give explicit SMs instead, e.g. `"80;86;89"` (A100 / 30xx /
  40xx).
- Optional llama.cpp CUDA knobs, all passed the same way:
  `-DGGML_CUDA_FA_ALL_QUANTS=ON` (flash-attention for all quant types),
  `-DGGML_CUDA_FORCE_MMQ=ON`, `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=<n>`.
- First configure still needs network access (FetchContent), and the CUDA
  backend adds a few minutes to the build.

**Run on the GPU**

The binaries are unchanged — you just have to ask for offload, which is
CPU-only by default:

```sh
# tree_client: offload every layer to GPU 0
./build/tree_client \
  --host 127.0.0.1:8000 --dtype fp16 \
  --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf \
  --prompt "Explain speculative decoding in one paragraph." \
  --max-new-tokens 64 \
  --n-gpu-layers -1

# graph_engine smoke test on the GPU
./build/graph_engine \
  --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf \
  --prompt "The capital of France is" --n-generate 16 \
  --n-gpu-layers -1
```

For the `client` batch binary, set it in the YAML:

```yaml
client:
  n_gpu_layers: -1   # -1 = all layers, 0 = CPU only
  main_gpu: 0        # which CUDA device to use
  single_gpu: true   # whole model on main_gpu only; false splits layers across all visible GPUs
```

`--n-gpu-layers -1` (or `n_gpu_layers: -1`) offloads all layers; a positive
value offloads that many. `graph_engine` and `tree_client` expose only
`--n-gpu-layers` and always use device 0 — select another card with
`CUDA_VISIBLE_DEVICES=1 ./build/tree_client …`. The `client` binary also
takes `main_gpu:` in the YAML to pick the CUDA device directly.

By default every binary now loads with llama.cpp's `LLAMA_SPLIT_MODE_NONE`,
so the draft model stays on a single GPU (`main_gpu`). Set `single_gpu: false`
in the `client` YAML to restore llama.cpp's default layer-split across all
visible CUDA devices. On load, llama.cpp logs lines like
`load_tensors: offloaded 29/29 layers to GPU` — check those to confirm the
draft model is actually on the GPU.

### Regenerating the gRPC stubs (only if needed)

```sh
protoc -I . \
  --cpp_out=src/specedge_grpc \
  --grpc_out=src/specedge_grpc \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  specedge.proto
```

Then fix the includes if your protoc layout differs from the committed
`#include "specedge_grpc/specedge.grpc.pb.h"` form.

---

## Models

All local binaries need a GGUF draft model. The default path everywhere is:

```
models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf
```

Download it (e.g. from Hugging Face) and place it there, or pass `--model` /
set `draft_model:` in the YAML. Paths are resolved relative to the current
working directory, so run the binaries **from the project root**.

The large model runs on the target server and is configured there, not here.

---

## Running

### 1. `graph_engine` — local smoke test (no server)

Prefills a prompt and greedily decodes locally. Proves the build links against
real llama.cpp and that prefill/forward/gather round-trip.

```sh
./build/graph_engine \
  --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf \
  --prompt "The capital of France is" \
  --n-generate 16
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <path>` | `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf` | GGUF model path |
| `--prompt <text>` | `"The capital of France is"` | prompt to complete |
| `--max-len <n>` | `256` | context / `max_len` |
| `--n-generate <n>` | `8` | tokens to greedily decode |
| `--n-gpu-layers <n>` | `0` | layers to offload (`-1` = all) |
| `--n-threads <n>` | llama.cpp default | decode threads |
| `--n-threads-batch <n>` | = `--n-threads` | batch threads |

### 2. Start a target server

Bring up a `SpecEdgeService` (from the Python `specedge` project, or a mock)
listening on some `host:port`. Note its model dtype (`SPECEDGE_DTYPE` /
`base.dtype`) — the `Validate` wire format carries no dtype tag, so the client's
`--dtype` / `dtype:` **must match it exactly** (`fp16` default, or `fp32` /
`bf16`). Also keep the client's `max_budget` equal to the server's
`SPECEDGE_MAX_BUDGET` (the server sizes its cache buffers from it).

### 3. `tree_client` — one prompt against the target

```sh
./build/tree_client \
  --host 127.0.0.1:8000 \
  --dtype fp16 \
  --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf \
  --prompt "Explain speculative decoding in one paragraph." \
  --max-new-tokens 64
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--host <host:port>` | `localhost:50555` | `SpecEdgeService` address |
| `--dtype <fp32\|fp16\|bf16>` | `fp16` | `attention_mask` wire dtype — **match the server** |
| `--model <path>` | `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf` | GGUF draft model |
| `--prompt <text>` | `"The capital of France is"` | prompt to complete |
| `--max-len <n>` | `256` | context budget (prompt + generation) |
| `--max-n-beams <n>` | `4` | candidates expanded per draft step |
| `--max-beam-len <n>` | `8` | draft steps (tree depth) per round |
| `--max-branch-width <n>` | `2` | children sampled per candidate |
| `--max-budget <n>` | `16` | draft-node budget per round — **match the server** |
| `--max-seqs <n>` | `0` (auto) | llama.cpp sequences; `0` derives an upper bound, capped at 256 |
| `--max-new-tokens <n>` | `32` | tokens to generate |
| `--n-gpu-layers <n>` | `0` | layers to offload (`-1` = all) |
| `--n-threads <n>` / `--n-threads-batch <n>` | llama.cpp default | thread counts |

### 4. `client` — batch run over a dataset

Reads a YAML config, loads a dataset from `./data`, builds the request order
(`req_offset` slice → `sample_req_cnt` stride → shuffle seeded by `client_idx`),
and runs every prompt through tree speculative decoding against the target.

```sh
cp config/client.example.yaml config/client.yaml
# edit host, dtype, draft_model, dataset, ...
./build/client --config config/client.yaml
```

`--config, -c <path>` defaults to `config/client.example.yaml`. Run from the
project root. Every option lives under a single top-level `client:` key — see
[`config/client.example.yaml`](config/client.example.yaml) for the full,
commented list. Key fields:

| Field | Meaning |
|-------|---------|
| `draft_model` | GGUF path (relative to cwd) |
| `host` | target `SpecEdgeService` address |
| `dtype` | `fp32` / `fp16` / `bf16` — must equal the server's model dtype |
| `max_len` | context budget shared by prompt + generation |
| `max_n_beams`, `max_beam_len`, `max_branch_width`, `max_budget` | SpecExec tree-drafting params (`max_budget` must equal the server's) |
| `max_seqs` | llama.cpp sequences; `null`/`0` auto-derives, cap 256 |
| `max_new_tokens` | tokens generated per request |
| `client_idx` | seeds the per-client request shuffle; names the output log |
| `dataset` | one of `mtbench`, `c4`, `oasst`, `wikitext`, `specbench` |
| `max_request_num` | `-1` = whole dataset, else absolute upper index |
| `req_offset`, `sample_req_cnt` | start index, and take every Nth prompt |
| `n_gpu_layers`, `main_gpu`, `n_threads`, `n_threads_batch` | llama.cpp placement / threading |

### 5. `parity_test` — wire-format check

No arguments; sends a fixed `Validate` request to `localhost:50555` with
`fp32` mask dtype and prints the response. Used to diff the C++ `GrpcClient`
against the Python client, both against `mock_server.py`.

```sh
./build/parity_test
```

---

## Output logs

- **`log/client_<client_idx>.jsonl`** — one JSON record per draft+verify round
  (timings, accepted-token counts). Written by `SpecExecClient`, so both
  `tree_client` and `client` produce it. Truncated on first open per process,
  appended thereafter.
- **`graph-engine.log`** — per-forward debug log from `LlamaCppEngine`. Goes to
  the current directory, or to `$SPECEDGE_RESULT_PATH/$SPECEDGE_EXP_NAME/` when
  both env vars are set.

## Analyzing a run

`src/metric/mobile.py` turns the JSONL logs into a latency / throughput table.
It needs **both** the client logs and the server's `server.jsonl` (copy the
server's log next to the client ones):

```sh
pip install polars rich
mkdir -p run && cp log/client_*.jsonl run/ && cp /path/to/server.jsonl run/
python src/metric/mobile.py --data run/
```

| Flag | Meaning |
|------|---------|
| `-d, --data <dir>` | folder holding `client_*.jsonl` and `server.jsonl` |
| `-s, --subset <name>` | `multi_turn`, `translation`, `summarization`, `question_answering`, `mathematical_reasoning`, `retrieval`, or `overall` (default) — slices by `req_idx` for a SpecBench-ordered run |
| `--plain` | tab-separated values instead of the rich table |

---

## Typical end-to-end flow

1. Build the binaries (`cmake … && cmake --build build -j`).
2. `./build/graph_engine …` to confirm the draft model loads and decodes.
3. Start the `SpecEdgeService` target; note its dtype and `max_budget`.
4. `./build/client --config config/client.yaml` (matching `dtype` / `max_budget`).
5. Collect `log/client_*.jsonl` + the server's `server.jsonl`.
6. `python src/metric/mobile.py --data run/` for the summary.
