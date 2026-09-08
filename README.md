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
`SpecEdgeService` from the Python `specedge` project (or a mock) to run
anything here.

---

## Repository layout

| Path | What it is |
|------|------------|
| `specedge.proto` | gRPC contract: `SpecEdgeService.Validate` / `.Sync` / `.Done`. Kept byte-identical to the Python `specedge` project's copy — that side owns it. |
| `src/specedge_grpc/` | **Pre-generated** protobuf/gRPC C++ stubs, checked in. Regenerate only with a matching `protoc` / `grpc_cpp_plugin` (see below). |
| `src/graph_engine.{h,cpp}` | `LlamaCppEngine` — wraps llama.cpp's KV cache. Linear mode (1 sequence) and tree mode (`kv_unified`, one sequence per draft branch). |
| `src/grpc_client.{h,cpp}` | `GrpcClient` — synchronous client for the `Validate` RPC. |
| `src/tree.{h,cpp}` | `Tree` — the client-side draft tree (slots, positions, parents, attention mask). |
| `src/spec_exec_client.{h,cpp}` | `SpecExecClient` — the draft + verify round loop (port of `specexec.py`). |
| `src/proactive_draft.{h,cpp}` | `ProactiveDraft` — bets on the bonus token and pre-grows next round's tree inside the `Validate` round-trip (port of `proactive.py`). |
| `src/script/client.cpp` | → `client` binary. Config-driven batch run over a dataset against a target server. |
| `src/local_test.cpp` | → `local_test` binary. Offline smoke test: prefill one prompt, greedily decode a few tokens through `LlamaCppEngine` (linear mode), print the completion. No target server. |
| `src/config.h` | Env-var config reader kept for parity with the Python launcher. **Not used by any current binary.** |
| `src/metric/mobile.py` | Post-run latency/throughput analysis of the JSONL result logs. |
| `src/test/` | Unit tests; build with `-DSPECEDGE_BUILD_TESTS=ON`. |
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

The build produces two binaries: `build/client` (the batch client) and
`build/local_test` (an offline llama.cpp smoke test — see
[`local_test`](#local_test--offline-smoke-test)).

### GPU / accelerator offload

llama.cpp backend options pass straight through at configure time, e.g.
`-DGGML_METAL=ON` (Apple), `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`. At run time set
`n_gpu_layers` in the YAML config: `-1` offloads all layers, `0` is CPU-only.

Two backends have their own sections below, since both need more than the one
flag: [CUDA](#cuda-build) for an NVIDIA edge box, and
[OpenCL](#android-build-opencl--adreno-gpu) for a Snapdragon phone's Adreno GPU.

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

The binary is unchanged — you just have to ask for offload in the YAML, which
is CPU-only by default:

```yaml
client:
  n_gpu_layers: -1   # -1 = all layers, 0 = CPU only
  main_gpu: 0        # which CUDA device to use
  single_gpu: true   # whole model on main_gpu only; false splits layers across all visible GPUs
```

`n_gpu_layers: -1` offloads all layers; a positive value offloads that many.
`main_gpu` picks the CUDA device directly, so `CUDA_VISIBLE_DEVICES` is not
needed.

The draft model loads with llama.cpp's `LLAMA_SPLIT_MODE_NONE`, so it stays on
a single GPU (`main_gpu`). Set `single_gpu: false` to restore llama.cpp's
default layer-split across all visible CUDA devices. On load, llama.cpp logs
lines like `load_tensors: offloaded 29/29 layers to GPU` — check those to
confirm the draft model is actually on the GPU.

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

`specedge.proto` is a copy of the Python `specedge` project's file and that
side owns it: when the contract moves there, copy it over and regenerate,
rather than editing this copy. Editing it here without regenerating is worse
than useless — the stubs keep the old contract while the `.proto` claims the
new one, and nothing in the build catches the divergence. The stubs' own
`#if PROTOBUF_VERSION != 7036000` guard pins the generating `protoc` to the
runtime you link (36.0); see [Building](#building).

### Android build (OpenCL / Adreno GPU)

Runs the draft model on a Snapdragon phone's Adreno GPU through llama.cpp's
OpenCL backend, instead of its CPU. Verified on a Snapdragon 8 Elite (Adreno
830) — see llama.cpp's [`docs/backend/OPENCL.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)
and Qualcomm's [backend announcement](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu).

**Prerequisites** — the Android NDK (r29 works; `brew install --cask
android-ndk` puts it in `/opt/homebrew/share/android-ndk`), Ninja, a host
`python3` (the kernel-embedding step runs `embed_kernel.py`), and gRPC /
Protobuf **cross-built for `arm64-v8a`** — the same prefix any Android build of
this project needs, since `CMakeLists.txt` takes them from the system rather
than `FetchContent`.

**1. OpenCL headers and ICD loader.** llama.cpp's own guide copies these into
the NDK's sysroot; installing them into the same prefix as gRPC keeps the NDK
untouched and is found the same way.

```sh
NDK=/opt/homebrew/share/android-ndk
PREFIX=$PWD/build/android-deps          # the prefix holding gRPC/Protobuf too

git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers
cmake -S OpenCL-Headers -B ocl-headers-build -G Ninja \
  -DBUILD_TESTING=OFF -DOPENCL_HEADERS_BUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build ocl-headers-build --target install

git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader
cmake -S OpenCL-ICD-Loader -B ocl-icd-build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOPENCL_ICD_LOADER_HEADERS_DIR="$PREFIX/include" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build ocl-icd-build && cmake --install ocl-icd-build
```

The `libOpenCL.so` this produces is only a **link-time stand-in**. On the
device the loader resolves to the vendor's real Adreno driver at
`/vendor/lib64/libOpenCL.so`, so it is never pushed to the phone.

**2. Configure and build.** Use a separate build directory so a CPU build in
`build/android` survives alongside it.

```sh
export PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig"   # pin pkg-config to the cross prefix
cmake -S . -B build/android-opencl -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_FIND_ROOT_PATH="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON -DPKG_CONFIG_ARGN=--static \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DOpenCL_INCLUDE_DIR="$PREFIX/include" \
  -DOpenCL_LIBRARY="$PREFIX/lib/libOpenCL.so"
cmake --build build/android-opencl -j
```

Configure should report `Found OpenCL … (found version "3.0")` and `OpenCL will
use matmul kernels optimized for Adreno`. Both OpenCL options are already ON by
default; they are passed explicitly so the intent is visible.

| flag | why |
|---|---|
| `GGML_OPENCL_EMBED_KERNELS` | compiles the `.cl` sources into the binary — nothing extra to push |
| `GGML_OPENCL_USE_ADRENO_KERNELS` | Adreno-tuned matmul kernels |
| `GGML_CPU_ARM_ARCH=…+dotprod+fp16+i8mm` | still worth setting: anything not offloaded runs on CPU, and cross-compiling cannot probe these (needs Snapdragon 8 Gen 1 or newer) |
| `OpenCL_INCLUDE_DIR` / `OpenCL_LIBRARY` | pointed at the prefix so `find_package(OpenCL)` does not pick up a host OpenCL |

**3. Deploy and run.** The binary links `libOpenCL.so`, resolved on-device from
`/vendor/lib64`; everything else is static.

```sh
adb push build/android-opencl/client /data/local/tmp/specedge/client-cl
adb shell chmod 755 /data/local/tmp/specedge/client-cl
```

Then set `n_gpu_layers: -1` in the YAML (`0` leaves it on the CPU — the flag is
what actually moves work to the GPU; building with OpenCL alone changes
nothing). Confirm the offload from the load log:

```
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
ggml_opencl: device: 'QUALCOMM Adreno(TM) 830 (OpenCL 3.0 Adreno(TM) 830)'
llama_prepare_model_devices: using device GPUOpenCL (QUALCOMM Adreno(TM) 830) - 4532 MiB free
load_tensors: offloaded 29/29 layers to GPU
```

`./local_test-cl --n-gpu-layers -1` is the quickest way to see those lines
without needing a target server.

**Troubleshooting**

| symptom | cause |
|---|---|
| `CANNOT LINK EXECUTABLE … library "libOpenCL.so" not found` | device has no OpenCL driver, or it is outside the linker namespace; check `ls /vendor/lib64/libOpenCL.so` |
| build fails in `find_package(OpenCL)` | headers/loader not installed into `$PREFIX`, or `CMAKE_FIND_ROOT_PATH` not pointing at it |
| runs but `offloaded 0/N layers` | `n_gpu_layers` still `0` in the YAML |

---

## Models

The `client` binary needs a GGUF draft model. The default path is:

```
models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf
```

Download it (e.g. from Hugging Face) and place it there, or set `draft_model:`
in the YAML. Paths are resolved relative to the current working directory, so
run the binary **from the project root**.

The large model runs on the target server and is configured there, not here.

---

## Running

### `local_test` — offline smoke test

No target server, no gRPC. Loads a GGUF draft model into `LlamaCppEngine` in
linear mode, prefills a prompt, greedily decodes a few tokens through
`forward()`, exercises `gather()`/`reset()`, and prints the completion. Use it
to confirm the llama.cpp build links and runs on this machine before wiring up
a target.

```sh
./build/local_test --prompt "The capital of France is" --n-generate 12
# Prompt: The capital of France is
# Completion:  Paris, and the capital of Italy is Rome. The capital
```

| Flag | Meaning |
|------|---------|
| `--model <path>` | GGUF model path (default `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf`) |
| `--prompt <text>` | prompt to complete (default `"The capital of France is"`) |
| `--max-len <n>` | context / `max_len` passed to `LlamaCppEngine` (default 256) |
| `--n-generate <n>` | tokens to greedily decode (default 8) |
| `--n-gpu-layers <n>` | layers to offload to GPU, `-1` for all (default 0) |
| `--n-threads <n>`, `--n-threads-batch <n>` | llama.cpp threading (default: llama.cpp's own) |

Run from the project root so the default model path resolves.

### 1. Start a target server

Bring up a `SpecEdgeService` (from the Python `specedge` project, or a mock)
listening on some `host:port`. Note its model dtype (`SPECEDGE_DTYPE` /
`base.dtype`) — the `Validate` wire format carries no dtype tag, so the client's
`dtype:` **must match it exactly** (`fp16` default, or `fp32` /
`bf16`). Also keep the client's `max_budget` equal to the server's
`SPECEDGE_MAX_BUDGET` (the server sizes its cache buffers from it).

### 2. `client` — batch run over a dataset

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
| `max_seqs` | llama.cpp sequences; `null`/`0` auto-derives (proactive branches included), cap 256 |
| `proactive.type` | `disabled` (default), `excluded`, or `included` — see below |
| `proactive.max_n_beams`, `.max_beam_len`, `.max_branch_width`, `.max_budget` | subtree shape; `max_branch_width` must be `<=` the top-level one |
| `max_new_tokens` | tokens generated per request |
| `client_idx` | seeds the per-client request shuffle; names the output log |
| `result_path`, `exp_name` | run outputs go to `<result_path>/<exp_name>/`; both empty ⇒ `./log` (see [Output logs](#output-logs)) |
| `dataset` | one of `mtbench`, `c4`, `oasst`, `wikitext`, `specbench` |
| `max_request_num` | `-1` = whole dataset, else absolute upper index |
| `req_offset`, `sample_req_cnt` | start index, and take every Nth prompt |
| `n_gpu_layers`, `main_gpu`, `n_threads`, `n_threads_batch` | llama.cpp placement / threading |

### Proactive draft

With `proactive.type` set to anything but `disabled`, the `Validate` RPC moves
to a worker thread and the draft model spends the round-trip betting on the
bonus token the target is about to return, pre-growing next round's tree from
that bet. A hit splices the subtree in; a miss discards it and costs only the
window it ran in. `excluded` banks the hit as a deeper tree; `included` banks
it as a shorter draft, reducing `max_beam_len` by `proactive.max_beam_len`
(which must therefore be strictly smaller).

Only work that fits inside the round-trip is free, so `proactive.max_beam_len`
is the parameter to tune — one unit costs roughly one draft level. Every round
logs what it needs to size it:

| field in `client_<idx>.jsonl` | |
|---|---|
| `target.client_wait` | the window available (RPC + overlapped draft) |
| `target.proactive_ms` | what the bet actually cost |
| `target.proactive_ran` | a bet was placed |
| `target.proactive` / `.prev_proactive` | it was right (this round / the previous one) |
| `target.proactive_nodes` | subtree size, spliced or discarded |

A hit rate near zero, or `proactive_ms` well over `client_wait`, means the
setting is costing more than it returns.

---

## Output logs

Everything a run writes goes to one directory, named by two config fields that
mirror the Python side's `base.result_path` / `base.exp_name`:

```yaml
client:
  result_path: "result/mobile"
  exp_name: "mobile"
```

With both set, the files below land in **`<result_path>/<exp_name>/`**; leave
either empty (or omit it) and they fall back to `./log`, the previous
behaviour. Paths are relative to the working directory, so run from the
project root. The `client` binary prints the directory it chose on startup.

- **`<log dir>/client_<client_idx>.jsonl`** — one JSON record per draft+verify round
  (timings, accepted-token counts). Written by `SpecExecClient`. Truncated on
  first open per process, appended thereafter. Per-round fields for bucketing a
  run the way `llama-bench`'s `-p` / `-d` sweeps do:

  | field | |
  |---|---|
  | `context_len` | committed KV depth this round conditions on (prompt + all accepted so far) |
  | `prompt_len` | this request's prompt token count |
  | `draft.n_nodes` | draft tree size shipped to the target; `num_accepted_tokens / draft.n_nodes` is draft efficiency |
- **`<log dir>/trace.txt`, `<log dir>/trace.jsonl`** — per-request prompt and
  completion, for diffing two runs against each other.
- **`graph-engine.log`** — per-forward debug log from `LlamaCppEngine`. Goes to
  the current directory, or to `$SPECEDGE_RESULT_PATH/$SPECEDGE_EXP_NAME/` when
  both env vars are set — which is exactly what `client` exports from
  `result_path` / `exp_name`, so it joins the rest of the run's files.

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

1. Build the binary (`cmake … && cmake --build build -j`).
2. Start the `SpecEdgeService` target; note its dtype and `max_budget`.
3. `./build/client --config config/client.yaml` (matching `dtype` / `max_budget`).
4. Collect `log/client_*.jsonl` + the server's `server.jsonl`.
5. `python src/metric/mobile.py --data run/` for the summary.
