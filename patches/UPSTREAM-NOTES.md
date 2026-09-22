# Hexagon backend limits worth fixing upstream

Findings from moving this project's draft sampler onto HTP0 (llama.cpp
`ce8caa6e6` / b11065, Snapdragon 8 Elite Gen 5, Hexagon v81, 8 MB VTCM).
Nothing here is patched locally — the client works around all of it. Each item
is small, self-contained and measurable, so each is worth filing.

Context: the draft model emits a vocabulary-wide logits row (151936 for
Qwen3-0.6B, 248320 for Qwen3.5-0.8B) and the sampler has to take a top-k and a
full-vocabulary log normalizer over it. That is an ordinary shape for a
language model, and it is exactly where the Hexagon backend's per-op limits
bite.

---

## 1. Elementwise ops reject vocabulary-wide rows (VTCM-TOO-SMALL)

**Where:** `ggml/src/ggml-hexagon/htp/binary-ops.c:783-822`, and the same
pattern in `htp/act-ops.c:556-578`.

**What happens:** the kernels stage whole rows in VTCM — for a binary op, two
source rows plus a destination row, double-buffered, per DSP thread:

```c
spad_row_total  = 2 * (src0_row_size_aligned + dst_row_size_aligned);  // scalar src1
rows_per_buffer = octx->ctx->vtcm_size / (n_threads * spad_row_total);
if (rows_per_buffer < 1) return HTP_STATUS_VTCM_TOO_SMALL;
```

A 151936-column f32 row is 593 KB, so `spad_row_total` is 2.4 MB and four
threads need 9.5 MB against 8 MB of VTCM. `ggml_sub(logits, max_logit)` is
therefore rejected **at run time**, after `supports_op` already claimed it.
`ggml_hexagon_supported_binary` (`ggml-hexagon.cpp:5061`) has no width check at
all, so the scheduler places the op on HTP0 and the graph aborts:

```
ggml-hex: HTP0 dspcall : dsp-rsp VTCM-TOO-SMALL
ggml-hexagon.cpp:6446: ggml-hex: HTP0 get-tensor-async failed : dsp-error VTCM-TOO-SMALL
```

**Two separate bugs.** The kernel cannot tile within a row, and `supports_op`
does not reflect the limit it will hit. The second is the more serious one: a
false positive from `supports_op` is an abort rather than a CPU fallback.

**Fix, in order of value:** tile the row (process it in VTCM-sized column
chunks, as `htp/argsort-ops.c`'s `op_top_k_single_row_threaded` already does for
TOP_K); failing that, at minimum make `ggml_hexagon_supported_binary` and
`ggml_hexagon_supported_unary` reject widths that will not fit, so the
scheduler falls back instead of aborting.

**Workaround here:** `src/topk_sampler.cpp` folds the vocabulary row into a
`[n / rows, rows]` shape before the elementwise work (`fold_rows()`). The ops
are elementwise, so the arithmetic is unchanged, and `sum_rows` over the folded
shape is a partial sum that a second `sum_rows` finishes. Every node then runs
on HTP0.

## 2. `TOP_K` corrupts a sampler's candidate output at vocabulary width

**Where:** `ggml/src/ggml-hexagon/htp/argsort-ops.c`'s
`op_top_k_single_row_threaded` (the chunk+merge path taken when
`total_rows == 1 && ne00 > 1024`), or the backend-sampler plumbing around it.

**What happens:** with one backend sampler per sequence over 192 sequences
(llama.cpp's `llama_sampler_seq_config`), a draft replay aborts deterministically
part way through:

```
init: invalid token[31] = 1065382555
decode: failed to initialize batch
```

`1065382555` is `0x3F807295`, a float of about 1.0035 — a *probability* sitting
in the token-id buffer that `llama_get_sampled_candidates_ith()` returns. Same
request, same batch row, same value across reruns, so not a race.

**It goes away with `GGML_HEXAGON_OPFILTER=TOP_K`**, everything else identical:
the same replay then completes, and its accepted-token counts match the CPU
reference round for round. So the values the sampler computes are right; one
candidate row's buffer gets a probability written over it.

Up to the abort, 34 of 34 rounds match the reference exactly, so this is rare and
data-dependent rather than systematic.

**Repro:** `experiments/09-21-26/q3-npu-backend-sampler/` in this repo has the
configs, the logs and the exact command line. Qwen3-0.6B-Q8_0, vocabulary 151936,
`k = 16`, 32-row batches, 192 sequences.

## 3. `TOP_K` at vocabulary width is 16x its own matmul

**Where:** same kernel.

**Measured** with `GGML_HEXAGON_PROFILE=1` on an S25 Ultra (Hexagon v81), over
1624 calls:

| op | shape | avg usec |
|---|---|---:|
| `TOP_K` | `151936:1 -> 16:1` | **1918.5** |
| `MUL_MAT` (the head that produced those logits) | `1024:151936 x 1024:32` | 3849 for 32 rows = **120 per row** |

Selecting the 16 largest values out of a row costs 16x the 155M-MAC q8_0 matmul
that produced it. For comparison, six ARM CPU threads do strictly more work over
the same row — a full log-softmax *and* the top-k — in 710 usec, 2.7x faster.

The other nine nodes of the sampler (`SUB`, `EXP`, `SUM_ROWS`, `LOG`, `ADD`,
`GET_ROWS`) total 124 usec per row, so the kernel is not merely "wide rows are
slow": it is specifically this one. Worth profiling the chunk+merge path's
`bitonic_sort_vtcm_desc` against a straight selection scan at `k << ne00`; a
16-element selection does not need a full sort of each chunk.

This is the reason speculative decoding cannot yet move its draft scoring onto
HTP, even though every op is now supported and the graph stays at 2 splits.

## 4. `SOFT_MAX`'s 131072-column cap is both too low and too high

**Where:** the cap is at `ggml/src/ggml-hexagon/ggml-hexagon.cpp:5289-5296`; the
allocation it is supposed to guard is at `htp/softmax-ops.c:379-405`.

```c
#define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision
if (ne0 > SOFTMAX_MAX_ROW_SIZE) return false;
```

**Too low** for its stated reason. Every current Qwen3 vocabulary is wider
(151936 for 0.6B, 248320 for the 3.5 hybrid), so a softmax over logits always
falls back to the CPU and splits the graph — which then drags the whole logits
row across the bus, the most expensive thing a sampler can do. The precision
justification does not hold up either: the kernel already subtracts the row
maximum before exponentiating (`hvx_vec_reduce_max_f32` at
`htp/softmax-ops.c:185`, `hvx_reduce_max_f32` at `:324`), so accumulating 151936
terms is not meaningfully worse than accumulating 131072. If the f32 accumulator
is the worry, a pairwise or Kahan sum fixes it at any width; a hard reject does
not.

**Too high** for the kernel behind it, which is the more serious half.
`execute_op_softmax_f32` stages **4 rows per thread in VTCM, for src0, src1 and
dst alike** — src1 unconditionally, even when there is no mask:

```c
octx->src0_spad.size_per_thread = hex_round_up(4 * src0_row_size, 128);
octx->src1_spad.size_per_thread = hex_round_up(4 * src1_row_size, 128);
octx->dst_spad.size_per_thread  = hex_round_up(4 * dst_row_size,  128);
...
if (octx->ctx->vtcm_size < spad_size) return HTP_STATUS_VTCM_TOO_SMALL;
```

That is `3 * 4 * ne0 * 4 * n_threads` = `48 * ne0 * n_threads` bytes. On an S25
Ultra (`ggml-hex: HTP0 hwinfo: threads 6, hvx 6, hmx 1, vtcm 8 MB`) the real
ceiling is:

| HVX threads | max `ne0` that fits 8 MB VTCM |
|---:|---:|
| 6 (default) | **29127** |
| 4 | 43690 |
| 2 | 87381 |
| 1 | 174762 |

So at the default thread count the kernel tops out around **29k columns, 4.5x
below the declared cap**. Any softmax between 29127 and 131072 columns passes
`ggml_hexagon_supported_softmax`, gets scheduled on HTP0, and then aborts at run
time with `VTCM-TOO-SMALL` — the same false-positive-`supports_op` failure as
item 1, which this project hit for real on `SUB`. Raising the constant to fit a
vocabulary would widen that window rather than close it.

**Fix:** the two halves need fixing together. Make the kernel tile within a row
(softmax needs a global max and sum, so either two passes over the row or an
online/streaming normalizer), and derive the `supports_op` bound from the same
VTCM arithmetic the kernel uses instead of a hard-coded constant. Skipping the
`src1` scratchpad when there is no mask is a free 33% on its own.

**Workaround here:** the sampler builds the normalizer by hand out of `SUB` /
`EXP` / `SUM_ROWS` / `LOG`, none of which are capped, over a folded `[ne0/8, 8]`
shape. Measured at 124 usec per 151936-wide row on HTP0 — so for this use case
the softmax kernel is not on the critical path at all even if it were fixed.
Item 3 is.

## 5. Multi-row `TOP_K` is capped at 65536 columns

**Where:** `ggml/src/ggml-hexagon/ggml-hexagon.cpp:5403-5410`.

```c
// Single row uses the threaded chunk+merge path. Multi-row uses one full
// buffer per thread, so it keeps the tighter 64K cap.
const int64_t max_ne00 = single_row ? (256*1024) : (64*1024);
```

The single-row path handles 262144 columns by splitting the row across threads
and merging. The multi-row path keeps the tighter cap only because it allocates
one full padded buffer per thread.

This matters for speculative decoding specifically: a draft tree level is 32
beams wide, and one batched `[n_vocab, 32]` top-k would be strictly better than
32 single-row ops. Applying the chunk+merge strategy per row would lift the cap
and remove the reason for the asymmetry.

## 6. `TOP_K` output order is undefined, and deliberately so

**Where:** `ggml/src/ggml-cpu/ops.cpp:8589-8592`.

```c
// emphasize that the order is not important
if (top_k > 1) std::swap(dst_data[0], dst_data[1]);
```

Defensible as a contract, but every caller that wants ranked candidates — which
is most of them, including llama.cpp's own top-k sampler feeding a chain — has
to re-sort. A documented `sorted` flag on the op, or at least a line in
`ggml.h` stating the contract (it is currently stated only by that swap), would
save each caller from rediscovering it.

**Workaround here:** `LlamaCppEngine::forward_batch_topk`
(`src/graph_engine.cpp`) sorts each row best-first after readback, since
`TopKRows` is documented as ordered and several callers read a row's leading
entries as its best ones.
