"""Decomposes draft.end_to_end into its timed spans.

Companion to mobile.py, which reports draft.end_to_end as a single number.
This splits it into the spans GrowTree emits:

    forward   engine_.forward_batch_topk  (decode + softmax + top-k, all on
                                           the backend)
    fork      seq_cp + tree_.add          (O(kept children))
    residual  everything else             (candidate scan, joint nth_element,
                                           TrimByBudget -- all O(max_budget))

A residual anywhere near the size of the spans means the cost is somewhere
this breakdown does not yet look, and the spans need extending.

Logs written while scoring still ran on the host also carry `softmax` and
`topk` spans. Those are picked up automatically when present, so old and
new runs can both be read with this script.

Usage:  python3 src/metric/draft_breakdown.py <data_folder> [--per-level]
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# Present in every log. Host-scoring spans are added per-file when found.
BASE_SPANS = ("forward", "fork")
LEGACY_SPANS = ("softmax", "topk")


def load(folder: Path):
    files = sorted(folder.glob("client_*.jsonl"))
    if not files:
        sys.exit(f"No client_*.jsonl in {folder}")
    rows = []
    for path in files:
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        sys.exit("No records found.")
    return rows


def resolve_spans(rows):
    """The spans this log actually carries, in report order."""
    missing = [s for s in BASE_SPANS if s not in rows[0]["draft"]]
    if missing:
        sys.exit(
            f"Records lack the {', '.join(missing)} span(s) -- this log predates "
            "the GrowTree instrumentation. Rebuild the client and re-run."
        )
    legacy = [s for s in LEGACY_SPANS if s in rows[0]["draft"]]
    spans = ("forward",) + tuple(legacy) + ("fork",)

    # Every vector is pushed once per tree level, including on the two
    # early-exit paths, so a ragged record means a level exited somewhere
    # that forgot to push.
    for i, r in enumerate(rows):
        lengths = {s: len(r["draft"][s]) for s in spans}
        if len(set(lengths.values())) != 1:
            sys.exit(f"Record {i} is ragged: {lengths}")
    if legacy:
        print("note: this log scored on the host (softmax/topk spans present)")
    return spans


def summarize(rows, label, spans):
    per_step = {s: [sum(r["draft"][s]) for r in rows] for s in spans}
    resid = [r["draft"]["residual"] for r in rows]
    e2e = [r["draft"]["end_to_end"] for r in rows]
    target = [r["target"]["end_to_end"] for r in rows]
    step = [d + t for d, t in zip(e2e, target)]
    accepted = [r["num_accepted_tokens"] for r in rows]

    draft_mean = st.mean(e2e)
    step_mean = st.mean(step)

    print(f"\n===== {label}  (n={len(rows)}) =====")
    print(f"{'span':<12}{'mean ms':>10}{'% draft':>10}{'% step':>9}")
    print("-" * 41)
    for s in spans:
        m = st.mean(per_step[s])
        print(f"{s:<12}{m:>10.2f}{100 * m / draft_mean:>9.1f}%{100 * m / step_mean:>8.1f}%")
    m = st.mean(resid)
    print(f"{'residual':<12}{m:>10.2f}{100 * m / draft_mean:>9.1f}%{100 * m / step_mean:>8.1f}%")
    print("-" * 41)
    print(f"{'draft total':<12}{draft_mean:>10.2f}{100.0:>9.1f}%{100 * draft_mean / step_mean:>8.1f}%")
    print(f"{'target':<12}{st.mean(target):>10.2f}{'':>10}{100 * st.mean(target) / step_mean:>8.1f}%")
    print(f"{'STEP':<12}{step_mean:>10.2f}")

    cpu = sum(st.mean(per_step[s]) for s in spans if s != "forward") + st.mean(resid)
    acc_mean = st.mean(accepted)
    print(f"\n  accepted/step {acc_mean:.3f}   ->  {step_mean / acc_mean:.2f} ms/tok"
          f"   ({1000 * acc_mean / step_mean:.1f} tok/s)")
    label = "+".join([s for s in spans if s != "forward"] + ["residual"])
    print(f"  draft host-side ({label}): {cpu:.2f} ms")
    print(f"  if that host cost went to zero: {(step_mean - cpu) / acc_mean:.2f} ms/tok"
          f"   ({1000 * acc_mean / (step_mean - cpu):.1f} tok/s)")


def per_level(rows, spans):
    """Cost by tree level. Host-side spans scale linearly with n_beams;
    forward grows more slowly, since it is one batched decode either way."""
    depth = max(len(r["draft"]["forward"]) for r in rows)
    print(f"\n===== per tree level =====")
    print(f"{'level':<7}{'beams':>7}" + "".join(f"{s:>10}" for s in spans))
    print("-" * (14 + 10 * len(spans)))
    for lvl in range(depth):
        at = [r for r in rows if len(r["draft"]["forward"]) > lvl]
        beams = st.mean(r["draft"]["n_beams"][lvl] for r in at)
        cells = "".join(f"{st.mean([r['draft'][s][lvl] for r in at]):>10.2f}" for s in spans)
        print(f"{lvl:<7}{beams:>7.1f}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_folder", type=Path)
    ap.add_argument("--per-level", action="store_true")
    args = ap.parse_args()

    rows = load(args.data_folder)
    spans = resolve_spans(rows)

    non_prefill = [r for r in rows if r["step_idx"] != 0]
    summarize(rows, "ALL STEPS", spans)
    summarize(non_prefill, "NON-PREFILL", spans)
    if args.per_level:
        per_level(non_prefill, spans)


if __name__ == "__main__":
    main()
