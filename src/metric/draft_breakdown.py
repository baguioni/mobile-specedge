"""Decomposes draft.end_to_end into its four timed spans.

Companion to mobile.py, which reports draft.end_to_end as a single number.
That number is ~60% of a Mobile-SpecEdge step and only ~24% of it is the
engine call, so this splits the rest into the spans GrowTree now emits:

    forward   engine_.forward_batch   (GPU decode + logit readback)
    softmax   row_max + sum_exp       (log-softmax denominator, O(beams*vocab))
    topk      TopKIndices             (O(beams*vocab))
    fork      seq_cp + tree_.add      (O(kept children))
    residual  everything else         (candidate scan, joint nth_element,
                                       TrimByBudget -- all O(max_budget))

A residual anywhere near the size of the spans means the cost is somewhere
this breakdown does not yet look, and the spans need extending.

Usage:  python3 src/metric/draft_breakdown.py <data_folder> [--per-level]
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

SPANS = ("forward", "softmax", "topk", "fork")


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


def check_schema(rows):
    missing = [s for s in SPANS if s not in rows[0]["draft"]]
    if missing:
        sys.exit(
            f"Records lack the {', '.join(missing)} span(s) -- this log predates "
            "the GrowTree instrumentation. Rebuild the client and re-run."
        )
    # All four vectors are pushed once per tree level, including on the two
    # early-exit paths, so a ragged record means a level exited somewhere
    # that forgot to push.
    for i, r in enumerate(rows):
        lengths = {s: len(r["draft"][s]) for s in SPANS}
        if len(set(lengths.values())) != 1:
            sys.exit(f"Record {i} is ragged: {lengths}")


def summarize(rows, label):
    per_step = {s: [sum(r["draft"][s]) for r in rows] for s in SPANS}
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
    for s in SPANS:
        m = st.mean(per_step[s])
        print(f"{s:<12}{m:>10.2f}{100 * m / draft_mean:>9.1f}%{100 * m / step_mean:>8.1f}%")
    m = st.mean(resid)
    print(f"{'residual':<12}{m:>10.2f}{100 * m / draft_mean:>9.1f}%{100 * m / step_mean:>8.1f}%")
    print("-" * 41)
    print(f"{'draft total':<12}{draft_mean:>10.2f}{100.0:>9.1f}%{100 * draft_mean / step_mean:>8.1f}%")
    print(f"{'target':<12}{st.mean(target):>10.2f}{'':>10}{100 * st.mean(target) / step_mean:>8.1f}%")
    print(f"{'STEP':<12}{step_mean:>10.2f}")

    cpu = sum(st.mean(per_step[s]) for s in ("softmax", "topk", "fork")) + st.mean(resid)
    acc_mean = st.mean(accepted)
    print(f"\n  accepted/step {acc_mean:.3f}   ->  {step_mean / acc_mean:.2f} ms/tok"
          f"   ({1000 * acc_mean / step_mean:.1f} tok/s)")
    print(f"  draft CPU (softmax+topk+fork+residual): {cpu:.2f} ms")
    print(f"  if that CPU cost went to zero: {(step_mean - cpu) / acc_mean:.2f} ms/tok"
          f"   ({1000 * acc_mean / (step_mean - cpu):.1f} tok/s)")


def per_level(rows):
    """Cost by tree level. softmax/topk should scale with n_beams; forward
    grows more slowly (one batched decode either way)."""
    depth = max(len(r["draft"]["forward"]) for r in rows)
    print(f"\n===== per tree level =====")
    print(f"{'level':<7}{'beams':>7}" + "".join(f"{s:>10}" for s in SPANS))
    print("-" * 54)
    for lvl in range(depth):
        at = [r for r in rows if len(r["draft"]["forward"]) > lvl]
        beams = st.mean(r["draft"]["n_beams"][lvl] for r in at)
        cells = "".join(f"{st.mean([r['draft'][s][lvl] for r in at]):>10.2f}" for s in SPANS)
        print(f"{lvl:<7}{beams:>7.1f}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_folder", type=Path)
    ap.add_argument("--per-level", action="store_true")
    args = ap.parse_args()

    rows = load(args.data_folder)
    check_schema(rows)

    non_prefill = [r for r in rows if r["step_idx"] != 0]
    summarize(rows, "ALL STEPS")
    summarize(non_prefill, "NON-PREFILL")
    if args.per_level:
        per_level(non_prefill)


if __name__ == "__main__":
    main()
