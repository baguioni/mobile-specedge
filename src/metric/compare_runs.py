"""Checks that two runs made the same decisions, step for step.

Written to answer one question -- did moving the draft scoring onto the
backend change what the tree does? -- and useful for any change that is
supposed to be decision-neutral: a llama.cpp bump, a tree-config tweak, a
refactor. Aggregate throughput cannot answer it, since two runs can share a
mean accept rate while diverging constantly, so this joins the per-step
records on (client_idx, req_idx, step_idx) and compares them exactly.

Two signals, both structural:

  num_accepted_tokens  how many draft tokens the target ratified. Depends on
                       every token in the tree and the order they were tried,
                       so it changes if scoring changes at all.
  draft.n_beams        beams expanded per tree level. A direct readout of
                       tree *shape*: which candidates survived the budget,
                       which is decided by the scores under comparison.

Identical on both, for every step, means the two runs made the same
decisions -- not merely that their averages agree.

Usage:  python3 src/metric/compare_runs.py <baseline_dir> <candidate_dir>
"""

import argparse
import json
import sys
from pathlib import Path


def load(folder: Path):
    files = sorted(folder.glob("client_*.jsonl"))
    if not files:
        sys.exit(f"No client_*.jsonl in {folder}")
    rows = {}
    for path in files:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows[(r["client_idx"], r["req_idx"], r["step_idx"])] = r
    if not rows:
        sys.exit(f"No records in {folder}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path, help="log dir of the known-good run")
    ap.add_argument("candidate_dir", type=Path, help="log dir of the run under test")
    ap.add_argument("--max-report", type=int, default=10)
    args = ap.parse_args()

    base = load(args.baseline_dir)
    cand = load(args.candidate_dir)

    print(f"baseline  {args.baseline_dir}: {len(base)} steps")
    print(f"candidate {args.candidate_dir}: {len(cand)} steps")

    only_base = base.keys() - cand.keys()
    only_cand = cand.keys() - base.keys()
    if only_base or only_cand:
        print(f"\nSTEP SETS DIFFER: {len(only_base)} only in baseline, "
              f"{len(only_cand)} only in candidate")
        for key in sorted(only_base)[:5]:
            print(f"   baseline-only  {key}")
        for key in sorted(only_cand)[:5]:
            print(f"   candidate-only {key}")

    shared = sorted(base.keys() & cand.keys())
    mismatches = []
    for key in shared:
        b, c = base[key], cand[key]
        diffs = []
        if b["num_accepted_tokens"] != c["num_accepted_tokens"]:
            diffs.append(
                f"accepted {b['num_accepted_tokens']} != {c['num_accepted_tokens']}")
        bn = b["draft"].get("n_beams")
        cn = c["draft"].get("n_beams")
        if bn is not None and cn is not None and bn != cn:
            diffs.append(f"n_beams {bn} != {cn}")
        if diffs:
            mismatches.append((key, diffs))

    print(f"\ncompared {len(shared)} shared steps")
    if not mismatches and not only_base and not only_cand:
        print("\nIDENTICAL: every step accepted the same number of tokens and "
              "expanded the same beams per level.")
        print("The two runs made the same decisions.")
        return 0

    print(f"{len(mismatches)} step(s) diverged "
          f"({100 * len(mismatches) / max(len(shared), 1):.2f}%)")
    for key, diffs in mismatches[:args.max_report]:
        print(f"   client={key[0]} req={key[1]} step={key[2]}: {'; '.join(diffs)}")
    if len(mismatches) > args.max_report:
        print(f"   ... and {len(mismatches) - args.max_report} more")

    # The first divergence is the one worth debugging: everything after it
    # runs on a tree that already differs.
    if mismatches:
        key = mismatches[0][0]
        print(f"\nfirst divergence at client={key[0]} req={key[1]} step={key[2]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
