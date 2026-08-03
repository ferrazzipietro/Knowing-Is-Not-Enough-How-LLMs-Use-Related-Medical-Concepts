"""
sweep_correlation_params.py

Runs correlation_matrix.py (unmodified) across a grid of hyperparameter
settings and reports how the resulting pair set changes, so you can back
a "these thresholds are robust" (or "aren't") claim with real numbers
instead of an anecdotal spot-check.

Two things are swept differently, deliberately:
  - min_phi, max_p, min_support, min_cooccurrence: swept on the FILTERED
    output (correlation_pairs_n_bins_<n>.xlsx), since that's what these
    parameters directly gate.
  - n_bins: swept using --save_full and checked against the UNFILTERED
    pair list (correlation_pairs_full.xlsx), not just the reported top-N.
    This matters: if you only look at the already-filtered pairs, n_bins
    can trivially show "no effect" simply because none of the surviving
    pairs involve a binnable item type (NUMBER / NUMBER_OR_CATEGORY) --
    see check_binnable_types.py. Checking the full, unfiltered candidate
    list is the only way to know whether n_bins does anything at all.

For each sweep, we report:
  - number of pairs retained
  - Jaccard similarity of the pair SET (as (item_a,value_a,item_b,value_b)
    tuples) against the default-parameter run
  - which pairs, if any, entered/left relative to the default run

Usage:
  python sweep_correlation_params.py --script correlation_matrix.py
  python sweep_correlation_params.py --script correlation_matrix.py --quick
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def pair_key(row):
    return (row["item_a"], row["value_a"], row["item_b"], row["value_b"])


def load_pairs(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_excel(path)
    return set(df.apply(pair_key, axis=1))


def run(script: str, args: list, out_dir: Path):
    cmd = [sys.executable, script] + args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  !! FAILED")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return None
    return result


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def report_sweep(name, baseline_pairs, runs):
    """runs: list of (label, pairs_set)"""
    print(f"\n{'='*90}\nSWEEP: {name}\n{'='*90}")
    print(f"{'setting':<25} {'n_pairs':>8} {'jaccard_vs_default':>20} {'entered':>10} {'left':>8}")
    for label, pairs in runs:
        j = jaccard(baseline_pairs, pairs)
        entered = len(pairs - baseline_pairs)
        left = len(baseline_pairs - pairs)
        print(f"{label:<25} {len(pairs):>8} {j:>20.3f} {entered:>10} {left:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="correlation_matrix.py", help="Path to correlation_matrix.py")
    parser.add_argument("--out_dir", default="data/eCream-gamma", help="OUT_DIR the script writes to (must match its internal OUT_DIR)")
    parser.add_argument("--quick", action="store_true", help="Smaller grid, for a fast sanity check")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ---- Baseline (all defaults) ----
    print("Running baseline (default parameters) …")
    run(args.script, [], out_dir)
    baseline_filtered = load_pairs(out_dir / "correlation_pairs_n_bins_5.xlsx")
    print(f"  Baseline: {len(baseline_filtered)} filtered pairs.")

    # ---- n_bins sweep, checked on the FULL unfiltered candidate list ----
    n_bins_grid = [3, 5] if args.quick else [3, 4, 5, 6, 7, 8, 10]
    print("\nRunning n_bins sweep (on --save_full, i.e. the unfiltered candidate list) …")
    run(args.script, ["--n_bins", "5", "--save_full"], out_dir)
    baseline_full = load_pairs(out_dir / "correlation_pairs_full.xlsx")
    n_bins_runs = [("n_bins=5 (default)", baseline_full)]
    for nb in n_bins_grid:
        if nb == 5:
            continue
        run(args.script, ["--n_bins", str(nb), "--save_full"], out_dir)
        pairs = load_pairs(out_dir / "correlation_pairs_full.xlsx")
        n_bins_runs.append((f"n_bins={nb}", pairs))
    report_sweep("n_bins (unfiltered candidate pairs)", baseline_full, n_bins_runs)

    # Also report it on the filtered/reported set for direct comparison
    n_bins_filtered_runs = []
    for nb in ([5] + n_bins_grid):
        run(args.script, ["--n_bins", str(nb)], out_dir)
        pairs = load_pairs(out_dir / f"correlation_pairs_n_bins_{nb}.xlsx")
        n_bins_filtered_runs.append((f"n_bins={nb}", pairs))
    report_sweep("n_bins (filtered/reported pairs, min_phi=0.3, max_p=0.05)", baseline_filtered, n_bins_filtered_runs)

    # ---- min_phi sweep ----
    min_phi_grid = [0.25, 0.3, 0.35] if args.quick else [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]
    print("\nRunning min_phi sweep …")
    min_phi_runs = []
    for mp in min_phi_grid:
        run(args.script, ["--min_phi", str(mp)], out_dir)
        pairs = load_pairs(out_dir / "correlation_pairs_n_bins_5.xlsx")
        min_phi_runs.append((f"min_phi={mp}", pairs))
    report_sweep("min_phi", baseline_filtered, min_phi_runs)

    # ---- min_support sweep ----
    min_support_grid = [5, 10, 20] if args.quick else [5, 8, 10, 15, 20]
    print("\nRunning min_support sweep …")
    min_support_runs = []
    for ms in min_support_grid:
        run(args.script, ["--min_support", str(ms)], out_dir)
        pairs = load_pairs(out_dir / "correlation_pairs_n_bins_5.xlsx")
        min_support_runs.append((f"min_support={ms}", pairs))
    report_sweep("min_support", baseline_filtered, min_support_runs)

    # ---- min_cooccurrence sweep ----
    min_cooc_grid = [3, 5, 10] if args.quick else [3, 5, 7, 10, 15]
    print("\nRunning min_cooccurrence sweep …")
    min_cooc_runs = []
    for mc in min_cooc_grid:
        run(args.script, ["--min_cooccurrence", str(mc)], out_dir)
        pairs = load_pairs(out_dir / "correlation_pairs_n_bins_5.xlsx")
        min_cooc_runs.append((f"min_cooccurrence={mc}", pairs))
    report_sweep("min_cooccurrence", baseline_filtered, min_cooc_runs)

    print("\nDone. Note: max_p is not swept -- your reported p-values are on the order of "
          "1e-300 to 1e-240 even for the weakest retained pairs, many orders of magnitude "
          "below the 0.05 threshold, so max_p is almost certainly never the binding "
          "constraint and sweeping it is unlikely to be informative.")


if __name__ == "__main__":
    main()