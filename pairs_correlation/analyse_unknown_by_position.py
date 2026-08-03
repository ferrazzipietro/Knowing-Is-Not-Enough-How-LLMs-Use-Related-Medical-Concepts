"""
analyze_unknown_by_position.py

Diagnostic script: for a results JSON file (one-shot / per-note style, where
a single generation produces values for ALL crf_items at once), this looks
at the raw_output of each note (the one row per note that has raw_output
!= null), parses the big JSON blob of item->value, and reports how often
"unknown" appears as a function of WHERE the item sits in that generated
JSON (first item, second item, ..., last item).

The purpose is to test whether extraction quality degrades over the course
of a single long generation (a known LLM failure mode for long structured
outputs), which would show up as an increasing "unknown" rate for items
that appear later in the schema/JSON.

Usage:
    python analyze_unknown_by_position.py results.json
    python analyze_unknown_by_position.py results.json --n_segments 10
    python analyze_unknown_by_position.py results.json --n_segments 10 --csv out.csv
    python analyze_unknown_by_position.py results.json --per_item
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# RESULTS_PATH = "gemma-3-27b-it_eCream-gamma_dev_by-group"
# RESULTS_PATH = "medgemma-27b-text-it_eCream-gamma_dev_by-group"
# RESULTS_PATH = "gemma-3-27b-it_eCream-gamma_dev_one-shot"
RESULTS_PATH = "medgemma-27b-text-it_eCream-gamma_dev_one-shot"

BASE_DIR     = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "results" / f"{RESULTS_PATH}.json"



def strip_code_fence(raw: str) -> str:
    """Remove ```json ... ``` (or bare ``` ... ```) wrapping, if present."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def is_unknown(entry) -> bool:
    """
    Decide whether a single item's parsed entry counts as "unknown".
    Entries look like {"first_measurement": "..."} or {"value": "..."} or
    {"first_measurement": "...", "second_measurement": "..."}.
    We judge on the PRIMARY slot (first_measurement, else value); a
    followup/second_measurement being "unknown" is normal/expected and not
    counted as an extraction failure on its own.
    """
    if isinstance(entry, str):
        return entry.strip().lower() == "unknown"
    if not isinstance(entry, dict):
        return False
    for key in ("first_measurement", "value"):
        if key in entry:
            return str(entry[key]).strip().lower() == "unknown"
    # Fallback: no recognized key, just look at whatever's there
    vals = list(entry.values())
    if vals:
        return str(vals[0]).strip().lower() == "unknown"
    return True


def load_note_level_outputs(results_path: str):
    """
    Returns a list of ordered-dicts: one per note, each mapping
    item_name -> parsed entry, taken from the single row per note that has
    a non-null raw_output.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    notes = []
    seen_note_ids = set()
    skipped_parse_errors = 0

    for row in data.get("results", []):
        raw = row.get("raw_output")
        if raw is None:
            continue
        note_id = row.get("note_id")
        if note_id in seen_note_ids:
            # Shouldn't normally happen (one raw_output per note), but guard anyway
            continue

        cleaned = strip_code_fence(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            skipped_parse_errors += 1
            continue

        if not isinstance(parsed, dict) or not parsed:
            continue

        notes.append(parsed)
        seen_note_ids.add(note_id)

    if skipped_parse_errors:
        print(f"  (warning: {skipped_parse_errors} raw_output blobs failed to parse and were skipped)",
              file=sys.stderr)

    return notes


def analyze(notes, n_segments: int):
    """
    For each note, split its ordered items into n_segments equal-ish chunks
    by POSITION (not by name), and tally unknown/total counts per segment
    across all notes.
    """
    seg_unknown = [0] * n_segments
    seg_total = [0] * n_segments

    for note in notes:
        items = list(note.items())  # preserves JSON order
        n = len(items)
        if n == 0:
            continue
        for idx, (item_name, entry) in enumerate(items):
            # map position idx (0..n-1) to a segment 0..n_segments-1
            seg = min(int(idx * n_segments / n), n_segments - 1)
            seg_total[seg] += 1
            if is_unknown(entry):
                seg_unknown[seg] += 1

    return seg_unknown, seg_total


def analyze_per_item(notes):
    """Per-item (by name) unknown rate, sorted by average position in the schema."""
    item_unknown = defaultdict(int)
    item_total = defaultdict(int)
    item_position_sum = defaultdict(float)

    for note in notes:
        items = list(note.items())
        n = len(items)
        if n == 0:
            continue
        for idx, (item_name, entry) in enumerate(items):
            item_total[item_name] += 1
            item_position_sum[item_name] += idx / n  # normalized position 0..1
            if is_unknown(entry):
                item_unknown[item_name] += 1

    rows = []
    for name in item_total:
        total = item_total[name]
        unk = item_unknown[name]
        avg_pos = item_position_sum[name] / total
        rows.append((name, avg_pos, unk, total, unk / total))
    rows.sort(key=lambda r: r[1])  # sort by average schema position
    return rows


def compute_gold_unknown_rates(results_path: str):
    """
    Compute, per crf_item, the fraction of ALL rows (not just the raw_output
    row) whose gold_label is "unknown". gold_label is present on every row
    regardless of which one carries raw_output, so this gives the true
    item-level applicability/documentation rate to compare pred against.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    gold_unknown = defaultdict(int)
    gold_total = defaultdict(int)

    for row in data.get("results", []):
        item = row.get("crf_item")
        if item is None:
            continue
        gold_total[item] += 1
        if is_unknown(row.get("gold_label")):
            gold_unknown[item] += 1

    return gold_unknown, gold_total


def compare_pred_vs_gold(notes, results_path: str, min_n: int = 20, top_n: int = 130):
    """
    For each item, compare pred's unknown rate (from raw_output, positional
    parsing) against gold's unknown rate (from gold_label, all rows).
    A large POSITIVE gap (pred >> gold) flags items the model is failing to
    extract even though gold shows they're often actually documented.
    """
    pred_unknown = defaultdict(int)
    pred_total = defaultdict(int)
    for note in notes:
        for item_name, entry in note.items():
            pred_total[item_name] += 1
            if is_unknown(entry):
                pred_unknown[item_name] += 1

    gold_unknown, gold_total = compute_gold_unknown_rates(results_path)

    rows = []
    for name in pred_total:
        if name not in gold_total or gold_total[name] < min_n:
            continue
        pred_rate = pred_unknown[name] / pred_total[name]
        gold_rate = gold_unknown[name] / gold_total[name]
        gap = pred_rate - gold_rate
        rows.append((name, pred_total[name], pred_rate, gold_total[name], gold_rate, gap))

    rows.sort(key=lambda r: r[5], reverse=True)  # biggest gap (pred >> gold) first
    return rows[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Analyze 'unknown' rate by position within one-shot raw_output JSON")
    parser.add_argument("--n_segments", type=int, default=10, help="Number of positional segments/deciles (default: 10)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write segment breakdown as CSV")
    parser.add_argument("--per_item", action="store_true", help="Also print a full per-item breakdown sorted by schema position")
    parser.add_argument("--compare_gold", action="store_true",
                         help="Compare pred's per-item unknown rate against gold's, to separate real extraction failures from legitimate schema sparsity")
    parser.add_argument("--min_n", type=int, default=20, help="Minimum gold sample size per item to include in --compare_gold (default: 20)")
    parser.add_argument("--top_n", type=int, default=130, help="Number of items to show in --compare_gold (default: 130)")
    args = parser.parse_args()

    print(f"Loading {RESULTS_PATH} ...")
    notes = load_note_level_outputs(RESULTS_PATH)
    print(f"  {len(notes)} notes with a parsed raw_output found.")
    if not notes:
        print("No notes with parsable raw_output found. Nothing to analyze.")
        return

    item_counts = [len(n) for n in notes]
    print(f"  Items per note: min={min(item_counts)}, max={max(item_counts)}, "
          f"mode-ish~{max(set(item_counts), key=item_counts.count)}")

    seg_unknown, seg_total = analyze(notes, args.n_segments)

    print("\n" + "=" * 60)
    print(f"'unknown' RATE BY POSITION IN GENERATED JSON ({args.n_segments} segments)")
    print("=" * 60)
    print(f"{'segment':>8} | {'range':>12} | {'n_total':>8} | {'n_unknown':>9} | {'pct_unknown':>11}")
    print("-" * 60)
    rows_for_csv = []
    for i in range(args.n_segments):
        lo = int(i * 100 / args.n_segments)
        hi = int((i + 1) * 100 / args.n_segments)
        total = seg_total[i]
        unk = seg_unknown[i]
        pct = (100 * unk / total) if total else float("nan")
        print(f"{i+1:>8} | {lo:>4}-{hi:>3}%   | {total:>8} | {unk:>9} | {pct:>10.1f}%")
        rows_for_csv.append({"segment": i + 1, "position_range": f"{lo}-{hi}%",
                              "n_total": total, "n_unknown": unk, "pct_unknown": pct})
    print("=" * 60)

    overall_total = sum(seg_total)
    overall_unknown = sum(seg_unknown)
    print(f"\nOverall: {overall_unknown}/{overall_total} = "
          f"{100*overall_unknown/overall_total:.1f}% unknown across all items/notes.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["segment", "position_range", "n_total", "n_unknown", "pct_unknown"])
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\nSegment breakdown saved -> {args.csv}")

    if args.per_item:
        print("\n" + "=" * 60)
        print("PER-ITEM 'unknown' RATE, sorted by average position in schema")
        print("=" * 60)
        print(f"{'item':<60} | {'avg_pos':>7} | {'n':>5} | {'pct_unknown':>11}")
        print("-" * 92)
        for name, avg_pos, unk, total, pct in analyze_per_item(notes):
            print(f"{name[:60]:<60} | {avg_pos:>7.2f} | {total:>5} | {100*pct:>10.1f}%")

    if args.compare_gold:
        print("\n" + "=" * 100)
        print(f"PRED vs GOLD 'unknown' RATE PER ITEM (top {args.top_n} by gap, pred >> gold first)")
        print("=" * 100)
        print(f"{'item':<55} | {'pred_n':>6} | {'pred_unk%':>9} | {'gold_n':>6} | {'gold_unk%':>9} | {'gap':>6}")
        print("-" * 100)
        for name, pn, prate, gn, grate, gap in compare_pred_vs_gold(notes, RESULTS_PATH, args.min_n, args.top_n):
            print(f"{name[:55]:<55} | {pn:>6} | {100*prate:>8.1f}% | {gn:>6} | {100*grate:>8.1f}% | {100*gap:>+5.1f}%")
        print("=" * 100)
        print("Large positive gap = model says 'unknown' far more often than gold does -> likely a real extraction failure on that item.")
        print("Gap near 0 = model's unknown rate matches gold's -> likely genuine schema sparsity, not a bug.")


if __name__ == "__main__":
    main()