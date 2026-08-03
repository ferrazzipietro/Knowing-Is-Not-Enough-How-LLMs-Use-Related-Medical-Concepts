"""
check_numeric_sparsity.py

Follow-up to check_binnable_types.py. Even for an item whose schema type IS
binnable (NUMBER / NUMBER_OR_CATEGORY), correlation_matrix.py's
bin_numeric_column() silently skips binning entirely if there are fewer
than n_bins*2 notes with an actual numeric (as opposed to categorical/
unknown) first-measurement value for that item:

    numeric_mask = series.apply(lambda x: x != UNKNOWN and _is_numeric(x))
    if numeric_mask.sum() < n_bins * 2:
        return series  # too sparse to bin

Since the numeric count doesn't change with n_bins but the required
threshold (n_bins*2) only grows, an item that's too sparse to bin at your
SMALLEST tested n_bins is guaranteed to be too sparse at every larger one
too -- which would fully explain a "zero effect across n_bins=3..10" sweep
result for that item, independent of anything else.

This script identifies the binnable-type items, extracts their raw
first-measurement values from train.json (same logic as
correlation_matrix.py's load_data, first-measurement branch only), and
reports, per item: how many notes have a numeric vs. non-numeric/unknown
value, and at which n_bins settings (if any) binning would actually have
triggered.

Usage:
  python check_numeric_sparsity.py \\
      --schema crf_schema/crf_schema_synthetic_26_03_2026_2_percent.json \\
      --train data/eCream-gamma/train.json
"""

import argparse
import json

UNKNOWN = "__unknown__"
BINNABLE_TYPES = ("NUMBER", "NUMBER_OR_CATEGORY")
N_BINS_TO_CHECK = [2, 3, 4, 5, 6, 7, 8, 10]


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, meta in raw.items():
        out[name.strip()] = {
            "type": meta.get("type", "YN").upper(),
            "allow_followup": bool(meta.get("allow_followup", False)),
        }
    return out


def first_measurement_values(train_path, item_name):
    """Replicates load_data's first-measurement extraction for one item."""
    with open(train_path, encoding="utf-8") as f:
        notes = json.load(f)

    values = []  # one entry per note that has this item at all
    for note in notes:
        first_val = None
        for span in note.get("spans", []):
            if span.get("crf_item", "").strip() != item_name:
                continue
            label = span.get("label_assigned")
            if label is None:
                continue
            first_val = str(label)
            break  # first occurrence only, matching correlation_matrix.py
        if first_val is not None:
            values.append(first_val)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--train", required=True)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    binnable_items = [name for name, meta in schema.items() if meta["type"] in BINNABLE_TYPES]

    print(f"{len(binnable_items)} items in schema have a binnable type ({BINNABLE_TYPES}):\n")

    for item in binnable_items:
        values = first_measurement_values(args.train, item)
        numeric_vals = [v for v in values if v != UNKNOWN and _is_numeric(v)]
        n_numeric = len(numeric_vals)
        n_total = len(values)

        print(f"'{item}':")
        print(f"  {n_total} notes have this item annotated at all "
              f"({n_numeric} numeric, {n_total - n_numeric} categorical/other).")

        if n_numeric == 0:
            print("  -> No numeric observations at all; this item's binnable branch "
                  "is empty regardless of n_bins.")
        else:
            triggers_at = [nb for nb in N_BINS_TO_CHECK if n_numeric >= nb * 2]
            skipped_at = [nb for nb in N_BINS_TO_CHECK if n_numeric < nb * 2]
            if triggers_at:
                print(f"  Binning WOULD trigger (numeric_mask.sum()={n_numeric} >= n_bins*2) "
                      f"at n_bins in {triggers_at}")
            if skipped_at:
                print(f"  Binning would be SKIPPED (too sparse) at n_bins in {skipped_at}")
        print()

    print("If a binnable item shows 'skipped' across your entire tested range (3-10), "
          "that item's contribution to the sweep result is a sparsity floor, not evidence "
          "that n_bins has no effect in general -- worth reporting as such rather than as "
          "a blanket 'n_bins doesn't matter' claim.")


if __name__ == "__main__":
    main()