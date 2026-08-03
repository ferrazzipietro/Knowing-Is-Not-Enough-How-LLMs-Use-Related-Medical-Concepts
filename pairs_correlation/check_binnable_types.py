"""
check_binnable_types.py

Diagnostic for the n_bins "no effect" finding: binarize() in
correlation_matrix.py only bins an item's values if its schema type is
exactly "NUMBER" or "NUMBER_OR_CATEGORY" -- "NUMBER_OR_YN" (and everything
else) is left as raw string values, completely unaffected by --n_bins.

This script loads the schema and reports, for every item that appears in
your reported pairs (edit ITEMS_TO_CHECK below, or pass --pairs_xlsx to
pull them automatically from a correlation_pairs*.xlsx file), whether its
type is one that binarize() actually bins.

Usage:
  python check_binnable_types.py --schema crf_schema/crf_schema_synthetic_26_03_2026_2_percent.json
  python check_binnable_types.py --schema <schema.json> --pairs_xlsx data/eCream-gamma/correlation_pairs.xlsx
"""

import argparse
import json
import re

FIRST_SUFFIX = " [first]"
FOLLOWING_SUFFIX = " [following]"
BINNABLE_TYPES = ("NUMBER", "NUMBER_OR_CATEGORY")


def strip_role_suffix(item_name: str) -> str:
    if item_name.endswith(FIRST_SUFFIX):
        return item_name[: -len(FIRST_SUFFIX)]
    if item_name.endswith(FOLLOWING_SUFFIX):
        return item_name[: -len(FOLLOWING_SUFFIX)]
    return item_name


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {name.strip(): meta.get("type", "YN").upper() for name, meta in raw.items()}


def items_from_pairs_xlsx(path):
    import pandas as pd
    df = pd.read_excel(path)
    items = set()
    for col in ("item_a", "item_b"):
        items.update(strip_role_suffix(str(v)) for v in df[col])
    return sorted(items)


# Fallback: the item names behind the 28 pairs reported in the paper,
# used if --pairs_xlsx is not given.
ITEMS_TO_CHECK = [
    "Antiepileptic therapy already in place", "History of epilepsy",
    "C-Reactive Protein", "Leukocytes",
    "Antihypertensive therapy", "Cardiovascular diseases (except for HF)",
    "Cardiac ultrasound, any abnormality", "Thoracic ultrasound, any abnormalities",
    "Blood potassium", "Blood sodium",
    "Foreign body in the airways", "Level of consciousness (GCS)",
    "Body temperature", "Respiratory Rate",
    "Blood calcium", "Lactates", "pH",
    "History of drug allergy", "TLoC during effort", "TLoC while supine",
    "Poly-pharmacological therapy (>= 5)", "Level of consciousness",
    "Heart Rate", "Anticoagulants or antiplatelet drug therapy",
    "Diffuse vascular disease", "Creatinine",
    "Administration of bronchodilators", "Administration of steroids",
    "ECG, any abnormality",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--pairs_xlsx", default=None,
                         help="If given, pull item names from this pairs file instead of the hardcoded list")
    args = parser.parse_args()

    schema_types = load_schema(args.schema)
    items = items_from_pairs_xlsx(args.pairs_xlsx) if args.pairs_xlsx else ITEMS_TO_CHECK

    print(f"{'Item':<55} {'Type':<20} {'Binnable?'}")
    print("-" * 90)
    n_binnable = 0
    for item in items:
        t = schema_types.get(item, "??? NOT FOUND IN SCHEMA ???")
        binnable = t in BINNABLE_TYPES
        n_binnable += binnable
        flag = "YES" if binnable else "no"
        print(f"{item:<55} {t:<20} {flag}")

    print("-" * 90)
    print(f"{n_binnable}/{len(items)} items are of a type that binarize() ever bins "
          f"(i.e. type in {BINNABLE_TYPES}).")
    if n_binnable == 0:
        print("\n=> This confirms n_bins could NOT have affected any of these pairs, "
              "for ANY value of n_bins -- the sweep result is a structural certainty, "
              "not a finding about the data's distribution.")


if __name__ == "__main__":
    main()