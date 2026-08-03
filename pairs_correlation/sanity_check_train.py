"""
sanity_check_train.py  (throwaway diagnostic -- delete after use)

Sanity check: score the TRAINING SET's own gold labels against the
correlation pairs (which were themselves derived from this same training
set by correlation_matrix.py). Since it's the source data scored against
itself, this tells us whether the scoring formula behaves sensibly at its
best case -- i.e. whether a perfectly self-consistent source actually
lands near +1, or whether the achievable ceiling is capped well below 1
even here (which would confirm that's just a property of moderate phi
values / imbalanced marginals, not a bug in the scorer).

Usage:
  python sanity_check_train.py
  python sanity_check_train.py --top_k 20 --min_phi 0.2
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Make sure Python can find score_correlation_consistency.py regardless of
# the working directory this script is invoked from -- explicitly add this
# script's own directory to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the exact same logic as the main scoring script so this check is
# apples-to-apples with whatever's currently in score_correlation_consistency.py
from score_correlation_consistency import (
    load_correlation_pairs,
    load_schema,
    score_source,
    normalize_value,
    UNKNOWN,
    FIRST_SUFFIX,
    FOLLOWING_SUFFIX,
    CORR_XLSX,
    SCHEMA_PATH,
    BASE_DIR,
)

TRAIN_PATH = BASE_DIR / "data" / "eCream-gamma" / "train.json"


def item_key(item: str, role: str) -> str:
    if role == "first":
        return item + FIRST_SUFFIX
    if role == "following":
        return item + FOLLOWING_SUFFIX
    return item


def collect_item_casings(entries):
    """
    Build {lowercased_item_name: set(original-case spellings seen in spans)}
    by scanning every span across the whole file once. Needed because the
    schema keys are lowercased, but spans/correlation pairs use whatever
    original casing correlation_matrix.py originally saw -- so when we
    stamp UNKNOWN for an item that had no span in a given note, we need to
    know what casing to stamp it under to actually match downstream.
    """
    casings = defaultdict(set)
    for entry in entries:
        for span in entry.get("spans", []):
            crf_item = span.get("crf_item")
            if crf_item:
                casings[crf_item.strip().lower()].add(crf_item)
    return casings


def build_train_value_sets(train_path: Path, schema: dict):
    """
    Build {(note_id, item_with_role): set(normalized values)} directly from
    train.json's "spans" annotations.

    Notes:
      - A (note_id, case_index) pair is treated as its own pseudo-note (key
        "{note_id}_{case_index}"), since a single note_id can be split into
        multiple annotated cases with different span sets.
      - allow_followup=True items: spans for the same (note, item) are
        ordered by their "start" character offset; the earliest is "first",
        the rest are "following" (accumulated into one set, matching how
        load_results handles multiple following-measurement rows).
      - allow_followup=False items: all spans for the same (note, item)
        contribute their label_assigned values to a single "single"-role set.
      - An item with NO span at all for a given note is stamped UNKNOWN for
        its "single"/"first" role slot -- mirroring extract_value's current
        behavior in the main pipeline (a missing value = an explicit
        UNKNOWN, not an absent dict entry). This matters specifically for
        phi<0 pairs: without this, a note where item_b is genuinely
        undocumented would be silently SKIPPED (set_b is None -> continue)
        instead of correctly counted as "item_b = unknown". A missing
        "following" slot is left absent, same as in the main pipeline,
        since that's a legitimate clinical state, not a data gap.
    """
    with open(train_path, encoding="utf-8") as f:
        entries = json.load(f)

    item_name_by_lower = collect_item_casings(entries)
    value_sets = defaultdict(set)
    all_schema_items = list(schema.keys())  # lowercased names, for lookup only

    for entry in entries:
        note_id = entry.get("note_id")
        case_index = entry.get("case_index")
        pseudo_note_id = f"{note_id}_{case_index}"

        spans_by_item = defaultdict(list)
        for span in entry.get("spans", []):
            crf_item = span.get("crf_item")
            label = span.get("label_assigned")
            start = span.get("start", 0)
            if crf_item is None or label is None:
                continue
            spans_by_item[crf_item].append((start, label))

        seen_items_this_note = set()
        for crf_item, occurrences in spans_by_item.items():
            allow_followup = schema.get(crf_item.strip().lower(), False)
            occurrences.sort(key=lambda x: x[0])  # order by position in text
            seen_items_this_note.add(crf_item.strip().lower())

            if not allow_followup:
                for _, label in occurrences:
                    value_sets[(pseudo_note_id, item_key(crf_item, "single"))].add(normalize_value(label))
            else:
                first_start, first_label = occurrences[0]
                value_sets[(pseudo_note_id, item_key(crf_item, "first"))].add(normalize_value(first_label))
                for _, label in occurrences[1:]:
                    value_sets[(pseudo_note_id, item_key(crf_item, "following"))].add(normalize_value(label))

        # Stamp UNKNOWN for every schema item that had no span at all in
        # this note -- mirrors extract_value's missing-value fallback.
        # We only know the item names actually referenced in correlation
        # pairs matter downstream, but stamping the full schema here keeps
        # this exactly consistent with what extract_value would do for a
        # real per-item/per-note results row that came back empty.
        for item_lower, item_allow_followup in schema.items():
            if item_lower in seen_items_this_note:
                continue
            # We don't have the original-cased item name here, only the
            # lowercased schema key -- but correlation pairs' item_a/item_b
            # come from correlation_matrix.py, which presumably used the
            # same original casing as the spans. Since spans never referenced
            # this item at all in this note, we stamp UNKNOWN under EVERY
            # casing variant we've seen used for this item elsewhere, to
            # avoid silently missing the match due to a casing mismatch.
            role = "first" if item_allow_followup else "single"
            # Stamp using whatever original-case spellings appeared anywhere
            # in this file for this item; if none appeared anywhere, skip
            # (we have no way to reconstruct the exact original casing).
            for original_name in item_name_by_lower.get(item_lower, []):
                value_sets[(pseudo_note_id, item_key(original_name, role))].add(UNKNOWN)

    return dict(value_sets)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check CCS scoring against the training set's own labels")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_phi", type=float, default=0.0)
    args = parser.parse_args()

    print("Loading correlation pairs …")
    corr_df = load_correlation_pairs(CORR_XLSX, args.top_k, args.min_phi)
    print(f"  {len(corr_df)} pairs loaded.")

    print("Loading schema …")
    schema = load_schema(SCHEMA_PATH)

    print(f"Loading training set from {TRAIN_PATH} …")
    train_values = build_train_value_sets(TRAIN_PATH, schema)
    all_note_ids = sorted({k[0] for k in train_values.keys()})
    print(f"  {len(all_note_ids)} pseudo-notes (note_id + case_index) found in training set.")

    pos_df = corr_df[corr_df["phi"] >= 0].reset_index(drop=True)
    neg_df = corr_df[corr_df["phi"] < 0].reset_index(drop=True)

    print("Scoring training set against itself …")
    overall_score, overall_stats = score_source(corr_df, train_values, all_note_ids)
    pos_score, pos_stats = score_source(pos_df, train_values, all_note_ids)
    neg_score, neg_stats = score_source(neg_df, train_values, all_note_ids)

    def fmt(x):
        return f"{x:.4f}" if x is not None else "N/A"

    def total_activated(stats):
        return sum(s["n_activated"] for s in stats)

    print("\n" + "=" * 60)
    print("TRAINING SET SELF-CONSISTENCY CHECK")
    print("=" * 60)
    print(f"  Overall score (train vs train) : {fmt(overall_score)}   (total n_activated: {total_activated(overall_stats)})")
    print(f"  phi>=0 score                   : {fmt(pos_score)}   (total n_activated: {total_activated(pos_stats)})")
    print(f"  phi<0  score                   : {fmt(neg_score)}   (total n_activated: {total_activated(neg_stats)})")
    print("=" * 60)
    print("\nCheck the n_activated counts above before trusting the scores --")
    print("a score based on very few activated instances is not meaningful,")
    print("regardless of how close to +1 it looks.")
    print("\nIf these are near +1 with healthy activation counts: the scorer")
    print("behaves as expected at the best case, and any gap you see on real")
    print("predictions reflects real model behavior. If phi>=0 in particular")
    print("is well below +1 even here, that confirms it's a property of")
    print("moderate phi / imbalanced value marginals -- not a scoring bug --")
    print("since this is the exact data the phi coefficients came from.")


if __name__ == "__main__":
    main()