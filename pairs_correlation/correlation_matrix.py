"""
correlation_matrix.py

Computes pairwise phi coefficients between all (item, value) binary variables
across clinical notes. Missing annotations are treated as "__unknown__".

For items with allow_followup=True, multiple measurements per note are split
into two separate pseudo-items: "Item [first]" (the first measurement) and
"Item [following]" (any subsequent measurements, possibly several). This
allows correlations between first and following measurements of the SAME
item to be discovered too (e.g. "Troponin [first]=high" vs
"Troponin [following]=high").

Each (item, value) pair becomes a binary column: 1 if the note has that value
for that item/role, 0 otherwise. For "[following]", a note's value set may
contain several distinct values (multiple follow-up measurements); the binary
column for a given value is 1 if ANY of the note's following measurements
equals that value.

Phi coefficient is then computed for all (item_a, value_a) × (item_b, value_b)
combinations, including same-item pairs across different roles (e.g.
"X [first]" vs "X [following]"), but same-item-same-role pairs are still
skipped since they are mutually exclusive by construction.

Outputs:
  - correlation_pairs.csv   : pairs passing --min_phi and --max_p, ranked
  - correlation_heatmap.png : heatmap of top --top_k pairs
  - correlation_pairs_full.csv : unfiltered pairs (optional, --save_full)

Usage:
  python scripts/correlation_matrix.py [options]

Hyperparameters:
  --min_phi          Minimum |phi| to include a pair (default: 0.3)
  --max_p            Maximum p-value (default: 0.05)
  --top_k            Top pairs shown in heatmap (default: 50)
  --n_bins           Quantile bins for numeric item values (default: 5)
  --group_filter     all | same | cross (default: all)
  --unknown_filter   none | both | all
                       none → keep all pairs (default)
                       both → exclude pairs where BOTH values are __unknown__
                       all  → exclude any pair where EITHER value is __unknown__
  --min_support      Minimum notes where each value must appear individually (default: 10)
  --min_cooccurrence Minimum notes where both values appear together (default: 5)
  --save_full        Also save unfiltered pairs CSV
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
# SCHEMA_PATH = BASE_DIR / "crf_schema" / "crf_schema_synthetic_26_03_2026_no-duplicates.json"
SCHEMA_PATH = BASE_DIR / "crf_schema" / "crf_schema_synthetic_26_03_2026_2_percent.json"
DATA_PATH   = BASE_DIR / "data" / "eCream-gamma" / "train.json"
OUT_DIR     = BASE_DIR / "data" / "eCream-gamma"

UNKNOWN = "__unknown__"

FIRST_SUFFIX     = " [first]"
FOLLOWING_SUFFIX = " [following]"

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema(path: Path) -> dict:
    """Returns {base_name_lower: {base_name, type, group, allow_followup}}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, meta in raw.items():
        out[name.strip().lower()] = {
            "base_name": name.strip(),
            "type": meta.get("type", "YN").upper(),
            "group": meta.get("group", ""),
            "allow_followup": bool(meta.get("allow_followup", False)),
        }
    return out

# ---------------------------------------------------------------------------
# Data loading → per-note item→value(s)
# ---------------------------------------------------------------------------

def load_data(data_path: Path, schema: dict):
    """
    Returns (df, followup_value_sets, item_columns):

      df : DataFrame, one row per note. Columns are pseudo-items:
           - "Item" for non-followup items (single value, or UNKNOWN)
           - "Item [first]" for the first measurement of allow_followup items
           Value is a string (or UNKNOWN).

      followup_value_sets : dict[(note_index, "Item [following]")] -> set of
           value strings (possibly empty) collected from all measurements
           after the first, for allow_followup items.

      item_columns : list of all pseudo-item column names that will end up
           in the binarized output (df columns + the followup pseudo-items),
           used downstream for group/type lookups.

    For allow_followup items, spans are taken in the order they appear in
    the note's span list (assumed to reflect document order / first→following).
    The first non-unknown-labeled span becomes "Item [first]"; every
    subsequent span (if any) contributes to "Item [following]".
    """
    with open(data_path, encoding="utf-8") as f:
        notes = json.load(f)

    item_lookup = {v["base_name"].lower(): v for v in schema.values()}

    # Non-followup items get a plain column; followup items get "[first]"
    plain_items  = [v["base_name"] for v in schema.values() if not v["allow_followup"]]
    first_items  = [v["base_name"] + FIRST_SUFFIX for v in schema.values() if v["allow_followup"]]
    df_columns   = plain_items + first_items

    rows = []
    followup_value_sets = {}

    for note_idx, note in enumerate(notes):
        row = {col: UNKNOWN for col in df_columns}
        # Track, per allow_followup base item, whether we've already assigned "first"
        first_assigned = set()
        following_vals = {}  # base_item -> set of values (role=following)

        for span in note.get("spans", []):
            crf_item = span.get("crf_item")
            if not crf_item:
                continue
            meta = item_lookup.get(crf_item.strip().lower())
            if meta is None:
                continue
            label = span.get("label_assigned")
            if label is None:
                continue
            label_str = str(label)
            base_name = meta["base_name"]

            if not meta["allow_followup"]:
                if row[base_name] == UNKNOWN:
                    row[base_name] = label_str
                continue

            # allow_followup item
            first_col = base_name + FIRST_SUFFIX
            if base_name not in first_assigned:
                row[first_col] = label_str
                first_assigned.add(base_name)
            else:
                following_vals.setdefault(base_name, set()).add(label_str)

        for base_name, vals in following_vals.items():
            followup_value_sets[(note_idx, base_name + FOLLOWING_SUFFIX)] = vals

        rows.append(row)

    df = pd.DataFrame(rows, columns=df_columns)

    following_items = [v["base_name"] + FOLLOWING_SUFFIX for v in schema.values() if v["allow_followup"]]
    item_columns = df_columns + following_items

    return df, followup_value_sets, item_columns

# ---------------------------------------------------------------------------
# Numeric binning
# ---------------------------------------------------------------------------

def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def bin_numeric_column(series: pd.Series, n_bins: int) -> pd.Series:
    """
    Bin numeric values into quantile buckets; non-numeric strings kept as-is.
    """
    numeric_mask = series.apply(lambda x: x != UNKNOWN and _is_numeric(x))
    if numeric_mask.sum() < n_bins * 2:
        return series  # too sparse to bin

    numeric_vals = series[numeric_mask].apply(lambda x: float(x.replace(",", ".")))
    try:
        binned = pd.qcut(numeric_vals, q=n_bins, duplicates="drop", labels=False)
        binned_str = binned.astype(str).apply(lambda x: f"bin_{x}")
    except Exception:
        return series

    result = series.copy()
    result[numeric_mask] = binned_str.values
    return result


def bin_value_set_column(value_sets: dict, note_indices: list, n_bins: int) -> dict:
    """
    Bin numeric values found across all "[following]" value sets for one
    pseudo-item. Returns a new dict[note_idx] -> set of (possibly binned)
    value strings. Non-numeric values are kept as-is.
    """
    # Collect all numeric values across all notes for this column to compute bin edges
    all_numeric = []
    for note_idx in note_indices:
        vals = value_sets.get(note_idx, set())
        for v in vals:
            if _is_numeric(v):
                all_numeric.append(float(v.replace(",", ".")))

    if len(all_numeric) < n_bins * 2:
        return value_sets  # too sparse, leave as-is

    try:
        _, bin_edges = pd.qcut(all_numeric, q=n_bins, duplicates="drop", retbins=True)
    except Exception:
        return value_sets

    def to_bin_label(x: float) -> str:
        b = np.digitize([x], bin_edges[1:-1], right=True)[0]
        return f"bin_{b}"

    new_value_sets = {}
    for note_idx in note_indices:
        vals = value_sets.get(note_idx, set())
        new_vals = set()
        for v in vals:
            if _is_numeric(v):
                new_vals.add(to_bin_label(float(v.replace(",", "."))))
            else:
                new_vals.add(v)
        new_value_sets[note_idx] = new_vals

    return new_value_sets

# ---------------------------------------------------------------------------
# Phi coefficient for two binary series
# ---------------------------------------------------------------------------

def phi_coefficient(a: pd.Series, b: pd.Series):
    """
    Compute phi coefficient and χ² p-value for two binary (0/1) series.
    Returns (phi, p_value) or (nan, nan) on failure.
    Phi is signed: positive means co-occurrence, negative means mutual exclusion.
    """
    contingency = pd.crosstab(a, b)
    if contingency.shape != (2, 2):
        return np.nan, np.nan

    for ax in [0, 1]:
        if list(contingency.axes[ax]) != [0, 1]:
            contingency = contingency.reindex(
                index=[0, 1] if ax == 0 else contingency.index,
                columns=[0, 1] if ax == 1 else contingency.columns,
                fill_value=0
            )

    n00 = contingency.loc[0, 0]
    n01 = contingency.loc[0, 1]
    n10 = contingency.loc[1, 0]
    n11 = contingency.loc[1, 1]

    n = n00 + n01 + n10 + n11
    if n == 0:
        return np.nan, np.nan

    denom = np.sqrt(
        (n10 + n11) * (n00 + n01) * (n00 + n10) * (n01 + n11)
    )
    if denom == 0:
        return np.nan, np.nan

    phi = (n11 * n00 - n10 * n01) / denom

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chi2, p, _, _ = chi2_contingency(contingency)
        except Exception:
            return np.nan, np.nan

    return float(phi), float(p)

# ---------------------------------------------------------------------------
# Binarize into (item_with_role, value) binary columns
# ---------------------------------------------------------------------------

def binarize(df: pd.DataFrame, followup_value_sets: dict, schema: dict, n_bins: int) -> pd.DataFrame:
    """
    Builds the full binary DataFrame, one column per (item_with_role, value).

    - Plain / "[first]" pseudo-items: one value per note (from df), numeric
      items are binned via bin_numeric_column.
    - "[following]" pseudo-items: a SET of values per note (from
      followup_value_sets), numeric items are binned via bin_value_set_column.
      A binary column "Item [following]||value" is 1 for a note iff that
      value is present anywhere in the note's following-measurement set.
    """
    item_meta = {v["base_name"]: v for v in schema.values()}
    binary_cols = {}
    n_notes = len(df)

    # --- Plain and [first] columns (single value per note) ---
    for col in df.columns:
        base_name = col[:-len(FIRST_SUFFIX)] if col.endswith(FIRST_SUFFIX) else col
        meta = item_meta.get(base_name)
        if meta and meta["type"] in ("NUMBER", "NUMBER_OR_CATEGORY"):
            series = bin_numeric_column(df[col], n_bins)
        else:
            series = df[col]

        for val in series.unique():
            col_name = f"{col}||{val}"
            binary_cols[col_name] = (series == val).astype(int)

    # --- [following] columns (set of values per note) ---
    following_items = [v["base_name"] for v in schema.values() if v["allow_followup"]]
    for base_name in following_items:
        col = base_name + FOLLOWING_SUFFIX
        # Gather this item's value sets across all notes (defaulting to empty set)
        value_sets = {idx: followup_value_sets.get((idx, col), set()) for idx in range(n_notes)}

        meta = item_meta.get(base_name)
        if meta and meta["type"] in ("NUMBER", "NUMBER_OR_CATEGORY"):
            value_sets = bin_value_set_column(value_sets, list(range(n_notes)), n_bins)

        # Collect all distinct values that appear anywhere
        all_vals = set()
        for vals in value_sets.values():
            all_vals |= vals

        for val in all_vals:
            col_name = f"{col}||{val}"
            binary_cols[col_name] = pd.Series(
                [1 if val in value_sets[idx] else 0 for idx in range(n_notes)],
                index=df.index,
            )

    return pd.DataFrame(binary_cols, index=df.index)

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(pairs_df: pd.DataFrame, top_k: int, out_path: Path):
    """
    Square heatmap over the items that appear in the top_k pairs.
    Axis labels are "item (value)"; cells show phi.
    """
    top = pairs_df.head(top_k)

    def fmt_label(col):
        item, val = col.split("||", 1)
        return f"{item}\n({val})"

    labels_a = top["item_a"].str.cat(top["value_a"].apply(lambda v: f"||{v}"))
    labels_b = top["item_b"].str.cat(top["value_b"].apply(lambda v: f"||{v}"))
    all_cols  = sorted(set(labels_a.tolist() + labels_b.tolist()))
    n = len(all_cols)
    idx = {c: i for i, c in enumerate(all_cols)}

    matrix = np.full((n, n), np.nan)
    for _, row in top.iterrows():
        ca = f"{row['item_a']}||{row['value_a']}"
        cb = f"{row['item_b']}||{row['value_b']}"
        i, j = idx[ca], idx[cb]
        matrix[i, j] = row["phi"]
        matrix[j, i] = row["phi"]
    np.fill_diagonal(matrix, 1.0)

    tick_labels = [fmt_label(c) for c in all_cols]
    fig_size = max(14, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        matrix,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=-1, vmax=1,
        center=0,
        cmap="RdBu_r",
        annot=(n <= 30),
        fmt=".2f",
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(f"Phi coefficient — top {top_k} pairs (|phi|)", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved → {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRF value-pair correlation via phi coefficient")
    parser.add_argument("--min_phi",       type=float, default=0.3,    help="Minimum |phi| (default: 0.3)")
    parser.add_argument("--max_p",         type=float, default=0.05,   help="Maximum p-value (default: 0.05)")
    parser.add_argument("--top_k",         type=int,   default=50,     help="Top pairs in heatmap (default: 50)")
    parser.add_argument("--n_bins",        type=int,   default=5,      help="Quantile bins for numeric items (default: 5)")
    parser.add_argument("--group_filter",  type=str,   default="all",  choices=["all", "same", "cross"],
                        help="Pair group filter: all | same | cross (default: all)")
    parser.add_argument("--unknown_filter",type=str,   default="none", choices=["none", "both", "all"],
                        help=(
                            "none → keep all pairs (default); "
                            "both → exclude pairs where BOTH values are __unknown__; "
                            "all  → exclude any pair where EITHER value is __unknown__"
                        ))
    parser.add_argument("--min_support",     type=int, default=10,      help="Minimum notes where each value must appear individually (default: 10)")
    parser.add_argument("--min_cooccurrence",type=int, default=5,       help="Minimum notes where both values appear together (default: 5)")
    parser.add_argument("--save_full",     action="store_true",         help="Also save unfiltered pairs CSV")
    args = parser.parse_args()

    print("Loading schema …")
    schema = load_schema(SCHEMA_PATH)
    item_meta = {v["base_name"]: v for v in schema.values()}
    print(f"  {len(schema)} items in schema "
          f"({sum(1 for v in schema.values() if v['allow_followup'])} allow_followup).")

    print("Loading data …")
    df, followup_value_sets, _ = load_data(DATA_PATH, schema)
    print(f"  {len(df)} notes loaded.")

    print("Binarizing (item, value) columns …")
    bin_df = binarize(df, followup_value_sets, schema, args.n_bins)
    print(f"  {len(bin_df.columns)} binary (item, value) columns.")

    # Build group lookup that resolves "[first]"/"[following]" suffixes back
    # to the base item's group/type metadata.
    def base_item_name(col: str) -> str:
        if col.endswith(FIRST_SUFFIX):
            return col[: -len(FIRST_SUFFIX)]
        if col.endswith(FOLLOWING_SUFFIX):
            return col[: -len(FOLLOWING_SUFFIX)]
        return col

    def col_meta(col):
        item, val = col.split("||", 1)
        return item, val

    all_cols = list(bin_df.columns)

    # Precompute support (number of notes where each value = 1) for fast filtering
    support = bin_df.sum(axis=0)

    # Pre-filter columns below min_support to shrink the pair space
    valid_cols = [c for c in all_cols if support[c] >= args.min_support]
    n_dropped  = len(all_cols) - len(valid_cols)
    if n_dropped:
        print(f"  Dropped {n_dropped} (item, value) columns below min_support={args.min_support}.")
    all_cols = valid_cols

    total = len(all_cols) * (len(all_cols) - 1) // 2
    print(f"Computing phi for {total:,} pairs …")

    records = []
    for i, (ca, cb) in enumerate(combinations(all_cols, 2)):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,}/{total:,} …", end="\r")

        item_a, val_a = col_meta(ca)
        item_b, val_b = col_meta(cb)

        # Skip same pseudo-item-and-role pairs (mutually exclusive by construction).
        # NOTE: same base item across different roles (e.g. "X [first]" vs
        # "X [following]") IS allowed and computed.
        if item_a == item_b:
            continue

        # Unknown filter
        if args.unknown_filter == "all" and (val_a == UNKNOWN or val_b == UNKNOWN):
            continue
        if args.unknown_filter == "both" and (val_a == UNKNOWN and val_b == UNKNOWN):
            continue

        # Group filter (resolved via base item name, ignoring role suffix)
        ga = item_meta[base_item_name(item_a)]["group"]
        gb = item_meta[base_item_name(item_b)]["group"]
        if args.group_filter == "same"  and ga != gb:
            continue
        if args.group_filter == "cross" and ga == gb:
            continue

        # Support and co-occurrence filters
        if support[ca] < args.min_support or support[cb] < args.min_support:
            continue
        n11 = int((bin_df[ca] & bin_df[cb]).sum())
        if n11 < args.min_cooccurrence:
            continue

        phi, p = phi_coefficient(bin_df[ca], bin_df[cb])
        if np.isnan(phi):
            continue

        records.append({
            "item_a":  item_a,
            "value_a": val_a,
            "item_b":  item_b,
            "value_b": val_b,
            "group_a": ga,
            "group_b": gb,
            "phi":     round(phi, 4),
            "abs_phi": round(abs(phi), 4),
            "p_value": p,
        })

    print(f"\n  {len(records):,} valid pairs computed.")

    all_pairs_df = (
        pd.DataFrame(records)
        .sort_values("abs_phi", ascending=False)
        .reset_index(drop=True)
    )

    if args.save_full:
        full_df = all_pairs_df.drop(columns="abs_phi")
        full_path = OUT_DIR / "correlation_pairs_full.xlsx"
        full_df.to_excel(full_path.with_suffix(".xlsx"), index=False)
        print(f"  Full pairs saved → {full_path}")

    filtered_df = all_pairs_df[
        (all_pairs_df["abs_phi"] >= args.min_phi) &
        (all_pairs_df["p_value"] <= args.max_p)
    ].reset_index(drop=True)

    print(f"  {len(filtered_df):,} pairs pass min_phi={args.min_phi}, max_p={args.max_p}, min_support={args.min_support}, min_cooccurrence={args.min_cooccurrence}.")

    out_df = filtered_df.drop(columns="abs_phi")    
    pairs_path = OUT_DIR / f"correlation_pairs_n_bins_{args.n_bins}.xlsx"
    out_df.to_excel(pairs_path.with_suffix(".xlsx"), index=False)
    
    print(f"  Filtered pairs saved → {pairs_path}")

    if filtered_df.empty:
        print("No pairs pass the filters — skipping heatmap. Try lowering --min_phi.")
        return

    heatmap_path = OUT_DIR / "correlation_heatmap.png"
    plot_heatmap(filtered_df, args.top_k, heatmap_path)

    print("\nTop 10 pairs:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()