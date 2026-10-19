"""Compare the curated family rate table to the step-log-derived master.

Emits `family_vs_current_comparison.csv` showing:
  - how the legacy (motif_family_3d, site_class_3d, direction_family_3d) keys
    in rate_lookup_table_3d_master.csv collapse into curated family buckets,
  - per-family count deltas vs the legacy aggregate,
  - high-level stats: row counts on each side, relative barriers when both
    sides cover the same mechanism.

Also emits a short text summary to stdout.

Inputs: classified_events_with_families.csv and rate_lookup_table_family.csv
(pylatkmc's family assignment), rate_lookup_table_3d_master.csv
(``python -m Analysis.build_rate_table``). Every path defaults to the output
directory ($PYKMC_ANALYSIS_OUTPUT_DIR, else Analysis/lattice_event_classification/).

Run from the repository root:
    python -m Analysis.compare_family_vs_current
    python -m Analysis.compare_family_vs_current \\
        --events  /path/to/classified_events_with_families.csv \\
        --current /path/to/rate_lookup_table_3d_master.csv \\
        --family  /path/to/rate_lookup_table_family.csv \\
        --output  /path/to/family_vs_current_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from Analysis._paths import CLASSIFICATION_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events",  type=Path,
                    default=CLASSIFICATION_DIR / "classified_events_with_families.csv")
    ap.add_argument("--current", type=Path,
                    default=CLASSIFICATION_DIR / "rate_lookup_table_3d_master.csv")
    ap.add_argument("--family",  type=Path,
                    default=CLASSIFICATION_DIR / "rate_lookup_table_family.csv")
    ap.add_argument("--output",  type=Path,
                    default=CLASSIFICATION_DIR / "family_vs_current_comparison.csv")
    args = ap.parse_args()

    print(f"[compare] reading {args.events}")
    events = pd.read_csv(args.events)
    print(f"  {len(events)} events")

    # Per-event cross-tab: legacy (motif × class × direction) → family_id (accepted only).
    accepted = events[events["assignment_status"] == "accepted"]
    legacy_cols = ["motif_family_3d", "site_class_3d", "direction_family_3d"]

    # Ensure legacy labels are clean strings (NaN → "(na)") so groupby never
    # silently drops a category.
    for c in legacy_cols:
        accepted = accepted.assign(**{c: accepted[c].fillna("(na)")})

    xtab = (accepted.groupby(legacy_cols + ["family_id"])
                    .size()
                    .reset_index(name="n_events"))

    # Attach counts from the legacy master CSV per (motif, class, direction)
    # so a reader can see row-count collapse.
    if args.current.exists():
        current = pd.read_csv(args.current)
        current_ev = (current.groupby(legacy_cols)["n_events"].sum()
                              .reset_index(name="current_master_n_events"))
        xtab = xtab.merge(current_ev, on=legacy_cols, how="left")
    else:
        xtab["current_master_n_events"] = 0

    # Attach per-family row counts from the curated family table.
    if args.family.exists():
        family = pd.read_csv(args.family)
        family_ev = (family.groupby("family_id")["n_events"].sum()
                             .reset_index(name="family_table_n_events"))
        xtab = xtab.merge(family_ev, on="family_id", how="left")
    else:
        xtab["family_table_n_events"] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    xtab.to_csv(args.output, index=False)
    print(f"[compare] wrote {args.output}  ({len(xtab)} rows)")

    # ── Text summary ────────────────────────────────────────────────────────
    print("\n=== legacy master → curated family mapping ===")
    n_legacy = int(pd.read_csv(args.current)["n_events"].sum()) if args.current.exists() else 0
    n_family = int(pd.read_csv(args.family)["n_events"].sum()) if args.family.exists() else 0
    print(f"  legacy master  total events: {n_legacy:,}")
    print(f"  curated family total events: {n_family:,}")
    if args.current.exists():
        legacy_rows = len(pd.read_csv(args.current))
        print(f"  legacy master  row count: {legacy_rows}")
    if args.family.exists():
        family_rows = len(pd.read_csv(args.family))
        print(f"  curated family row count: {family_rows}")

    # Flag any legacy key that splits across multiple families (interesting
    # cases where the direction/motif tuple no longer determines the family).
    splits = (xtab.groupby(legacy_cols)["family_id"]
                   .nunique()
                   .reset_index(name="n_unique_families"))
    multi = splits[splits["n_unique_families"] > 1]
    if len(multi) > 0:
        print(f"\n  {len(multi)} legacy keys split across >1 curated family "
              f"(often a sign of Stage 1g z-layer refinement).")
        print(multi.head(5).to_string(index=False))
    else:
        print("\n  no legacy key is split across multiple families.")

    # Show top 10 rows by event count for a quick eyeball check.
    top = xtab.sort_values("n_events", ascending=False).head(10)
    print("\n  top 10 (legacy key → curated family, by event count):")
    with pd.option_context("display.max_rows", None, "display.width", 140,
                           "display.max_colwidth", 32):
        print(top[legacy_cols + ["family_id", "n_events"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
