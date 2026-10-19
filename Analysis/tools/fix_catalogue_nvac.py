"""Recompute the catalogue's n_vac_nn1 columns using the corrected formula.

Original (broken) formula in 04b_ml_descriptors.py:
    n_vac_nn1 = max(0, 12 - coord_mover)
This treats every atom as bulk-coordinated (12 neighbors), so a surface
atom (coord=8) reports n_vac_nn1=4 even when isolated.

Corrected formula:
    expected_coord = 8 if surface_layer_initial == 0 else 12
    n_vac_nn1 = max(0, expected_coord - coord_mover)

The fix only requires existing columns: coord_mover_initial,
coord_mover_final, surface_layer_initial, surface_layer_final.

Input:  classified_events.csv (before family assignment, recommended) or
        classified_events_with_families.csv (also re-derives family_bucket_id)
Output: the same table with corrected n_vac_nn1_* and added n_vac_nn2_initial
        (``--output`` may equal ``--input``)

Usage (from the repository root):
    python -m Analysis.tools.fix_catalogue_nvac \
        --input  Analysis/lattice_event_classification/classified_events.csv \
        --output Analysis/lattice_event_classification/classified_events.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def corrected_nvac_nn1(coord, layer):
    """8 - coord for surface (layer=0), 12 - coord for bulk."""
    if pd.isna(coord) or pd.isna(layer):
        return np.nan
    expected = 8 if int(layer) == 0 else 12
    return float(max(0, int(expected) - int(coord)))


def corrected_nvac_nn2(nn2_total, layer):
    """Expected 2NN shell: 5 for surface (top layer), 6 for bulk-like."""
    if pd.isna(nn2_total) or pd.isna(layer):
        return np.nan
    expected = 5 if int(layer) == 0 else 6
    return float(max(0, int(expected) - int(nn2_total)))


def fix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add corrected n_vac_nn1_initial / _final columns (clobbers if present).

    Also re-derives `family_bucket_id` for families that bucket on n_vac_nn1
    so the rate-table builder picks up the corrected counts.
    """
    n_changed_init = 0
    n_changed_final = 0

    if "coord_mover_initial" in df.columns and "surface_layer_initial" in df.columns:
        new_initial = df.apply(
            lambda r: corrected_nvac_nn1(
                r["coord_mover_initial"], r["surface_layer_initial"]
            ),
            axis=1,
        )
        if "n_vac_nn1_initial" in df.columns:
            n_changed_init = int(
                (df["n_vac_nn1_initial"].fillna(-1) != new_initial.fillna(-1)).sum()
            )
        df["n_vac_nn1_initial"] = new_initial.astype("Int64")

    if "coord_mover_final" in df.columns and "surface_layer_final" in df.columns:
        new_final = df.apply(
            lambda r: corrected_nvac_nn1(
                r["coord_mover_final"], r["surface_layer_final"]
            ),
            axis=1,
        )
        if "n_vac_nn1_final" in df.columns:
            n_changed_final = int(
                (df["n_vac_nn1_final"].fillna(-1) != new_final.fillna(-1)).sum()
            )
        df["n_vac_nn1_final"] = new_final.astype("Int64")

    print(f'  changed n_vac_nn1_initial: {n_changed_init}')
    print(f'  changed n_vac_nn1_final:   {n_changed_final}')

    # --- n_vac_nn2_initial (using nn2_count_* columns) ---
    nn2_cols = [c for c in df.columns if c.startswith("nn2_count_")]
    if nn2_cols and "surface_layer_initial" in df.columns:
        df["nn2_total_atoms"] = df[nn2_cols].sum(axis=1)
        new_nv2 = df.apply(
            lambda r: corrected_nvac_nn2(r["nn2_total_atoms"], r["surface_layer_initial"]),
            axis=1,
        )
        df["n_vac_nn2_initial"] = new_nv2.astype("Int64")
        print(f'  added n_vac_nn2_initial (from nn2_count_* sum): '
              f'{df["n_vac_nn2_initial"].notna().sum()} non-null')
        print(f'    distribution:')
        print(df["n_vac_nn2_initial"].value_counts(dropna=False).sort_index().to_string())

    # Re-derive family_bucket_id.
    if "family_bucket_id" in df.columns and "family_id" in df.columns:
        # Two-axis (nv1, nv2) bucketing for the dominant 1NN single-atom hop families.
        # Separates truly-isolated (nv2=0) from clustered (nv2>=1) cases — this is
        # the fix for the remaining MSD overshoot after the nv1 correction.
        nv1_nv2_families = {
            "surface_1NN_inplane",
            "subsurface_1NN_inplane",
            "bulk_1NN_inplane",   # already used 2-axis
        }
        # Single-axis (nv1) bucketing for everything else.
        nv1_only_families = {
            "surface_2NN_diagonal", "subsurface_2NN_diagonal",
            "surface_subsurface_exchange_up",
            "surface_subsurface_exchange_down",
            "adatom_attachment",
            "adatom_detachment",
            # Legacy id of adatom_attachment, kept so pre-rename CSVs still
            # get corrected buckets (pylatkmc LEGACY_FAMILY_ALIASES).
            "surface_subsurface_exchange_lateral",
            "subsurface_migration_interlayer",
        }
        n_bucket_changed_2 = 0
        if "n_vac_nn2_initial" in df.columns:
            mask_2 = df["family_id"].isin(nv1_nv2_families)
            if mask_2.any():
                old_buckets = df.loc[mask_2, "family_bucket_id"].copy()
                def _two_axis(r):
                    n1 = r.get("n_vac_nn1_initial")
                    n2 = r.get("n_vac_nn2_initial")
                    if pd.isna(n1):
                        return "nv1=?"
                    if pd.isna(n2):
                        return f"nv1={int(n1)}"
                    return f"nv1={int(n1)}_nv2={int(n2)}"
                new_buckets = df.loc[mask_2].apply(_two_axis, axis=1)
                n_bucket_changed_2 = int((old_buckets != new_buckets).sum())
                df.loc[mask_2, "family_bucket_id"] = new_buckets.values
        print(f'  changed family_bucket_id (1NN, 2-axis nv1+nv2): {n_bucket_changed_2}')

        n_bucket_changed_1 = 0
        mask_1 = df["family_id"].isin(nv1_only_families)
        if mask_1.any():
            old_buckets = df.loc[mask_1, "family_bucket_id"].copy()
            new_buckets = df.loc[mask_1, "n_vac_nn1_initial"].apply(
                lambda v: f"nv1={int(v)}" if pd.notna(v) else "nv1=?"
            )
            n_bucket_changed_1 = int((old_buckets != new_buckets).sum())
            df.loc[mask_1, "family_bucket_id"] = new_buckets.values
        print(f'  changed family_bucket_id (other nv1-only):       {n_bucket_changed_1}')

    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="catalogue CSV to read (classified_events.csv or "
                        "classified_events_with_families.csv)")
    p.add_argument("--output", required=True, type=Path,
                   help="output CSV path")
    args = p.parse_args()

    print(f'reading {args.input}')
    df = pd.read_csv(args.input)
    print(f'  {len(df)} rows, {len(df.columns)} columns')

    fix_dataframe(df)

    print(f'writing {args.output}')
    df.to_csv(args.output, index=False)
    print(f'  wrote {len(df)} rows')

    # Quick verification
    if 'family_id' in df.columns:
        s1 = df[df['family_id'] == 'surface_1NN_inplane']
        print(f'\nsurface_1NN_inplane n_vac_nn1_initial distribution after fix:')
        print(s1['n_vac_nn1_initial'].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
