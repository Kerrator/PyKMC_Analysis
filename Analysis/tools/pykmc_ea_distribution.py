"""Extract the Ea distribution of events selected by a pyKMC reference run.

Usage (from the repository root):
    python -m Analysis.tools.pykmc_ea_distribution \\
        --sim /path/to/run \\
        --out pykmc_ea_distribution.csv

What it does:
    1. Parse pykmc.out to get per-step (ref_event, Ea, dt, ...).
    2. Load the reference_table.pickle to get the cluster geometry of each
       catalogue entry.
    3. Classify each catalogue entry's move_type via classify_event_from_row.
    4. Aggregate per move_type: how many times each was selected, and the
       step-weighted Ea statistics.
    5. Print headline numbers and write a CSV.

Why this matters: an on-lattice model built from the catalogue (e.g. with
pylatkmc) should reproduce the barriers pyKMC actually selects. This gives
that reference per move type, to compare against the barriers the
on-lattice model uses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from Analysis.event_viewer.data import _load_pickle
from Analysis.lattice_map_events import (
    classify_event_from_row,
    parse_pykmc_out,
)


def analyse_sim(sim_dir: Path) -> dict:
    """Run the full Ea-distribution analysis on one sim directory."""
    sim_dir = Path(sim_dir)
    pykmc_out = sim_dir / "pykmc.out"
    ref_pkl = sim_dir / "reference_table.pickle"
    if not pykmc_out.exists():
        raise FileNotFoundError(f"missing pykmc.out: {pykmc_out}")
    if not ref_pkl.exists():
        raise FileNotFoundError(f"missing reference_table.pickle: {ref_pkl}")

    # 1. Step log
    df_steps = parse_pykmc_out(pykmc_out)
    df_steps = df_steps[df_steps["ref_event"].notna()]   # drop the empty step-0 row
    df_steps["ref_event"] = df_steps["ref_event"].astype(int)
    n_steps = len(df_steps)
    if n_steps == 0:
        raise RuntimeError(f"pykmc.out at {pykmc_out} has zero usable steps")

    # 2. Reference table
    ref = _load_pickle(ref_pkl)
    if "idx_ref" not in ref.columns:
        raise RuntimeError("reference_table missing idx_ref column")

    # 3. Classify each catalogue entry (cache by idx_ref)
    classifications: dict[int, dict] = {}
    for _, row in ref.iterrows():
        idx = int(row["idx_ref"])
        try:
            cls = classify_event_from_row(row, include_saddle_features=False)
        except Exception as exc:
            cls = {"move_type": "_classify_failed", "error": str(exc)}
        classifications[idx] = cls

    # 4. Join + per-step move_type (note: classify_event_from_row returns
    # the type under key 'event_type', not 'move_type')
    def _get_type(i):
        c = classifications.get(int(i), {})
        return c.get("event_type") or c.get("move_type") or "_missing_idx_ref"

    df_steps["move_type"] = df_steps["ref_event"].map(_get_type)
    df_steps["move_dist"] = df_steps["ref_event"].map(
        lambda i: classifications.get(int(i), {}).get("move_dist", float("nan"))
    )
    df_steps["move_dz"] = df_steps["ref_event"].map(
        lambda i: classifications.get(int(i), {}).get("move_dz", float("nan"))
    )

    # 5. Aggregate per move_type
    agg = (
        df_steps.groupby("move_type")
        .agg(
            n_steps_fired=("Ea", "size"),
            n_unique_idx_ref=("ref_event", "nunique"),
            Ea_mean=("Ea", "mean"),
            Ea_std=("Ea", "std"),
            Ea_min=("Ea", "min"),
            Ea_max=("Ea", "max"),
            Ea_median=("Ea", "median"),
            time_total_s=("dt", "sum"),
        )
        .reset_index()
        .sort_values("n_steps_fired", ascending=False)
    )

    # Read input.in for context (T, n_steps, basin)
    input_in = sim_dir / "input.in"
    cfg = {}
    if input_in.exists():
        for line in input_in.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("[") and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()

    return {
        "sim": str(sim_dir),
        "T_K": float(cfg.get("T", "0")),
        "basin": cfg.get("basin", "?"),
        "n_steps_total": int(n_steps),
        "n_unique_ref_events_fired": int(df_steps["ref_event"].nunique()),
        "n_catalogue_entries": int(len(ref)),
        "Ea_overall_mean": float(df_steps["Ea"].mean()),
        "Ea_overall_std": float(df_steps["Ea"].std()),
        "total_time_s": float(df_steps["dt"].sum()),
        "agg": agg,
        "df_steps": df_steps,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", required=True, type=Path,
                   help="path to a basin=False pyKMC simulation directory")
    p.add_argument("--out", required=True, type=Path,
                   help="output CSV path for per-move_type aggregate stats")
    args = p.parse_args()

    res = analyse_sim(args.sim)

    print(f"\n{'=' * 70}")
    print(f"pyKMC reference Ea distribution — {Path(res['sim']).name}")
    print('=' * 70)
    print(f"  T = {res['T_K']:.0f} K, basin = {res['basin']}")
    print(f"  n_steps     = {res['n_steps_total']}")
    print(f"  unique events fired = {res['n_unique_ref_events_fired']} "
          f"(out of {res['n_catalogue_entries']} in catalogue)")
    print(f"  total sim time = {res['total_time_s']:.3e} s")
    print(f"  overall Ea mean = {res['Ea_overall_mean']:.4f} ± {res['Ea_overall_std']:.4f} eV")
    print()
    print(f"  Per move_type breakdown:")
    print(res["agg"].to_string(index=False, float_format="%.4f"))

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res["agg"].to_csv(args.out, index=False)
    print()
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
