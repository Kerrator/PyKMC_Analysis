"""Catalogue → ProcessDescriptor list. Pilot scope: one family at a time.

The pilot translates a single family (`surface_1NN_inplane` is the canonical
target) into kmos `Process` descriptors.

Pilot simplifications:
  - Single `mover_species` (default `Ni`).
  - Single event-weighted-mean Ea across all buckets, rather than per-bucket
    rate selection. This loses the n_vac dependence but is enough to verify
    the translator pipeline. A full version would emit one process per bucket.
  - No bystanders → no OTF rate modulation → uses kmos `rate_constant=`
    (local_smart backend).
  - For the in-plane family, emits 8 processes covering ±x, ±y on both
    sites a and b (FCC(100) two-basis lattice). Other families are not yet
    translated.

The translator's output is our IR (`ProcessDescriptor` / `KmosProjectDescriptor`).
The IR-to-kmos render lives in `emitter.py`; this module never imports kmos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Analysis.kmos_export.lattice_fcc100 import (
    A_LAT_DEFAULT,
    LAYER_NAME,
    SITE_A,
    SITE_B,
    build_layer_descriptor,
    first_neighbours,
)
from Analysis.kmos_export.schema import (
    ActionDescriptor,
    ConditionDescriptor,
    CoordOffset,
    KmosProjectDescriptor,
    ProcessDescriptor,
)


@dataclass(frozen=True)
class FamilyEaSummary:
    """Bucket statistics + the event-weighted-mean Ea used by the pilot."""

    family_id: str
    n_events: int
    n_buckets: int
    Ea_weighted_mean_eV: float
    Ea_min_eV: float
    Ea_max_eV: float
    bucket_means: dict[str, float]
    bucket_counts: dict[str, int]


def summarize_family(
    df: pd.DataFrame, family_id: str, *,
    accepted_only: bool = True,
) -> FamilyEaSummary:
    """Filter the curated CSV to one family and compute per-bucket statistics.

    `df` is `classified_events_with_families.csv` (or any superset).
    `family_id` matches the CSV's `family_id` column.
    """
    sub = df[df["family_id"] == family_id]
    if accepted_only and "assignment_status" in sub.columns:
        sub = sub[sub["assignment_status"] == "accepted"]
    if sub.empty:
        raise ValueError(
            f"no accepted events for family_id={family_id!r} in dataframe"
        )

    g = sub.groupby("family_bucket_id")["energy_barrier"].agg(["mean", "count"])
    means = {str(k): float(v) for k, v in g["mean"].items()}
    counts = {str(k): int(v) for k, v in g["count"].items()}
    Ea_arr = sub["energy_barrier"].to_numpy(dtype=float)
    weights = np.ones_like(Ea_arr)   # each row is one event
    Ea_weighted = float(np.average(Ea_arr, weights=weights))
    return FamilyEaSummary(
        family_id=family_id,
        n_events=len(sub),
        n_buckets=g.shape[0],
        Ea_weighted_mean_eV=Ea_weighted,
        Ea_min_eV=float(Ea_arr.min()),
        Ea_max_eV=float(Ea_arr.max()),
        bucket_means=means,
        bucket_counts=counts,
    )


# ------------------------------------------------------------------
# In-plane 1NN process generator
# ------------------------------------------------------------------
_DIRECTION_LABELS: dict[tuple[int, int, int], str] = {
    (1, 0, 0):  "px",
    (-1, 0, 0): "mx",
    (0, 1, 0):  "py",
    (0, -1, 0): "my",
}

# For the kmos backend we just need the parameter name in the rate formula.
# Single Ea param per family in the pilot.
_RATE_FORMULA_TEMPLATE = "k0 * exp(-{Ea_param} * eV / (kboltzmann*T))"


def _direction_label(target_site: str, di: int, dj: int, dk: int) -> str:
    """Short, kmos-identifier-safe label for a (target, offset) tuple.
    Maps the in-plane offsets ±x, ±y to 'px', 'mx', 'py', 'my'."""
    label = _DIRECTION_LABELS.get((di, dj, dk))
    if label is None:
        # Out-of-plane: build something like 'k_p1' or 'i_m1_k_p1' — only
        # needed once translation extends beyond in-plane.
        parts: list[str] = []
        for axis_name, val in (("i", di), ("j", dj), ("k", dk)):
            if val == 0:
                continue
            parts.append(f"{axis_name}{'p' if val > 0 else 'm'}{abs(val)}")
        label = "_".join(parts) if parts else "self"
    return f"{target_site}_{label}"


def _is_in_plane(target_site: str, di: int, dj: int, dk: int,
                 from_site: str) -> bool:
    """An in-plane 1NN of an FCC(100) site stays on the same z-layer.
    With our two-basis cell (a at z=0, b at z=a_lat/2) and cell_c = a_lat,
    'in-plane' means dk == 0 AND target_site == from_site."""
    return dk == 0 and target_site == from_site


def emit_in_plane_processes(
    family_id: str,
    Ea_param_name: str,
    *,
    mover_species: str = "Ni",
    name_prefix: str | None = None,
) -> list[ProcessDescriptor]:
    """Build the 8 in-plane 1NN hop processes for the FCC(100) two-basis cell.

    For each of {site_a, site_b}, emits 4 processes for ±x, ±y in-plane hops.
    All 8 share a single rate-constant referencing `Ea_param_name`.
    """
    if name_prefix is None:
        name_prefix = family_id

    processes: list[ProcessDescriptor] = []
    rate_formula = _RATE_FORMULA_TEMPLATE.format(Ea_param=Ea_param_name)

    for from_site in (SITE_A, SITE_B):
        for target_site, di, dj, dk in first_neighbours(from_site):
            if not _is_in_plane(target_site, di, dj, dk, from_site):
                continue
            label = _direction_label(target_site, di, dj, dk)
            origin_coord = CoordOffset(site=from_site)
            target_coord = CoordOffset(site=target_site, i=di, j=dj, k=dk)
            proc = ProcessDescriptor(
                name=f"{name_prefix}_{mover_species}_{from_site}_to_{label}",
                conditions=(
                    ConditionDescriptor(coord=origin_coord, species=mover_species),
                    ConditionDescriptor(coord=target_coord, species="Vacant"),
                ),
                actions=(
                    ActionDescriptor(coord=origin_coord, species="Vacant"),
                    ActionDescriptor(coord=target_coord, species=mover_species),
                ),
                rate_formula=rate_formula,
                family_id=family_id,
                notes=(
                    f"In-plane 1NN hop on FCC(100), site={from_site} → "
                    f"target={target_site}.({di},{dj},{dk}); "
                    f"pilot uses single bucket-aggregated Ea."
                ),
            )
            processes.append(proc)
    return processes


# ------------------------------------------------------------------
# Top-level pilot translator
# ------------------------------------------------------------------
def translate_pilot_surface_1NN(
    catalogue: pd.DataFrame,
    *,
    project_name: str = "fcc100_pilot_surface_1NN",
    mover_species: str = "Ni",
    a_lat: float = A_LAT_DEFAULT,
    temperature_K: float = 500.0,
    k0_Hz: float = 1.0e13,
) -> tuple[KmosProjectDescriptor, FamilyEaSummary]:
    """Pilot entry point: translate `surface_1NN_inplane` end-to-end.

    Returns a `(project_descriptor, family_summary)` tuple. The descriptor
    is renderable by `emitter.descriptor_to_kmos_project()`; the summary
    holds the per-bucket statistics for reporting / debugging.
    """
    family_id = "surface_1NN_inplane"
    summary = summarize_family(catalogue, family_id)

    Ea_param_name = f"Ea_{family_id}"
    processes = emit_in_plane_processes(
        family_id=family_id,
        Ea_param_name=Ea_param_name,
        mover_species=mover_species,
    )

    lattice = build_layer_descriptor(a_lat=a_lat, layer_name=LAYER_NAME,
                                     default_species="Vacant")
    species_tuple = ("Vacant", mover_species)

    parameters: dict[str, float] = {
        "T": float(temperature_K),
        "k0": float(k0_Hz),
        Ea_param_name: float(summary.Ea_weighted_mean_eV),
    }

    proj = KmosProjectDescriptor(
        name=project_name,
        lattice=lattice,
        species=species_tuple,
        parameters=parameters,
        processes=tuple(processes),
    )
    return proj, summary
