"""Tests for the catalogue → ProcessDescriptor translator.

These tests use a synthetic fixture DataFrame (small) so they're fast and
deterministic — no dependency on the live curated CSV.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from Analysis.kmos_export.lattice_fcc100 import SITE_A, SITE_B
from Analysis.kmos_export.schema import (
    ActionDescriptor,
    ConditionDescriptor,
    KmosProjectDescriptor,
    ProcessDescriptor,
)
from Analysis.kmos_export.translator import (
    FamilyEaSummary,
    emit_in_plane_processes,
    summarize_family,
    translate_pilot_surface_1NN,
)


# ------------------------------------------------------------------
# Test fixture: a small synthetic catalogue
# ------------------------------------------------------------------
def _synthetic_catalogue(
    family_id: str = "surface_1NN_inplane",
    bucket_means: dict[str, float] | None = None,
    n_per_bucket: int = 100,
    mover: str = "Ni",
    *,
    include_other_family: bool = True,
    include_rejected: bool = True,
) -> pd.DataFrame:
    """Build a deterministic DataFrame matching the curated-CSV columns the
    translator actually reads."""
    if bucket_means is None:
        bucket_means = {"nv1=4": 0.90, "nv1=5": 0.74, "nv1=6": 0.54}
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for bucket, mean_Ea in bucket_means.items():
        for _ in range(n_per_bucket):
            rows.append({
                "family_id": family_id,
                "family_bucket_id": bucket,
                "energy_barrier": float(mean_Ea + rng.normal(0, 0.05)),
                "mover_species_ml": mover,
                "assignment_status": "accepted",
            })
    if include_other_family:
        for _ in range(50):
            rows.append({
                "family_id": "other_family",
                "family_bucket_id": "*",
                "energy_barrier": 1.5,
                "mover_species_ml": mover,
                "assignment_status": "accepted",
            })
    if include_rejected:
        for _ in range(20):
            rows.append({
                "family_id": family_id,
                "family_bucket_id": "nv1=4",
                "energy_barrier": 99.0,   # garbage; should be filtered out
                "mover_species_ml": mover,
                "assignment_status": "rejected",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# summarize_family
# ------------------------------------------------------------------
def test_summarize_family_counts_only_accepted_by_default() -> None:
    df = _synthetic_catalogue(n_per_bucket=100)
    summary = summarize_family(df, "surface_1NN_inplane")
    # 3 buckets × 100 events = 300; 20 rejected rows excluded.
    assert summary.n_events == 300
    assert summary.n_buckets == 3


def test_summarize_family_buckets_align_with_inputs() -> None:
    df = _synthetic_catalogue(
        bucket_means={"nv1=5": 0.74, "nv1=6": 0.54}, n_per_bucket=200,
    )
    summary = summarize_family(df, "surface_1NN_inplane")
    assert set(summary.bucket_means) == {"nv1=5", "nv1=6"}
    # Bucket counts match the synthetic fixture.
    assert summary.bucket_counts == {"nv1=5": 200, "nv1=6": 200}


def test_summarize_family_weighted_mean_close_to_bucket_average() -> None:
    """For two equal-count buckets, the weighted mean is the simple average
    of bucket means (within sampling noise)."""
    df = _synthetic_catalogue(
        bucket_means={"nv1=5": 0.50, "nv1=7": 1.00}, n_per_bucket=500,
    )
    summary = summarize_family(df, "surface_1NN_inplane")
    expected = 0.75   # equal counts
    assert math.isclose(summary.Ea_weighted_mean_eV, expected, rel_tol=0.05)


def test_summarize_family_raises_when_no_matching_rows() -> None:
    df = _synthetic_catalogue(n_per_bucket=10, include_other_family=False)
    with pytest.raises(ValueError, match="no accepted events"):
        summarize_family(df, "totally_unknown_family")


def test_summarize_family_returns_dataclass_shape() -> None:
    df = _synthetic_catalogue(n_per_bucket=100)
    summary = summarize_family(df, "surface_1NN_inplane")
    assert isinstance(summary, FamilyEaSummary)
    assert summary.family_id == "surface_1NN_inplane"
    assert summary.Ea_min_eV <= summary.Ea_weighted_mean_eV <= summary.Ea_max_eV


# ------------------------------------------------------------------
# emit_in_plane_processes — process shape
# ------------------------------------------------------------------
def test_emit_in_plane_processes_count_is_8() -> None:
    """4 directions × 2 sites = 8 in-plane hop processes."""
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
    )
    assert len(procs) == 8


def test_emit_in_plane_process_names_unique() -> None:
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
    )
    names = [p.name for p in procs]
    assert len(set(names)) == len(names)


def test_emit_in_plane_processes_cover_all_in_plane_directions() -> None:
    """Each process should target one of the four ±x, ±y in-plane offsets,
    and each (from_site, direction) pair should appear exactly once."""
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
    )
    in_plane_offsets = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)}
    seen: set[tuple[str, tuple[int, int, int]]] = set()
    for p in procs:
        # Origin is the from_site, target is the in-plane neighbour.
        cond_origin = p.conditions[0]
        cond_target = p.conditions[1]
        offset = (cond_target.coord.i, cond_target.coord.j, cond_target.coord.k)
        assert offset in in_plane_offsets
        seen.add((cond_origin.coord.site, offset))
    assert len(seen) == 8


def test_emit_in_plane_each_process_has_2_conditions_2_actions() -> None:
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
    )
    for p in procs:
        assert len(p.conditions) == 2
        assert len(p.actions) == 2
        # Origin condition has mover species; target has Vacant.
        assert p.conditions[0].species == "Ni"
        assert p.conditions[1].species == "Vacant"
        # Action mirrors: origin becomes Vacant; target becomes mover.
        assert p.actions[0].species == "Vacant"
        assert p.actions[1].species == "Ni"


def test_emit_in_plane_processes_share_rate_formula() -> None:
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_test",
    )
    formulas = {p.rate_formula for p in procs}
    assert len(formulas) == 1
    assert "Ea_test" in formulas.pop()


def test_emit_in_plane_processes_carry_family_id() -> None:
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
    )
    assert all(p.family_id == "surface_1NN_inplane" for p in procs)


def test_emit_in_plane_processes_other_mover() -> None:
    """Translator should accept Fe / Cr movers for alloy catalogues."""
    procs = emit_in_plane_processes(
        family_id="surface_1NN_inplane",
        Ea_param_name="Ea_x",
        mover_species="Fe",
    )
    for p in procs:
        assert p.conditions[0].species == "Fe"
        assert p.actions[1].species == "Fe"


# ------------------------------------------------------------------
# translate_pilot_surface_1NN — full pipeline
# ------------------------------------------------------------------
def test_translate_pilot_returns_valid_project_descriptor() -> None:
    df = _synthetic_catalogue(n_per_bucket=200)
    proj, summary = translate_pilot_surface_1NN(df)
    assert isinstance(proj, KmosProjectDescriptor)
    assert proj.name == "fcc100_pilot_surface_1NN"
    assert proj.species == ("Vacant", "Ni")
    # Lattice has the canonical two basis sites.
    assert {s.name for s in proj.lattice.sites} == {SITE_A, SITE_B}
    # 8 in-plane processes.
    assert len(proj.processes) == 8


def test_translate_pilot_parameters_carry_required_names() -> None:
    df = _synthetic_catalogue(n_per_bucket=200)
    proj, summary = translate_pilot_surface_1NN(df, temperature_K=600.0)
    assert "T" in proj.parameters
    assert proj.parameters["T"] == 600.0
    assert "k0" in proj.parameters
    assert "Ea_surface_1NN_inplane" in proj.parameters
    # Ea param matches the summary's weighted mean.
    assert math.isclose(
        proj.parameters["Ea_surface_1NN_inplane"],
        summary.Ea_weighted_mean_eV, rel_tol=1e-6,
    )


def test_translate_pilot_descriptor_passes_cross_reference_checks() -> None:
    """KmosProjectDescriptor enforces site / species / formula symbol
    cross-references at construction time. If translate_pilot returns one,
    the IR is internally consistent."""
    df = _synthetic_catalogue(n_per_bucket=200)
    proj, _ = translate_pilot_surface_1NN(df)
    # Just calling .model_dump round-trip is the tightest check; pydantic
    # re-validates on construction.
    blob = proj.model_dump_json()
    KmosProjectDescriptor.model_validate_json(blob)


def test_translate_pilot_rate_formula_uses_kmos_builtins() -> None:
    df = _synthetic_catalogue(n_per_bucket=200)
    proj, _ = translate_pilot_surface_1NN(df)
    formula = proj.processes[0].rate_formula
    assert "k0" in formula
    assert "exp" in formula
    # kmos's local_smart backend reads `rate_constant`,
    # not `otf_rate`. Our formula must be written for that backend.
    assert "kboltzmann" in formula
    assert "eV" in formula
