"""Tests for the FCC(100) two-basis lattice factory.

Two layers of test:
  - Pure-Python tests on `build_layer_descriptor()` and `first_neighbours()`.
    These run in CI without kmos installed.
  - Optional kmos integration test on `populate_kmos_project()` — skipped if
    `kmos` is not importable.
"""
from __future__ import annotations

from math import isclose, sqrt

import pytest

from Analysis.kmos_export.lattice_fcc100 import (
    A_LAT_DEFAULT,
    LAYER_NAME,
    SITE_A,
    SITE_B,
    build_layer_descriptor,
    first_neighbours,
    populate_kmos_project,
)
from Analysis.kmos_export.schema import LatticeDescriptor


# ------------------------------------------------------------------
# Descriptor shape
# ------------------------------------------------------------------
def test_default_descriptor_uses_ni_lattice_constant() -> None:
    desc = build_layer_descriptor()
    d = A_LAT_DEFAULT / sqrt(2.0)
    assert desc.cell_a == (d, 0.0, 0.0)
    assert desc.cell_b == (0.0, d, 0.0)
    assert desc.cell_c == (0.0, 0.0, A_LAT_DEFAULT)


def test_descriptor_has_two_sites_at_canonical_positions() -> None:
    desc = build_layer_descriptor()
    assert len(desc.sites) == 2
    assert desc.sites[0].name == SITE_A
    assert desc.sites[0].pos_frac == (0.0, 0.0, 0.0)
    assert desc.sites[1].name == SITE_B
    assert desc.sites[1].pos_frac == (0.5, 0.5, 0.5)


def test_descriptor_layer_name_default() -> None:
    desc = build_layer_descriptor()
    assert desc.name == LAYER_NAME


def test_descriptor_default_species_propagates() -> None:
    desc = build_layer_descriptor(default_species="Vacant")
    assert desc.sites[0].default_species == "Vacant"
    assert desc.sites[1].default_species == "Vacant"


def test_descriptor_rejects_zero_lattice_constant() -> None:
    with pytest.raises(ValueError, match="a_lat must be positive"):
        build_layer_descriptor(a_lat=0.0)


def test_descriptor_rejects_negative_lattice_constant() -> None:
    with pytest.raises(ValueError, match="a_lat must be positive"):
        build_layer_descriptor(a_lat=-1.0)


def test_descriptor_round_trips_through_pydantic() -> None:
    desc = build_layer_descriptor()
    blob = desc.model_dump_json()
    restored = LatticeDescriptor.model_validate_json(blob)
    assert restored == desc


# ------------------------------------------------------------------
# 1NN geometry — the critical check: all 12 neighbours of site_a must
# sit at distance d in absolute coords.
# ------------------------------------------------------------------
def _absolute(target_site: str, di: int, dj: int, dk: int, a_lat: float
              ) -> tuple[float, float, float]:
    """Cartesian position of a target_site at offset (di, dj, dk) relative
    to the unit cell at origin. Mirrors the lattice math we expect kmos to
    apply when building the explicit neighbour table at codegen time."""
    d = a_lat / sqrt(2.0)
    cell_origin = (di * d, dj * d, dk * a_lat)
    if target_site == SITE_A:
        site_frac = (0.0, 0.0, 0.0)
    elif target_site == SITE_B:
        site_frac = (0.5, 0.5, 0.5)
    else:
        raise AssertionError(target_site)
    abs_xyz = (
        cell_origin[0] + site_frac[0] * d,
        cell_origin[1] + site_frac[1] * d,
        cell_origin[2] + site_frac[2] * a_lat,
    )
    return abs_xyz


def _dist(p: tuple[float, float, float]) -> float:
    return sqrt(p[0]**2 + p[1]**2 + p[2]**2)


def test_site_a_first_neighbours_count_is_12() -> None:
    nns = first_neighbours(SITE_A)
    assert len(nns) == 12


def test_site_b_first_neighbours_count_is_12() -> None:
    nns = first_neighbours(SITE_B)
    assert len(nns) == 12


def test_site_a_all_first_neighbours_at_distance_d() -> None:
    a_lat = A_LAT_DEFAULT
    d = a_lat / sqrt(2.0)
    for target, di, dj, dk in first_neighbours(SITE_A):
        pos = _absolute(target, di, dj, dk, a_lat)
        dist = _dist(pos)
        assert isclose(dist, d, rel_tol=1e-9), (
            f"NN ({target}, {di}, {dj}, {dk}) at distance {dist}, expected {d}"
        )


def test_site_b_all_first_neighbours_at_distance_d() -> None:
    a_lat = A_LAT_DEFAULT
    d = a_lat / sqrt(2.0)
    # site_b is at (d/2, d/2, a_lat/2) — adjust origin accordingly
    site_b_origin = (d / 2.0, d / 2.0, a_lat / 2.0)
    for target, di, dj, dk in first_neighbours(SITE_B):
        pos = _absolute(target, di, dj, dk, a_lat)
        dx = pos[0] - site_b_origin[0]
        dy = pos[1] - site_b_origin[1]
        dz = pos[2] - site_b_origin[2]
        dist = sqrt(dx*dx + dy*dy + dz*dz)
        assert isclose(dist, d, rel_tol=1e-9), (
            f"NN ({target}, {di}, {dj}, {dk}) at distance {dist}, expected {d}"
        )


def test_first_neighbours_unique_offsets() -> None:
    """No (target_site, di, dj, dk) tuple appears twice for either site."""
    for site in (SITE_A, SITE_B):
        nns = first_neighbours(site)
        assert len(set(nns)) == len(nns), (
            f"site {site}: duplicate NN offsets {[n for n in nns if nns.count(n) > 1]}"
        )


def test_first_neighbours_unknown_site_raises() -> None:
    with pytest.raises(ValueError, match="unknown site name"):
        first_neighbours("c")


# ------------------------------------------------------------------
# Optional kmos-integration test
# ------------------------------------------------------------------
def _kmos_available() -> bool:
    try:
        import kmos.types  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _kmos_available(), reason="kmos not installed")
def test_populate_kmos_project_adds_layer_and_sites() -> None:
    from kmos.types import Project   # type: ignore[import-not-found]

    pt = Project()
    pt.meta.author = "test"
    pt.meta.email = "test@example.com"
    pt.meta.model_name = "fcc100_test"
    pt.meta.model_dimension = 3
    pt.add_species(name="Vacant", representation="")
    pt.add_species(name="Ni", representation="Atoms('Ni')")

    populate_kmos_project(pt)

    # Inspect the layer kmos now holds.
    assert len(pt.layer_list) == 1
    layer = pt.layer_list[0]
    assert layer.name == LAYER_NAME
    site_names = {s.name for s in layer.sites}
    assert site_names == {SITE_A, SITE_B}
    # cell_size text uses the d / a_lat values we expect
    cell_parts = pt.lattice.cell_size.split()
    assert len(cell_parts) == 3
    d_str, _, a_lat_str = cell_parts
    assert isclose(float(d_str), A_LAT_DEFAULT / sqrt(2.0), rel_tol=1e-6)
    assert isclose(float(a_lat_str), A_LAT_DEFAULT, rel_tol=1e-6)
