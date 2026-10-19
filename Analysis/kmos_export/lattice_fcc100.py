"""FCC(100) two-basis Layer factory for kmos.

Encodes the geometry:

    cell_a = (d, 0, 0)        where d = a_lat / sqrt(2)  ≈ 2.49 Å for Ni
    cell_b = (0, d, 0)
    cell_c = (0, 0, a_lat)    one A-B layer pair per unit cell

    site_a at fractional (0.0, 0.0, 0.0)    — even (A) layers
    site_b at fractional (0.5, 0.5, 0.5)    — odd  (B) layers,
                                              shifted by (d/2, d/2, a_lat/2)

Verifying 1NN distances (12 neighbours, all at distance d):
  * 4 in-plane: (±1, 0, 0).a or (0, ±1, 0).a — distance d
  * 4 above:    site_b at offsets (0,0,0)/(-1,0,0)/(0,-1,0)/(-1,-1,0)
                — distance √((d/2)² + (d/2)² + (a_lat/2)²) = d  (since a_lat = d√2)
  * 4 below:    site_b at offsets (...,0,-1) — by symmetry, also d

This module exposes two layers of API:
  - `build_layer_descriptor(...)` — produces our IR's `LatticeDescriptor`
    (no kmos import; testable in CI without the Fortran toolchain).
  - `populate_kmos_project(...)` — given a kmos `Project`, adds the
    matching `Layer` + `Site` definitions. Imports kmos lazily so this
    module still loads when kmos is absent.

Both halves intentionally agree: anything in the descriptor must show up
identically in the kmos project.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from Analysis.kmos_export.schema import LatticeDescriptor, SiteDescriptor

if TYPE_CHECKING:
    from kmos.types import Project as KmosProject     # type: ignore[import-not-found]


# Default lattice constant for the Ni / Fe / Cr alloy family. The catalogue
# uses Ni's a_lat as the FCC parameter (alloying with 5–10 % Cr / Fe shifts
# this by < 0.5 % at room T — well below our coord-tolerance budget).
A_LAT_NI = 3.524
A_LAT_DEFAULT = A_LAT_NI


# Canonical site names used everywhere downstream — translator, emitter,
# tests. Keep the names short so kmos's coord notation
# (`a.(1,0,0).fcc100`) stays readable.
SITE_A = "a"
SITE_B = "b"
LAYER_NAME = "fcc100"


def _nn_dist(a_lat: float) -> float:
    return a_lat / (2.0 ** 0.5)


def build_layer_descriptor(
    a_lat: float = A_LAT_DEFAULT,
    *,
    layer_name: str = LAYER_NAME,
    default_species: str = "Vacant",
) -> LatticeDescriptor:
    """Return a `LatticeDescriptor` for an FCC(100) two-basis cell.

    The cell is (d, d, a_lat) with d = a_lat/sqrt(2). Two sites: site_a
    (A-layer atom, frac 0,0,0) and site_b (B-layer atom, frac 0.5, 0.5,
    0.5). The B site offset is (d/2, d/2, a_lat/2) in absolute coords.
    """
    if a_lat <= 0.0:
        raise ValueError(f"a_lat must be positive, got {a_lat}")

    d = _nn_dist(a_lat)
    return LatticeDescriptor(
        name=layer_name,
        cell_a=(d, 0.0, 0.0),
        cell_b=(0.0, d, 0.0),
        cell_c=(0.0, 0.0, a_lat),
        sites=(
            SiteDescriptor(
                name=SITE_A,
                pos_frac=(0.0, 0.0, 0.0),
                default_species=default_species,
            ),
            SiteDescriptor(
                name=SITE_B,
                pos_frac=(0.5, 0.5, 0.5),
                default_species=default_species,
            ),
        ),
    )


def first_neighbours(site: str) -> tuple[tuple[str, int, int, int], ...]:
    """Return the 12 1NN (target_site, di, dj, dk) for a site at (0, 0, 0).

    Used by the translator to enumerate hop directions per process.
    Verified by `tests/test_lattice_fcc100.py`: all 12 are at distance d.
    """
    if site == SITE_A:
        # 4 in-plane neighbours of A — all on site_a in the same cell row
        in_plane: list[tuple[str, int, int, int]] = [
            (SITE_A, 1, 0, 0), (SITE_A, -1, 0, 0),
            (SITE_A, 0, 1, 0), (SITE_A, 0, -1, 0),
        ]
        # 4 above (k = +0): site_b at (0,0,0), (-1,0,0), (0,-1,0), (-1,-1,0)
        above: list[tuple[str, int, int, int]] = [
            (SITE_B, 0, 0, 0),
            (SITE_B, -1, 0, 0),
            (SITE_B, 0, -1, 0),
            (SITE_B, -1, -1, 0),
        ]
        # 4 below (k = -1): site_b in the cell below, same xy pattern
        below: list[tuple[str, int, int, int]] = [
            (SITE_B, 0, 0, -1),
            (SITE_B, -1, 0, -1),
            (SITE_B, 0, -1, -1),
            (SITE_B, -1, -1, -1),
        ]
        return tuple(in_plane + above + below)

    if site == SITE_B:
        # By symmetry: in-plane on site_b row, then site_a above and below
        in_plane = [
            (SITE_B, 1, 0, 0), (SITE_B, -1, 0, 0),
            (SITE_B, 0, 1, 0), (SITE_B, 0, -1, 0),
        ]
        # 4 above (k = +1): site_a in the cell above, mirror of A's pattern
        above = [
            (SITE_A, 0, 0, 1),
            (SITE_A, 1, 0, 1),
            (SITE_A, 0, 1, 1),
            (SITE_A, 1, 1, 1),
        ]
        # 4 below (k = +0): site_a in the same cell row, same xy pattern
        below = [
            (SITE_A, 0, 0, 0),
            (SITE_A, 1, 0, 0),
            (SITE_A, 0, 1, 0),
            (SITE_A, 1, 1, 0),
        ]
        return tuple(in_plane + above + below)

    raise ValueError(f"unknown site name {site!r}; expected {SITE_A!r} or {SITE_B!r}")


def populate_kmos_project(
    pt: "KmosProject",
    *,
    a_lat: float = A_LAT_DEFAULT,
    layer_name: str = LAYER_NAME,
    default_species: str = "Vacant",
) -> None:
    """Add an FCC(100) Layer to a kmos Project. Imports kmos lazily.

    Mutates `pt` in place: sets `pt.lattice.cell_size`, adds the Layer with
    two sites, and sets `pt.lattice.default_layer = layer_name`.
    `pt.species_list.default_species` is set to `default_species` so site
    defaults are consistent across the project.
    """
    from kmos.types import Layer, Site   # type: ignore[import-not-found]

    d = _nn_dist(a_lat)
    pt.lattice.cell_size = f"{d} {d} {a_lat}"
    pt.lattice.default_layer = layer_name

    layer = Layer(name=layer_name)
    layer.sites.append(
        Site(name=SITE_A, pos="0.0 0.0 0.0", default_species=default_species)
    )
    layer.sites.append(
        Site(name=SITE_B, pos="0.5 0.5 0.5", default_species=default_species)
    )
    pt.add_layer(layer)
    pt.species_list.default_species = default_species
