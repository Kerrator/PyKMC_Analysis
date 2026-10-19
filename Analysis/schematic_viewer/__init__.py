"""Flat 2D triptych renderer for KMC events — schematic counterpart to
:mod:`Analysis.blender_viewer`.

Reads the same ``.npz`` event files produced by
:func:`Analysis.blender_viewer.prepare.main` and produces APE-style
init / saddle / final figures suitable for slides and thesis chapters.

See ``README.md`` in the package directory.
"""

from __future__ import annotations

from Analysis.schematic_viewer.cli import main
from Analysis.schematic_viewer.constants import (
    ACTIVE_HEX,
    ARROW_HEX,
    DEFAULT_COORD_THR,
    DEFAULT_DPI,
    DEFAULT_RNEI_A,
    MOVER_HEX,
    SUBSTRATE_HEX,
)
from Analysis.schematic_viewer.coordination import (
    DEFAULT_DISP_THR_A,
    coordination_counts,
    find_active_neighbours,
    find_active_set,
    find_co_movers,
)
from Analysis.schematic_viewer.projection import project_positions, union_bbox
from Analysis.schematic_viewer.render import make_triptych_figure

__all__ = [
    "main",
    "make_triptych_figure",
    "project_positions",
    "union_bbox",
    "coordination_counts",
    "find_active_neighbours",
    "find_co_movers",
    "find_active_set",
    "DEFAULT_DISP_THR_A",
    "MOVER_HEX",
    "ACTIVE_HEX",
    "SUBSTRATE_HEX",
    "ARROW_HEX",
    "DEFAULT_COORD_THR",
    "DEFAULT_RNEI_A",
    "DEFAULT_DPI",
]
