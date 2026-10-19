"""Visual conventions for the APE-style 2D schematic renderer.

Complements :mod:`Analysis.blender_viewer.constants` (which holds the
photorealistic-render conventions). Element colours and metallic radii are
shared via re-export from blender_viewer — only the schematic-specific
overrides (flat palette, matplotlib line widths, default coordination
threshold) live here.

The palette is chosen to match the Automatic Process Explorer (APE) figure
aesthetic from Trochet et al.: a vivid orange primary mover against a
muted grey substrate, with yellow accents on the immediate
under-coordinated neighbours of the defect.
"""

from __future__ import annotations

from Analysis.blender_viewer.constants import (
    COVALENT_RADIUS_A,
    JMOL_HEX,
    METALLIC_RADIUS_A,
    display_radius,
)

# ---------------------------------------------------------------------------
# APE palette (flat, schematic-only)
# ---------------------------------------------------------------------------

#: Vivid orange — primary mover (the canonical ``mover_idx`` atom).
MOVER_HEX: str = "#FF6633"

#: Yellow — undercoordinated atoms in the immediate neighbourhood of the
#: defect. Picked out by :func:`Analysis.schematic_viewer.coordination.find_active_neighbours`.
ACTIVE_HEX: str = "#F4C430"

#: Light grey — regular substrate atoms.
SUBSTRATE_HEX: str = "#CCCCCC"

#: Outline colour for a (rare) vacancy site rendered as a hollow dashed circle.
VACANCY_OUTLINE_HEX: str = "#666666"

#: Dark navy — between-panel flow arrows.
ARROW_HEX: str = "#1F3A5F"

#: Black silhouette around every atom — the APE "strong outline" cue.
ATOM_EDGE_COLOR: str = "#000000"

#: Atom silhouette line width in points (matplotlib units, not Å).
ATOM_EDGE_WIDTH_PT: float = 0.6

# ---------------------------------------------------------------------------
# Radii scaling
# ---------------------------------------------------------------------------

#: Multiplier on :data:`METALLIC_RADIUS_A` (or fallback :data:`COVALENT_RADIUS_A`)
#: to get the rendered disc radius in Å. Sits between blender_viewer's
#: ``VESTA_RADIUS_SCALE`` (0.40) and ``EVENT_RADIUS_SCALE`` (0.55).
SCHEMATIC_RADIUS_SCALE: float = 0.45

#: Multiplier on the schematic radius for the mover atom — same convention
#: as :data:`Analysis.blender_viewer.constants.MOVER_RADIUS_BOOST`.
MOVER_RADIUS_BOOST: float = 1.15

# ---------------------------------------------------------------------------
# Coordination-based role inference
# ---------------------------------------------------------------------------

#: First-shell cutoff radius (Å) used to count neighbours for the
#: under-coordinated test. 3.5 Å sits cleanly above the FCC Ni 1NN
#: distance (~2.49 Å) and below the 2NN distance (~3.52 Å).
DEFAULT_RNEI_A: float = 3.5

#: Coordination threshold — atoms with strictly fewer than this many
#: 1NN neighbours are flagged as "under-coordinated". 9 is the value
#: used by the basin module's fingerprint pre-filter for FCC slabs.
DEFAULT_COORD_THR: int = 9

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

#: Default figure size for a 1-row × 3-column triptych, in inches.
DEFAULT_FIGSIZE_IN: tuple[float, float] = (12.0, 4.5)

#: Default DPI for the PNG sibling of the vector PDF output.
DEFAULT_DPI: int = 300

#: Padding (Å) added around the union bounding box of init/saddle/final
#: positions before locking all three panels to a shared axis range.
PANEL_MARGIN_A: float = 1.5

# Re-export the shared blender_viewer lookups so callers do not need to
# import from two places. Suppressed from the module's public ``__all__``
# below because they're not schematic-specific — but the import keeps
# ``Analysis.schematic_viewer.constants.METALLIC_RADIUS_A`` resolvable.
_ = (METALLIC_RADIUS_A, COVALENT_RADIUS_A, JMOL_HEX, display_radius)

__all__ = [
    "MOVER_HEX",
    "ACTIVE_HEX",
    "SUBSTRATE_HEX",
    "VACANCY_OUTLINE_HEX",
    "ARROW_HEX",
    "ATOM_EDGE_COLOR",
    "ATOM_EDGE_WIDTH_PT",
    "SCHEMATIC_RADIUS_SCALE",
    "MOVER_RADIUS_BOOST",
    "DEFAULT_RNEI_A",
    "DEFAULT_COORD_THR",
    "DEFAULT_FIGSIZE_IN",
    "DEFAULT_DPI",
    "PANEL_MARGIN_A",
    "METALLIC_RADIUS_A",
    "COVALENT_RADIUS_A",
    "JMOL_HEX",
    "display_radius",
]
