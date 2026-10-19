"""Move-type palette and classification thresholds for the scatter view.

The move-type labels are copied (not imported) from
`Analysis/lattice_map_events.py` so the Dash app does not need to import the
classification engine at runtime. Lattice distances come from
`Analysis/_paths.py`.
"""

from __future__ import annotations

# Lattice thresholds — single source in Analysis/_paths.py (KMCConfig).
from Analysis._paths import A_NI, NN1_DIST, NN2_DIST  # noqa: E402

LAYER_CHANGE_DZ = 1.0           # Å — out-of-plane threshold for layer change

# Move-type labels (must match lattice_map_events.TYPE_* string values) ------
TYPE_1NN = "1NN_hop"
TYPE_1NN_COOP = "1NN_hop_cooperative"
TYPE_1NN_TRIPLE = "1NN_hop_triple"
TYPE_2NN = "2NN_hop"
TYPE_EXCHANGE_UP = "exchange_up"
TYPE_EXCHANGE_DOWN = "exchange_down"
TYPE_EXCHANGE_LAT = "exchange_lateral"
TYPE_CONCERTED = "concerted"
TYPE_RELAXATION = "relaxation"
TYPE_COMPLEX = "complex"
TYPE_INTERLAYER_HOP_UP = "interlayer_hop_up"
TYPE_INTERLAYER_HOP_DOWN = "interlayer_hop_down"
TYPE_INPLANE_COMPLEX = "inplane_complex"
TYPE_SUBSURFACE_MIGRATION = "subsurface_migration"
TYPE_SUBSURFACE_HOP = "subsurface_hop"
TYPE_SUBSURFACE_1NN_HOP = "subsurface_1NN_hop"
TYPE_SUBSURFACE_OTHER = "subsurface_other"
# Not a classifier label: the viewer's placeholder for events the
# classifier CSV does not cover (data.load_events / single-run mode).
TYPE_UNCLASSIFIED = "unclassified"

ALL_TYPES = [
    TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN,
    TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN, TYPE_EXCHANGE_LAT,
    TYPE_CONCERTED, TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN,
    TYPE_INPLANE_COMPLEX, TYPE_COMPLEX, TYPE_RELAXATION,
    TYPE_SUBSURFACE_MIGRATION, TYPE_SUBSURFACE_HOP,
    TYPE_SUBSURFACE_1NN_HOP, TYPE_SUBSURFACE_OTHER,
    TYPE_UNCLASSIFIED,
]


def base_type(move_type: str) -> str:
    """Collapse direction suffixes (_up/_down/_lateral) to a base type."""
    for suffix in ("_up", "_down", "_lateral"):
        if move_type.endswith(suffix):
            stem = move_type[: -len(suffix)]
            if stem == "exchange":
                return "exchange"
            if stem == "interlayer_hop":
                return "interlayer_hop"
    return move_type


ALL_TYPES_BASE: list[str] = []
for _t in ALL_TYPES:
    _b = base_type(_t)
    if _b not in ALL_TYPES_BASE:
        ALL_TYPES_BASE.append(_b)


CUSTOM_COLORS: dict[str, str] = {
    TYPE_1NN: "#4DAF4A",
    TYPE_1NN_COOP: "#8DD35F",
    TYPE_1NN_TRIPLE: "#006400",
    TYPE_2NN: "#377EB8",
    TYPE_EXCHANGE_UP: "#E41A1C",
    TYPE_EXCHANGE_DOWN: "#FF6666",
    TYPE_EXCHANGE_LAT: "#FF7F00",
    TYPE_CONCERTED: "#984EA3",
    TYPE_INTERLAYER_HOP_UP: "#F781BF",
    TYPE_INTERLAYER_HOP_DOWN: "#A65628",
    TYPE_INPLANE_COMPLEX: "#BBBBBB",
    TYPE_RELAXATION: "#DDDDDD",
    TYPE_COMPLEX: "#999999",
    TYPE_SUBSURFACE_MIGRATION: "#6A3D9A",
    TYPE_SUBSURFACE_HOP: "#CAB2D6",
    TYPE_SUBSURFACE_1NN_HOP: "#1F78B4",
    TYPE_SUBSURFACE_OTHER: "#666666",
    TYPE_UNCLASSIFIED: "#607D8B",
    # Merged (up/down collapsed)
    "exchange": "#E41A1C",
    "interlayer_hop": "#F781BF",
}


def color_for(move_type: str) -> str:
    """Return the palette color for a move_type, or a neutral grey fallback."""
    return CUSTOM_COLORS.get(move_type, "#888888")


# Family palette ------------------------------------------------------------
# Keyed on family_id (pylatkmc.ingest.families.FAMILY_REGISTRY). Colors are
# grouped by mechanism family so related variants read as hue siblings:
#   greens  — single-atom 1NN in-plane hops (surface / subsurface / bulk)
#   blues   — 2NN diagonal hops
#   pinks   — interlayer hops
#   reds    — surface/subsurface exchanges (up / down)
#   oranges — adatom attachment / detachment
#   purples — subsurface migrations
#   greys   — multisite catch-alls

FAMILY_COLORS: dict[str, str] = {
    "surface_1NN_inplane":               "#4DAF4A",
    "subsurface_1NN_inplane":            "#1F7A1F",
    "bulk_1NN_inplane":                  "#006400",
    "surface_2NN_diagonal":              "#377EB8",
    "subsurface_2NN_diagonal":           "#1F4E79",
    "surface_interlayer_hop":            "#F781BF",
    "subsurface_interlayer_hop":         "#A65628",
    "surface_subsurface_exchange_up":    "#E41A1C",
    "surface_subsurface_exchange_down":  "#FF6666",
    "adatom_attachment":                 "#FF7F00",
    "adatom_detachment":                 "#FDBF6F",
    # Legacy id (pre-2026-08 rename of adatom_attachment) — kept so CSVs
    # written before the rename still color correctly.
    "surface_subsurface_exchange_lateral": "#FF7F00",
    "subsurface_migration_axial":        "#6A3D9A",
    "subsurface_migration_interlayer":   "#CAB2D6",
    "concerted_multisite":               "#999999",
    "unresolved_multisite":              "#BBBBBB",
}


def color_for_family(family_id: str) -> str:
    """Return the palette color for a family_id, or a neutral grey fallback."""
    return FAMILY_COLORS.get(family_id, "#888888")


def _load_family_ids_from_registry() -> list[str]:
    """Source the canonical family_id order from pylatkmc's ``FAMILY_REGISTRY``.

    pylatkmc is optional (``pip install -e .[families]``); without it this
    falls back to ``FAMILY_COLORS.keys()``.
    """
    try:
        from pylatkmc.ingest.families import FAMILY_REGISTRY  # type: ignore[import-not-found]
        return [f.family_id for f in FAMILY_REGISTRY]
    except Exception:
        return list(FAMILY_COLORS.keys())


ALL_FAMILY_IDS: list[str] = _load_family_ids_from_registry()


# Direction palette ---------------------------------------------------------
# Keyed on `direction_family_3d` (see Analysis/lattice_3d.py
# classify_direction_xyz). Colors are intentionally distinct from
# FAMILY_COLORS (greens/blues/pinks) and CUSTOM_COLORS so the three scatter
# color modes read as three independent palettes.

DIRECTION_IDS: list[str] = [
    "<110>_inplane",       # surface 1NN axial
    "<100>_inplane",       # 2NN diagonal
    "<111>_interlayer",    # 1NN cross-layer
    "<001>_exchange",      # 2NN axial
    "unresolved",
]

DIRECTION_COLORS: dict[str, str] = {
    "<110>_inplane":      "#00897B",  # teal
    "<100>_inplane":      "#FBC02D",  # amber
    "<111>_interlayer":   "#8E24AA",  # violet
    "<001>_exchange":     "#D81B60",  # magenta
    "unresolved":         "#9E9E9E",  # grey
}


def color_for_direction(direction_id: str) -> str:
    """Return the palette color for a direction_family_3d label."""
    return DIRECTION_COLORS.get(direction_id, "#888888")
