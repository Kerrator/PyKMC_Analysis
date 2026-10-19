"""Pure 3D lattice helpers for the KMC on-lattice pipeline.

These extend the original surface-only (2D) mapping to the full slab:
subsurface, exchange and interlayer events. All functions are pure
(no global state, no file I/O), take numpy/scalar inputs, and return
plain dicts/arrays. Unit-tested in Analysis/tests/. The method is
described in lattice_event_classification/EVENT_CLASSIFICATION.md.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_SQRT2 = np.sqrt(2.0)


def min_image(d: float, L: float) -> float:
    """Minimum-image wrap of a 1-D displacement to (-L/2, L/2].

    Used by callers that have a scalar Cartesian delta and the corresponding
    cell length. Assumes the true hop is shorter than L/2 — safe for any NN
    or 2NN pair in a standard periodic simulation box.
    """
    if d >  0.5 * L: return d - L
    if d < -0.5 * L: return d + L
    return d


def compute_3d_site_graph(
    ideal_sites: np.ndarray,
    cell: np.ndarray,
    nn_dist: float,
) -> dict[int, dict]:
    """Build a full 3D site graph with 1NN and 2NN adjacency.

    Reuses the PBC-aware cKDTree pattern from
    build_rate_table.build_lattice_infrastructure, but unified in a pure
    function that works for any slab orientation where consecutive layers
    are spaced by nn_dist / sqrt(2).

    Parameters
    ----------
    ideal_sites : np.ndarray, shape (N, 3)
        Cartesian coordinates of every ideal lattice site in the slab.
    cell : np.ndarray, shape (3,)
        Simulation cell lengths [Lx, Ly, Lz] for PBC images.
    nn_dist : float
        Nearest-neighbor distance in the ideal lattice.

    Returns
    -------
    dict mapping site_index -> {
        'position':    np.ndarray(3,),
        'layer_index': int,   # 0 = bottom, monotonic in z
        '1NN_3d':      list[int],
        '2NN_3d':      list[int],
    }
    """
    n = len(ideal_sites)
    Lx, Ly, _Lz = float(cell[0]), float(cell[1]), float(cell[2])
    offsets_xy = [
        (ox, oy, 0.0)
        for ox in (-Lx, 0.0, Lx)
        for oy in (-Ly, 0.0, Ly)
    ]
    images = np.vstack([ideal_sites + np.array(o) for o in offsets_xy])
    tree = cKDTree(images)

    cutoff_1nn = nn_dist * 1.1
    cutoff_2nn = nn_dist * np.sqrt(2.0) * 1.1

    # Layer index: bin z by (nn_dist / sqrt(2)) (the FCC(100) interlayer spacing).
    layer_spacing = nn_dist / np.sqrt(2.0)
    zs = ideal_sites[:, 2]
    # Round each z to the nearest layer-spacing multiple, then rank.
    z_bins = np.round(zs / layer_spacing).astype(int)
    unique_bins = np.unique(z_bins)
    unique_bins.sort()
    bin_rank = {int(b): r for r, b in enumerate(unique_bins)}

    graph: dict[int, dict] = {}
    for i in range(n):
        pos = ideal_sites[i]
        near_1nn = tree.query_ball_point(pos, cutoff_1nn)
        near_2nn = tree.query_ball_point(pos, cutoff_2nn)
        nn1 = sorted({j % n for j in near_1nn if (j % n) != i})
        nn2_raw = sorted({j % n for j in near_2nn if (j % n) != i})
        nn2 = [j for j in nn2_raw if j not in nn1]
        graph[i] = {
            "position": pos.copy(),
            "layer_index": bin_rank[int(z_bins[i])],
            "1NN_3d": nn1,
            "2NN_3d": nn2,
        }
    return graph


def assign_3d_site_classes(site_graph: dict[int, dict]) -> dict[int, str]:
    """Classify each site as surface / subsurface / bulk_like based on layer.

    Top layer -> 'surface'; second-from-top -> 'subsurface'; all others
    -> 'bulk_like'. Mirrors the convention used in
    build_rate_table.process_simulation().
    """
    if not site_graph:
        return {}
    top = max(info["layer_index"] for info in site_graph.values())
    out: dict[int, str] = {}
    for i, info in site_graph.items():
        li = info["layer_index"]
        if li == top:
            out[i] = "surface"
        elif li == top - 1:
            out[i] = "subsurface"
        else:
            out[i] = "bulk_like"
    return out


# Motif family names below are matched as strings by pylatkmc's family
# registry seed rules (pylatkmc/ingest/families.py): renaming one silently
# breaks family assignment there.
_MOTIF_TABLE = {
    "1NN_hop": "surface_1nn_translation",
    "1NN_hop_cooperative": "surface_1nn_translation",
    "1NN_hop_triple": "surface_1nn_translation",
    "2NN_hop": "surface_2nn_translation",
    "subsurface_1NN_hop": "subsurface_1nn_translation",
    "exchange_up": "surface_subsurface_exchange",
    "exchange_down": "surface_subsurface_exchange",
    "exchange_lateral": "surface_subsurface_exchange",
    "interlayer_hop_up": "interlayer_translation",
    "interlayer_hop_down": "interlayer_translation",
    # subsurface_hop is the reclassify_by_zlayer output for interlayer_hop_*
    # events below the top layer; same motif (interlayer_translation), the
    # bulk_like / subsurface distinction is carried by site_class_3d.
    "subsurface_hop": "interlayer_translation",
    "subsurface_migration": "subsurface_exchange",
}


def classify_3d_motif(row: dict) -> str:
    """Map an event row to a 3D motif family.

    Pure post-hoc relabeling; does not recompute any event geometry.
    Reads 'move_type', 'n_moved', and optionally 'surface_layer_initial'.
    subsurface_migration gets its own family per design decision
    (distinct from surface-initiated interlayer translations).
    """
    mt = row.get("move_type", "")
    n_moved = int(row.get("n_moved", 0))
    if mt in _MOTIF_TABLE:
        base = _MOTIF_TABLE[mt]
        # Surface 1NN-like events initiated from subsurface: demote label.
        if (
            base == "surface_1nn_translation"
            and int(row.get("surface_layer_initial", 0)) >= 1
        ):
            return "subsurface_1nn_translation"
        return base
    if mt == "concerted":
        return "concerted_3d"
    # 'complex' and 'inplane_complex' remain unresolved regardless of n_moved:
    # they have no single-motif interpretation.
    return "unresolved_multisite"


def extract_3d_occupancy_transition(
    row: dict,
    site_graph: dict[int, dict],
    occupancy: np.ndarray,
) -> dict:
    """Express an event as a minimal occupancy transition.

    Returns occ_init, occ_final, changed_sites (set), and the vacancy
    origin/destination site indices. Single-vacancy hop case. Concerted
    or multi-atom events return occ_init == occ_final and an empty
    changed_sites set — callers should branch on len(changed_sites).
    """
    vac_origin = int(row.get("vac_origin_site", -1))
    vac_dest = int(row.get("vac_dest_site", -1))
    occ_init = occupancy.copy()
    occ_final = occupancy.copy()
    if vac_origin >= 0 and vac_dest >= 0 and vac_origin != vac_dest:
        occ_final[vac_origin] = 1
        occ_final[vac_dest] = 0
        changed: set[int] = {vac_origin, vac_dest}
    else:
        changed = set()
    # site_graph parameter retained for API completeness and future extensions
    # (e.g., validating that vac_origin/vac_dest are valid site indices).
    _ = site_graph
    return {
        "occ_init": occ_init,
        "occ_final": occ_final,
        "changed_sites": changed,
        "vac_origin_3d": vac_origin if vac_origin >= 0 else None,
        "vac_dest_3d": vac_dest if vac_dest >= 0 else None,
    }


def compute_3d_environment_descriptor(
    vac_origin: int,
    vac_dest: int,
    occupancy: np.ndarray,
    site_graph: dict[int, dict],
    species: np.ndarray | None = None,
) -> dict:
    """Compute 3D environment counts around the vacancy origin.

    Counts vacancies in 1NN and 2NN shells (all layers), splits the 1NN
    shell into above/below by relative layer index. When `species` is
    provided, also returns species-histogram keys for each shell.
    """
    info = site_graph[vac_origin]
    layer0 = info["layer_index"]
    nn1 = info["1NN_3d"]
    nn2 = info["2NN_3d"]
    n_vac_1nn = sum(1 for j in nn1 if occupancy[j] == 0 and j != vac_dest)
    n_vac_2nn = sum(1 for j in nn2 if occupancy[j] == 0 and j != vac_dest)
    n_above = sum(
        1
        for j in nn1
        if site_graph[j]["layer_index"] > layer0
        and occupancy[j] == 0
        and j != vac_dest
    )
    n_below = sum(
        1
        for j in nn1
        if site_graph[j]["layer_index"] < layer0
        and occupancy[j] == 0
        and j != vac_dest
    )
    top = max(g["layer_index"] for g in site_graph.values())
    if layer0 == top:
        sc = "surface"
    elif layer0 == top - 1:
        sc = "subsurface"
    else:
        sc = "bulk_like"
    out: dict = {
        "n_vacant_1nn_3d": n_vac_1nn,
        "n_vacant_2nn_3d": n_vac_2nn,
        "n_vacant_above": n_above,
        "n_vacant_below": n_below,
        "site_class": sc,
    }
    if species is not None:
        for shell_name, shell in (("1nn", nn1), ("2nn", nn2)):
            for elem in np.unique(species):
                out[f"species_count_{shell_name}_{elem}"] = sum(
                    1 for j in shell if occupancy[j] == 1 and species[j] == elem
                )
    return out


def assign_direction_family(di: int, dj: int, dz: int) -> str:
    """DEPRECATED: integer-lattice-index direction classifier.

    Kept for backward-compatibility only. Call sites that have access to
    Cartesian displacements should use :func:`classify_direction_xyz`
    instead — the integer-index form cannot distinguish a 1NN cross-layer
    hop (<111>_interlayer) from a 2NN-axial exchange (<001>_exchange) when
    only the LAYER delta is available (hop_di, hop_dj = 0, 0 both cases).

    Convention: cubic-frame Miller indices. On FCC(100) the in-plane 1NN
    vectors running along the unit-cell edges correspond to cubic <110>
    directions; in-plane 2NN vectors (face-diagonals within the layer)
    correspond to cubic <100>; interlayer 1NN vectors are <111>; pure
    z-exchange with no in-plane shift is <001>.
    """
    adi, adj, adz = abs(int(di)), abs(int(dj)), abs(int(dz))
    if adz == 0 and (adi + adj) == 1:
        return "<110>_inplane"
    if adz == 0 and adi == 1 and adj == 1:
        return "<100>_inplane"
    if adz >= 1 and (adi + adj) == 1:
        return "<111>_interlayer"
    if adz >= 1 and (adi + adj) == 0:
        return "<001>_exchange"
    return "unresolved"


def classify_direction_xyz(dx: float, dy: float, dz: float, nn_dist: float) -> str:
    """Cartesian-vector direction classifier for FCC(100).

    Prefer this over :func:`assign_direction_family` whenever the raw
    displacement vector is available — it correctly distinguishes a 1NN
    cross-layer hop (|d| ≈ nn_dist, |dz| ≈ layer_spacing) from a
    2NN-axial exchange (|d_xy| ≈ 0, |dz| ≈ 2·layer_spacing).

    Mirrors the classifier in pylatkmc's `tools/build_initial_config.py`,
    so the rate-table training pipeline and the runtime agree on edge labels.

    Convention (FCC(100) with layer spacing ls = nn_dist / sqrt(2)):
      |dz| ≈ 0, |d_xy| ≈ nn_dist                 → <110>_inplane (1NN axial)
      |dz| ≈ 0, |d_xy| ≈ sqrt(2)·nn_dist         → <100>_inplane (2NN diag)
      |dz| ≈ ls                                  → <111>_interlayer (1NN cross-layer)
      |dz| ≈ 2·ls, |d_xy| ≈ 0                    → <001>_exchange (2NN axial)
      anything else                              → unresolved
    """
    ls = nn_dist / _SQRT2
    tol = 0.1 * nn_dist
    d_xy = np.hypot(dx, dy)
    adz = abs(dz)
    if adz < tol:
        if abs(d_xy - nn_dist)           < tol: return "<110>_inplane"
        if abs(d_xy - _SQRT2 * nn_dist)  < tol: return "<100>_inplane"
        return "unresolved"
    if abs(adz - ls) < tol:
        return "<111>_interlayer"
    if abs(adz - 2.0 * ls) < tol and d_xy < tol:
        return "<001>_exchange"
    return "unresolved"
