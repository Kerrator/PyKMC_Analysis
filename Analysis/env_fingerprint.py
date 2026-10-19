"""Lattice-environment fingerprints for FCC(100) event identification.

Builds a canonical (per-site occupancy + hop vector) fingerprint for each
KMC event. Two events with identical local-lattice configurations (modulo
D4 surface symmetry) yield the same fingerprint string and feed the same
rate-table row.

Layout:
- Shell templates (surface / bulk) — 13 / 18 lattice sites in 1NN+2NN
- D4 group operations — 8 elements (4 rotations + 4 mirrors), preserve z
- Permutation tables — pre-computed mapping of slot index under each op
- Per-event extraction functions — `compute_env_fingerprint(row, ...)`

Units (integer tuples): site offsets are in (nn1/2, nn1/2, a/2) where
nn1 = NN1_DIST = a/sqrt(2). This packs both the in-plane sub-lattice
shifts (±0.5*nn1 between layers) and the interlayer spacing into a single
integer grid.

Used by ``lattice_map_events.compute_env_fingerprint`` (ML Tier 5 columns of
classified_events.csv); tested in ``Analysis/tests/test_env_fingerprint.py``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Shell templates
# ---------------------------------------------------------------------------
# Site offsets in integer units of (nn1/2, nn1/2, a/2).
# Surface (top-layer) site has 13 neighbours in 1NN+2NN; bulk has 18.

# --- 1NN ---
_1NN_INPLANE = [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0)]
_1NN_BELOW = [(1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)]
_1NN_ABOVE = [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]

# --- 2NN ---
_2NN_INPLANE = [(2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0)]
_2NN_BELOW = [(0, 0, -2)]
_2NN_ABOVE = [(0, 0, 2)]

# Surface (top layer): 4 in-plane 1NN + 4 below 1NN + 4 in-plane 2NN + 1 below 2NN
SHELL_REF_POSITIONS_SURFACE: List[Tuple[int, int, int]] = (
    _1NN_INPLANE + _1NN_BELOW + _2NN_INPLANE + _2NN_BELOW
)

# Bulk: 4 in-plane 1NN + 4 above + 4 below 1NN + 4 in-plane 2NN + 1 above + 1 below 2NN
SHELL_REF_POSITIONS_BULK: List[Tuple[int, int, int]] = (
    _1NN_INPLANE + _1NN_ABOVE + _1NN_BELOW
    + _2NN_INPLANE + _2NN_ABOVE + _2NN_BELOW
)

assert len(SHELL_REF_POSITIONS_SURFACE) == 13
assert len(SHELL_REF_POSITIONS_BULK) == 18


# ---------------------------------------------------------------------------
# D4 symmetry group: 8 ops preserving the slab z-axis.
# ---------------------------------------------------------------------------
D4_OPS: List[np.ndarray] = [
    # 4 rotations about z-axis
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),     # identity
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),    # rot 90°
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),   # rot 180°
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),    # rot 270°
    # 4 mirror planes (all perpendicular to xy plane)
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),    # mirror x-axis
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),    # mirror y-axis
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),     # mirror diag y=x
    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),   # mirror diag y=-x
]


def _build_d4_permutations(ref_positions: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Pre-compute slot permutations under each D4 operation.

    perms[op_idx, i] = j means: under operation op_idx, the site at
    reference slot i maps to the site at reference slot j.

    Returns
    -------
    np.ndarray (n_ops, n_slots) of int
    """
    n_slots = len(ref_positions)
    pos_arr = np.array(ref_positions, dtype=int)
    pos_to_idx: Dict[Tuple[int, int, int], int] = {
        tuple(p): i for i, p in enumerate(pos_arr)
    }
    perms = np.zeros((len(D4_OPS), n_slots), dtype=int)
    for op_idx, R in enumerate(D4_OPS):
        for i, p in enumerate(pos_arr):
            new_p = tuple((R @ p).astype(int))
            if new_p not in pos_to_idx:
                raise ValueError(
                    f"D4 op {op_idx} maps slot {i}={tuple(p)} to {new_p} "
                    f"which is not in the shell template — broken closure."
                )
            perms[op_idx, i] = pos_to_idx[new_p]
    return perms


def _build_inverse_permutations(perms: np.ndarray) -> np.ndarray:
    """Inverse of each row: inv_perms[op, perms[op, i]] = i."""
    n_ops, n_slots = perms.shape
    inv = np.zeros_like(perms)
    for op_idx in range(n_ops):
        inv[op_idx] = np.argsort(perms[op_idx])
    return inv


PERMS_SURFACE: np.ndarray = _build_d4_permutations(SHELL_REF_POSITIONS_SURFACE)
PERMS_BULK: np.ndarray = _build_d4_permutations(SHELL_REF_POSITIONS_BULK)
INV_PERMS_SURFACE: np.ndarray = _build_inverse_permutations(PERMS_SURFACE)
INV_PERMS_BULK: np.ndarray = _build_inverse_permutations(PERMS_BULK)


# ---------------------------------------------------------------------------
# Sanity checks at import — fail loudly if shell / D4 closure is broken.
# ---------------------------------------------------------------------------

def _validate_d4_group() -> None:
    """Verify D4_OPS form a closed group of 8 elements."""
    assert len(D4_OPS) == 8
    # Check determinants are ±1 (proper / improper rotation)
    for i, R in enumerate(D4_OPS):
        det = np.linalg.det(R)
        assert abs(abs(det) - 1.0) < 1e-9, f"op {i} det={det}"

    # Check closure: every product is in D4_OPS
    op_set = {tuple(R.flatten().tolist()) for R in D4_OPS}
    for i, R1 in enumerate(D4_OPS):
        for j, R2 in enumerate(D4_OPS):
            R12 = R1 @ R2
            key = tuple(R12.flatten().astype(int).tolist())
            assert key in op_set, f"op {i} ∘ op {j} not in D4 group"


def _validate_permutations(perms: np.ndarray) -> None:
    """Each row of perms must be a valid permutation of range(n)."""
    n_ops, n_slots = perms.shape
    for op_idx in range(n_ops):
        row = sorted(perms[op_idx].tolist())
        assert row == list(range(n_slots)), \
            f"perms[{op_idx}] is not a permutation: {perms[op_idx]}"


_validate_d4_group()
_validate_permutations(PERMS_SURFACE)
_validate_permutations(PERMS_BULK)


# ---------------------------------------------------------------------------
# Species encoding
# ---------------------------------------------------------------------------
SPECIES_CODE = {"Ni": 1, "Fe": 2, "Cr": 3}
SPECIES_VACANT = 0
SPECIES_OTHER = 4
SHELL_SLOT_OUT_OF_RANGE = -1   # sentinel for slab-edge / missing site


# ---------------------------------------------------------------------------
# Site → shell-slot mapping
# ---------------------------------------------------------------------------

def _wrap_xy(delta: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Minimum-image wrap of delta (3-vector) in xy."""
    out = delta.copy()
    out[0] -= round(out[0] / Lx) * Lx
    out[1] -= round(out[1] / Ly) * Ly
    return out


def extract_shell_indices(
    site_graph: Dict[int, Dict],
    central_site: int,
    ref_positions: Sequence[Tuple[int, int, int]],
    nn_dist: float,
    cell_xy: Tuple[float, float],
) -> np.ndarray:
    """Map ideal-lattice neighbour indices into shell-template slot order.

    Returns an array of length len(ref_positions). Slot i contains the
    ideal-lattice site index whose offset from central_site equals
    ref_positions[i] (in units of (nn1/2, nn1/2, a/2)). Slots whose
    expected position falls outside the slab (or is otherwise missing)
    are set to SHELL_SLOT_OUT_OF_RANGE.
    """
    n_slots = len(ref_positions)
    out = np.full(n_slots, SHELL_SLOT_OUT_OF_RANGE, dtype=int)

    if central_site not in site_graph:
        return out

    central_pos = site_graph[central_site]["position"]
    Lx, Ly = cell_xy
    unit_xy = nn_dist / 2.0
    unit_z = nn_dist / np.sqrt(2.0)

    # Gather candidate sites: 1NN + 2NN of central
    candidates = list(site_graph[central_site].get("1NN_3d", []))
    candidates += list(site_graph[central_site].get("2NN_3d", []))

    pos_to_slot: Dict[Tuple[int, int, int], int] = {
        tuple(p): i for i, p in enumerate(ref_positions)
    }

    for j in candidates:
        if j == central_site:
            continue
        delta = site_graph[j]["position"] - central_pos
        delta = _wrap_xy(delta, Lx, Ly)
        di = (
            int(round(delta[0] / unit_xy)),
            int(round(delta[1] / unit_xy)),
            int(round(delta[2] / unit_z)),
        )
        slot = pos_to_slot.get(di)
        if slot is not None and out[slot] == SHELL_SLOT_OUT_OF_RANGE:
            out[slot] = j

    return out


# ---------------------------------------------------------------------------
# Occupancy + species extraction
# ---------------------------------------------------------------------------

def extract_env_vectors_from_cluster(
    positions: np.ndarray,
    types: Sequence,
    mover_idx: int,
    ref_positions: Sequence[Tuple[int, int, int]],
    nn_dist: float,
    occ_tol: float = 0.6,
    cluster_margin: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract shell occupancy from a cluster's positions, mover-relative.

    Reference-table clusters are stored in pARTn's centered frame: the mover
    sits at some "canonical" position (typically near the cell centre), and
    the cluster contains all atoms within a few Å of the mover. We compute
    shell occupancy by checking, for each shell-template offset, whether
    there's a cluster atom at the expected mover-relative position.

    Each shell slot is one of:
      - 1 (occupied): atom found within `occ_tol` of the expected position
      - 0 (vacant): expected position is INSIDE the cluster's spatial reach
        but no atom is there (a real vacancy)
      - -1 (out-of-range): expected position is past the cluster's extent
        — the cluster doesn't reach this shell slot, so we can't tell

    `cluster_margin` adds a buffer to the cluster bounding box: a shell slot
    is considered "out-of-range" only if its expected position is beyond
    cluster_max + cluster_margin. This avoids false-vacancies from atoms
    that happen to sit just inside the cluster boundary.
    """
    n_slots = len(ref_positions)
    occ_vec = np.zeros(n_slots, dtype=np.int8)
    species_vec = np.zeros(n_slots, dtype=np.int8)
    atom_tree = cKDTree(positions)
    mover_pos = positions[mover_idx]
    unit_xy = nn_dist / 2.0
    unit_z = nn_dist / np.sqrt(2.0)

    # Cluster bounding box (with margin) — used to detect out-of-range slots
    pos_min = positions.min(axis=0) - cluster_margin
    pos_max = positions.max(axis=0) + cluster_margin

    for i, (rx, ry, rz) in enumerate(ref_positions):
        expected = mover_pos + np.array(
            [rx * unit_xy, ry * unit_xy, rz * unit_z]
        )
        # Check if expected position is within cluster reach
        if (expected < pos_min).any() or (expected > pos_max).any():
            occ_vec[i] = SHELL_SLOT_OUT_OF_RANGE
            species_vec[i] = SHELL_SLOT_OUT_OF_RANGE
            continue
        d, atom_idx = atom_tree.query(expected)
        if d > occ_tol or atom_idx == mover_idx:
            # Vacant (no atom near expected position) — leave as 0
            continue
        occ_vec[i] = 1
        sp_str = str(types[atom_idx]) if types is not None else ""
        species_vec[i] = SPECIES_CODE.get(sp_str, SPECIES_OTHER)

    return occ_vec, species_vec


# ---------------------------------------------------------------------------
# Hop vector + canonicalisation
# ---------------------------------------------------------------------------

def compute_hop_vector(
    src_pos: np.ndarray,
    dst_pos: np.ndarray,
    nn_dist: float,
    cell_xy: Tuple[float, float],
) -> Tuple[int, int, int]:
    """Return integer hop vector in (nn1/2, nn1/2, a/2) units."""
    Lx, Ly = cell_xy
    unit_xy = nn_dist / 2.0
    unit_z = nn_dist / np.sqrt(2.0)
    delta = dst_pos - src_pos
    delta = _wrap_xy(delta, Lx, Ly)
    return (
        int(round(delta[0] / unit_xy)),
        int(round(delta[1] / unit_xy)),
        int(round(delta[2] / unit_z)),
    )


def canonicalise(
    env_vec: np.ndarray,
    hop_vec: Tuple[int, int, int],
    perms: np.ndarray,
) -> Tuple[Tuple[int, ...], Tuple[int, int, int]]:
    """Apply each D4 op consistently to (env_vec, hop_vec) and return lex-min.

    Active rotation by R: env_rotated[R(p)] = env_original[p], i.e.,
    env_rotated = env_original[perms[R^-1]] = env_original[INV_PERMS[R]].
    Hop rotation: hop_rotated = R @ hop. Both transformations use the
    SAME R, so the orbit covers consistent (env, hop) pairs.

    Returns
    -------
    (canonical_env_tuple, canonical_hop_tuple)
    """
    candidates = []
    hop_arr = np.array(hop_vec, dtype=int)
    inv_perms = _build_inverse_permutations(perms)
    for op_idx in range(len(D4_OPS)):
        env_perm = tuple(int(x) for x in env_vec[inv_perms[op_idx]])
        hop_rot = tuple(int(x) for x in (D4_OPS[op_idx] @ hop_arr).tolist())
        candidates.append((env_perm, hop_rot))
    return min(candidates)


# ---------------------------------------------------------------------------
# Fingerprint string format
# ---------------------------------------------------------------------------

def fingerprint_string(
    env_vec_canonical: Sequence[int],
    hop_vec_canonical: Tuple[int, int, int],
    kind: str = "occ",
) -> str:
    """Compact human-readable fingerprint.

    Format: '{kind}:1NN+2NN/{env_str}|h:{hx},{hy},{hz}'
    where env_str is dash-separated runs of 4 digits.
    """
    env_str = "-".join(
        "".join(str(int(x)) if x >= 0 else "x" for x in env_vec_canonical[i:i + 4])
        for i in range(0, len(env_vec_canonical), 4)
    )
    hop_str = ",".join(str(int(h)) for h in hop_vec_canonical)
    return f"{kind}:1NN+2NN/{env_str}|h:{hop_str}"
