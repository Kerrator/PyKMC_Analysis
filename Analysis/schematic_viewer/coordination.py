"""Role inference: identify "active neighbour" atoms around a defect.

Mirrors the count-based logic in
``pyKMC-develop/pykmc/basins/basin.py:735-739``
(``cKDTree.query_ball_point`` + per-atom neighbour count, minus 1 for the
self-match) without dragging in any LAMMPS / MPI / Config machinery.

For the schematic renderer's APE-style yellow atoms we want the *tight*
ring around the defect, not every undercoordinated atom in the slab —
:func:`find_active_neighbours` intersects the global under-coordinated
set with the mover's own 1NN sphere to get that ring.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from Analysis.schematic_viewer.constants import DEFAULT_COORD_THR, DEFAULT_RNEI_A

#: Displacement threshold (Å) for the motion-based co-mover detector. Same
#: value as :data:`Analysis.blender_viewer.constants.COOP_DISPLACEMENT_THR_A`
#: (well above thermal noise, well below a real 1NN hop).
DEFAULT_DISP_THR_A: float = 0.5


def coordination_counts(
    positions: np.ndarray,
    rnei: float = DEFAULT_RNEI_A,
) -> np.ndarray:
    """Per-atom 1st-shell neighbour counts.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic coordinates (Å).
    rnei : float
        First-shell cutoff radius (Å).

    Returns
    -------
    np.ndarray, shape (N,), dtype int32
        Neighbour count for each atom, excluding self.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")
    tree = cKDTree(positions)
    neighbour_lists = tree.query_ball_point(positions, r=float(rnei))
    return np.array([len(ni) - 1 for ni in neighbour_lists], dtype=np.int32)


def find_active_neighbours(
    positions: np.ndarray,
    mover_idx: int,
    rnei: float = DEFAULT_RNEI_A,
    coord_thr: int = DEFAULT_COORD_THR,
) -> np.ndarray:
    """Indices of "active" undercoordinated atoms near the mover.

    Two-step filter, designed to reproduce the APE figure's yellow-atom
    selection rather than the bare basin.py "all undercoordinated" rule:

    1. Atoms with neighbour count strictly less than ``coord_thr``
       (basin.py rule — under-coordinated atoms).
    2. AND within ``rnei`` Å of the mover (the "tight ring around the
       defect" interpretation).

    The mover atom itself is always excluded from the returned set.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic coordinates (Å). Typically the *initial-state* positions
        for the event being rendered, but any single state works — the
        active set is computed once and held constant across the
        triptych for visual continuity.
    mover_idx : int
        Index of the canonical hopping atom in ``positions``.
    rnei : float
        First-shell cutoff (Å). Same value used for both coordination
        counting and the mover-neighbour filter.
    coord_thr : int
        Undercoordination threshold (default 9 — matches basin.py's FCC
        slab convention).

    Returns
    -------
    np.ndarray, shape (M,), dtype int64
        Indices of active-neighbour atoms, sorted ascending. Returns an
        empty array if no atoms qualify.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_atoms = positions.shape[0]
    if not 0 <= int(mover_idx) < n_atoms:
        raise IndexError(
            f"mover_idx={mover_idx} out of range for {n_atoms} atoms"
        )

    counts = coordination_counts(positions, rnei=rnei)
    undercoordinated = counts < int(coord_thr)

    # Atoms within rnei of the mover (includes the mover itself).
    tree = cKDTree(positions)
    near_mover_indices = np.array(
        tree.query_ball_point(positions[mover_idx], r=float(rnei)),
        dtype=np.int64,
    )
    near_mover_mask = np.zeros(n_atoms, dtype=bool)
    near_mover_mask[near_mover_indices] = True
    near_mover_mask[int(mover_idx)] = False  # exclude the mover

    active_mask = undercoordinated & near_mover_mask
    return np.flatnonzero(active_mask).astype(np.int64)


def find_co_movers(
    init_positions: np.ndarray,
    final_positions: np.ndarray,
    mover_idx: int,
    disp_thr: float = DEFAULT_DISP_THR_A,
) -> np.ndarray:
    """Atoms (other than the mover) that move significantly during the hop.

    Pure motion-based detector — independent of coordination, so works on
    any subset of atoms. Mirrors the
    :data:`Analysis.blender_viewer.constants.COOP_DISPLACEMENT_THR_A`
    rule used by ``cooperative_highlight`` in the photorealistic
    renderer.

    Parameters
    ----------
    init_positions, final_positions : np.ndarray, shape (N, 3)
        Initial and final atomic coordinates. Same atom ordering.
    mover_idx : int
        Canonical mover index — always excluded from the returned set.
    disp_thr : float
        Net |final - init| threshold (Å) above which an atom counts as
        an active co-mover.

    Returns
    -------
    np.ndarray, shape (M,), dtype int64
        Indices of co-mover atoms, sorted ascending. Empty array if
        none qualify.
    """
    init = np.asarray(init_positions, dtype=np.float64)
    final = np.asarray(final_positions, dtype=np.float64)
    if init.shape != final.shape:
        raise ValueError(
            f"init and final must share shape; got {init.shape} vs {final.shape}"
        )
    disp = np.linalg.norm(final - init, axis=1)
    moving = disp > float(disp_thr)
    moving[int(mover_idx)] = False  # always exclude the canonical mover
    return np.flatnonzero(moving).astype(np.int64)


def find_active_set(
    init_positions: np.ndarray,
    final_positions: np.ndarray,
    mover_idx: int,
    rnei: float = DEFAULT_RNEI_A,
    coord_thr: int = DEFAULT_COORD_THR,
    disp_thr: float = DEFAULT_DISP_THR_A,
) -> np.ndarray:
    """Union of the coordination-based and motion-based active sets.

    Both rules are useful for different inputs:

    * **Coordination** (:func:`find_active_neighbours`) — relies on a
      reliable per-atom 1NN count. Works on full-system positions; can
      fail to flag anything when given a small neighbour-subset (e.g.
      the 127-atom slices in ``blender_viewer/prepare.py`` npz files,
      where every atom looks under-coordinated relative to its trimmed
      neighbourhood).
    * **Motion** (:func:`find_co_movers`) — flags any atom whose net
      ``|final - init|`` exceeds ``disp_thr``. Robust to subset
      effects; catches the 2 partner atoms in a ``1NN_hop_triple``
      cooperative hop.

    Taking the union gives the right behaviour in both regimes — the
    immediate undercoordinated ring on full-system inputs *and* the
    explicit co-movers in mechanism-rich events.

    Returns
    -------
    np.ndarray, shape (M,), dtype int64
        Sorted, deduplicated indices of active atoms.
    """
    coord_set = find_active_neighbours(
        init_positions, mover_idx=mover_idx, rnei=rnei, coord_thr=coord_thr,
    )
    motion_set = find_co_movers(
        init_positions, final_positions,
        mover_idx=mover_idx, disp_thr=disp_thr,
    )
    return np.array(sorted(set(coord_set.tolist()) | set(motion_set.tolist())),
                    dtype=np.int64)


__all__ = [
    "coordination_counts",
    "find_active_neighbours",
    "find_co_movers",
    "find_active_set",
    "DEFAULT_DISP_THR_A",
]
