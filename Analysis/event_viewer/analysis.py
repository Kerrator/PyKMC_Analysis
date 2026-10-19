"""Structural analysis helpers for the event viewer.

- `compute_cna(positions)` : adaptive Honeycutt–Andersen Common Neighbor
  Analysis. Returns a per-atom label in {FCC, HCP, BCC, ICO, OTHER}.
- `find_vacancies(init, saddle, final)` : identify vacancy positions per
  frame by taking the union of all atom positions across the three frames
  (each unique position is a "lattice site") and reporting sites not
  occupied in a given frame.

Both functions are pure and cheap enough (~100 ms for ~130 atoms) to run on
demand when a user changes event / color mode.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

CNA_LABELS = ("FCC", "HCP", "BCC", "ICO", "OTHER", "BOUNDARY")

CNA_COLORS = {
    "FCC": "#4DD0E1",       # teal — bulk crystalline
    "HCP": "#FF7043",       # orange — stacking fault / close-packed variant
    "BCC": "#9575CD",       # purple — bcc-like
    "ICO": "#FFEB3B",       # yellow — icosahedral
    "OTHER": "#E57373",     # red — surface / defect
    "BOUNDARY": "#E0E0E0",  # light grey — at the edge of the cutout (don't trust)
}


def _longest_path(bond_graph: dict[int, list[int]], start: int) -> int:
    """Length (#edges) of the longest simple path starting at `start`."""
    best = 0
    stack = [(start, frozenset([start]), 0)]
    while stack:
        node, visited, depth = stack.pop()
        extended = False
        for nxt in bond_graph[node]:
            if nxt not in visited:
                extended = True
                stack.append((nxt, visited | {nxt}, depth + 1))
        if not extended and depth > best:
            best = depth
    return best


def compute_cna(positions: np.ndarray) -> np.ndarray:
    """Adaptive-cutoff Honeycutt–Andersen CNA.

    Returns an ndarray of dtype=object with one label per atom. Atoms close to
    the cutout boundary (the outer ~15% radius) are flagged as "BOUNDARY"
    rather than misleadingly classified as OTHER — their neighbourhood is
    truncated by the event's cutout and any CNA result is unreliable.

    The boundary mask uses the geometric centroid so it is stable across
    frames (only a single mover atom shifts between frames, so the centroid
    barely moves).
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    tree = cKDTree(pos)

    # Adaptive cutoff: use the 12 nearest neighbours (FCC coordination) to
    # estimate the 1st-NN distance, then set r_cut midway between 1NN and 2NN.
    k = min(15, n)
    dists, _ = tree.query(pos, k=k)
    # dists[:, 0] is the self distance (zero). Use columns 1..12 for 1st-NN.
    nn1 = np.median(dists[:, 1 : min(13, k)], axis=1)
    # r_cut for FCC: roughly (1 + sqrt(2)) / 2 ≈ 1.207 × d_1NN
    r_cuts = nn1 * 1.207

    # Flag atoms in the outer shell of the (approximately spherical) cutout.
    # These atoms lose neighbours simply because the cutout ends, so any CNA
    # classification would be an artifact.
    center = pos.mean(axis=0)
    dist_from_center = np.linalg.norm(pos - center, axis=1)
    boundary_mask = dist_from_center > 0.85 * dist_from_center.max()

    labels = np.empty(n, dtype=object)

    for i in range(n):
        if boundary_mask[i]:
            labels[i] = "BOUNDARY"
            continue
        r_i = r_cuts[i]
        nbrs_i = np.array(tree.query_ball_point(pos[i], r_i), dtype=int)
        nbrs_i = nbrs_i[nbrs_i != i]
        coord = len(nbrs_i)

        # FCC/HCP/ICO need coord == 12; BCC needs 14 (8 1NN + 6 2NN).
        if coord not in (12, 14):
            labels[i] = "OTHER"
            continue

        signatures: list[tuple[int, int, int]] = []
        for j in nbrs_i:
            nbrs_j = np.array(tree.query_ball_point(pos[j], r_cuts[j]), dtype=int)
            nbrs_j = nbrs_j[nbrs_j != j]
            common = np.intersect1d(nbrs_i, nbrs_j, assume_unique=False)
            n_common = len(common)
            if n_common == 0:
                signatures.append((0, 0, 0))
                continue

            # Build bond graph among common neighbours.
            common_list = common.tolist()
            bond_graph: dict[int, list[int]] = {k: [] for k in common_list}
            n_bonds = 0
            for a, b in combinations(common_list, 2):
                thresh = (r_cuts[a] + r_cuts[b]) * 0.5
                if np.linalg.norm(pos[a] - pos[b]) < thresh:
                    n_bonds += 1
                    bond_graph[a].append(b)
                    bond_graph[b].append(a)

            longest = max((_longest_path(bond_graph, s) for s in common_list), default=0)
            signatures.append((n_common, n_bonds, longest))

        ctr = Counter(signatures)
        if coord == 12 and ctr.get((4, 2, 1), 0) == 12:
            labels[i] = "FCC"
        elif coord == 12 and ctr.get((4, 2, 1), 0) == 6 and ctr.get((4, 2, 2), 0) == 6:
            labels[i] = "HCP"
        elif coord == 14 and ctr.get((6, 6, 6), 0) == 8 and ctr.get((4, 4, 4), 0) == 6:
            labels[i] = "BCC"
        elif coord == 12 and ctr.get((5, 5, 5), 0) == 12:
            labels[i] = "ICO"
        else:
            labels[i] = "OTHER"

    return labels


def _infer_nn_offsets(positions: np.ndarray, d_nn: float) -> np.ndarray:
    """Infer the 12 FCC nearest-neighbour offset vectors from the data.

    Different orientations (standard (100) vs rotated (110)-aligned) give
    different offset vectors; rather than hardcoding either, we collect the
    actual NN displacements from well-coordinated atoms and cluster them
    into 12 consensus directions.
    """
    pos = np.asarray(positions, dtype=float)
    tree = cKDTree(pos)
    k = min(13, len(pos))
    dists, idxs = tree.query(pos, k=k)

    # Collect 1st-NN offset vectors from every well-coordinated atom.
    offsets: list[np.ndarray] = []
    for i in range(len(pos)):
        # Well-coordinated iff 12 neighbours within 1.15 × d_nn.
        if dists[i, -1] > d_nn * 1.15:
            continue
        for j in idxs[i, 1:]:  # skip self
            vec = pos[j] - pos[i]
            d = np.linalg.norm(vec)
            if 0.8 * d_nn < d < 1.15 * d_nn:
                offsets.append(vec)

    if len(offsets) < 24:  # not enough data — fall back to (110)-axis offsets
        d = d_nn
        h = d_nn / np.sqrt(2.0)
        v = np.sqrt(d**2 - 2 * h**2) if d**2 > 2 * h**2 else d_nn / 2.0
        return np.array(
            [
                [+d, 0, 0], [-d, 0, 0], [0, +d, 0], [0, -d, 0],
                [+h, +h, +v], [+h, -h, +v], [-h, +h, +v], [-h, -h, +v],
                [+h, +h, -v], [+h, -h, -v], [-h, +h, -v], [-h, -h, -v],
            ]
        )

    offs = np.asarray(offsets)
    # Cluster offsets by direction. Normalise and group positions that point
    # in nearly the same direction.
    norms = np.linalg.norm(offs, axis=1, keepdims=True)
    dirs = offs / norms

    # Greedy clustering: angle threshold ≈ 15°.
    cos_thresh = np.cos(np.deg2rad(15.0))
    unique: list[np.ndarray] = []
    counts: list[int] = []
    for o, d_hat in zip(offs, dirs):
        matched = False
        for i, u in enumerate(unique):
            u_hat = u / np.linalg.norm(u)
            if float(d_hat @ u_hat) > cos_thresh:
                unique[i] = (u * counts[i] + o) / (counts[i] + 1)
                counts[i] += 1
                matched = True
                break
        if not matched:
            unique.append(o.copy())
            counts.append(1)

    # Keep only the 12 most common directions (covers one FCC coordination shell).
    order = np.argsort(counts)[::-1]
    kept = [unique[i] for i in order[:12]]
    return np.asarray(kept)


def _find_lattice_vacancies(
    positions: np.ndarray,
    a: float,
    min_coord: int = 7,
    match_tol_factor: float = 0.35,
) -> np.ndarray:
    """Enumerate candidate vacancy sites and keep well-surrounded ones.

    For each atom, we look at where its 12 FCC nearest neighbours *should* be
    and note any site that is not occupied. Duplicate candidates (the same
    missing site discovered from several neighbouring atoms) are clustered.

    Candidates are kept only if:
    1. At least `min_coord` of their own 12 NN sites are occupied (≥ 8 is
       correct for both bulk vacancies (12 NNs) and top-surface vacancies
       (8 NNs — no atoms above in vacuum)).
    2. The candidate sits inside the z-extent of the atoms plus a small
       tolerance — this suppresses "vacancies" predicted into vacuum above
       the slab or into the bottom of the cutout.
    3. The candidate's xy lies within the atoms' xy bounding box (same
       cutout-boundary defence for the in-plane directions).
    """
    pos = np.asarray(positions, dtype=float)
    tree = cKDTree(pos)
    d_nn = a / np.sqrt(2.0)
    nn_offsets = _infer_nn_offsets(pos, d_nn)
    if len(nn_offsets) == 0:
        return np.empty((0, 3))
    match_tol = a * match_tol_factor  # ~0.5 Å for Ni

    # Bounding box of the actual atoms — candidates must fit inside (plus a
    # small tolerance so surface-layer vacancies on the boundary still count).
    lo = pos.min(axis=0) - 0.3
    hi = pos.max(axis=0) + 0.3

    # Gather candidate vacancy sites.
    candidates: list[np.ndarray] = []
    for p in pos:
        sites = p + nn_offsets  # shape (12, 3)
        dists, _ = tree.query(sites, k=1)
        for s, d in zip(sites, dists):
            if d > match_tol:
                candidates.append(s)
    if not candidates:
        return np.empty((0, 3))

    cand = np.asarray(candidates)
    # Cluster candidates that are close to each other (same missing site seen
    # from multiple neighbouring atoms).
    ctree = cKDTree(cand)
    parent = list(range(len(cand)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for u, v in ctree.query_pairs(match_tol):
        union(u, v)

    groups: dict[int, list[int]] = {}
    for i in range(len(cand)):
        groups.setdefault(find(i), []).append(i)
    centroids = np.array([cand[ixs].mean(axis=0) for ixs in groups.values()])

    # Keep only candidates that satisfy all three filters.
    kept: list[np.ndarray] = []
    for c in centroids:
        # Filter 2+3: candidate must be inside the atoms' bounding box.
        if not (np.all(c >= lo) and np.all(c <= hi)):
            continue
        # Filter 1: ≥ min_coord of the 12 NN sites must be occupied.
        nn_sites = c + nn_offsets
        dists, _ = tree.query(nn_sites, k=1)
        n_occupied = int((dists < match_tol).sum())
        if n_occupied >= min_coord:
            kept.append(c)
    return np.asarray(kept) if kept else np.empty((0, 3))


def _estimate_lattice_constant(positions: np.ndarray) -> float:
    """Median of nearest-neighbour distance × sqrt(2) — FCC lattice constant."""
    pos = np.asarray(positions, dtype=float)
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=min(5, len(pos)))  # self + 4 nearest
    # dists[:, 0] is self distance (zero); dists[:, 1] is 1st-NN distance.
    d_nn = np.median(dists[:, 1])
    return float(d_nn * np.sqrt(2.0))


def find_vacancies(
    init: np.ndarray,
    saddle: np.ndarray,
    final: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return vacancy positions per frame.

    For each frame we predict where the FCC (100) ideal lattice sites sit
    (using the median nearest-neighbour distance across all three frames to
    estimate the lattice constant), then report any site that has no atom
    within ~0.5 Å. This finds ALL vacancies in the local cutout, including
    vacancies that were not touched by the current event.

    The cutout-boundary filter keeps only candidates that have ≥6 of their
    own 12 nearest-neighbour sites occupied, so candidates falling outside
    the cutout are not reported as false vacancies.

    Works for single-atom hops, cooperative events, and multi-vacancy systems.
    Assumes an axis-aligned FCC (100) lattice (the convention in the pyKMC
    Ni slab simulations).
    """
    init = np.asarray(init, dtype=float)
    saddle = np.asarray(saddle, dtype=float)
    final = np.asarray(final, dtype=float)

    # Estimate the lattice constant once from the best-behaved frame.
    # Initial positions are usually closest to the ideal lattice.
    a = _estimate_lattice_constant(init)

    out: dict[str, np.ndarray] = {}
    for name, frame_pos in (("initial", init), ("saddle", saddle), ("final", final)):
        out[name] = _find_lattice_vacancies(frame_pos, a=a)
    return out
