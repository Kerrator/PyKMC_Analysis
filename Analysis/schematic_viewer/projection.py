"""3D → 2D orthographic projection for the schematic triptych.

Pure numpy. Two responsibilities:

1. :func:`project_positions` — drop one axis (or project onto an arbitrary
   view vector) and return (x2d, y2d, depth) for atom-drawing.
2. :func:`union_bbox` — compute a shared axis range across multiple frames
   so init / saddle / final panels lock to identical xlim/ylim, which is
   essential for the eye to track mover motion between panels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from Analysis.schematic_viewer.constants import PANEL_MARGIN_A

# Axis-aligned views map "the dropped axis" → (which axes survive as x2d/y2d,
# index of the depth axis). Kept explicit so the layout is unambiguous.
_AXIS_ALIASES: dict[str, tuple[int, int, int]] = {
    # view name → (x2d source index, y2d source index, depth axis index)
    "x": (1, 2, 0),
    "y": (0, 2, 1),
    "z": (0, 1, 2),
}


def _view_to_basis(view: str | np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the orthonormal (u, v, w) basis for a given view vector.

    Parameters
    ----------
    view : {'x', 'y', 'z'} or array-like of shape (3,)
        Camera +z direction. Axis-aligned strings use a canonical basis;
        arbitrary 3-vectors are normalised and an in-plane (u, v) basis is
        constructed via Gram-Schmidt against the world +z (or world +x if
        the view itself is aligned with world +z).

    Returns
    -------
    u, v, w : np.ndarray, shape (3,)
        Right-handed orthonormal basis. u, v span the projection plane;
        w is the depth axis (camera +z).
    """
    if isinstance(view, str):
        key = view.lower()
        if key not in _AXIS_ALIASES:
            raise ValueError(
                f"view={view!r}: must be one of {sorted(_AXIS_ALIASES)} "
                "or a (3,) array."
            )
        xi, yi, wi = _AXIS_ALIASES[key]
        u = np.zeros(3, dtype=np.float64)
        u[xi] = 1.0
        v = np.zeros(3, dtype=np.float64)
        v[yi] = 1.0
        w = np.zeros(3, dtype=np.float64)
        w[wi] = 1.0
        return u, v, w

    w = np.asarray(view, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        raise ValueError("view vector has near-zero magnitude")
    w = w / norm

    # Pick a "world up" reference that's least aligned with w. World +z by
    # default; if w is itself near world +z, fall back to world +x so the
    # cross product doesn't collapse.
    world_z = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(w, world_z))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = world_z

    u = np.cross(ref, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    # Already unit because w and u are unit and orthogonal.
    return u, v, w


def project_positions(
    positions: np.ndarray,
    view: str | np.ndarray | Sequence[float] = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N, 3) positions onto a 2D viewing plane.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic coordinates in Å.
    view : {'x', 'y', 'z'} or array-like of shape (3,)
        Camera +z direction. Axis-aligned strings are O(N) coordinate
        drops; arbitrary 3-vectors trigger a Gram-Schmidt basis build
        and an O(N) matmul.

    Returns
    -------
    x2d, y2d : np.ndarray, shape (N,)
        Projected coordinates in the viewing plane (Å).
    depth : np.ndarray, shape (N,)
        Distance along the view axis. Larger values mean closer to the
        camera. Used by :func:`Analysis.schematic_viewer.render.draw_atoms`
        to z-sort discs before drawing.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")

    if isinstance(view, str) and view.lower() in _AXIS_ALIASES:
        xi, yi, wi = _AXIS_ALIASES[view.lower()]
        return positions[:, xi].copy(), positions[:, yi].copy(), positions[:, wi].copy()

    u, v, w = _view_to_basis(view)
    x2d = positions @ u
    y2d = positions @ v
    depth = positions @ w
    return x2d, y2d, depth


def union_bbox(
    *position_arrays: np.ndarray,
    view: str | np.ndarray | Sequence[float] = "z",
    margin_a: float = PANEL_MARGIN_A,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared (xlim, ylim) across multiple position arrays.

    Used to lock all panels of the triptych to identical axis ranges so
    that motion between init / saddle / final reads cleanly in the eye.

    Parameters
    ----------
    *position_arrays : np.ndarray, each (N, 3)
        Position snapshots from successive states of the same event.
        N may differ between arrays (not expected here, but allowed).
    view : {'x', 'y', 'z'} or array-like of shape (3,)
        Same view used for the actual rendering.
    margin_a : float
        Padding around the tight bounding box in Å.

    Returns
    -------
    xlim, ylim : tuple of (float, float)
        Limits to pass to ``ax.set_xlim`` / ``ax.set_ylim`` for every
        panel.
    """
    if not position_arrays:
        raise ValueError("union_bbox requires at least one positions array")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for pos in position_arrays:
        x2d, y2d, _ = project_positions(pos, view=view)
        xs.append(x2d)
        ys.append(y2d)
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    xlim = (float(all_x.min() - margin_a), float(all_x.max() + margin_a))
    ylim = (float(all_y.min() - margin_a), float(all_y.max() + margin_a))
    return xlim, ylim


__all__ = ["project_positions", "union_bbox"]
