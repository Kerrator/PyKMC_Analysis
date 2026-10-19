"""Matplotlib drawing primitives for the schematic triptych.

Pure matplotlib (no Blender, no ASE). All atoms drawn as
:class:`matplotlib.patches.Circle` instances with true Å-scale radii in
data coordinates — *not* :meth:`Axes.scatter`, whose marker size is in
points² and would not scale correctly with axis zoom.

The public entry point is :func:`make_triptych_figure`, which composes
the three init/saddle/final panels, shared axis limits, and the arrow
glyphs in the gutters between subplots.
"""

from __future__ import annotations

import logging
import warnings
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, ConnectionPatch

from Analysis.schematic_viewer.constants import (
    ACTIVE_HEX,
    ARROW_HEX,
    ATOM_EDGE_COLOR,
    ATOM_EDGE_WIDTH_PT,
    COVALENT_RADIUS_A,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE_IN,
    METALLIC_RADIUS_A,
    MOVER_HEX,
    MOVER_RADIUS_BOOST,
    SCHEMATIC_RADIUS_SCALE,
    SUBSTRATE_HEX,
)
from Analysis.schematic_viewer.projection import project_positions, union_bbox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role → colour / radius lookups
# ---------------------------------------------------------------------------


def make_role_colors(
    n_atoms: int,
    mover_idx: int,
    active_idx: np.ndarray | Sequence[int],
) -> list[str]:
    """Assign a fill colour per atom by role.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms in the rendered set.
    mover_idx : int
        Canonical mover index — gets :data:`MOVER_HEX`.
    active_idx : array-like of int
        Indices to colour with :data:`ACTIVE_HEX`. The mover index is
        excluded if it happens to be in this set.

    Returns
    -------
    list[str]
        Hex colour string per atom, length ``n_atoms``.
    """
    colors = [SUBSTRATE_HEX] * int(n_atoms)
    for i in np.asarray(active_idx, dtype=np.int64).tolist():
        if 0 <= i < n_atoms and i != int(mover_idx):
            colors[i] = ACTIVE_HEX
    if 0 <= int(mover_idx) < n_atoms:
        colors[int(mover_idx)] = MOVER_HEX
    return colors


def make_role_radii(
    elements: Sequence[str],
    mover_idx: int,
    scale: float = SCHEMATIC_RADIUS_SCALE,
) -> np.ndarray:
    """Per-atom rendered disc radius (Å).

    Falls back to :data:`COVALENT_RADIUS_A` if an element is missing
    from :data:`METALLIC_RADIUS_A`, and to a flat 1.30 Å if both miss.
    The mover gets an extra :data:`MOVER_RADIUS_BOOST` factor so the
    eye lands on it first.

    Parameters
    ----------
    elements : sequence of str
        Element symbols, one per atom.
    mover_idx : int
    scale : float

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
    """
    radii = np.empty(len(elements), dtype=np.float64)
    for i, elt in enumerate(elements):
        r = METALLIC_RADIUS_A.get(elt) or COVALENT_RADIUS_A.get(elt, 1.30)
        radii[i] = float(r) * float(scale)
    if 0 <= int(mover_idx) < radii.size:
        radii[int(mover_idx)] *= MOVER_RADIUS_BOOST
    return radii


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def draw_atoms(
    ax: Axes,
    x2d: np.ndarray,
    y2d: np.ndarray,
    depth: np.ndarray,
    radii: np.ndarray,
    fill_colors: Sequence[str],
    edge_color: str = ATOM_EDGE_COLOR,
    edge_width: float = ATOM_EDGE_WIDTH_PT,
) -> None:
    """Render atoms as back-to-front Circle patches.

    Atoms are sorted by ``depth`` ascending (farthest first) so the front
    ones over-paint the back ones — a cheap painter's-algorithm depth cue.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes. Should have ``set_aspect('equal')`` and identical
        xlim/ylim across all triptych panels.
    x2d, y2d, depth : np.ndarray, shape (N,)
        Projected coordinates and depth from
        :func:`Analysis.schematic_viewer.projection.project_positions`.
    radii : np.ndarray, shape (N,)
        Disc radius per atom (data coords, Å).
    fill_colors : sequence of str
        Hex colour per atom.
    edge_color : str
    edge_width : float
        matplotlib points; matches APE's strong silhouette cue.
    """
    order = np.argsort(depth)
    for i in order:
        circle = Circle(
            (float(x2d[i]), float(y2d[i])),
            radius=float(radii[i]),
            facecolor=fill_colors[i],
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=2.0 + 1e-3 * float(depth[i]),
        )
        ax.add_patch(circle)


def draw_arrow_between_axes(
    fig: Figure,
    from_ax: Axes,
    to_ax: Axes,
    color: str = ARROW_HEX,
    label: str | None = None,
) -> None:
    """Draw a flow arrow in the gutter between two subplot panels.

    Uses :class:`matplotlib.patches.ConnectionPatch` anchored at the right
    edge of ``from_ax`` (in axes coordinates) and the left edge of
    ``to_ax`` so the arrow sits cleanly in the gutter regardless of figure
    size.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    from_ax, to_ax : matplotlib.axes.Axes
        Source and destination subplot axes (left and right of the gutter).
    color : str
    label : str or None
        If given, place text just above the arrow midpoint in figure
        coordinates.
    """
    arrow = ConnectionPatch(
        xyA=(1.02, 0.5),
        coordsA=from_ax.transAxes,
        xyB=(-0.02, 0.5),
        coordsB=to_ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color=color,
        zorder=5,
    )
    fig.add_artist(arrow)
    if label:
        # Midpoint of the two anchor points, expressed in figure coordinates.
        a_disp = from_ax.transAxes.transform((1.02, 0.5))
        b_disp = to_ax.transAxes.transform((-0.02, 0.5))
        mid_disp = ((a_disp[0] + b_disp[0]) / 2.0,
                    (a_disp[1] + b_disp[1]) / 2.0 + 12.0)
        mid_fig = fig.transFigure.inverted().transform(mid_disp)
        fig.text(
            mid_fig[0], mid_fig[1], label,
            ha="center", va="bottom",
            color=color, fontsize=9,
        )


def draw_cell_outline(
    ax: Axes,
    cell: np.ndarray | None,
    view: str | np.ndarray | Sequence[float],
) -> None:
    """Project a unit cell parallelepiped onto the viewing plane.

    Not currently exercised — the npz files written by
    :mod:`Analysis.blender_viewer.prepare` do not carry cell information.
    Stub kept here so a future caller that has cell data can plug straight
    in. Raises :class:`FutureWarning` if invoked with ``cell is None``.
    """
    if cell is None:
        warnings.warn(
            "draw_cell_outline: cell=None (the blender_viewer npz pipeline "
            "does not currently carry cell data); skipping cell box.",
            FutureWarning,
            stacklevel=2,
        )
        return
    cell = np.asarray(cell, dtype=np.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3); got {cell.shape}")
    # 8 corners of the parallelepiped.
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    ) @ cell
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    x2d, y2d, _ = project_positions(corners, view=view)
    for a, b in edges:
        ax.plot(
            [x2d[a], x2d[b]], [y2d[a], y2d[b]],
            color="#888888", linewidth=0.5, linestyle="--", zorder=1,
        )


# ---------------------------------------------------------------------------
# Triptych composer
# ---------------------------------------------------------------------------


def make_triptych_figure(
    positions_triple: tuple[np.ndarray, np.ndarray, np.ndarray],
    elements: Sequence[str],
    mover_idx: int,
    active_idx: np.ndarray | Sequence[int],
    view: str | np.ndarray | Sequence[float] = "z",
    barrier_ev: float | None = None,
    titles: tuple[str, str, str] = ("init", "saddle", "final"),
    show_cell: bool = False,
    cell: np.ndarray | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE_IN,
    dpi: int = DEFAULT_DPI,
) -> Figure:
    """Build a 1×3 triptych figure for one event.

    Parameters
    ----------
    positions_triple : tuple of three (N, 3) np.ndarrays
        ``(init, saddle, final)`` atomic positions, all sharing the same
        atom ordering.
    elements : sequence of str, length N
        Element symbols, one per atom.
    mover_idx : int
        Index of the primary mover (orange).
    active_idx : array-like of int
        Indices of "active" undercoordinated neighbours (yellow). Pass
        an empty array to suppress yellow atoms entirely.
    view : str or (3,) array-like
        Camera direction. Default ``'z'`` (drop the z coordinate; look
        straight down).
    barrier_ev : float or None
        If given, placed as the saddle panel's title (``"Ea = X.XXX eV"``).
    titles : tuple of three str
        Panel titles. ``barrier_ev`` overrides the middle one if set.
    show_cell : bool
        Draw the unit cell outline. Currently a no-op when ``cell`` is
        not provided (the npz pipeline does not yet carry cell info).
    cell : (3, 3) np.ndarray or None
    figsize : (float, float)
        Inches.
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    init_pos, saddle_pos, final_pos = positions_triple
    for label, arr in (("init", init_pos), ("saddle", saddle_pos), ("final", final_pos)):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{label}: positions must be (N, 3); got {arr.shape}")
    if len(elements) != init_pos.shape[0]:
        raise ValueError(
            f"len(elements) ({len(elements)}) != N atoms ({init_pos.shape[0]})"
        )

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=figsize, dpi=dpi,
        gridspec_kw={"wspace": 0.12, "left": 0.01, "right": 0.99,
                     "top": 0.92, "bottom": 0.02},
    )

    # Shared bounding box.
    xlim, ylim = union_bbox(init_pos, saddle_pos, final_pos, view=view)

    radii = make_role_radii(elements, mover_idx=mover_idx)
    colors = make_role_colors(
        n_atoms=init_pos.shape[0], mover_idx=mover_idx, active_idx=active_idx,
    )

    panel_titles = list(titles)
    if barrier_ev is not None:
        panel_titles[1] = f"saddle  (Ea = {float(barrier_ev):.3f} eV)"

    for ax, pos, title in zip(
        axes,
        (init_pos, saddle_pos, final_pos),
        panel_titles,
        strict=True,
    ):
        x2d, y2d, depth = project_positions(pos, view=view)
        if show_cell:
            draw_cell_outline(ax, cell, view)
        draw_atoms(ax, x2d, y2d, depth, radii, colors)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    draw_arrow_between_axes(fig, axes[0], axes[1])
    draw_arrow_between_axes(fig, axes[1], axes[2])

    return fig


__all__ = [
    "make_role_colors",
    "make_role_radii",
    "draw_atoms",
    "draw_arrow_between_axes",
    "draw_cell_outline",
    "make_triptych_figure",
]
