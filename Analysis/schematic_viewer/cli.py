"""CLI entry point for the schematic triptych renderer.

Run from the repository root::

    python -m Analysis.schematic_viewer.cli \\
        --npz /path/to/event_<...>.npz \\
        --view z \\
        --out-dir schematics

Outputs a PNG (300 dpi) and a PDF (vector) side-by-side, in ``--out-dir``
(default: the current directory).

This is the *only* module in :mod:`Analysis.schematic_viewer` that emits
console output. Library callers should use
:func:`Analysis.schematic_viewer.render.make_triptych_figure` directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

# Force a non-interactive backend BEFORE importing pyplot so headless runs
# (macOS without a display, cluster nodes) don't try to open a window.
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from Analysis.blender_viewer.data import (  # noqa: E402
    EventPair,
    EventSide,
    load_event_pair_npz,
)
from Analysis.schematic_viewer.constants import (  # noqa: E402
    DEFAULT_COORD_THR,
    DEFAULT_RNEI_A,
)
from Analysis.schematic_viewer.coordination import (  # noqa: E402
    DEFAULT_DISP_THR_A,
    find_active_set,
)
from Analysis.schematic_viewer.projection import project_positions  # noqa: E402
from Analysis.schematic_viewer.render import make_triptych_figure  # noqa: E402

logger = logging.getLogger("schematic_viewer")

Side = Literal["forward", "backward"]


def _pick_side(pair: EventPair, side: Side) -> EventSide:
    if side == "forward":
        return pair.forward
    if side == "backward":
        return pair.backward
    raise ValueError(f"side={side!r}: expected 'forward' or 'backward'")


def _verify(
    fig,
    side: EventSide,
    view: str | np.ndarray,
    out_png: Path,
) -> None:
    """Programmatic post-render checks. Raises AssertionError on failure."""
    axes = fig.axes
    if len(axes) < 3:
        raise AssertionError(f"expected 3 panels, got {len(axes)}")
    a, b, c = axes[0], axes[1], axes[2]

    # 1. Shared axis limits.
    for name, getter in (("xlim", "get_xlim"), ("ylim", "get_ylim")):
        la, lb, lc = (getattr(ax, getter)() for ax in (a, b, c))
        if not (np.allclose(la, lb) and np.allclose(lb, lc)):
            raise AssertionError(
                f"{name} mismatch across panels: {la} vs {lb} vs {lc}"
            )

    # 2. Mover projection round-trip — for each state, the projected
    #    coordinates of the mover atom must match.
    for label, positions in (
        ("init", side.init_positions),
        ("saddle", side.saddle_positions),
        ("final", side.final_positions),
    ):
        x2d, y2d, _ = project_positions(positions, view=view)
        i = int(side.mover_idx)
        x_actual, y_actual = float(x2d[i]), float(y2d[i])
        # We don't have a direct "projected mover position" lookup here
        # but we can re-project the single atom and confirm consistency.
        x_check, y_check, _ = project_positions(positions[i : i + 1], view=view)
        if not (np.isclose(x_actual, x_check[0], atol=0.01)
                and np.isclose(y_actual, y_check[0], atol=0.01)):
            raise AssertionError(
                f"{label}: mover projection mismatch "
                f"({x_actual}, {y_actual}) vs ({x_check[0]}, {y_check[0]})"
            )

    # 3. Output sanity check.
    size = out_png.stat().st_size
    if size <= 5_000:
        raise AssertionError(
            f"output PNG suspiciously small: {size} bytes — backend may have "
            "produced a blank canvas"
        )


def main(
    npz_path: Path | str,
    view: str | np.ndarray = "z",
    side: Side = "forward",
    show_cell: bool = False,
    barrier_ev: float | None = None,
    save: bool = True,
    out_dir: Path | str | None = None,
    active_idx: list[int] | None = None,
    coord_thr: int = DEFAULT_COORD_THR,
    rnei: float = DEFAULT_RNEI_A,
    disp_thr: float = DEFAULT_DISP_THR_A,
) -> Path:
    """Render one event's init/saddle/final schematic triptych.

    Parameters
    ----------
    npz_path : pathlib.Path or str
        Path to a prepared event npz from
        :mod:`Analysis.blender_viewer.prepare`.
    view : {'x', 'y', 'z'} or array-like of shape (3,)
        Camera +z direction. Default 'z' (look straight down).
    side : {'forward', 'backward'}
        Which direction of the event pair to render.
    show_cell : bool
        Draw the unit cell outline. Currently a no-op because the npz
        pipeline does not yet carry cell information — emits a
        FutureWarning when True.
    barrier_ev : float or None
        Saddle-panel barrier label override. Defaults to the value
        stored in the npz for the chosen side.
    save : bool
        If True, write PNG and PDF; if False, build the figure only and
        return the path the PNG *would* have been written to (does not
        actually save).
    out_dir : pathlib.Path or str or None
        Output directory. Default: the current working directory.
    active_idx : list of int or None
        Manual override for the "active neighbour" set. When None, the
        set is computed via
        :func:`Analysis.schematic_viewer.coordination.find_active_neighbours`.
    coord_thr : int
    rnei : float
    disp_thr : float
        Thresholds when auto-inferring ``active_idx``. The auto rule
        unions:
          * undercoordinated atoms within ``rnei`` of the mover
            (``coord_thr``, ``rnei``);
          * atoms whose net |final - init| exceeds ``disp_thr``.
        All three are ignored when ``active_idx`` is given explicitly.

    Returns
    -------
    pathlib.Path
        Path to the written PNG (regardless of ``save``).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    pair = load_event_pair_npz(npz_path)
    chosen = _pick_side(pair, side)

    if active_idx is None:
        active = find_active_set(
            chosen.init_positions,
            chosen.final_positions,
            mover_idx=int(chosen.mover_idx),
            rnei=float(rnei),
            coord_thr=int(coord_thr),
            disp_thr=float(disp_thr),
        )
        logger.info(
            "active neighbours (auto, union): rnei=%.2f thr=%d disp=%.2f "
            "→ %d atoms %s",
            rnei, coord_thr, disp_thr, len(active), active.tolist(),
        )
    else:
        active = np.asarray(active_idx, dtype=np.int64)
        logger.info("active neighbours (manual): %s", active.tolist())

    if barrier_ev is None:
        barrier_ev = float(chosen.energy_barrier)

    fig = make_triptych_figure(
        positions_triple=(
            chosen.init_positions,
            chosen.saddle_positions,
            chosen.final_positions,
        ),
        elements=chosen.types,
        mover_idx=int(chosen.mover_idx),
        active_idx=active,
        view=view,
        barrier_ev=barrier_ev,
        show_cell=show_cell,
    )

    out_dir_path = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"schematic_{int(chosen.idx_ref)}_{pair.sim_label}"
    out_png = out_dir_path / f"{basename}.png"
    out_pdf = out_dir_path / f"{basename}.pdf"

    if save:
        fig.savefig(out_png, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
        logger.info("wrote %s (%.1f kB)", out_png, out_png.stat().st_size / 1024)
        logger.info("wrote %s (%.1f kB)", out_pdf, out_pdf.stat().st_size / 1024)
        _verify(fig, chosen, view=view, out_png=out_png)
        logger.info("verified — shared axes, mover projection, file size OK")

    # Free up the figure to avoid leaks in batch loops.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
    return out_png


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path,
                        help="prepared event npz from blender_viewer.prepare")
    parser.add_argument("--view", default="z",
                        help="'x' / 'y' / 'z' or three comma-separated floats "
                             "(e.g. '1,1,0') for an arbitrary view vector")
    parser.add_argument("--side", default="forward",
                        choices=["forward", "backward"])
    parser.add_argument("--show-cell", action="store_true",
                        help="(currently a no-op; npz lacks cell info)")
    parser.add_argument("--barrier-ev", type=float, default=None,
                        help="override the saddle panel's Ea label")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: current directory)")
    parser.add_argument("--active-idx", default=None,
                        help="comma-separated indices to colour yellow; "
                             "overrides auto-detection")
    parser.add_argument("--coord-thr", type=int, default=DEFAULT_COORD_THR,
                        help=f"undercoordination threshold "
                             f"(default {DEFAULT_COORD_THR})")
    parser.add_argument("--rnei", type=float, default=DEFAULT_RNEI_A,
                        help=f"1NN cutoff radius (Å, default {DEFAULT_RNEI_A})")
    parser.add_argument("--disp-thr", type=float, default=DEFAULT_DISP_THR_A,
                        help=f"displacement threshold for the motion-based "
                             f"co-mover rule (Å, default {DEFAULT_DISP_THR_A})")
    parser.add_argument("--no-save", action="store_true",
                        help="build but do not write PNG/PDF")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )

    # Parse view: comma-separated triple → np.array, else pass as str.
    view: str | np.ndarray
    if "," in args.view:
        view = np.array([float(t) for t in args.view.split(",")], dtype=np.float64)
        if view.size != 3:
            raise SystemExit("--view requires 3 comma-separated floats")
    else:
        view = args.view

    active = None
    if args.active_idx is not None:
        active = [int(i) for i in args.active_idx.split(",")]

    out_png = main(
        npz_path=args.npz,
        view=view,
        side=args.side,
        show_cell=args.show_cell,
        barrier_ev=args.barrier_ev,
        save=not args.no_save,
        out_dir=args.out_dir,
        active_idx=active,
        coord_thr=args.coord_thr,
        rnei=args.rnei,
        disp_thr=args.disp_thr,
    )
    print(out_png)


if __name__ == "__main__":
    _cli()
