"""Data access for the Blender viewer.

Two import modes:

- Full mode (pandas + pyKMC deps available): ``load_event_pair`` reads
  ``reference_table.pickle`` directly. Used by host-venv scripts and tests.

- Lite mode (numpy only, for Blender's bundled Python): ``load_event_pair_npz``
  reads a pre-dumped intermediate file produced by ``dump_event_pair_npz``.
  Blender's numpy often mismatches the numpy that saved the pickle (numpy
  1.x vs 2.x), so we avoid pickle unpickling inside bpy by going through
  an npz intermediate. numpy is the only shared dependency.

Why two separate object sets per pair
-------------------------------------
Forward and backward events each store a 127-atom neighbor subset centered on
their own mover. Those two movers sit at neighboring lattice sites ~2.5 Å
apart (the hop endpoints), so the two neighbor balls overlap most of their
interior but differ at the edges. We keep the two event subsets as separate
atom sets in Blender and visibility-toggle between them on the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EventSide:
    """One direction (forward or backward) of a KMC transition."""

    idx_ref: int
    types: list[str]
    init_positions: np.ndarray    # (N, 3)
    saddle_positions: np.ndarray  # (N, 3)
    final_positions: np.ndarray   # (N, 3)
    mover_idx: int
    energy_barrier: float


@dataclass
class EventPair:
    """Forward + backward sides of a single KMC transition."""

    forward: EventSide
    backward: EventSide
    sim_label: str


def load_event_pair(sim_path: Path, idx_ref_fwd: int) -> EventPair:
    """Load the forward + backward event from ``reference_table.pickle``.

    Requires pandas and the host pyKMC venv's pickle compatibility layer.
    Do NOT call this from inside Blender's Python — use
    :func:`load_event_pair_npz` instead (after running
    :func:`dump_event_pair_npz` in the host venv).
    """
    import pandas as pd  # noqa: PLC0415

    from Analysis.event_viewer.data import _load_pickle  # noqa: PLC0415

    sim_path = Path(sim_path)
    table_path = sim_path / "reference_table.pickle"
    if not table_path.exists():
        raise FileNotFoundError(table_path)

    df: pd.DataFrame = _load_pickle(table_path)

    fwd_rows = df[df.idx_ref == idx_ref_fwd]
    if fwd_rows.empty:
        raise KeyError(f"idx_ref={idx_ref_fwd} not found in {table_path}")
    fwd = fwd_rows.iloc[0]

    idx_ref_bwd = int(fwd.idx_backward)
    if idx_ref_bwd == idx_ref_fwd:
        raise ValueError(
            f"event {idx_ref_fwd} is symmetric (idx_backward == idx_ref); "
            "this workflow expects a distinct backward partner"
        )
    bwd_rows = df[df.idx_ref == idx_ref_bwd]
    if bwd_rows.empty:
        raise KeyError(f"idx_backward={idx_ref_bwd} not found in {table_path}")
    bwd = bwd_rows.iloc[0]

    forward = _row_to_side(fwd)
    backward = _row_to_side(bwd)

    # Short human-readable label from the sim path — parent dirs carry
    # composition/nvac/temp. Not required for correctness; only used in
    # the saved .blend filename and collection names.
    parts = sim_path.resolve().parts
    try:
        comp = parts[-4]          # e.g. 100Ni
        nvac = parts[-3]          # e.g. 10vac
        temp = parts[-2]          # e.g. T1200
        sim_label = f"{comp}_{nvac}_{temp}"
    except IndexError:
        sim_label = sim_path.name

    return EventPair(forward=forward, backward=backward, sim_label=sim_label)


def _row_to_side(row) -> EventSide:  # pd.Series — unannotated to keep module import-light
    # Newer pyKMC reference tables name the column ``types`` (no ``initial_types``).
    types = row["initial_types"] if "initial_types" in row.index else row["types"]
    return EventSide(
        idx_ref=int(row.idx_ref),
        types=[str(t) for t in types],
        init_positions=np.asarray(row.initial_positions, dtype=np.float64),
        saddle_positions=np.asarray(row.saddle_positions, dtype=np.float64),
        final_positions=np.asarray(row.final_positions, dtype=np.float64),
        mover_idx=int(row.move_atom_idx),
        energy_barrier=float(row.energy_barrier),
    )


def mover_displacement(side: EventSide) -> float:
    """|final - init| of the mover atom (Å) — useful for sanity-checking selection."""
    return float(
        np.linalg.norm(
            side.final_positions[side.mover_idx] - side.init_positions[side.mover_idx]
        )
    )


# ---------------------------------------------------------------------------
# NPZ intermediate: host venv writes, Blender reads
# ---------------------------------------------------------------------------

_SIDES = ("forward", "backward")


def dump_event_pair_npz(pair: EventPair, out_path: Path) -> None:
    """Serialize an EventPair to a single .npz file Blender can load."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"sim_label": np.array(pair.sim_label)}
    for side_name in _SIDES:
        side: EventSide = getattr(pair, side_name)
        payload[f"{side_name}_idx_ref"] = np.array(side.idx_ref)
        payload[f"{side_name}_types"] = np.array(side.types)
        payload[f"{side_name}_init"] = side.init_positions
        payload[f"{side_name}_saddle"] = side.saddle_positions
        payload[f"{side_name}_final"] = side.final_positions
        payload[f"{side_name}_mover_idx"] = np.array(side.mover_idx)
        payload[f"{side_name}_energy_barrier"] = np.array(side.energy_barrier)
    np.savez(out_path, **payload)


def load_event_pair_npz(npz_path: Path) -> EventPair:
    """Read the npz intermediate back into an EventPair (numpy-only dependency)."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    def _side(prefix: str) -> EventSide:
        return EventSide(
            idx_ref=int(data[f"{prefix}_idx_ref"]),
            types=[str(t) for t in data[f"{prefix}_types"]],
            init_positions=np.asarray(data[f"{prefix}_init"], dtype=np.float64),
            saddle_positions=np.asarray(data[f"{prefix}_saddle"], dtype=np.float64),
            final_positions=np.asarray(data[f"{prefix}_final"], dtype=np.float64),
            mover_idx=int(data[f"{prefix}_mover_idx"]),
            energy_barrier=float(data[f"{prefix}_energy_barrier"]),
        )

    return EventPair(
        forward=_side("forward"),
        backward=_side("backward"),
        sim_label=str(data["sim_label"]),
    )
