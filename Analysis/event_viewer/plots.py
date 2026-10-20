"""Plotly 3D figure builder for a single KMC event.

Renders one of three frames (initial / saddle / final) with:
- Element-colored atoms from `initial_types`
- Mover atom (from `move_atom_idx`) scaled up with a black outline
- Translucent "ghost" copies of the other two frames so motion is visible
- The mover atom's initial -> saddle -> final path drawn as a red polyline
- Fixed axis ranges across all three frames so the viewport does not jump

Kept pure (no Dash imports, no module globals mutated at call time) so it can be
reused from a notebook or unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Analysis.event_viewer.analysis import (
    CNA_COLORS,
    _estimate_lattice_constant,
    _infer_nn_offsets,
    compute_cna,
    find_vacancies,
)
from Analysis.event_viewer.scatter_palette import DIRECTION_COLORS
from Analysis.event_viewer.spheres import make_spheres_trace
from Analysis.lattice_3d import classify_direction_xyz
from Analysis.reference_table import event_types

# Element -> color.
ELEMENT_COLORS = {
    "Ni": "#8C9EFF",
    "Fe": "#FF9E80",
    "Cr": "#B9F6CA",
    "Cu": "#FFB74D",
    "Al": "#FFF59D",
    "H": "#FFFFFF",
}
DEFAULT_COLOR = "#CCCCCC"
FRAME_NAMES = ("initial", "saddle", "final")

# Distinctive outline for the mover in marker mode (independent of element).
MOVER_OUTLINE_COLOR = "#FFC400"  # amber/gold — visible against any element color


def _lighten_hex(hex_color: str, amount: float = 0.35) -> str:
    """Mix a hex color with white by `amount` (0–1). Used to tint the mover
    sphere so it's distinguishable from other atoms of the same element."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_color
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

# Cache CNA results per event so re-rendering frames is cheap.
_CNA_CACHE: dict[tuple[int, int, int, str], np.ndarray] = {}


def _color_for(type_str: str) -> str:
    return ELEMENT_COLORS.get(str(type_str), DEFAULT_COLOR)


def _clip_line_to_box(
    p0: np.ndarray, v: np.ndarray, mins: np.ndarray, maxs: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (p_lo, p_hi) where the infinite line p0 + t·v enters/exits the
    axis-aligned box [mins, maxs]. Returns None if there is no intersection
    (should not happen in practice since p0 is inside the box).
    """
    v = np.asarray(v, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    t_lo, t_hi = -np.inf, np.inf
    for i in range(3):
        if abs(v[i]) < 1e-12:
            if not (mins[i] - 1e-9 <= p0[i] <= maxs[i] + 1e-9):
                return None
            continue
        t_a = (mins[i] - p0[i]) / v[i]
        t_b = (maxs[i] - p0[i]) / v[i]
        lo, hi = (t_a, t_b) if t_a < t_b else (t_b, t_a)
        if lo > t_lo:
            t_lo = lo
        if hi < t_hi:
            t_hi = hi
    if t_lo >= t_hi:
        return None
    return p0 + t_lo * v, p0 + t_hi * v


def _canonicalise_direction(v: np.ndarray) -> tuple[float, float, float]:
    """Collapse ±v and small numerical jitter into one hash key.

    Normalises to a unit vector, flips sign so the first significant
    component is positive, and rounds to 2 decimal places. The rounding
    tolerance is deliberately coarse (~0.01 unit ≈ 0.5° angular) so that
    data-inferred NN offsets with ~1% noise collapse onto the 6 canonical
    FCC directions in each family.
    """
    vv = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(vv))
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    u = vv / n
    # Canonical sign: first component with |c| > 0.05 must be positive.
    for c in u:
        if c > 0.05:
            break
        if c < -0.05:
            u = -u
            break
    return tuple(float(x) for x in np.round(u, 2))  # type: ignore[return-value]


def _direction_vectors_for_overlay(
    init: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Bucket canonical FCC direction vectors by direction_family_3d label.

    Uses the data-inferred 12 NN offsets for `<110>_inplane` and
    `<111>_interlayer`, and adds 2NN vectors (`<100>_inplane` diagonals and
    `<001>_exchange` axial) analytically since those are not in the 1NN set.
    """
    a = _estimate_lattice_constant(init)
    d_nn = a / np.sqrt(2.0)
    ls = d_nn / np.sqrt(2.0)  # (100) layer spacing = a/2

    buckets: dict[str, list[tuple[float, float, float]]] = {}
    seen: dict[str, set[tuple[float, float, float]]] = {}

    def _add(vec: np.ndarray, label: str) -> None:
        k = _canonicalise_direction(vec)
        s = seen.setdefault(label, set())
        if k in s:
            return
        s.add(k)
        buckets.setdefault(label, []).append(np.asarray(k, dtype=float))

    # 1NN offsets (12): yields `<110>_inplane` × 4 and `<111>_interlayer` × 8
    for v in _infer_nn_offsets(init, d_nn):
        lbl = classify_direction_xyz(float(v[0]), float(v[1]), float(v[2]), d_nn)
        if lbl in ("<110>_inplane", "<111>_interlayer"):
            _add(np.asarray(v, dtype=float), lbl)

    # 2NN in-plane diagonals — the four (±d, ±d, 0)/√2 · √2 = (±d, ±d, 0)
    # pattern with magnitude d_nn·√2. Four unique directions after ±-dedup.
    for sx in (+1, -1):
        for sy in (+1, -1):
            _add(np.array([sx * d_nn, sy * d_nn, 0.0]), "<100>_inplane")

    # 2NN axial exchange — straight up/down at twice the layer spacing.
    _add(np.array([0.0, 0.0, 2.0 * ls]), "<001>_exchange")

    return {k: list(vs) for k, vs in buckets.items()}


def _cna_for_event(event_row: pd.Series, frame: str) -> np.ndarray:
    """Return per-atom CNA labels for a specific (event, frame), cached."""
    # Key on event identity (idx_ref + array data hash) + frame.
    pos = np.asarray(event_row[f"{frame}_positions"], dtype=float)
    key = (int(event_row["idx_ref"]), len(pos), hash(pos.tobytes()), frame)
    if key not in _CNA_CACHE:
        _CNA_CACHE[key] = compute_cna(pos)
    return _CNA_CACHE[key]


def make_event_figure(
    event_row: pd.Series,
    frame: str = "initial",
    color_mode: str = "element",
    show_vacancies: bool = True,
    atom_size: float = 1.25,
    render_style: str = "markers",
    overlay_110_inplane: bool = False,
    overlay_100_inplane: bool = False,
    overlay_111_interlayer: bool = False,
    overlay_001_exchange: bool = False,
    camera_state: dict | None = None,
) -> go.Figure:
    """Build a Plotly 3D scatter figure for one event at the selected frame.

    The figure intentionally does NOT set `scene.camera` — projection and
    camera orientation are left to the client side (via `uirevision` to
    preserve the user's rotation and a clientside callback to toggle
    perspective/orthographic).

    Parameters
    ----------
    event_row : one row from the DataFrame returned by `data.load_events`
    frame : "initial", "saddle", or "final"
    color_mode : "element" (Ni/Fe/Cr colors) or "cna" (FCC/HCP/BCC/ICO/OTHER)
    show_vacancies : whether to draw vacancy markers (open diamonds)
    atom_size : approximate atomic radius in Ångstroms (0.3 — 3.0 Å is the
        sensible range; 1.25 Å ≈ Ni metallic radius). For markers it scales the
        pixel size; for spheres it is the actual radius in data units.
    render_style : "markers" (Scatter3d sprites — pixel-fixed size) or
        "spheres" (Mesh3d real 3D spheres — closer atoms appear larger).
    """
    if frame not in FRAME_NAMES:
        raise ValueError(f"frame must be one of {FRAME_NAMES}, got {frame!r}")
    if color_mode not in ("element", "cna"):
        raise ValueError(f"color_mode must be 'element' or 'cna', got {color_mode!r}")
    if render_style not in ("markers", "spheres"):
        raise ValueError(f"render_style must be 'markers' or 'spheres', got {render_style!r}")
    s = float(atom_size)

    init = np.asarray(event_row["initial_positions"], dtype=float)
    sadd = np.asarray(event_row["saddle_positions"], dtype=float)
    final = np.asarray(event_row["final_positions"], dtype=float)
    frames = {"initial": init, "saddle": sadd, "final": final}

    types = [str(t) for t in event_types(event_row)]
    n_atoms = len(types)
    mover_idx = int(event_row["move_atom_idx"])

    idx_ref = int(event_row["idx_ref"])
    ea = float(event_row["energy_barrier"])
    move_type = str(event_row.get("move_type", "unclassified"))

    # Fix the camera box over all frames so switching frames does not zoom.
    # In spheres mode the atoms have physical radius `s` Å in data units, so
    # spheres at the boundary clip against the axis box unless we pad by the
    # radius (plus a little extra for the slightly-bigger mover).
    all_pos = np.vstack([init, sadd, final])
    pad = 0.5 + (s * 1.15 if render_style == "spheres" else 0.0)
    mins = all_pos.min(axis=0) - pad
    maxs = all_pos.max(axis=0) + pad

    # Per-atom colors. Mover is drawn as a separate trace so it can carry an
    # outline (Plotly Scatter3d.marker.line.width must be a scalar).
    other_mask = np.ones(n_atoms, dtype=bool)
    other_mask[mover_idx] = False

    if color_mode == "cna":
        cna_labels = _cna_for_event(event_row, frame)
        colors = np.array([CNA_COLORS.get(lbl, DEFAULT_COLOR) for lbl in cna_labels])
        hover_extra = [f"{t} | {lbl}" for t, lbl in zip(types, cna_labels)]
    else:
        colors = np.array([_color_for(t) for t in types])
        hover_extra = list(types)

    frame_pos = frames[frame]
    fig = go.Figure()

    # Ghost markers for the two non-active frames (translucent, small).
    for gname in FRAME_NAMES:
        if gname == frame:
            continue
        gpos = frames[gname]
        fig.add_trace(
            go.Scatter3d(
                x=gpos[:, 0],
                y=gpos[:, 1],
                z=gpos[:, 2],
                mode="markers",
                marker=dict(size=2.5 * s, color=colors.tolist(), opacity=0.18),
                name=f"{gname} (ghost)",
                hoverinfo="skip",
            )
        )

    # Non-mover atoms at the selected frame.
    other_hover = [f"atom {i} — {hover_extra[i]}" for i in range(n_atoms) if i != mover_idx]
    other_pos = frame_pos[other_mask]

    if render_style == "spheres":
        # Real 3D spheres — closer atoms appear larger under perspective.
        # Atom radius is `s` Å; mover gets a slight bump so it stays visible.
        fig.add_trace(
            make_spheres_trace(
                centers=other_pos,
                radius=s,
                colors=colors[other_mask].tolist(),
                name=f"atoms — {frame}",
                hovertext=other_hover,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=other_pos[:, 0],
                y=other_pos[:, 1],
                z=other_pos[:, 2],
                mode="markers",
                marker=dict(
                    size=5 * s,
                    color=colors[other_mask].tolist(),
                    opacity=0.95,
                    line=dict(color="black", width=0),
                ),
                text=other_hover,
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                name=f"atoms — {frame}",
            )
        )

    # Mover atom (highlighted). 1.1× size bump only — the distinctiveness comes
    # from a different outline (markers) or a lightened shade (spheres).
    mover_pos_this = frame_pos[mover_idx]
    mover_base_color = _color_for(types[mover_idx])
    if render_style == "spheres":
        fig.add_trace(
            make_spheres_trace(
                centers=mover_pos_this[None, :],
                radius=s * 1.1,
                colors=[_lighten_hex(mover_base_color, 0.35)],
                name="mover",
                hovertext=[f"MOVER: atom {mover_idx} — {types[mover_idx]}"],
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=[mover_pos_this[0]],
                y=[mover_pos_this[1]],
                z=[mover_pos_this[2]],
                mode="markers",
                marker=dict(
                    size=5 * s * 1.1,
                    color=mover_base_color,
                    opacity=1.0,
                    line=dict(color=MOVER_OUTLINE_COLOR, width=3),
                ),
                text=[f"MOVER: atom {mover_idx} — {types[mover_idx]}"],
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                name="mover",
            )
        )

    # Red polyline showing the mover atom's initial -> saddle -> final path.
    mover_path = np.vstack([init[mover_idx], sadd[mover_idx], final[mover_idx]])
    fig.add_trace(
        go.Scatter3d(
            x=mover_path[:, 0],
            y=mover_path[:, 1],
            z=mover_path[:, 2],
            mode="lines+markers",
            line=dict(color="red", width=5),
            marker=dict(size=3, color="red"),
            name="mover path",
            hoverinfo="skip",
        )
    )

    # Vacancy markers — open diamonds at lattice sites that are empty in this
    # frame. Also draw their full-trajectory polyline in muted blue so users
    # can see how the vacancy moves as the frame changes.
    if show_vacancies:
        vac_per_frame = find_vacancies(init, sadd, final)
        vac_here = vac_per_frame[frame]
        if len(vac_here) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=vac_here[:, 0],
                    y=vac_here[:, 1],
                    z=vac_here[:, 2],
                    mode="markers",
                    marker=dict(
                        size=16 * s,
                        color="rgba(0,0,0,0)",  # hollow fill
                        symbol="diamond-open",
                        line=dict(color="#2962FF", width=4),
                    ),
                    text=[f"VACANCY ({frame})" for _ in vac_here],
                    hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                    name="vacancy",
                )
            )
        # Ghost diamonds for the other two frames to hint where vacancies move.
        for other_frame in FRAME_NAMES:
            if other_frame == frame:
                continue
            vac_other = vac_per_frame[other_frame]
            if len(vac_other) == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=vac_other[:, 0],
                    y=vac_other[:, 1],
                    z=vac_other[:, 2],
                    mode="markers",
                    marker=dict(
                        size=10 * s,
                        color="rgba(41,98,255,0.22)",
                        symbol="diamond-open",
                        line=dict(color="rgba(41,98,255,0.4)", width=2),
                    ),
                    name=f"vacancy ({other_frame})",
                    hoverinfo="skip",
                )
            )

    # Crystallographic direction overlays — drawn through the mover's
    # initial-frame position, clipped to the current axis box so autoscale
    # stays intact. Each toggled family emits one Scatter3d(mode="lines")
    # trace per unique direction; legend shows one entry per family.
    _overlay_flags = {
        "<110>_inplane": overlay_110_inplane,
        "<100>_inplane": overlay_100_inplane,
        "<111>_interlayer": overlay_111_interlayer,
        "<001>_exchange": overlay_001_exchange,
    }
    if any(_overlay_flags.values()):
        p0_overlay = init[mover_idx]
        dir_buckets = _direction_vectors_for_overlay(init)
        for family, on in _overlay_flags.items():
            if not on:
                continue
            vecs = dir_buckets.get(family, [])
            first_in_group = True
            for v in vecs:
                endpoints = _clip_line_to_box(p0_overlay, v, mins, maxs)
                if endpoints is None:
                    continue
                p_lo, p_hi = endpoints
                fig.add_trace(
                    go.Scatter3d(
                        x=[p_lo[0], p_hi[0]],
                        y=[p_lo[1], p_hi[1]],
                        z=[p_lo[2], p_hi[2]],
                        mode="lines",
                        line=dict(
                            color=DIRECTION_COLORS.get(family, "#888888"),
                            width=3,
                            dash="dash",
                        ),
                        name=family,
                        legendgroup=family,
                        showlegend=first_in_group,
                        hoverinfo="skip",
                    )
                )
                first_in_group = False

    # One stable revision key per event preserves user-driven figure state
    # (camera, zoom, legend click-to-hide, modebar mode) across frame /
    # overlay / color / atom-size toggles on the *same* event. A fresh event
    # (different idx_ref) gets a new key so the view refits automatically.
    #
    # `uirevision` is valid at two levels here: the top-level layout
    # (preserves legend/toolbar state) and `scene` (preserves 3D camera).
    # It is NOT a property of scene.xaxis/yaxis/zaxis — putting it there
    # raises "Bad property path: uirevision" in plotly.
    ui_rev = f"event-{idx_ref}"

    scene_dict: dict = dict(
        xaxis=dict(range=[mins[0], maxs[0]], title="x (Å)"),
        yaxis=dict(range=[mins[1], maxs[1]], title="y (Å)"),
        zaxis=dict(range=[mins[2], maxs[2]], title="z (Å)"),
        aspectmode="data",
        # uirevision preserves legend state, modebar, etc. across rebuilds.
        # The user's camera is plumbed through camera_state below — Plotly's
        # uirevision heuristic alone wasn't enough because the per-frame
        # axis ranges (atoms move) appear to defeat the camera carry-over.
        uirevision=ui_rev,
    )
    if camera_state:
        scene_dict["camera"] = camera_state

    fig.update_layout(
        title=f"Event #{idx_ref} — {move_type} — {ea:.3f} eV — {frame} frame",
        uirevision=ui_rev,
        scene=scene_dict,
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )
    return fig


def empty_figure(message: str = "Select a run and event to view") -> go.Figure:
    """Placeholder figure shown before a selection is made or on failure."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=16, color="#888"),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
