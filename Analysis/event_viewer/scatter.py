"""Displacement scatter figure for the event_viewer audit UI."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from Analysis.event_viewer.scatter_palette import (
    ALL_FAMILY_IDS,
    ALL_TYPES,
    ALL_TYPES_BASE,
    DIRECTION_IDS,
    LAYER_CHANGE_DZ,
    NN1_DIST,
    NN2_DIST,
    base_type,
    color_for,
    color_for_direction,
    color_for_family,
)


def build_scatter_figure(
    df: pd.DataFrame,
    active_types: set[str],
    audited: set[tuple[str, int, int, int]],
    *,
    merge_updown: bool,
    show_only_unaudited: bool,
    x_col: str = "move_dist",
    y_col: str = "move_dz",
    color_col: str = "move_type",
    point_size: int = 7,
    active_families: set[str] | None = None,
    active_buckets: set[str] | None = None,
    active_directions: set[str] | None = None,
    selected_xy: tuple[float, float] | None = None,
) -> go.Figure:
    """Return a Plotly scatter coloured by move_type, family_id, or a descriptor.

    Each point carries customdata = [composition, nvac, temp, idx_ref,
    energy_barrier, move_type] so a click callback can resolve it back to an
    event.

    Parameters
    ----------
    x_col, y_col : str
        Column names to use for x and y axes (default: move_dist, move_dz).
    color_col : str
        Column name for point color. ``"move_type"`` / ``"move_type_base"``
        uses the move-type palette; ``"family_id"`` uses the family palette;
        any numeric column uses a Viridis scale.
    active_families, active_buckets : set[str] | None
        If provided (and non-empty), only rows whose ``family_id`` /
        ``family_bucket_id`` is in the set survive. ``None`` means no filter.
    """
    fig = go.Figure()

    # Determine axis labels
    _LABELS = {
        "move_dist": "Primary atom displacement |d| (Å)",
        "move_dz": "Out-of-plane |dz| (Å)",
    }
    x_label = _LABELS.get(x_col, x_col)
    y_label = _LABELS.get(y_col, y_col)

    if df.empty:
        fig.update_layout(
            title="No events match current filters",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=480,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
        )
        return fig

    # Check required columns
    if x_col not in df.columns or y_col not in df.columns:
        fig.update_layout(
            title=f"Column '{x_col}' or '{y_col}' not in data",
            height=480,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
        )
        return fig

    type_col = "move_type_base" if merge_updown and "move_type_base" in df.columns else "move_type"
    if merge_updown and "move_type_base" not in df.columns:
        df = df.assign(_base=df["move_type"].map(base_type))
        type_col = "_base"

    type_order = ALL_TYPES_BASE if merge_updown else ALL_TYPES

    # Family / bucket filters. Each non-empty set is applied independently so
    # the user can narrow the scatter to a single family and/or bucket.
    if active_families and "family_id" in df.columns:
        df = df[df["family_id"].astype(str).isin(active_families)]
    if active_buckets and "family_bucket_id" in df.columns:
        df = df[df["family_bucket_id"].astype(str).isin(active_buckets)]
    if active_directions and "direction_family_3d" in df.columns:
        df = df[df["direction_family_3d"].astype(str).isin(active_directions)]

    if show_only_unaudited and audited:
        key_tuples = list(zip(
            df["composition"].astype(str),
            df["nvac"].astype(int),
            df["temp"].astype(int),
            df["idx_ref"].astype(int),
        ))
        keep = [k not in audited for k in key_tuples]
        df = df.loc[keep].copy()

    # Build audited flag for dark-outline styling
    if audited:
        key_tuples = list(zip(
            df["composition"].astype(str),
            df["nvac"].astype(int),
            df["temp"].astype(int),
            df["idx_ref"].astype(int),
        ))
        df = df.assign(_audited=[k in audited for k in key_tuples])
    else:
        df = df.assign(_audited=False)

    # --- Compute x/y values ---
    x_vals = df[x_col]
    y_vals = df[y_col].abs() if y_col == "move_dz" else df[y_col]

    # --- Determine color mode ---
    categorical_cols = {"move_type", "move_type_base", "family_id", "direction_family_3d"}
    use_continuous = color_col not in categorical_cols
    use_family = color_col == "family_id"
    use_direction = color_col == "direction_family_3d"

    if use_direction and "direction_family_3d" in df.columns:
        # Categorical trace per direction_id using DIRECTION_COLORS palette.
        for did in DIRECTION_IDS:
            mask = df["direction_family_3d"].astype(str) == did
            if not mask.any():
                continue
            sub = df.loc[mask]
            sub_types = sub[type_col] if type_col in sub.columns else sub.get("move_type")
            if active_types:
                sub = sub[sub_types.isin(active_types)] if sub_types is not None else sub
                if sub.empty:
                    continue
            custom = sub[[
                "composition", "nvac", "temp", "idx_ref", "energy_barrier",
            ]].copy()
            custom["move_type"] = sub["move_type"].values
            hover = (
                f"<b>{did}</b><br>"
                "%{customdata[5]}<br>"
                "%{customdata[0]} · %{customdata[1]} vac · T=%{customdata[2]} K<br>"
                "idx_ref=%{customdata[3]} · Ea=%{customdata[4]:.3f} eV<br>"
                f"{x_col}=%{{x:.3f}}, {y_col}=%{{y:.3f}}<extra></extra>"
            )
            fig.add_trace(go.Scattergl(
                x=sub[x_col] if x_col != "move_dz" else sub[x_col].abs(),
                y=sub[y_col].abs() if y_col == "move_dz" else sub[y_col],
                mode="markers",
                name=f"{did} ({len(sub)})",
                marker={
                    "size": point_size,
                    "color": color_for_direction(did),
                    "line": {
                        "width": [1.3 if a else 0.4 for a in sub["_audited"]],
                        "color": ["#111" if a else "#ffffff"
                                  for a in sub["_audited"]],
                    },
                    "opacity": 0.75,
                },
                customdata=custom.to_numpy(),
                hovertemplate=hover,
            ))
    elif use_family and "family_id" in df.columns:
        # Categorical trace per family_id using FAMILY_COLORS palette.
        for fid in ALL_FAMILY_IDS:
            mask = df["family_id"].astype(str) == fid
            if not mask.any():
                continue
            sub = df.loc[mask]
            sub_types = sub[type_col] if type_col in sub.columns else sub.get("move_type")
            if active_types:
                sub = sub[sub_types.isin(active_types)] if sub_types is not None else sub
                if sub.empty:
                    continue
            custom = sub[[
                "composition", "nvac", "temp", "idx_ref", "energy_barrier",
            ]].copy()
            custom["move_type"] = sub["move_type"].values
            hover = (
                f"<b>{fid}</b><br>"
                "%{customdata[5]}<br>"
                "%{customdata[0]} · %{customdata[1]} vac · T=%{customdata[2]} K<br>"
                "idx_ref=%{customdata[3]} · Ea=%{customdata[4]:.3f} eV<br>"
                f"{x_col}=%{{x:.3f}}, {y_col}=%{{y:.3f}}<extra></extra>"
            )
            fig.add_trace(go.Scattergl(
                x=sub[x_col] if x_col != "move_dz" else sub[x_col].abs(),
                y=sub[y_col].abs() if y_col == "move_dz" else sub[y_col],
                mode="markers",
                name=f"{fid} ({len(sub)})",
                marker={
                    "size": point_size,
                    "color": color_for_family(fid),
                    "line": {
                        "width": [1.3 if a else 0.4 for a in sub["_audited"]],
                        "color": ["#111" if a else "#ffffff"
                                  for a in sub["_audited"]],
                    },
                    "opacity": 0.75,
                },
                customdata=custom.to_numpy(),
                hovertemplate=hover,
            ))
    elif use_continuous and color_col in df.columns:
        # Single trace with continuous Viridis colorscale
        c_vals = pd.to_numeric(df[color_col], errors="coerce")
        custom = df[["composition", "nvac", "temp", "idx_ref", "energy_barrier"]].copy()
        custom["move_type"] = df["move_type"].values
        hover = (
            "<b>%{customdata[5]}</b><br>"
            "%{customdata[0]} · %{customdata[1]} vac · T=%{customdata[2]} K<br>"
            f"{x_col}=%{{x:.3f}}, {y_col}=%{{y:.3f}}<br>"
            f"{color_col}=%{{marker.color:.3f}}<extra></extra>"
        )
        fig.add_trace(go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=color_col,
            marker={
                "size": point_size,
                "color": c_vals,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_col, "thickness": 15},
                "opacity": 0.75,
                "line": {
                    "width": [1.3 if a else 0.4 for a in df["_audited"]],
                    "color": ["#111" if a else "#ffffff"
                              for a in df["_audited"]],
                },
            },
            customdata=custom.to_numpy(),
            hovertemplate=hover,
        ))
    else:
        # Categorical per-type traces (original logic)
        for mtype in type_order:
            if mtype not in active_types:
                continue
            mask = df[type_col] == mtype
            if not mask.any():
                continue
            sub = df.loc[mask]
            custom = sub[[
                "composition", "nvac", "temp", "idx_ref", "energy_barrier",
            ]].copy()
            custom["move_type"] = sub["move_type"].values
            hover = (
                "<b>%{customdata[5]}</b><br>"
                "%{customdata[0]} · %{customdata[1]} vac · T=%{customdata[2]} K<br>"
                "idx_ref=%{customdata[3]} · Ea=%{customdata[4]:.3f} eV<br>"
                f"{x_col}=%{{x:.3f}}, {y_col}=%{{y:.3f}}<extra></extra>"
            )
            fig.add_trace(go.Scattergl(
                x=sub[x_col] if x_col != "move_dz" else sub[x_col].abs(),
                y=sub[y_col].abs() if y_col == "move_dz" else sub[y_col],
                mode="markers",
                name=f"{mtype} ({len(sub)})",
                marker={
                    "size": point_size,
                    "color": color_for(mtype),
                    "line": {
                        "width": [1.3 if a else 0.4 for a in sub["_audited"]],
                        "color": ["#111" if a else "#ffffff"
                                  for a in sub["_audited"]],
                    },
                    "opacity": 0.75,
                },
                customdata=custom.to_numpy(),
                hovertemplate=hover,
            ))

    # ── Reference lines (PR7): only when using the default axes ──
    show_ref_lines = (x_col == "move_dist" and y_col == "move_dz")
    shapes: list[dict] = []
    annotations: list[dict] = []
    _mono = {"family": '"JetBrains Mono", ui-monospace, Menlo, monospace',
             "size": 10, "color": "#6a625c"}
    if show_ref_lines:
        shapes = [
            {"type": "line", "x0": NN1_DIST, "x1": NN1_DIST,
             "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#b4aea8", "dash": "dash", "width": 1}},
            {"type": "line", "x0": NN2_DIST, "x1": NN2_DIST,
             "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#b4aea8", "dash": "dash", "width": 1}},
            {"type": "line", "x0": 0, "x1": 1,
             "y0": LAYER_CHANGE_DZ, "y1": LAYER_CHANGE_DZ,
             "xref": "paper",
             "line": {"color": "#b4aea8", "dash": "dot", "width": 1}},
        ]
        annotations = [
            {"x": NN1_DIST, "y": 1, "yref": "paper", "showarrow": False,
             "text": f"1NN {NN1_DIST:.2f} Å",
             "font": _mono, "yanchor": "bottom", "xshift": 4},
            {"x": NN2_DIST, "y": 1, "yref": "paper", "showarrow": False,
             "text": f"2NN {NN2_DIST:.2f} Å",
             "font": _mono, "yanchor": "bottom", "xshift": 4},
            {"x": 1, "xref": "paper", "y": LAYER_CHANGE_DZ,
             "showarrow": False,
             "text": f"layer Δz = {LAYER_CHANGE_DZ:.1f} Å",
             "font": _mono,
             "xanchor": "right", "yanchor": "bottom",
             "xshift": -6, "yshift": 2},
        ]

    # ── Selected-point gold halo (PR7) ──
    if selected_xy is not None:
        sx, sy = selected_xy
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy],
            mode="markers",
            marker={
                "size": 20,
                "symbol": "circle-open",
                "line": {"color": "#fabd0f", "width": 2},
                "color": "rgba(0, 0, 0, 0)",
            },
            name="_halo",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Layout — design tokens (PR7) ──
    _sans_font = {
        "family": '"Open Sans", -apple-system, BlinkMacSystemFont, '
                  '"Segoe UI", Roboto, sans-serif',
        "color": "#212121",
        "size": 12,
    }
    _axis_common = {
        "gridcolor":     "#e6e7e8",   # --g-200
        "zerolinecolor": "#d1d3d4",   # --g-300
        "linecolor":     "#6a625c",   # --ls-dark
        "tickfont":      _mono,
        "title":         {"font": {**_sans_font, "size": 11.5,
                                   "color": "#58595b"}},
    }
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=_axis_common,
        yaxis=_axis_common,
        height=520,
        margin={"l": 55, "r": 20, "t": 24, "b": 48},
        paper_bgcolor="#fbfaf7",   # --paper
        plot_bgcolor="#ffffff",    # --card
        font=_sans_font,
        shapes=shapes,
        annotations=annotations,
        legend={"itemsizing": "constant",
                "font": {**_sans_font, "size": 11}},
        clickmode="event+select",
        hovermode="closest",
        dragmode="select",
        newselection={"line": {"color": "#002452",
                               "dash": "dash", "width": 1.5}},
        activeselection={"fillcolor": "rgba(0, 36, 82, 0.08)",
                         "opacity": 1.0},
    )
    return fig
