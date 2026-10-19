"""Data access layer for the KMC event viewer.

Responsibilities:
- Discover pyKMC simulation directories under the data root (or given roots)
- Build cascading-filter index (composition -> nvac -> temperature)
- Load the event classifier output once
- Lazy-load per-run `reference_table.pickle` and join `move_type`
- Build a scatter frame straight from one run's pickle (single-run mode)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._libs.arrays import NDArrayBacked
from pandas.core.arrays.string_ import StringDtype

from Analysis import _paths
from Analysis._paths import CLASSIFIED_CSV, CLASSIFIED_WITH_FAMILIES_CSV
from Analysis.sim_discovery import SimInfo, find_simulations, sim_label

_EVENT_CACHE: dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# Compatibility unpickler for legacy reference_table.pickle files.
#
# Some older runs were pickled by an older pandas that stored
# NDArrayBacked ExtensionArray state as a 2-tuple (dtype, ndarray). Modern
# pandas (>=2.3) expects either a dict {"_ndarray": ..., "_dtype": ...} or a
# (ndarray, dtype) tuple — and raises NotImplementedError otherwise.
#
# We override the BUILD opcode to detect that specific situation and rewrite
# the state into the modern dict form before __setstate__ is called.
# ---------------------------------------------------------------------------


class _CompatUnpickler(pickle._Unpickler):  # type: ignore[misc]
    """Pickle.Unpickler that can read legacy pandas NDArrayBacked states."""


def _compat_load_build(self: pickle._Unpickler) -> None:
    stack = self.stack  # type: ignore[attr-defined]
    state = stack.pop()
    inst = stack[-1]
    if (
        isinstance(inst, NDArrayBacked)
        and isinstance(state, tuple)
        and len(state) == 2
    ):
        a, b = state
        arr = a if isinstance(a, np.ndarray) else b
        dt_like = b if isinstance(a, np.ndarray) else a
        if isinstance(dt_like, str):
            try:
                dt: object = np.dtype(dt_like)
            except TypeError:
                dt = StringDtype(dt_like) if dt_like.startswith("string") else np.dtype("O")
        else:
            dt = dt_like
        state = {"_ndarray": arr, "_dtype": dt}
    stack.append(state)
    pickle._Unpickler.load_build(self)  # type: ignore[attr-defined]


_CompatUnpickler.dispatch = pickle._Unpickler.dispatch.copy()  # type: ignore[attr-defined]
_CompatUnpickler.dispatch[pickle.BUILD[0]] = _compat_load_build  # type: ignore[attr-defined]


def _load_pickle(path: Path) -> pd.DataFrame:
    """Load a reference_table pickle, falling back to CompatUnpickler on legacy format."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except NotImplementedError:
        with open(path, "rb") as f:
            return _CompatUnpickler(f).load()


def discover_runs(roots: list[Path] | None = None) -> list[SimInfo]:
    """Walk the supplied roots and return discovered simulation directories.

    ``roots=None`` means the configured data root (``_paths.find_data_root()``,
    i.e. ``PYKMC_ANALYSIS_DATA_ROOT``); with none configured there are no runs
    and the app needs ``--run``. Duplicates (same path reached via multiple
    roots) are collapsed.
    """
    if roots is None:
        root = _paths.find_data_root()
        roots = [root] if root is not None else []
    seen: set[Path] = set()
    unique: list[SimInfo] = []
    for root in roots:
        if not root.exists():
            continue
        for sim in find_simulations(root):
            if sim.path in seen:
                continue
            seen.add(sim.path)
            unique.append(sim)
    return unique


def build_filter_index(
    sims: list[SimInfo],
) -> tuple[dict[str, dict[int, list[float]]], dict[tuple[str, int, float], list[SimInfo]]]:
    """Return (cascade, resolver).

    `cascade[composition][nvac]` is a sorted list of available temperatures.
    `resolver[(composition, nvac, temp)]` is the list of SimInfos matching that
    triple (usually one, but multiple coloring variants are possible).
    """
    cascade: dict[str, dict[int, set[float]]] = {}
    resolver: dict[tuple[str, int, float], list[SimInfo]] = {}
    for sim in sims:
        cascade.setdefault(sim.composition, {}).setdefault(sim.n_vacancies, set()).add(
            sim.temperature
        )
        resolver.setdefault((sim.composition, sim.n_vacancies, sim.temperature), []).append(sim)

    out: dict[str, dict[int, list[float]]] = {}
    for comp, nvac_map in cascade.items():
        out[comp] = {nvac: sorted(temps) for nvac, temps in nvac_map.items()}
    return out, resolver


_CLASSIFIER_CORE_COLS = [
    "sim_label", "idx_ref", "move_type", "energy_barrier", "nvac", "temp",
]
_CLASSIFIER_SCATTER_COLS = [
    "composition", "sim_path", "move_type_base",
    "move_dist", "move_dz", "move_dz_signed",
    "direction_family_3d",
    # ML Tier 1: mover-only
    "disp_top1", "disp_top2", "disp_top3", "disp_top4", "disp_top5",
    "mover_depth_initial", "mover_depth_final", "mover_depth_delta",
    "Ea_over_kT", "mover_species_ml",
    "nn1_count_Ni", "nn1_count_Fe", "nn1_count_Cr",
    "nn2_count_Ni", "nn2_count_Fe", "nn2_count_Cr",
    # ML Tier 2: neighborhood geometry
    "coord_mover_initial", "coord_mover_final",
    "n_vac_nn1_initial", "n_vac_nn1_final",
    "mean_nn1_bond_length_initial", "dist_to_nearest_undercoord", "local_rmsd_5A",
    # ML Tier 3: saddle-path
    "path_straightness", "coord_mover_saddle",
    "bonds_broken_init_to_saddle", "bonds_formed_init_to_saddle",
    # ML Tier 4: metadata
    "reverse_barrier", "Ea_gap", "saddle_asymmetry_oob", "kmc_fire_count",
]
_CLASSIFIER_FAMILY_COLS = [
    "family_id", "family_name", "family_bucket_id", "family_bucket_name",
    "assignment_status", "assignment_note", "audit_excluded",
]


def load_classifications() -> pd.DataFrame:
    """Read the classifier CSV once; return empty frame if missing.

    Prefers ``classified_events_with_families.csv`` (which carries the
    curated-family columns) and falls back to ``classified_events.csv`` when
    the family-tagged export is absent. Missing columns are tolerated for
    back-compat with pre-audit CSVs: in that case ``composition`` defaults to
    the single-composition corpus name (``100Ni``) and ``sim_path`` is left
    unset.
    """
    source = (CLASSIFIED_WITH_FAMILIES_CSV if CLASSIFIED_WITH_FAMILIES_CSV.exists()
              else CLASSIFIED_CSV)
    if not source.exists():
        return pd.DataFrame(
            columns=_CLASSIFIER_CORE_COLS + _CLASSIFIER_SCATTER_COLS + _CLASSIFIER_FAMILY_COLS
        )
    header = pd.read_csv(source, nrows=0).columns.tolist()
    wanted = [
        c for c in _CLASSIFIER_CORE_COLS + _CLASSIFIER_SCATTER_COLS + _CLASSIFIER_FAMILY_COLS
        if c in header
    ]
    df = pd.read_csv(source, usecols=wanted)
    if "composition" not in df.columns:
        df["composition"] = "100Ni"
    # Fill absent family columns with empty strings so scatter/audit code can
    # read them unconditionally.
    for col in _CLASSIFIER_FAMILY_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def load_scatter_frame() -> pd.DataFrame:
    """Return the classifier DataFrame filtered to rows that have scatter geometry.

    The event_viewer scatter needs `move_dist`, `move_dz`, `move_type`, and an
    event identity. If the source CSV lacks displacement columns we return an
    empty frame and let the UI show an informative empty state.
    """
    df = load_classifications()
    required = {"move_dist", "move_dz", "move_type", "idx_ref", "nvac", "temp"}
    if not required.issubset(df.columns):
        return df.iloc[0:0]
    return df.dropna(subset=["move_dist", "move_dz"])


def run_scatter_frame(sim: SimInfo, events: pd.DataFrame) -> pd.DataFrame:
    """Scatter frame for one run, built from its loaded reference table.

    Unlike ``load_scatter_frame`` this needs no classifier CSV and keeps every
    event, including those above the classifier's barrier cutoff. Geometry is
    the primary mover's initial -> final displacement (the same quantity
    ``lattice_map_events.classify_event_mechanism`` reports, without its
    alternate-mover fallback); labels are whatever ``load_events`` joined, so
    events the CSV does not cover stay ``unclassified``. The direction label
    falls back to ``live_direction_id`` when the join left it empty.
    """
    if events.empty:
        return pd.DataFrame(columns=["composition", "nvac", "temp", "idx_ref",
                                     "energy_barrier", "move_type",
                                     "move_dist", "move_dz"])
    disp = np.full((len(events), 3), np.nan)
    for pos, (init, final, mover) in enumerate(zip(
        events["initial_positions"], events["final_positions"],
        events["move_atom_idx"],
    )):
        try:
            m = int(mover)
            disp[pos] = np.asarray(final, dtype=float)[m] - np.asarray(init, dtype=float)[m]
        except (TypeError, ValueError, IndexError):
            continue

    direction = (events["direction_family_3d"].astype(str)
                 if "direction_family_3d" in events.columns
                 else pd.Series("", index=events.index))
    direction = [d if d and d != "nan" else live_direction_id(row)
                 for d, (_, row) in zip(direction, events.iterrows())]

    out = pd.DataFrame({
        "composition": sim.composition,
        "nvac": sim.n_vacancies,
        "temp": sim.temperature,
        "sim_label": sim_label(sim),
        "sim_path": str(sim.path),
        "idx_ref": events["idx_ref"].astype(int).to_numpy(),
        "energy_barrier": events["energy_barrier"].astype(float).to_numpy(),
        "move_type": events["move_type"].astype(str).to_numpy(),
        "family_id": events.get("family_id", pd.Series("", index=events.index))
                           .fillna("").astype(str).to_numpy(),
        "family_bucket_id": events.get("family_bucket_id",
                                       pd.Series("", index=events.index))
                                  .fillna("").astype(str).to_numpy(),
        "direction_family_3d": direction,
        "move_dist": np.linalg.norm(disp, axis=1),
        "move_dz": np.abs(disp[:, 2]),
        "move_dz_signed": disp[:, 2],
    })
    return out.dropna(subset=["move_dist", "move_dz"]).reset_index(drop=True)


def numeric_scatter_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of numeric columns available for scatter axes."""
    exclude = {"idx_ref", "nvac", "temp", "_audited"}
    return sorted(c for c in df.select_dtypes(include="number").columns if c not in exclude)


def live_direction_id(event_row: pd.Series) -> str:
    """Cartesian direction classifier computed from the event geometry.

    Falls back to the canonical ``classify_direction_xyz`` rule on the
    most-moved atom's initial→final displacement vector. Used when
    ``direction_family_3d`` is missing from the classifications CSV (e.g.
    legacy exports) or for the 3D overlay's own use.
    """
    try:
        init = np.asarray(event_row["initial_positions"], dtype=float)
        final = np.asarray(event_row["final_positions"], dtype=float)
        mover = int(event_row["move_atom_idx"])
    except (KeyError, TypeError, ValueError):
        return "unresolved"

    from Analysis.event_viewer.scatter_palette import NN1_DIST  # noqa: PLC0415
    from Analysis.lattice_3d import classify_direction_xyz  # noqa: PLC0415

    dx, dy, dz = final[mover] - init[mover]
    return classify_direction_xyz(float(dx), float(dy), float(dz), float(NN1_DIST))


def load_events(sim: SimInfo, classifications: pd.DataFrame) -> pd.DataFrame:
    """Load a run's `reference_table.pickle` and attach `move_type`.

    Caches per-run so re-selecting a run is free. If the pickle is missing or
    cannot be unpickled, returns an empty DataFrame (caller decides what to
    display).
    """
    key = str(sim.path)
    if key in _EVENT_CACHE:
        return _EVENT_CACHE[key]

    pickle_path = sim.path / "reference_table.pickle"
    try:
        events = _load_pickle(pickle_path)
    except (FileNotFoundError, pickle.UnpicklingError, NotImplementedError, EOFError) as exc:
        print(f"[event_viewer] WARN: could not load {pickle_path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()

    events = events.copy()

    # Join move_type. The classifier CSV's `sim_label` is not composition-aware
    # (e.g. "4vac_T500" collapses 100Ni, 95Ni_5Cr, 95Ni_5Fe into one label), so
    # matching on sim_label alone mis-attributes move_types across compositions.
    # Disambiguate by also requiring the `energy_barrier` to match within a tiny
    # tolerance, which is unique per event within each run.
    full_label = sim_label(sim)
    short_label = f"{sim.n_vacancies}vac_T{int(sim.temperature)}"
    sub = classifications[classifications["sim_label"].isin([full_label, short_label])]
    if sub.empty and {"nvac", "temp"}.issubset(classifications.columns):
        sub = classifications[
            (classifications["nvac"] == sim.n_vacancies)
            & (classifications["temp"] == sim.temperature)
        ]

    if sub.empty:
        print(f"[event_viewer] no classifications for {full_label}")
        events["move_type"] = "unclassified"
        for col in ("family_id", "family_bucket_id", "assignment_status"):
            events[col] = ""
    else:
        attached = _assign_classifier_fields(events, sub)
        for col, series in attached.items():
            events[col] = series

    _EVENT_CACHE[key] = events
    return events


_CLASSIFIER_JOIN_FIELDS = (
    "move_type", "family_id", "family_bucket_id", "assignment_status",
    "direction_family_3d",
)


def _assign_classifier_fields(
    events: pd.DataFrame, classifier_sub: pd.DataFrame, tol: float = 1e-6
) -> dict[str, pd.Series]:
    """Return {field: Series} aligned to ``events`` matching on (idx_ref, energy_barrier).

    If the classifier CSV mixes compositions under the same sim_label, idx_ref
    alone is ambiguous. Use the energy_barrier as a tiebreaker: within a single
    run, barriers are distinct at double precision. Missing fields default to
    ``"unclassified"`` for ``move_type`` and ``""`` for the others.
    """
    has_barrier = "energy_barrier" in classifier_sub.columns
    defaults = {f: ("unclassified" if f == "move_type" else "")
                for f in _CLASSIFIER_JOIN_FIELDS}
    results: dict[str, pd.Series] = {
        f: pd.Series([defaults[f]] * len(events), index=events.index, dtype=object)
        for f in _CLASSIFIER_JOIN_FIELDS
    }

    # Group classifier candidates by idx_ref for O(1) lookup.
    grouped: dict[int, pd.DataFrame] = {
        int(k): g for k, g in classifier_sub.groupby("idx_ref")
    }

    for pos, row in enumerate(events.itertuples(index=False)):
        candidates = grouped.get(int(row.idx_ref))
        if candidates is None or candidates.empty:
            continue
        if has_barrier and len(candidates) > 1:
            diffs = (candidates["energy_barrier"] - float(row.energy_barrier)).abs()
            idx = int(diffs.argmin())
            if diffs.min() > tol:
                continue
            best = candidates.iloc[idx]
        else:
            best = candidates.iloc[0]
        for field in _CLASSIFIER_JOIN_FIELDS:
            if field in candidates.columns:
                val = best.get(field, defaults[field])
                if pd.isna(val):
                    val = defaults[field]
                results[field].iat[pos] = str(val)
    return results


def event_dropdown_label(row: pd.Series) -> str:
    """Human-friendly one-line summary for the event dropdown."""
    return f"#{int(row['idx_ref'])} — {float(row['energy_barrier']):.3f} eV — {row['move_type']}"
