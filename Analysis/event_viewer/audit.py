"""CSV-backed audit log for ML-masking validation.

Each row records a human verdict on one classified event identified by
``(composition, nvac, temp, idx_ref)``. Re-saving the same key upserts.

The log is ``audit_log.csv`` in the output directory (``PYKMC_ANALYSIS_OUTPUT_DIR``,
default ``Analysis/lattice_event_classification/``; see ``Analysis._paths``). It
is per-user state, not tracked by git; a missing file reads as an empty log and
is created on the first save.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from Analysis._paths import AUDIT_CSV

KEY_COLS = ["composition", "nvac", "temp", "idx_ref"]
ALL_COLS = KEY_COLS + [
    "original_move_type", "verdict", "override_move_type",
    "override_family_id", "override_bucket_id", "override_direction_id",
    "notes", "updated_at",
    # render_flag: orthogonal to verdict. "1" = include in the curated
    # animation showreel (Analysis.blender_viewer.showreel.select_curated).
    # "" or absent = not flagged. Coexists with any verdict value.
    "render_flag",
]


def _truthy(val: object) -> bool:
    """Normalise a render_flag input to bool. Handles "1"/"0"/True/False/etc."""
    if val is True:
        return True
    if val is False or val is None:
        return False
    s = str(val).strip().lower()
    if s in {"", "nan", "none", "0", "0.0", "false", "no", "n", "off"}:
        return False
    if s in {"1", "1.0", "true", "yes", "y", "on", "render"}:
        return True
    # Numeric fallback: any non-zero number is truthy.
    try:
        return float(s) != 0.0
    except ValueError:
        return False


def load_audit() -> pd.DataFrame:
    """Read the audit CSV; return an empty frame if the file does not exist.

    Missing columns (from pre-family audit logs) are added with empty strings
    so callers can index into ``override_family_id`` / ``override_bucket_id``
    unconditionally.
    """
    if not AUDIT_CSV.exists():
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_csv(AUDIT_CSV)
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = "" if col in (
                "override_family_id", "override_bucket_id",
                "override_direction_id", "render_flag",
            ) else pd.NA
    # Normalise render_flag to "1"/"" strings — pandas auto-coerces a column
    # with mixed empty + "1" entries to float64 (1.0/NaN), which breaks
    # ``str(val) == "1"`` comparisons downstream.
    df["render_flag"] = (
        df["render_flag"].fillna("").astype(str)
        .map(lambda s: "1" if _truthy(s) else "")
    )
    return df[ALL_COLS]


def save_audit_row(
    composition: str,
    nvac: int,
    temp: float,
    idx_ref: int,
    original_move_type: str,
    verdict: str,
    override_move_type: str | None,
    notes: str | None,
    override_family_id: str | None = None,
    override_bucket_id: str | None = None,
    override_direction_id: str | None = None,
    render_flag: bool | str | None = None,
) -> pd.DataFrame:
    """Upsert one audit row keyed by (composition, nvac, temp, idx_ref).

    ``render_flag``:
    - ``None`` → preserve the existing value (or default to "" for new rows).
    - truthy (``True``, ``"1"``, ``1``) → write ``"1"``.
    - falsy   (``False``, ``""``, ``0``) → write ``""``.
    """
    df = load_audit()
    mask = (
        (df["composition"].astype(str) == str(composition))
        & (df["nvac"].astype(int) == int(nvac))
        & (df["temp"].astype(float) == float(temp))
        & (df["idx_ref"].astype(int) == int(idx_ref))
    )
    # Resolve render_flag: None preserves existing, otherwise normalise.
    if render_flag is None:
        if mask.any():
            render_flag_str = "1" if _truthy(df.loc[mask, "render_flag"].iloc[0]) else ""
        else:
            render_flag_str = ""
    else:
        render_flag_str = "1" if _truthy(render_flag) else ""
    new_row = {
        "composition": composition,
        "nvac": int(nvac),
        "temp": float(temp),
        "idx_ref": int(idx_ref),
        "original_move_type": original_move_type or "",
        "verdict": verdict,
        "override_move_type": override_move_type or "",
        "override_family_id": override_family_id or "",
        "override_bucket_id": override_bucket_id or "",
        "override_direction_id": override_direction_id or "",
        "notes": notes or "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "render_flag": render_flag_str,
    }
    if mask.any():
        for col, val in new_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT_CSV, index=False)
    return df




def save_audit_bulk(
    events: list[tuple[str, int, float, int, str]],
    verdict: str,
    override_move_type: str | None,
    notes: str | None,
    override_family_id: str | None = None,
    override_bucket_id: str | None = None,
    override_direction_id: str | None = None,
    render_flag: bool | str | None = None,
) -> int:
    """Upsert the same verdict/overrides/notes for a batch of events.

    ``events`` is a list of ``(composition, nvac, temp, idx_ref, original_move_type)``.
    Returns the number of rows written.

    ``render_flag``:
    - ``None`` → preserve existing value per row.
    - truthy/falsy → apply uniformly to every event in the batch.
    """
    if not events:
        return 0
    df = load_audit()
    now = datetime.now().isoformat(timespec="seconds")
    override_val = override_move_type or ""
    family_val = override_family_id or ""
    bucket_val = override_bucket_id or ""
    direction_val = override_direction_id or ""
    notes_val = notes or ""
    flag_uniform: str | None = None
    if render_flag is not None:
        flag_uniform = "1" if _truthy(render_flag) else ""
    updates: list[dict] = []
    for composition, nvac, temp, idx_ref, original_move_type in events:
        mask = (
            (df["composition"].astype(str) == str(composition))
            & (df["nvac"].astype(int) == int(nvac))
            & (df["temp"].astype(float) == float(temp))
            & (df["idx_ref"].astype(int) == int(idx_ref))
        )
        # Resolve flag value for this row.
        if flag_uniform is not None:
            row_flag = flag_uniform
        elif mask.any():
            row_flag = "1" if _truthy(df.loc[mask, "render_flag"].iloc[0]) else ""
        else:
            row_flag = ""
        row = {
            "composition": composition,
            "nvac": int(nvac),
            "temp": float(temp),
            "idx_ref": int(idx_ref),
            "original_move_type": original_move_type or "",
            "verdict": verdict,
            "override_move_type": override_val,
            "override_family_id": family_val,
            "override_bucket_id": bucket_val,
            "override_direction_id": direction_val,
            "notes": notes_val,
            "updated_at": now,
            "render_flag": row_flag,
        }
        if mask.any():
            for col, val in row.items():
                df.loc[mask, col] = val
        else:
            updates.append(row)
    if updates:
        df = pd.concat([df, pd.DataFrame(updates)], ignore_index=True)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT_CSV, index=False)
    return len(events)


def flagged_keys(df: pd.DataFrame | None = None) -> set[tuple[str, int, int, int]]:
    """Return the set of (composition, nvac, temp, idx_ref) tuples flagged for animation."""
    if df is None:
        df = load_audit()
    if df.empty or "render_flag" not in df.columns:
        return set()
    flagged = df[df["render_flag"].apply(_truthy)]
    return {
        (str(r.composition), int(r.nvac), int(float(r.temp)), int(r.idx_ref))
        for r in flagged.itertuples(index=False)
    }


def audited_keys(df: pd.DataFrame | None = None) -> set[tuple[str, int, int, int]]:
    """Return the set of (composition, nvac, temp, idx_ref) tuples already audited.

    Temperatures are normalized to ``int`` to match the integer values used in
    the classifier export and the SimInfo cascade.
    """
    if df is None:
        df = load_audit()
    if df.empty:
        return set()
    return {
        (str(r.composition), int(r.nvac), int(float(r.temp)), int(r.idx_ref))
        for r in df.itertuples(index=False)
    }
