"""Per-event camera defaults — disk-backed storage.

Lets the user click "Lock view" in the 3D viewer to persist the current
Plotly camera (eye, center, up) for a specific event. The next time
anyone opens that event the camera opens at the saved angle instead of
Plotly's auto-fit.

Storage: ``view_defaults.json`` in the output directory
(``PYKMC_ANALYSIS_OUTPUT_DIR``, default ``Analysis/lattice_event_classification/``;
see ``Analysis._paths``), keyed by ``f"{composition}|{nvac}|{temp:.1f}|{idx_ref}"``
(matches the audit_log primary key shape but joined as a single string for JSON).

Concurrency model: single-user app, so we read the whole file on every
write and re-serialise. Atomic write via temp-file rename.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Analysis._paths import CLASSIFICATION_DIR

# Sits alongside audit_log.csv in the output directory (shares the
# (comp, nvac, temp, idx_ref) key with the audit log). Per-user state, not
# tracked by git.
VIEW_DEFAULTS_PATH: Path = CLASSIFICATION_DIR / "view_defaults.json"

# In-process cache so render_graph doesn't hit the disk on every event load.
# Invalidated on save_default / delete_default.
_CACHE: dict[str, dict[str, Any]] | None = None


def _key(composition: str, nvac: int, temp: float, idx_ref: int) -> str:
    """Build the canonical storage key for a (run, event) tuple."""
    return f"{composition}|{int(nvac)}|{float(temp):.1f}|{int(idx_ref)}"


def _load_raw() -> dict[str, dict[str, Any]]:
    if not VIEW_DEFAULTS_PATH.exists():
        return {}
    try:
        text = VIEW_DEFAULTS_PATH.read_text()
        if not text.strip():
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        # Corrupt file is non-fatal — log and start fresh. Backing up the
        # corrupt file would surprise the user; let them notice via missing
        # defaults if it's catastrophic.
        return {}


def load_all_defaults() -> dict[str, dict[str, Any]]:
    """Read the whole view_defaults.json. Cached after first call."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_raw()
    return _CACHE


def get_default(
    composition: str, nvac: int, temp: float, idx_ref: int,
) -> dict[str, Any] | None:
    """Return the saved camera dict (eye/center/up) or None if absent."""
    return load_all_defaults().get(_key(composition, nvac, temp, idx_ref))


def save_default(
    composition: str,
    nvac: int,
    temp: float,
    idx_ref: int,
    eye: dict[str, float],
    center: dict[str, float] | None,
    up: dict[str, float] | None,
) -> None:
    """Persist a camera as the default for this event. Atomic write."""
    global _CACHE
    data = _load_raw()
    record: dict[str, Any] = {
        "eye":    {k: float(v) for k, v in eye.items()},
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if center is not None:
        record["center"] = {k: float(v) for k, v in center.items()}
    if up is not None:
        record["up"] = {k: float(v) for k, v in up.items()}
    data[_key(composition, nvac, temp, idx_ref)] = record

    # Atomic write: temp file in same dir, then rename.
    VIEW_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="view_defaults.", suffix=".json.tmp",
        dir=str(VIEW_DEFAULTS_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, VIEW_DEFAULTS_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _CACHE = data


def delete_default(
    composition: str, nvac: int, temp: float, idx_ref: int,
) -> bool:
    """Remove a saved default. Returns True if a record was removed."""
    global _CACHE
    data = _load_raw()
    key = _key(composition, nvac, temp, idx_ref)
    if key not in data:
        return False
    del data[key]
    VIEW_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="view_defaults.", suffix=".json.tmp",
        dir=str(VIEW_DEFAULTS_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, VIEW_DEFAULTS_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _CACHE = data
    return True
