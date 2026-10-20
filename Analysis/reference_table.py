"""Column access for rows of a pyKMC ``reference_table.pickle``.

pyKMC renamed the per-atom element column: older reference tables call it
``initial_types``, newer ones ``types``. Read it through ``event_types`` rather
than indexing either name directly. (``blender_viewer/data.py`` keeps its own
copy of this fallback because it is imported inside Blender.)
"""

from __future__ import annotations

TYPES_COLUMNS = ("initial_types", "types")


def event_types(row):
    """Return the per-atom element types of one event row, or ``None`` if absent.

    ``row`` is a ``pd.Series`` (or any mapping with ``.get``). The first of
    ``TYPES_COLUMNS`` holding a non-null value wins.
    """
    for col in TYPES_COLUMNS:
        value = row.get(col)
        if value is not None:
            return value
    return None
