"""Both reference-table schemas (``initial_types`` and newer ``types``) are read."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Analysis.event_viewer.plots import make_event_figure
from Analysis.lattice_map_events import classify_event_from_row
from Analysis.reference_table import event_types
from Analysis.tests.fixtures import make_tiny_fcc100_slab


def _event_row(types_column: str) -> pd.Series:
    sites, _cell, nn_dist = make_tiny_fcc100_slab()
    init = sites
    final = init.copy()
    final[0] += [nn_dist, 0.0, 0.0]
    types = np.array(["Fe"] + ["Ni"] * (len(init) - 1))
    return pd.Series({
        "idx_ref": 0,
        "energy_barrier": 0.6,
        "initial_positions": init,
        "saddle_positions": (init + final) / 2,
        "final_positions": final,
        "move_atom_idx": 0,
        types_column: types,
    })


@pytest.mark.parametrize("column", ["initial_types", "types"])
def test_event_types_reads_either_column(column):
    assert list(event_types(_event_row(column)))[:2] == ["Fe", "Ni"]


def test_event_types_missing_is_none():
    assert event_types(pd.Series({"idx_ref": 0})) is None


@pytest.mark.parametrize("column", ["initial_types", "types"])
def test_event_figure_builds(column):
    make_event_figure(_event_row(column), "initial")


@pytest.mark.parametrize("column", ["initial_types", "types"])
def test_species_extracted(column):
    result = classify_event_from_row(_event_row(column), include_species=True)
    assert result["mover_species"] == "Fe"
