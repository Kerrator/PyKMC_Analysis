"""Parametrized tests for classify_3d_motif — event type to motif family mapping."""
from __future__ import annotations

import pytest

from Analysis.lattice_3d import classify_3d_motif


@pytest.mark.parametrize(
    "move_type,n_moved,surf_layer,expected",
    [
        ("1NN_hop", 1, 0, "surface_1nn_translation"),
        ("1NN_hop_cooperative", 2, 0, "surface_1nn_translation"),
        ("1NN_hop_triple", 3, 0, "surface_1nn_translation"),
        ("2NN_hop", 1, 0, "surface_2nn_translation"),
        ("subsurface_1NN_hop", 1, 1, "subsurface_1nn_translation"),
        ("exchange_up", 2, 0, "surface_subsurface_exchange"),
        ("exchange_down", 2, 0, "surface_subsurface_exchange"),
        ("exchange_lateral", 2, 0, "surface_subsurface_exchange"),
        ("interlayer_hop_up", 1, 0, "interlayer_translation"),
        ("interlayer_hop_down", 1, 0, "interlayer_translation"),
        ("subsurface_migration", 2, 1, "subsurface_exchange"),
        ("subsurface_other", 1, 1, "unresolved_multisite"),
        ("inplane_complex", 3, 0, "unresolved_multisite"),
        ("complex", 3, 0, "unresolved_multisite"),
        ("concerted", 4, 0, "concerted_3d"),
        ("relaxation", 0, 0, "unresolved_multisite"),
    ],
)
def test_classify_3d_motif(move_type, n_moved, surf_layer, expected):
    row = {
        "move_type": move_type,
        "n_moved": n_moved,
        "surface_layer_initial": surf_layer,
    }
    assert classify_3d_motif(row) == expected


def test_classify_3d_motif_demotes_surface_1nn_for_subsurface_origin():
    """A 1NN_hop initiated from a subsurface site should demote to subsurface_1nn_translation."""
    row = {"move_type": "1NN_hop", "n_moved": 1, "surface_layer_initial": 1}
    assert classify_3d_motif(row) == "subsurface_1nn_translation"
