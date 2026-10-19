"""Unit tests for the 3D lattice helpers in Analysis/lattice_3d.py."""
from __future__ import annotations

import numpy as np
import pytest

from Analysis.lattice_3d import (
    assign_3d_site_classes,
    assign_direction_family,
    compute_3d_environment_descriptor,
    compute_3d_site_graph,
    extract_3d_occupancy_transition,
)
from Analysis.tests.fixtures import make_tiny_fcc100_slab


def test_compute_3d_site_graph_shape():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab()
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    assert len(g) == 36
    for _, info in g.items():
        assert set(info.keys()) == {"position", "layer_index", "1NN_3d", "2NN_3d"}
        assert info["position"].shape == (3,)


def test_compute_3d_site_graph_symmetry():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab()
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    for i, info in g.items():
        for j in info["1NN_3d"]:
            assert i in g[j]["1NN_3d"], f"Asymmetric: {i} in {j}'s 1NN but not reverse"


def test_compute_3d_site_graph_layers():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab(nlayers=2)
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    layers = {info["layer_index"] for info in g.values()}
    assert layers == {0, 1}


def test_assign_3d_site_classes_2layer():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab(nlayers=2)
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    classes = assign_3d_site_classes(g)
    assert all(c in {"surface", "subsurface"} for c in classes.values())
    assert "bulk_like" not in classes.values()


def test_assign_3d_site_classes_3layer():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab(nlayers=3)
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    classes = assign_3d_site_classes(g)
    assert set(classes.values()) == {"surface", "subsurface", "bulk_like"}


def test_extract_occupancy_transition_simple():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab()
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    occ = np.ones(len(g), dtype=int)
    occ[5] = 0  # vacancy at site 5
    row = {"vac_origin_site": 5, "vac_dest_site": 8}
    t = extract_3d_occupancy_transition(row, g, occ)
    assert t["changed_sites"] == {5, 8}
    assert t["occ_init"][5] == 0 and t["occ_final"][5] == 1
    assert t["occ_init"][8] == 1 and t["occ_final"][8] == 0
    assert t["vac_origin_3d"] == 5 and t["vac_dest_3d"] == 8


def test_extract_occupancy_transition_missing_fields():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab()
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    occ = np.ones(len(g), dtype=int)
    row = {}  # no vac_origin_site, no vac_dest_site
    t = extract_3d_occupancy_transition(row, g, occ)
    assert t["changed_sites"] == set()
    assert t["vac_origin_3d"] is None and t["vac_dest_3d"] is None


def test_compute_3d_env_descriptor_isolated_vac():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab(nlayers=3)
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    occ = np.ones(len(g), dtype=int)
    top = max(info["layer_index"] for info in g.values())
    origin = next(i for i, info in g.items() if info["layer_index"] == top)
    dest = g[origin]["1NN_3d"][0]
    occ[origin] = 0
    env = compute_3d_environment_descriptor(origin, dest, occ, g)
    assert env["n_vacant_1nn_3d"] == 0  # no other vacancies
    assert env["site_class"] in {"surface", "subsurface", "bulk_like"}
    assert "n_vacant_above" in env and "n_vacant_below" in env


def test_compute_3d_env_descriptor_counts_cluster():
    ideal_sites, cell, nn_dist = make_tiny_fcc100_slab(nlayers=3)
    g = compute_3d_site_graph(ideal_sites, cell, nn_dist)
    occ = np.ones(len(g), dtype=int)
    top = max(info["layer_index"] for info in g.values())
    origin = next(i for i, info in g.items() if info["layer_index"] == top)
    nn_of_origin = g[origin]["1NN_3d"]
    dest = nn_of_origin[0]
    occ[origin] = 0
    # Add two more vacancies in origin's 1NN shell (other than dest)
    extra = [j for j in nn_of_origin if j != dest][:2]
    for j in extra:
        occ[j] = 0
    env = compute_3d_environment_descriptor(origin, dest, occ, g)
    assert env["n_vacant_1nn_3d"] == 2
    assert env["n_vacant_above"] + env["n_vacant_below"] <= 2


@pytest.mark.parametrize(
    "di,dj,dz,expected",
    [
        (1, 0, 0, "<110>_inplane"),
        (-1, 0, 0, "<110>_inplane"),
        (0, 1, 0, "<110>_inplane"),
        (1, 1, 0, "<100>_inplane"),
        (-1, 1, 0, "<100>_inplane"),
        (1, 0, 1, "<111>_interlayer"),
        (0, -1, -1, "<111>_interlayer"),
        (0, 0, 1, "<001>_exchange"),
        (0, 0, -1, "<001>_exchange"),
        (2, 0, 0, "unresolved"),
    ],
)
def test_assign_direction_family(di, dj, dz, expected):
    assert assign_direction_family(di, dj, dz) == expected
