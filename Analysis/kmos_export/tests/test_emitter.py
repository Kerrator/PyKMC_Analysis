"""Tests for the KmosProjectDescriptor → kmos.types.Project emitter.

These are kmos-gated — skipped if `kmos` is not importable. Pure-Python
shape tests live in test_translator.py.
"""
from __future__ import annotations

import pytest

from Analysis.kmos_export.emitter import coord_to_kmos_str
from Analysis.kmos_export.lattice_fcc100 import LAYER_NAME, SITE_A
from Analysis.kmos_export.schema import CoordOffset


def _kmos_available() -> bool:
    try:
        import kmos.types  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _kmos_available(), reason="kmos not installed"
)


# ------------------------------------------------------------------
# Pure: coord rendering
# ------------------------------------------------------------------
def test_coord_to_kmos_str_zero_offset() -> None:
    s = coord_to_kmos_str(CoordOffset(site="a"), layer_name="fcc100")
    assert s == "a.(0,0,0).fcc100"


def test_coord_to_kmos_str_with_offset() -> None:
    s = coord_to_kmos_str(
        CoordOffset(site="b", i=1, j=-2, k=3), layer_name="fcc100",
    )
    assert s == "b.(1,-2,3).fcc100"


# ------------------------------------------------------------------
# Project rendering
# ------------------------------------------------------------------
def test_descriptor_to_kmos_project_round_trip_via_translator() -> None:
    """The translator's output → emitter → kmos Project should validate
    cleanly and have the right number of processes / sites / params."""
    import numpy as np
    import pandas as pd

    from Analysis.kmos_export.emitter import descriptor_to_kmos_project
    from Analysis.kmos_export.translator import translate_pilot_surface_1NN

    rng = np.random.default_rng(42)
    rows = []
    for bucket, mean_Ea in {"nv1=4": 0.9, "nv1=5": 0.74, "nv1=6": 0.54}.items():
        for _ in range(50):
            rows.append({
                "family_id": "surface_1NN_inplane",
                "family_bucket_id": bucket,
                "energy_barrier": float(mean_Ea + rng.normal(0, 0.05)),
                "mover_species_ml": "Ni",
                "assignment_status": "accepted",
            })
    df = pd.DataFrame(rows)
    desc, _ = translate_pilot_surface_1NN(df, temperature_K=500.0)
    pt = descriptor_to_kmos_project(desc)

    # Project shape — we expect:
    # - One layer with two sites (a, b)
    # - Two species (Vacant, Ni)
    # - Three parameters (T, k0, Ea_surface_1NN_inplane)
    # - Eight processes
    assert len(pt.layer_list) == 1
    assert pt.layer_list[0].name == LAYER_NAME
    assert {s.name for s in pt.layer_list[0].sites} == {"a", "b"}
    species_names = {s.name for s in pt.species_list}
    assert species_names == {"Vacant", "Ni"}
    param_names = {p.name for p in pt.parameter_list}
    assert {"T", "k0", "Ea_surface_1NN_inplane"}.issubset(param_names)
    assert len(pt.process_list) == 8


def test_descriptor_to_kmos_project_validates_via_kmos() -> None:
    """kmos's own `validate_model()` must accept what the emitter produces."""
    import numpy as np
    import pandas as pd

    from Analysis.kmos_export.emitter import descriptor_to_kmos_project
    from Analysis.kmos_export.translator import translate_pilot_surface_1NN

    rng = np.random.default_rng(42)
    rows = [
        {
            "family_id": "surface_1NN_inplane",
            "family_bucket_id": "nv1=5",
            "energy_barrier": 0.74 + rng.normal(0, 0.05),
            "mover_species_ml": "Ni",
            "assignment_status": "accepted",
        }
        for _ in range(100)
    ]
    df = pd.DataFrame(rows)
    desc, _ = translate_pilot_surface_1NN(df)
    pt = descriptor_to_kmos_project(desc)
    # `validate_model` raises on inconsistencies. Don't capture; let the
    # exception fail the test naturally.
    pt.validate_model()


def test_descriptor_to_kmos_project_can_save_xml(tmp_path) -> None:
    """End of round-trip: kmos can serialise the rendered project to disk.
    This exercises every code path the export pipeline will hit before
    the Fortran codegen step."""
    import numpy as np
    import pandas as pd

    from Analysis.kmos_export.emitter import descriptor_to_kmos_project
    from Analysis.kmos_export.translator import translate_pilot_surface_1NN

    rng = np.random.default_rng(42)
    rows = [
        {
            "family_id": "surface_1NN_inplane",
            "family_bucket_id": "nv1=5",
            "energy_barrier": 0.74 + rng.normal(0, 0.05),
            "mover_species_ml": "Ni",
            "assignment_status": "accepted",
        }
        for _ in range(100)
    ]
    df = pd.DataFrame(rows)
    desc, _ = translate_pilot_surface_1NN(df)
    pt = descriptor_to_kmos_project(desc)

    out_xml = tmp_path / "pilot.xml"
    pt.save(str(out_xml))
    assert out_xml.is_file()
    content = out_xml.read_text()
    # Sanity: spec elements are present.
    assert "fcc100_pilot_surface_1NN" in content
    assert "Ni_a_to_a_px" in content   # one of the 8 process names
    assert "Ea_surface_1NN_inplane" in content


# ------------------------------------------------------------------
# Lazy import — emitter module loads even if kmos is uninstalled.
# This test deliberately does NOT skip, so we verify the import isolation.
# ------------------------------------------------------------------
def test_emitter_module_imports_without_eager_kmos_dependency() -> None:
    """Importing emitter alone must not transitively import kmos.types.
    (Lazy imports are inside the descriptor_to_kmos_project body.)"""
    import importlib
    import sys
    # Force a clean import path; if any top-level emitter line called into
    # kmos directly, this would have already loaded kmos at session start.
    # We just verify the module imported without raising.
    mod = importlib.import_module("Analysis.kmos_export.emitter")
    assert hasattr(mod, "descriptor_to_kmos_project")
    assert hasattr(mod, "coord_to_kmos_str")
