"""kmos_export — translate the curated FCC event catalogue into kmos OTF Process descriptions.

Experimental. Pipeline:
    classified_events_with_families.csv      (curated catalogue)
              │
              ▼  translator.py
    list[ProcessDescriptor]                  (intermediate IR — schema.py)
              │
              ▼  emitter.py
    kmos project file (XML)                  (kmos's native form)
              │
              ▼  kmos's own export-to-fortran
    compiled kmos binary

Public API:
    from Analysis.kmos_export import (
        ProcessDescriptor, ConditionDescriptor, ActionDescriptor,
        BystanderDescriptor, CoordOffset, SiteDescriptor,
        LatticeDescriptor, KmosProjectDescriptor,
    )

Implemented so far: translator + emitter for the `surface_1NN_inplane`
pilot. Not yet implemented: per-bucket OTF barriers, the full catalogue, and
bystander-aware OTF processes.
"""
from __future__ import annotations

from Analysis.kmos_export.lattice_fcc100 import (
    A_LAT_DEFAULT,
    LAYER_NAME,
    SITE_A,
    SITE_B,
    build_layer_descriptor,
    first_neighbours,
    populate_kmos_project,
)
from Analysis.kmos_export.schema import (
    ActionDescriptor,
    BystanderDescriptor,
    ConditionDescriptor,
    CoordOffset,
    KmosProjectDescriptor,
    LatticeDescriptor,
    ProcessDescriptor,
    SiteDescriptor,
)
from Analysis.kmos_export.translator import (
    FamilyEaSummary,
    emit_in_plane_processes,
    summarize_family,
    translate_pilot_surface_1NN,
)

__all__ = [
    "ActionDescriptor",
    "A_LAT_DEFAULT",
    "BystanderDescriptor",
    "ConditionDescriptor",
    "CoordOffset",
    "FamilyEaSummary",
    "KmosProjectDescriptor",
    "LAYER_NAME",
    "LatticeDescriptor",
    "ProcessDescriptor",
    "SITE_A",
    "SITE_B",
    "SiteDescriptor",
    "build_layer_descriptor",
    "emit_in_plane_processes",
    "first_neighbours",
    "populate_kmos_project",
    "summarize_family",
    "translate_pilot_surface_1NN",
]
