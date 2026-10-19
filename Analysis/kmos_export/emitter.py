"""KmosProjectDescriptor → kmos.types.Project rendering.

This module imports kmos lazily so the rest of the kmos_export package
remains testable without the kmos Fortran toolchain. Call
`descriptor_to_kmos_project()` only when kmos is available.

The rendering is mechanical (the IR is intentionally kmos-shaped):

    LatticeDescriptor    → pt.lattice cell_size + Layer with Sites
    species (tuple)      → pt.add_species() per species
    parameters (dict)    → pt.add_parameter() per name
    ProcessDescriptor    → pt.add_process(name, condition_list, action_list,
                                          bystander_list, rate_constant)

Bystanders + otf_rate are emitted only if the descriptor has bystanders
(the pilot has none — local_smart with rate_constant). With bystanders,
the emitter switches to `otf_rate=` and `bystander_list=`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from Analysis.kmos_export.lattice_fcc100 import populate_kmos_project
from Analysis.kmos_export.schema import (
    BystanderDescriptor,
    ConditionDescriptor,
    CoordOffset,
    KmosProjectDescriptor,
    ProcessDescriptor,
)

if TYPE_CHECKING:
    from kmos.types import Project as KmosProject  # type: ignore[import-not-found]


# ------------------------------------------------------------------
# kmos coord notation
# ------------------------------------------------------------------
def coord_to_kmos_str(coord: CoordOffset, *, layer_name: str) -> str:
    """Render an IR coord as a kmos coord string: 'site.(i,j,k).layer'.

    kmos's `pt.lattice.generate_coord('a.(1,0,0).fcc100')` is the canonical
    form — site name, optional offset tuple, layer name. We always emit the
    full triple even when the offset is zero, because kmos's coord parser
    handles both with the same code path.
    """
    return f"{coord.site}.({coord.i},{coord.j},{coord.k}).{layer_name}"


# ------------------------------------------------------------------
# Project-level emit
# ------------------------------------------------------------------
def _add_species(pt: "KmosProject", species_tuple: tuple[str, ...]) -> None:
    """kmos requires `pt.add_species()` per species name. The first species
    is treated as the lattice's default; we set that explicitly via
    `populate_kmos_project()` and don't need to make kmos guess."""
    for name in species_tuple:
        # kmos accepts an optional ASE-Atoms representation; for our purposes
        # bare species (no atomic detail) is sufficient. The pilot only ever
        # calls `model.proclist.<name_lower>` to read enum codes.
        repr_str = ""
        if name == "Ni":
            repr_str = "Atoms('Ni')"
        elif name == "Fe":
            repr_str = "Atoms('Fe')"
        elif name == "Cr":
            repr_str = "Atoms('Cr')"
        pt.add_species(name=name, representation=repr_str)


def _add_parameters(pt: "KmosProject", parameters: dict[str, float]) -> None:
    """One kmos parameter per IR parameter. All non-adjustable except `T`,
    which we expose as adjustable so users can override it without rebuilding
    the kmos Fortran source."""
    for name, value in parameters.items():
        kwargs: dict[str, Any] = {"name": name, "value": float(value)}
        if name == "T":
            kwargs.update(adjustable=True, min=100.0, max=2000.0)
        pt.add_parameter(**kwargs)


def _add_process(
    pt: "KmosProject", proc: ProcessDescriptor, layer_name: str,
) -> None:
    """One kmos `pt.add_process(...)` call. Switches between local_smart
    (rate_constant) and OTF (otf_rate + bystander_list) based on whether
    the descriptor has bystanders."""
    from kmos.types import Action, Bystander, Condition  # type: ignore[import-not-found]

    def _coord_obj(co: CoordOffset) -> Any:
        return pt.lattice.generate_coord(coord_to_kmos_str(co, layer_name=layer_name))

    cond_list = [
        Condition(coord=_coord_obj(c.coord), species=c.species)
        for c in proc.conditions
    ]
    action_list = [
        Action(coord=_coord_obj(a.coord), species=a.species)
        for a in proc.actions
    ]

    add_kwargs: dict[str, Any] = {
        "name": proc.name,
        "condition_list": cond_list,
        "action_list": action_list,
    }
    if proc.bystanders:
        # OTF backend: rate_formula → otf_rate, bystander_list populated.
        bystander_list = [
            Bystander(
                coord=_coord_obj(b.coord),
                allowed_species=list(b.allowed_species),
                flag=b.flag,
            )
            for b in proc.bystanders
        ]
        add_kwargs["bystander_list"] = bystander_list
        add_kwargs["otf_rate"] = proc.rate_formula
    else:
        # local_smart backend: rate_formula → rate_constant.
        add_kwargs["rate_constant"] = proc.rate_formula

    pt.add_process(**add_kwargs)


def descriptor_to_kmos_project(
    desc: KmosProjectDescriptor,
    *,
    a_lat: float | None = None,
    author: str = "kmos_export",
    email: str = "noreply@example.com",
) -> "KmosProject":
    """Build a fresh kmos `Project` from an IR descriptor.

    Imports kmos lazily so callers in CI without the Fortran toolchain
    can build descriptors and round-trip them through JSON without paying
    for the kmos import.
    """
    from kmos.types import Project  # type: ignore[import-not-found]

    pt = Project()
    pt.meta.author = author
    pt.meta.email = email
    pt.meta.model_name = desc.name
    pt.meta.model_dimension = 3

    _add_species(pt, desc.species)

    # Lattice goes after species so the layer's `default_species` references
    # are valid. We trust the IR: the descriptor's lattice and the default
    # species ('Vacant') are pre-validated by KmosProjectDescriptor's own
    # cross-reference checks.
    layer_name = desc.lattice.name
    if a_lat is None:
        # Recover a_lat from cell_c (third row of LatticeDescriptor).
        a_lat = float(desc.lattice.cell_c[2])
    populate_kmos_project(
        pt, a_lat=a_lat, layer_name=layer_name,
        default_species=desc.species[0],
    )

    _add_parameters(pt, desc.parameters)
    for proc in desc.processes:
        _add_process(pt, proc, layer_name=layer_name)

    return pt
