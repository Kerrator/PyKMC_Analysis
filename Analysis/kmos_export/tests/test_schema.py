"""Tests for the kmos_export.schema intermediate representation.

Coverage targets:
  - Construct minimal valid descriptors
  - All identifier/format validators reject bad inputs
  - Coord uniqueness within a role enforced
  - Cross-references (site name, species, formula symbols) checked at project level
  - Multi-action processes (concerted events) supported
  - Counter-name machinery (declared vs referenced) works
  - Round-trip through JSON
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from Analysis.kmos_export.schema import (
    ActionDescriptor,
    BystanderDescriptor,
    ConditionDescriptor,
    CoordOffset,
    KmosProjectDescriptor,
    LatticeDescriptor,
    MAX_ACTIONS_PER_PROCESS,
    MAX_CONDITIONS_PER_PROCESS,
    MAX_PROCESSES,
    ProcessDescriptor,
    RESERVED_VACANT,
    SiteDescriptor,
)


# ------------------------------------------------------------------
# Helpers — minimal valid factories so each test doesn't repeat boilerplate.
# ------------------------------------------------------------------
def _site(name: str = "a", default_species: str = RESERVED_VACANT) -> SiteDescriptor:
    return SiteDescriptor(name=name, pos_frac=(0.0, 0.0, 0.0),
                          default_species=default_species)


def _lattice(*, name: str = "fcc100",
             sites: tuple[SiteDescriptor, ...] | None = None) -> LatticeDescriptor:
    return LatticeDescriptor(
        name=name,
        cell_a=(1.0, 0.0, 0.0),
        cell_b=(0.0, 1.0, 0.0),
        cell_c=(0.0, 0.0, 1.0),
        sites=sites if sites is not None else (_site(),),
    )


def _minimal_hop_process(*, name: str = "hop_Ni") -> ProcessDescriptor:
    """Single-vacancy 1NN hop in (i=1, 0, 0) direction. Vacancy at site a
    moves into site at offset (1,0,0); the Ni atom moves the other way."""
    return ProcessDescriptor(
        name=name,
        conditions=(
            ConditionDescriptor(coord=CoordOffset(site="a"), species="Vacant"),
            ConditionDescriptor(coord=CoordOffset(site="a", i=1), species="Ni"),
        ),
        actions=(
            ActionDescriptor(coord=CoordOffset(site="a"), species="Ni"),
            ActionDescriptor(coord=CoordOffset(site="a", i=1), species="Vacant"),
        ),
        rate_formula="k0 * exp(-Ea0 / kT)",
    )


def _minimal_project(**overrides) -> KmosProjectDescriptor:
    base = dict(
        name="ni_fe_cr_test",
        lattice=_lattice(),
        species=("Vacant", "Ni"),
        parameters={"k0": 1e13, "Ea0": 0.51},
        processes=(_minimal_hop_process(),),
    )
    base.update(overrides)
    return KmosProjectDescriptor(**base)


# ------------------------------------------------------------------
# CoordOffset
# ------------------------------------------------------------------
def test_coord_offset_default_renders_bare_site_name() -> None:
    c = CoordOffset(site="a")
    assert c.as_kmos_coord_str() == "a"


def test_coord_offset_with_offset_renders_kmos_form() -> None:
    c = CoordOffset(site="a", i=1, j=-2, k=3)
    assert c.as_kmos_coord_str() == "a.(1,-2,3)"


def test_coord_offset_rejects_invalid_site_name() -> None:
    with pytest.raises(ValidationError, match="simple identifier"):
        CoordOffset(site="1bad")


def test_coord_offset_rejects_double_underscore() -> None:
    with pytest.raises(ValidationError, match="simple identifier"):
        CoordOffset(site="a__b")


# ------------------------------------------------------------------
# ConditionDescriptor / ActionDescriptor
# ------------------------------------------------------------------
def test_condition_rejects_bad_species() -> None:
    with pytest.raises(ValidationError, match="simple identifier"):
        ConditionDescriptor(coord=CoordOffset(site="a"), species="Ni 4")


def test_action_rejects_bad_species() -> None:
    with pytest.raises(ValidationError, match="simple identifier"):
        ActionDescriptor(coord=CoordOffset(site="a"), species="")


# ------------------------------------------------------------------
# BystanderDescriptor
# ------------------------------------------------------------------
def test_bystander_counter_naming_matches_kmos_convention() -> None:
    """kmos generates `nr_<species>_<flag>` per io/__init__.py:2825."""
    b = BystanderDescriptor(
        coord=CoordOffset(site="a", i=1),
        allowed_species=("Cr", "Fe"),
        flag="nn1",
    )
    assert b.counter_name("Cr") == "nr_Cr_nn1"
    assert b.counter_name("Fe") == "nr_Fe_nn1"


def test_bystander_counter_name_for_unknown_species_raises() -> None:
    b = BystanderDescriptor(
        coord=CoordOffset(site="a", i=1),
        allowed_species=("Cr",),
        flag="nn1",
    )
    with pytest.raises(ValueError, match="not in this bystander"):
        b.counter_name("Fe")


def test_bystander_rejects_empty_allowed_species() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        BystanderDescriptor(
            coord=CoordOffset(site="a"), allowed_species=(), flag="nn1",
        )


def test_bystander_rejects_duplicate_allowed_species() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        BystanderDescriptor(
            coord=CoordOffset(site="a"), allowed_species=("Cr", "Cr"), flag="nn1",
        )


def test_bystander_rejects_bad_flag() -> None:
    with pytest.raises(ValidationError, match="simple identifier"):
        BystanderDescriptor(
            coord=CoordOffset(site="a"),
            allowed_species=("Cr",),
            flag="1nn",   # leading digit
        )


# ------------------------------------------------------------------
# ProcessDescriptor — invariants
# ------------------------------------------------------------------
def test_process_minimal_construct() -> None:
    p = _minimal_hop_process()
    assert len(p.conditions) == 2
    assert len(p.actions) == 2
    assert p.bystanders == ()
    assert p.referenced_counter_names() == set()


def test_process_rejects_zero_conditions() -> None:
    with pytest.raises(ValidationError, match="at least one condition"):
        ProcessDescriptor(
            name="bad",
            conditions=(),
            actions=(ActionDescriptor(
                coord=CoordOffset(site="a"), species="Ni",
            ),),
            rate_formula="1.0",
        )


def test_process_rejects_zero_actions() -> None:
    with pytest.raises(ValidationError, match="at least one action"):
        ProcessDescriptor(
            name="bad",
            conditions=(ConditionDescriptor(
                coord=CoordOffset(site="a"), species="Vacant",
            ),),
            actions=(),
            rate_formula="1.0",
        )


def test_process_rejects_duplicate_condition_coords() -> None:
    with pytest.raises(ValidationError, match="duplicate coord in conditions"):
        ProcessDescriptor(
            name="bad",
            conditions=(
                ConditionDescriptor(coord=CoordOffset(site="a"), species="Vacant"),
                ConditionDescriptor(coord=CoordOffset(site="a"), species="Ni"),
            ),
            actions=(ActionDescriptor(
                coord=CoordOffset(site="a"), species="Ni",
            ),),
            rate_formula="1.0",
        )


def test_process_allows_same_coord_across_roles() -> None:
    """Site `a` can be both a condition AND an action — that's a normal hop."""
    p = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        rate_formula="1.0",
    )
    assert p.conditions[0].coord == p.actions[0].coord


def test_process_rate_formula_must_parse() -> None:
    with pytest.raises(ValidationError, match="not a valid Python expression"):
        ProcessDescriptor(
            name="bad",
            conditions=(ConditionDescriptor(
                coord=CoordOffset(site="a"), species="Vacant",
            ),),
            actions=(ActionDescriptor(
                coord=CoordOffset(site="a"), species="Ni",
            ),),
            rate_formula="k0 * (((",  # syntax error
        )


def test_process_too_many_conditions_rejected() -> None:
    with pytest.raises(ValidationError, match="too many conditions"):
        ProcessDescriptor(
            name="bigp",
            conditions=tuple(
                ConditionDescriptor(
                    coord=CoordOffset(site="a", i=i), species="Ni",
                )
                for i in range(MAX_CONDITIONS_PER_PROCESS + 1)
            ),
            actions=(ActionDescriptor(
                coord=CoordOffset(site="a"), species="Vacant",
            ),),
            rate_formula="1.0",
        )


def test_process_concerted_multi_action_construct() -> None:
    """A concerted exchange: two atoms swap. Maps to kmos's multi-action
    pattern (multidentate.py:41-48)."""
    p = ProcessDescriptor(
        name="ss_exchange",
        conditions=(
            ConditionDescriptor(coord=CoordOffset(site="a"), species="Ni"),
            ConditionDescriptor(coord=CoordOffset(site="b"), species="Vacant"),
        ),
        actions=(
            ActionDescriptor(coord=CoordOffset(site="a"), species="Vacant"),
            ActionDescriptor(coord=CoordOffset(site="b"), species="Ni"),
        ),
        rate_formula="k0 * exp(-Ea_exchange / kT)",
        family_id="surface_subsurface_exchange_up",
    )
    assert len(p.actions) == 2
    assert {a.coord.site for a in p.actions} == {"a", "b"}


def test_process_referenced_counter_names_picks_up_nr_symbols() -> None:
    p = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        bystanders=(BystanderDescriptor(
            coord=CoordOffset(site="a", i=1),
            allowed_species=("Cr",),
            flag="nn1",
        ),),
        rate_formula="k0 * exp(-(Ea0 + dEa_Cr * nr_Cr_nn1) / kT)",
    )
    assert p.referenced_counter_names() == {"nr_Cr_nn1"}
    assert p.declared_counter_names() == {"nr_Cr_nn1"}


# ------------------------------------------------------------------
# SiteDescriptor / LatticeDescriptor
# ------------------------------------------------------------------
def test_site_pos_frac_must_be_in_unit_cell() -> None:
    with pytest.raises(ValidationError, match="out of"):
        SiteDescriptor(name="a", pos_frac=(1.0, 0.0, 0.0))


def test_lattice_rejects_duplicate_site_names() -> None:
    with pytest.raises(ValidationError, match="duplicate site names"):
        LatticeDescriptor(
            name="bad",
            cell_a=(1.0, 0.0, 0.0),
            cell_b=(0.0, 1.0, 0.0),
            cell_c=(0.0, 0.0, 1.0),
            sites=(_site("a"), _site("a")),
        )


def test_lattice_rejects_zero_sites() -> None:
    with pytest.raises(ValidationError, match="at least one site"):
        LatticeDescriptor(
            name="bad",
            cell_a=(1.0, 0.0, 0.0),
            cell_b=(0.0, 1.0, 0.0),
            cell_c=(0.0, 0.0, 1.0),
            sites=(),
        )


# ------------------------------------------------------------------
# KmosProjectDescriptor — cross-references
# ------------------------------------------------------------------
def test_minimal_project_constructs() -> None:
    proj = _minimal_project()
    assert proj.name == "ni_fe_cr_test"
    assert proj.species[0] == RESERVED_VACANT
    assert len(proj.processes) == 1


def test_project_rejects_unknown_site_in_process() -> None:
    bad_proc = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="nonexistent"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="nonexistent"), species="Ni",
        ),),
        rate_formula="1.0",
    )
    with pytest.raises(ValidationError, match="not in the lattice"):
        _minimal_project(processes=(bad_proc,))


def test_project_rejects_unknown_species_in_condition() -> None:
    bad_proc = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Mythril",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        rate_formula="1.0",
    )
    with pytest.raises(ValidationError, match="unknown species 'Mythril'"):
        _minimal_project(processes=(bad_proc,))


def test_project_rejects_unknown_species_in_bystander() -> None:
    bad_proc = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        bystanders=(BystanderDescriptor(
            coord=CoordOffset(site="a", i=1),
            allowed_species=("Mythril",),
            flag="nn1",
        ),),
        rate_formula="k0 * nr_Mythril_nn1",
    )
    with pytest.raises(ValidationError, match="unknown species 'Mythril'"):
        _minimal_project(processes=(bad_proc,))


def test_project_rejects_undeclared_formula_symbol() -> None:
    bad_proc = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        rate_formula="UndeclaredParam * exp(-Ea0 / kT)",
    )
    with pytest.raises(ValidationError, match="unknown name 'UndeclaredParam'"):
        _minimal_project(processes=(bad_proc,))


def test_project_rejects_undeclared_bystander_counter_in_formula() -> None:
    """Process references nr_Cr_nn1 but declares no bystander for Cr."""
    bad_proc = ProcessDescriptor(
        name="hop",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        rate_formula="k0 * exp(-(Ea0 + nr_Cr_nn1 * dEa_Cr) / kT)",
    )
    # `dEa_Cr` is undeclared; nr_Cr_nn1 is undeclared. Both should trip.
    with pytest.raises(ValidationError, match="unknown name"):
        _minimal_project(parameters={"k0": 1e13, "Ea0": 0.5},
                         processes=(bad_proc,))


def test_project_species_must_start_with_vacant() -> None:
    with pytest.raises(ValidationError, match=r"start with 'Vacant'"):
        _minimal_project(species=("Ni", "Vacant"))


def test_project_rejects_duplicate_process_names() -> None:
    p1 = _minimal_hop_process(name="hop")
    p2 = _minimal_hop_process(name="hop")
    with pytest.raises(ValidationError, match="duplicate process names"):
        _minimal_project(processes=(p1, p2))


def test_project_accepts_realistic_otf_process() -> None:
    """A realistic OTF process with bystanders + parametric rate, all
    cross-references declared. This is the shape a full translator emits."""
    sites = (_site("a"), _site("b", default_species="Vacant"))
    lattice = _lattice(sites=sites)
    proc = ProcessDescriptor(
        name="surface_1NN_inplane_Ni_with_Cr_bystanders",
        conditions=(
            ConditionDescriptor(coord=CoordOffset(site="a"), species="Vacant"),
            ConditionDescriptor(coord=CoordOffset(site="a", i=1), species="Ni"),
        ),
        actions=(
            ActionDescriptor(coord=CoordOffset(site="a"), species="Ni"),
            ActionDescriptor(coord=CoordOffset(site="a", i=1), species="Vacant"),
        ),
        bystanders=(
            BystanderDescriptor(
                coord=CoordOffset(site="a", j=1),
                allowed_species=("Cr",), flag="nn1",
            ),
            BystanderDescriptor(
                coord=CoordOffset(site="a", j=-1),
                allowed_species=("Cr",), flag="nn1",
            ),
        ),
        rate_formula="k0 * exp(-(Ea0 + dEa_Cr * nr_Cr_nn1) / kT)",
        family_id="surface_1NN_inplane",
    )
    proj = KmosProjectDescriptor(
        name="ni_cr_alloy_otf",
        lattice=lattice,
        species=("Vacant", "Ni", "Cr"),
        parameters={"k0": 1e13, "Ea0": 0.51, "dEa_Cr": 0.05},
        processes=(proc,),
    )
    assert proj.processes[0].declared_counter_names() == {"nr_Cr_nn1"}


# ------------------------------------------------------------------
# Round-trip serialisation
# ------------------------------------------------------------------
def test_round_trip_via_json() -> None:
    proj = _minimal_project()
    blob = proj.model_dump_json()
    restored = KmosProjectDescriptor.model_validate_json(blob)
    assert restored == proj


def test_round_trip_preserves_bystanders_and_offsets() -> None:
    """A process with non-trivial bystander offsets should round-trip exactly."""
    sites = (_site("a"),)
    lattice = _lattice(sites=sites)
    proc = ProcessDescriptor(
        name="hop_with_offsets",
        conditions=(ConditionDescriptor(
            coord=CoordOffset(site="a"), species="Vacant",
        ),),
        actions=(ActionDescriptor(
            coord=CoordOffset(site="a"), species="Ni",
        ),),
        bystanders=(
            BystanderDescriptor(
                coord=CoordOffset(site="a", i=2, j=-3, k=1),
                allowed_species=("Cr", "Fe"), flag="far",
            ),
        ),
        rate_formula="k0 * exp(-Ea0 / kT)",
    )
    proj = KmosProjectDescriptor(
        name="x",
        lattice=lattice,
        species=("Vacant", "Ni", "Cr", "Fe"),
        parameters={"k0": 1e13, "Ea0": 0.5},
        processes=(proc,),
    )
    blob = json.loads(proj.model_dump_json())
    assert blob["processes"][0]["bystanders"][0]["coord"] == {
        "site": "a", "i": 2, "j": -3, "k": 1,
    }
    restored = KmosProjectDescriptor.model_validate(blob)
    assert restored == proj


# ------------------------------------------------------------------
# Smoke test: all reserved names render correctly to kmos coord notation.
# ------------------------------------------------------------------
def test_kmos_coord_notation_smoke() -> None:
    cases = [
        (CoordOffset(site="a"), "a"),
        (CoordOffset(site="a", i=1), "a.(1,0,0)"),
        (CoordOffset(site="hollow", i=1, j=-1, k=2), "hollow.(1,-1,2)"),
        (CoordOffset(site="b", k=-3), "b.(0,0,-3)"),
    ]
    for coord, expected in cases:
        assert coord.as_kmos_coord_str() == expected
