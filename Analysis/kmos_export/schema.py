"""Process descriptor schema — the intermediate representation between the
curated FCC event catalogue and a kmos project file.

Design rationale:

The natural Python form of a kmos `Process` (`kmos.types.Process`)
already does what we need for serialisation, but it depends on importing kmos
itself. We want an IR that:

  1. Doesn't depend on kmos being installed (so the translator is testable
     in CI without the kmos Fortran toolchain).
  2. Is explicitly validated with pydantic — typos in coord names or species
     surface as ValidationErrors, not silent kmos run failures hours later.
  3. Round-trips through JSON so curated catalogues can ship .json sidecars.
  4. Mirrors kmos's mental model 1:1 so the emitter is mechanical.

The mapping is direct:

  ProcessDescriptor    ↔  kmos Process (name, conditions, actions, rate_constant)
                          + Bystander list + otf_rate string for OTF backend
  ConditionDescriptor  ↔  kmos Condition (coord, species)
  ActionDescriptor     ↔  kmos Action    (coord, species)
  BystanderDescriptor  ↔  kmos Bystander (coord, allowed_species, flag)
  CoordOffset          ↔  kmos Coord  (i, j, k, site_name)  — stencil offset
  SiteDescriptor       ↔  kmos Site   (name, pos, default_species)
  LatticeDescriptor    ↔  kmos Lattice / Layer
  KmosProjectDescriptor↔  kmos Project (lattice + species + parameters + processes)

Pattern follows `pylatkmc/pylatkmc/spec.py` (frozen pydantic models with
multiple validator passes) so anyone who's read that file finds this familiar.
"""
from __future__ import annotations

import ast
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ------------------------------------------------------------------
# Compile-time limits. These guard against accidental runaway specs that
# would explode the kmos project file or take hours to validate.
# ------------------------------------------------------------------
MAX_PROCESSES = 1000             # one project shouldn't need more than this
MAX_CONDITIONS_PER_PROCESS = 12  # FCC 1NN coordination = 12 — generous
MAX_ACTIONS_PER_PROCESS = 8      # multi-site exchange + concerted, with margin
MAX_BYSTANDERS_PER_PROCESS = 24  # 12 1NN + 12 2NN at most

# Reserved species name. kmos uses 'default_species' or 'empty'; we standardise
# on 'Vacant' to match pylatkmc/runtime/src/core/events_base.h:Species.
RESERVED_VACANT = "Vacant"

# Identifier pattern for names that ultimately become Fortran symbol names.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_simple_identifier(s: str) -> bool:
    return bool(_IDENT_RE.match(s)) and "__" not in s


# ------------------------------------------------------------------
# CoordOffset — stencil position in (i, j, k, site_name) form.
# ------------------------------------------------------------------
class CoordOffset(BaseModel):
    """A position in the lattice stencil, expressed as a unit-cell offset
    plus a site name within that cell.

    Mirrors kmos's `coord('site_name.(i,j,k)')` syntax. The site_name must
    refer to a `SiteDescriptor.name` declared on the project's lattice.
    Cross-reference is checked at the project level, not here, since this
    type intentionally has no knowledge of the surrounding lattice.
    """

    model_config = ConfigDict(frozen=True)

    site: str
    i: int = 0
    j: int = 0
    k: int = 0

    @field_validator("site")
    @classmethod
    def _site_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"site name {v!r} must be a simple identifier")
        return v

    def as_kmos_coord_str(self) -> str:
        """Render as the kmos coord notation: 'site.(i,j,k)'."""
        if self.i == self.j == self.k == 0:
            return self.site
        return f"{self.site}.({self.i},{self.j},{self.k})"


# ------------------------------------------------------------------
# Condition / Action — pattern predicates and updates at a stencil coord.
# ------------------------------------------------------------------
class ConditionDescriptor(BaseModel):
    """Predicate that gates whether a process can fire: at this stencil
    coord, the site must currently have this species."""

    model_config = ConfigDict(frozen=True)

    coord: CoordOffset
    species: str

    @field_validator("species")
    @classmethod
    def _species_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"species {v!r} must be a simple identifier")
        return v


class ActionDescriptor(BaseModel):
    """State update applied when the process fires: at this stencil coord,
    set the species to this value. For concerted multi-site events, a
    process has multiple actions at distinct coords."""

    model_config = ConfigDict(frozen=True)

    coord: CoordOffset
    species: str

    @field_validator("species")
    @classmethod
    def _species_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"species {v!r} must be a simple identifier")
        return v


# ------------------------------------------------------------------
# BystanderDescriptor — coords whose species count modulates the rate
# but does NOT gate the process firing.
# ------------------------------------------------------------------
class BystanderDescriptor(BaseModel):
    """A bystander: a stencil coord whose species count contributes to the
    OTF rate formula via the auto-generated counter `nr_<species>_<flag>`.
    Bystanders do NOT gate process firing — they only modulate the rate.

    `allowed_species` lists the species names whose count we want to track
    at this coord (kmos generates one counter per (species, flag) pair that
    appears in any bystander).
    `flag` is the tag that names the counter; flags should group bystanders
    that share a physical interpretation (e.g. flag='nn1' for first-shell).
    """

    model_config = ConfigDict(frozen=True)

    coord: CoordOffset
    allowed_species: tuple[str, ...]
    flag: str

    @field_validator("flag")
    @classmethod
    def _flag_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"bystander flag {v!r} must be a simple identifier")
        return v

    @field_validator("allowed_species")
    @classmethod
    def _species_nonempty_and_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) == 0:
            raise ValueError("bystander allowed_species cannot be empty")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate species in allowed_species: {v}")
        for s in v:
            if not _is_simple_identifier(s):
                raise ValueError(f"species name {s!r} must be a simple identifier")
        return v

    def counter_name(self, species: str) -> str:
        """The kmos counter variable name for this bystander group + species.
        Mirrors kmos's `nr_<species>_<flag>` convention from
        `kmos/io/__init__.py` (upstream kmos)."""
        if species not in self.allowed_species:
            raise ValueError(
                f"species {species!r} not in this bystander's allowed_species"
            )
        return f"nr_{species}_{self.flag}"


# ------------------------------------------------------------------
# ProcessDescriptor — one full kmos process spec.
# ------------------------------------------------------------------
class ProcessDescriptor(BaseModel):
    """A single kmos Process in our IR.

    Conditions gate firing; actions update state on fire; bystanders feed
    the rate formula via auto-named counters; rate_formula is a Python
    expression string referencing those counters plus parameters from the
    project (e.g. `T`, `kT`, `k0`).

    Example (a single-vacancy 1NN hop with Cr-trap effect):

        ProcessDescriptor(
            name="surface_1NN_inplane_Ni_mover",
            conditions=[
                ConditionDescriptor(coord=CoordOffset(site="a"), species="Vacant"),
                ConditionDescriptor(
                    coord=CoordOffset(site="a", i=1), species="Ni",
                ),
            ],
            actions=[
                ActionDescriptor(coord=CoordOffset(site="a"), species="Ni"),
                ActionDescriptor(
                    coord=CoordOffset(site="a", i=1), species="Vacant",
                ),
            ],
            bystanders=[
                BystanderDescriptor(
                    coord=CoordOffset(site="a", j=1),
                    allowed_species=("Cr", "Fe"), flag="nn1",
                ),
                # ... 11 more 1NN bystanders ...
            ],
            rate_formula="k0 * exp(-(Ea0 + dEa_Cr_nn1*nr_Cr_nn1) / kT)",
            family_id="surface_1NN_inplane",
            notes="Linear-fit Ea(n_Cr_nn1) from catalogue bucket nv1=4..8",
        )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    conditions: tuple[ConditionDescriptor, ...]
    actions: tuple[ActionDescriptor, ...]
    bystanders: tuple[BystanderDescriptor, ...] = ()
    rate_formula: str
    family_id: str = ""
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _name_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"process name {v!r} must be a simple identifier")
        return v

    @field_validator("conditions")
    @classmethod
    def _conditions_nonempty_and_capped(
        cls, v: tuple[ConditionDescriptor, ...]
    ) -> tuple[ConditionDescriptor, ...]:
        if len(v) == 0:
            raise ValueError("a process must have at least one condition")
        if len(v) > MAX_CONDITIONS_PER_PROCESS:
            raise ValueError(
                f"too many conditions ({len(v)} > {MAX_CONDITIONS_PER_PROCESS})"
            )
        return v

    @field_validator("actions")
    @classmethod
    def _actions_nonempty_and_capped(
        cls, v: tuple[ActionDescriptor, ...]
    ) -> tuple[ActionDescriptor, ...]:
        if len(v) == 0:
            raise ValueError("a process must have at least one action")
        if len(v) > MAX_ACTIONS_PER_PROCESS:
            raise ValueError(
                f"too many actions ({len(v)} > {MAX_ACTIONS_PER_PROCESS})"
            )
        return v

    @field_validator("bystanders")
    @classmethod
    def _bystanders_capped(
        cls, v: tuple[BystanderDescriptor, ...]
    ) -> tuple[BystanderDescriptor, ...]:
        if len(v) > MAX_BYSTANDERS_PER_PROCESS:
            raise ValueError(
                f"too many bystanders ({len(v)} > {MAX_BYSTANDERS_PER_PROCESS})"
            )
        return v

    @model_validator(mode="after")
    def _coords_unique_per_role(self) -> ProcessDescriptor:
        """Within conditions, no two predicates at the same coord; same for
        actions and bystanders. (Different roles can share a coord — e.g. a
        coord that is BOTH a condition and an action — that's normal.)"""
        for label, items in (
            ("conditions", self.conditions),
            ("actions", self.actions),
            ("bystanders", self.bystanders),
        ):
            coords = [it.coord for it in items]
            if len(set(coords)) != len(coords):
                raise ValueError(
                    f"duplicate coord in {label}: each coord may appear only "
                    f"once per role within a process"
                )
        return self

    @model_validator(mode="after")
    def _rate_formula_parses_as_python(self) -> ProcessDescriptor:
        """`rate_formula` must be a valid Python expression. We don't
        evaluate it here (that needs the runtime parameter dict), but we
        catch syntax errors at descriptor creation time."""
        try:
            ast.parse(self.rate_formula, mode="eval")
        except SyntaxError as e:
            raise ValueError(
                f"rate_formula {self.rate_formula!r} is not a valid Python "
                f"expression: {e}"
            ) from None
        return self

    def referenced_counter_names(self) -> set[str]:
        """Return the set of bystander-counter names actually used in the
        rate formula. Useful for sanity-checking (a counter referenced but
        not declared as a bystander is a bug)."""
        try:
            tree = ast.parse(self.rate_formula, mode="eval")
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("nr_"):
                names.add(node.id)
        return names

    def declared_counter_names(self) -> set[str]:
        """Return the set of counter names this process's bystanders declare."""
        out: set[str] = set()
        for b in self.bystanders:
            for sp in b.allowed_species:
                out.add(b.counter_name(sp))
        return out


# ------------------------------------------------------------------
# Lattice — sites and unit cell.
# ------------------------------------------------------------------
class SiteDescriptor(BaseModel):
    """One site within the unit cell. Maps to kmos's `Site(name, pos,
    default_species)`."""

    model_config = ConfigDict(frozen=True)

    name: str
    pos_frac: tuple[float, float, float]    # fractional coords inside the cell
    default_species: str = RESERVED_VACANT

    @field_validator("name")
    @classmethod
    def _name_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"site name {v!r} must be a simple identifier")
        return v

    @field_validator("default_species")
    @classmethod
    def _default_species_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"default_species {v!r} must be a simple identifier")
        return v

    @field_validator("pos_frac")
    @classmethod
    def _pos_in_unit_cell(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        for x in v:
            if not (0.0 <= x < 1.0):
                raise ValueError(
                    f"pos_frac component {x} out of [0, 1); "
                    "use a within-cell fractional position"
                )
        return v


class LatticeDescriptor(BaseModel):
    """Crystal lattice: cell vectors (in Å) and the list of sites within
    the unit cell. The current iteration is FCC-only (one Layer in kmos
    parlance) but the schema doesn't enforce that — multiple layers can be
    added later by extending KmosProjectDescriptor."""

    model_config = ConfigDict(frozen=True)

    name: str
    cell_a: tuple[float, float, float]
    cell_b: tuple[float, float, float]
    cell_c: tuple[float, float, float]
    sites: tuple[SiteDescriptor, ...]

    @field_validator("name")
    @classmethod
    def _name_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"lattice name {v!r} must be a simple identifier")
        return v

    @field_validator("sites")
    @classmethod
    def _sites_nonempty_unique(
        cls, v: tuple[SiteDescriptor, ...]
    ) -> tuple[SiteDescriptor, ...]:
        if len(v) == 0:
            raise ValueError("lattice must declare at least one site")
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            dups = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate site names: {sorted(dups)}")
        return v

    def site_names(self) -> set[str]:
        return {s.name for s in self.sites}


# ------------------------------------------------------------------
# Top-level project.
# ------------------------------------------------------------------
class KmosProjectDescriptor(BaseModel):
    """The full description: lattice + species + parameters + processes.

    Maps to a kmos `Project` (`kmos.types.Project`). Renderable
    to a kmos project XML or to a Python script that calls
    `pt.add_layer(...)`, `pt.add_species(...)`, `pt.add_process(...)`.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    lattice: LatticeDescriptor
    species: tuple[str, ...]              # first must be the "vacant" name
    parameters: dict[str, float] = Field(default_factory=dict)
    processes: tuple[ProcessDescriptor, ...]

    @field_validator("name")
    @classmethod
    def _name_simple(cls, v: str) -> str:
        if not _is_simple_identifier(v):
            raise ValueError(f"project name {v!r} must be a simple identifier")
        return v

    @field_validator("species")
    @classmethod
    def _species_starts_with_vacant_and_unique(
        cls, v: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not v or v[0] != RESERVED_VACANT:
            raise ValueError(
                f"species list must start with {RESERVED_VACANT!r}; "
                f"got {v[0] if v else '<empty>'!r}"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate species names: {v}")
        for s in v:
            if not _is_simple_identifier(s):
                raise ValueError(f"species name {s!r} must be a simple identifier")
        return v

    @field_validator("processes")
    @classmethod
    def _processes_capped_and_unique(
        cls, v: tuple[ProcessDescriptor, ...]
    ) -> tuple[ProcessDescriptor, ...]:
        if len(v) > MAX_PROCESSES:
            raise ValueError(
                f"too many processes ({len(v)} > {MAX_PROCESSES})"
            )
        names = [p.name for p in v]
        if len(set(names)) != len(names):
            dups = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate process names: {sorted(dups)}")
        return v

    @model_validator(mode="after")
    def _all_coords_reference_known_sites(self) -> KmosProjectDescriptor:
        """Every CoordOffset.site referenced anywhere must exist on the
        project's lattice. Catches site-name typos at descriptor build time."""
        known_sites = self.lattice.site_names()
        for proc in self.processes:
            for items in (proc.conditions, proc.actions, proc.bystanders):
                for it in items:
                    if it.coord.site not in known_sites:
                        raise ValueError(
                            f"process {proc.name!r} references site "
                            f"{it.coord.site!r} which is not in the lattice "
                            f"(known: {sorted(known_sites)})"
                        )
        return self

    @model_validator(mode="after")
    def _all_species_known(self) -> KmosProjectDescriptor:
        """Every species named in conditions, actions, bystanders, or as a
        site default_species must be in the project's species list."""
        species_set = set(self.species)
        for site in self.lattice.sites:
            if site.default_species not in species_set:
                raise ValueError(
                    f"site {site.name!r} default_species "
                    f"{site.default_species!r} not in project species "
                    f"{sorted(species_set)}"
                )
        for proc in self.processes:
            for items, label in (
                (proc.conditions, "conditions"),
                (proc.actions, "actions"),
            ):
                for it in items:
                    if it.species not in species_set:
                        raise ValueError(
                            f"process {proc.name!r} {label} reference "
                            f"unknown species {it.species!r}"
                        )
            for b in proc.bystanders:
                for sp in b.allowed_species:
                    if sp not in species_set:
                        raise ValueError(
                            f"process {proc.name!r} bystander references "
                            f"unknown species {sp!r}"
                        )
        return self

    @model_validator(mode="after")
    def _rate_formulas_only_use_known_symbols(self) -> KmosProjectDescriptor:
        """Every Name in a rate_formula must be either:
          (a) a parameter declared in self.parameters,
          (b) a bystander counter declared by the process's bystanders, or
          (c) a builtin we whitelist (math functions, constants).

        This catches typos and undeclared parameters at build time, before
        they become silent kmos compile errors."""
        whitelist = _BUILTIN_FORMULA_NAMES | set(self.parameters)
        for proc in self.processes:
            try:
                tree = ast.parse(proc.rate_formula, mode="eval")
            except SyntaxError:
                # Already caught at process level; skip here.
                continue
            allowed = whitelist | proc.declared_counter_names()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id not in allowed:
                    raise ValueError(
                        f"process {proc.name!r} rate_formula references "
                        f"unknown name {node.id!r}; "
                        f"declare it as a parameter, a bystander counter, "
                        f"or a builtin (allowed builtins: "
                        f"{sorted(_BUILTIN_FORMULA_NAMES)})"
                    )
        return self


# Names allowed in any rate_formula without being declared as parameters or
# bystander counters. Conservative whitelist — extend deliberately.
#
# Two groups:
#   - Math: standard transcendental functions and constants.
#   - kmos built-ins: physical constants kmos's formula parser injects
#     automatically. Sourced from `kmos.types.Project.parser`
#     conventions and example projects (e.g. AB_model.py uses `bar`, `eV`,
#     `kboltzmann`, `h`, `beta`, `umass`).
_BUILTIN_FORMULA_NAMES: frozenset[str] = frozenset({
    # Math
    "exp", "log", "sqrt", "abs", "min", "max", "pi", "e",
    # kmos physical constants (parsed as units / scalars by its formula engine)
    "kboltzmann", "eV", "bar", "umass", "h", "beta", "amu", "angstrom",
    # Common shorthand parameter names; OK to also be in self.parameters
    "T", "kT", "kB",
})
