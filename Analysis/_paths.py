"""Central path and tool resolution for the Analysis package.

Every module that needs a location on disk or an external program asks this
module instead of hard-coding one. Configuration is by environment variable;
every tool that uses a location also takes a command-line flag that wins over
the variable.

``PYKMC_ANALYSIS_DATA_ROOT``
    Directory holding your pyKMC runs, laid out as
    ``{composition}/{N}vac/T{temp}/{variant}/`` (see ``Analysis.sim_discovery``).
    Only the corpus-wide tools need it; single-run tools take a run directory.
``PYKMC_ANALYSIS_OUTPUT_DIR``
    Where classification outputs and the audit log are read and written.
    Default: ``Analysis/lattice_event_classification/`` inside this checkout.
``PYKMC_ANALYSIS_BLENDER`` / ``PYKMC_ANALYSIS_LAMMPS`` / ``PYKMC_ANALYSIS_MPIRUN``
    External programs. Default: the first match on ``PATH``.
``PYKMC_ANALYSIS_DYLD_LIBRARY_PATH``
    macOS only, NEB renders: library path for a LAMMPS build that needs one.

Resolution order for the data root: explicit argument, then
``PYKMC_ANALYSIS_DATA_ROOT``, then the legacy ``PYKMC_KMC_ROOT`` (a tree that
contains ``Data/Research/``), then a walk up from this file looking for a
``Data/Research/`` sibling. Nothing here raises at import time; ``data_root()``
raises a ``RuntimeError`` naming the variable only when a corpus is needed and
none is configured. ``KMC_ROOT`` and ``DATA_RESEARCH`` remain as lazy aliases.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KMCConfig:
    """Physical + analysis parameters for one dataset.

    The single place to change when switching composition / potential / cutoff.
    FCC(100) Ni defaults. ``nn1_dist`` / ``nn2_dist`` are derived from the
    lattice constant (1NN = a/√2, 2NN = a on the (100) surface).
    """

    lattice_constant_A: float = 3.524          # Ni FCC lattice constant (Å)
    barrier_cutoff_eV: float = 1.2             # classify only events below this
    required_potential: str = "NiAlH"          # process sims whose potential matches

    @property
    def nn1_dist(self) -> float:
        return self.lattice_constant_A / math.sqrt(2)

    @property
    def nn2_dist(self) -> float:
        return self.lattice_constant_A


#: The active dataset config. Override by replacing this (or its fields) for a
#: different composition/potential; every consumer imports from here.
CONFIG: KMCConfig = KMCConfig()

# Backward-compatible module-level constants (single source = CONFIG). Modules
# that historically defined these locally now import them from here.
A_NI: float = CONFIG.lattice_constant_A
NN1_DIST: float = CONFIG.nn1_dist
NN2_DIST: float = CONFIG.nn2_dist
BARRIER_CUTOFF: float = CONFIG.barrier_cutoff_eV
REQUIRED_POTENTIAL: str = CONFIG.required_potential


def find_kmc_root() -> Path | None:
    """Return the legacy workspace root (a tree holding ``Data/Research/``), or ``None``.

    A ``PYKMC_KMC_ROOT`` that is set but wrong still raises: that is a
    misconfiguration, not an absent corpus.
    """
    env = os.environ.get("PYKMC_KMC_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "Data" / "Research").is_dir():
            return p
        raise RuntimeError(
            f"PYKMC_KMC_ROOT={env!r} does not contain a Data/Research/ directory."
        )
    for parent in Path(__file__).resolve().parents:
        if (parent / "Data" / "Research").is_dir():
            return parent
    return None


def find_data_root(explicit: str | os.PathLike | None = None) -> Path | None:
    """Return the directory holding the pyKMC runs, or ``None`` if none is configured.

    A location that is given (argument or ``PYKMC_ANALYSIS_DATA_ROOT``) but is
    not a directory raises, as it is a misconfiguration.
    """
    for source, value in (("argument", explicit),
                          ("PYKMC_ANALYSIS_DATA_ROOT", os.environ.get("PYKMC_ANALYSIS_DATA_ROOT"))):
        if value:
            p = Path(value).expanduser().resolve()
            if not p.is_dir():
                raise RuntimeError(f"Data root from {source} ({value!r}) is not a directory.")
            return p
    legacy = find_kmc_root()
    return legacy / "Data" / "Research" if legacy else None


def data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Like ``find_data_root`` but raises a helpful ``RuntimeError`` when unset."""
    root = find_data_root(explicit)
    if root is None:
        raise RuntimeError(
            "No pyKMC data directory configured. Set PYKMC_ANALYSIS_DATA_ROOT to the "
            "directory holding your runs ({composition}/{N}vac/T{temp}/{variant}/), "
            "or pass --data-root."
        )
    return root


def output_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Directory for classification outputs and the audit log (argument, env, default)."""
    value = explicit or os.environ.get("PYKMC_ANALYSIS_OUTPUT_DIR")
    if value:
        return Path(value).expanduser().resolve()
    return ANALYSIS_ROOT / "lattice_event_classification"


_TOOL_CANDIDATES = {
    "blender": ["blender"],
    "lammps": ["lmp", "lmp_mpi", "lmp_serial"],
    "mpirun": ["mpirun", "mpiexec"],
}


def tool(name: str, explicit: str | None = None) -> str | None:
    """Locate an external program: argument, ``PYKMC_ANALYSIS_<NAME>``, then ``PATH``.

    Returns ``None`` when nothing is found; callers raise with their own context.
    """
    value = explicit or os.environ.get(f"PYKMC_ANALYSIS_{name.upper()}")
    if value:
        return value
    for candidate in _TOOL_CANDIDATES.get(name, [name]):
        found = shutil.which(candidate)
        if found:
            return found
    if name == "blender" and sys.platform == "darwin":
        app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if app.exists():
            return str(app)
    return None


def resolve_sim_path(path: str | os.PathLike, root: str | os.PathLike | None = None) -> Path:
    """Turn a ``sim_path`` read from an output CSV into a directory on this machine.

    Relative paths are joined to the data root. An absolute path that does not
    exist here (a CSV made on another machine) is re-rooted on the part after
    ``Data/Research/`` when that marker is present. Anything else is returned
    unchanged.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        return data_root(root) / p
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts) - 1):
        if parts[i] == "Data" and parts[i + 1] == "Research":
            base = find_data_root(root)
            if base is not None:
                return base.joinpath(*parts[i + 2:])
            break
    return p


def __getattr__(name: str) -> Path:
    # PEP 562: KMC_ROOT / DATA_RESEARCH are resolved lazily (see module docstring).
    if name == "DATA_RESEARCH":
        return data_root()
    if name == "KMC_ROOT":
        root = find_kmc_root()
        if root is None:
            raise RuntimeError(
                "KMC_ROOT is the legacy workspace root; set PYKMC_KMC_ROOT, or use "
                "data_root() with PYKMC_ANALYSIS_DATA_ROOT."
            )
        return root
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


ANALYSIS_ROOT: Path = Path(__file__).resolve().parent
CLASSIFICATION_DIR: Path = output_dir()

CLASSIFIED_CSV: Path = CLASSIFICATION_DIR / "classified_events.csv"
CLASSIFIED_WITH_FAMILIES_CSV: Path = (
    CLASSIFICATION_DIR / "classified_events_with_families.csv"
)
AUDIT_CSV: Path = CLASSIFICATION_DIR / "audit_log.csv"
