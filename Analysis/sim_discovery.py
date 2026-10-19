"""Discover pyKMC run directories under a data root.

Layout: {composition}/{nvac}vac/T{temp}/{coloring}/  (e.g. 100Ni/4vac/T600/full/)

``parse_run_path`` is the one place that layout is parsed; use it rather than
re-parsing directory names. ``sim_from_dir`` describes a single run directory
that need not sit in the layout.

Usage:
    from Analysis.sim_discovery import find_simulations, sim_label

    for sim in find_simulations(Path(".")):
        print(sim_label(sim), sim.path)
"""

import configparser
import re
from pathlib import Path
from typing import NamedTuple


class SimInfo(NamedTuple):
    """Metadata for a simulation directory."""

    path: Path
    composition: str  # e.g. "100Ni", "95Ni_5Fe"
    n_vacancies: int  # e.g. 5
    temperature: float  # e.g. 400.0
    coloring: str  # e.g. "full", "grey", "unknown"


SIM_MARKER_FILES = ["trajkmc.xyz", "input.in", "reference_table.pickle"]

_NVAC_RE = re.compile(r"(\d+)vac")
_TEMP_RE = re.compile(r"T(\d+(?:\.\d+)?)")


def parse_run_path(run_dir: Path) -> tuple[str, int, float, str] | None:
    """Parse ``.../{composition}/{N}vac/T{temp}/{coloring}`` into its four fields.

    Returns ``(composition, n_vacancies, temperature, coloring)``, or ``None``
    when ``run_dir`` does not end in that layout.
    """
    run_dir = Path(run_dir)
    temp_dir = run_dir.parent
    nvac_dir = temp_dir.parent
    t_match = _TEMP_RE.fullmatch(temp_dir.name)
    n_match = _NVAC_RE.fullmatch(nvac_dir.name)
    if not (t_match and n_match and nvac_dir.parent.name):
        return None
    return nvac_dir.parent.name, int(n_match.group(1)), float(t_match.group(1)), run_dir.name


SKIP_DIRS = {
    "cluster_scripts",
    "templates",
    "analysis",
    ".ipynb_checkpoints",
    "__pycache__",
}


def find_simulations(research_dir: Path) -> list[SimInfo]:
    """Walk the 4-level hierarchy and return all simulation directories.

    Expects: research_dir/{composition}/{nvac}vac/T{temp}/{coloring}/
    """
    sims = []
    for comp_dir in sorted(research_dir.iterdir()):
        if not comp_dir.is_dir() or comp_dir.name in SKIP_DIRS:
            continue
        if comp_dir.name.startswith("."):
            continue
        # Must look like a composition directory
        if not any(c.isdigit() for c in comp_dir.name):
            continue

        for nvac_dir in sorted(comp_dir.iterdir()):
            if not nvac_dir.is_dir() or "vac" not in nvac_dir.name:
                continue

            for temp_dir in sorted(nvac_dir.iterdir()):
                if not temp_dir.is_dir() or not temp_dir.name.startswith("T"):
                    continue

                for color_dir in sorted(temp_dir.iterdir()):
                    if not color_dir.is_dir():
                        continue
                    if not any((color_dir / f).exists() for f in SIM_MARKER_FILES):
                        continue
                    parsed = parse_run_path(color_dir)
                    if parsed is None:
                        continue
                    composition, nvac, temp, coloring = parsed
                    sims.append(
                        SimInfo(
                            path=color_dir,
                            composition=composition,
                            n_vacancies=nvac,
                            temperature=temp,
                            coloring=coloring,
                        )
                    )
    return sims


def sim_label(sim: SimInfo) -> str:
    """Human-readable label for a simulation (for display, filtering, dict keys)."""
    return f"{sim.composition}_{sim.n_vacancies}vac_T{int(sim.temperature)}_{sim.coloring}"


def sim_relpath(sim: SimInfo, research_dir: Path) -> str:
    """Relative path from research_dir (for summary.json dirname field)."""
    return str(sim.path.relative_to(research_dir))


def sim_from_dir(
    run_dir: Path,
    composition: str | None = None,
    n_vacancies: int | None = None,
    temperature: float | None = None,
) -> SimInfo:
    """Describe a single run directory (one that holds ``reference_table.pickle``).

    Each field comes from, in order: the explicit argument; the
    ``{composition}/{N}vac/T{temp}/{coloring}/`` layout when the directory sits
    in one; ``input.in`` (``[RateConstant] T``) for the temperature; else a
    default (composition = directory name, 0 vacancies, 0 K). The coloring is
    always the directory name, as in ``find_simulations``.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not (run_dir / "reference_table.pickle").is_file():
        raise FileNotFoundError(f"{run_dir} has no reference_table.pickle")

    parsed = parse_run_path(run_dir)
    if parsed is not None:
        composition = composition or parsed[0]
        n_vacancies = parsed[1] if n_vacancies is None else n_vacancies
        temperature = parsed[2] if temperature is None else temperature

    if temperature is None and (run_dir / "input.in").is_file():
        cfg = configparser.ConfigParser()
        try:
            cfg.read(run_dir / "input.in")
            temperature = cfg.getfloat("RateConstant", "T", fallback=None)
        except (configparser.Error, ValueError):
            temperature = None

    return SimInfo(
        path=run_dir,
        composition=composition or run_dir.name,
        n_vacancies=0 if n_vacancies is None else int(n_vacancies),
        # The viewer keys runs on integer kelvin (scatter clicks round-trip
        # through int), so store a whole number.
        temperature=float(round(temperature or 0.0)),
        coloring=run_dir.name,
    )
