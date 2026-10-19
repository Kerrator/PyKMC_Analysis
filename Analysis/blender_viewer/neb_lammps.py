"""LAMMPS NEB runner for pyKMC events.

Given an ``EventPair`` (the neighbour subset with init / saddle / final
positions), run a LAMMPS Nudged Elastic Band calculation and return the
intermediate positions along the minimum-energy path, suitable for use as
animation keyframes.

Architecture
------------
LAMMPS NEB needs one MPI rank per replica, so the LAMMPS executable is
launched under MPI:

    mpirun -n 11 lmp -partition 11x1 -in neb.in

The executable must be built with the REPLICA package. The LAMMPS Python
module used by pyKMC is often built without it, which is why NEB is driven
through the executable rather than the Python API.

Configuration (an explicit argument wins over the environment variable):

* LAMMPS executable: ``lmp_binary`` argument, ``PYKMC_ANALYSIS_LAMMPS``, then
  ``lmp`` / ``lmp_mpi`` / ``lmp_serial`` on ``PATH``.
* MPI launcher: ``mpirun`` argument, ``PYKMC_ANALYSIS_MPIRUN``, then
  ``mpirun`` / ``mpiexec`` on ``PATH``. Use the launcher from the same MPI
  build LAMMPS was linked against. The ``--map-by :OVERSUBSCRIBE`` option
  passed below is Open MPI syntax.
* Potential file: the ``potential`` argument (required; no default).
* macOS only: if the executable finds its shared libraries through a
  relative path, set ``PYKMC_ANALYSIS_DYLD_LIBRARY_PATH`` (or pass
  ``lmp_lib_dir``) to the directory holding them; it is prepended to
  ``DYLD_LIBRARY_PATH`` for the LAMMPS run only.

Subset NEB with frozen outer shell
-----------------------------------
Atoms within ``mover_ball_A`` of the mover's initial position are mobile;
atoms beyond are frozen (``fix setforce 0.0 0.0 0.0``). This approximates
the full-system constraint without needing the full simulation cell
(which the pickle does not preserve).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from Analysis import _paths
from Analysis.blender_viewer.data import EventPair, EventSide

# ---------------------------------------------------------------------------
# External programs
# ---------------------------------------------------------------------------


def _require_tool(name: str, explicit: str | os.PathLike | None, hint: str) -> str:
    """Resolve ``name`` via ``_paths.tool`` or raise with setup instructions."""
    found = _paths.tool(name, str(explicit) if explicit else None)
    if found is None:
        raise FileNotFoundError(
            f"{name} executable not found. {hint} Put it on PATH, set "
            f"PYKMC_ANALYSIS_{name.upper()}, or pass --{name}."
        )
    # LAMMPS runs with cwd = the NEB directory, so make relative paths absolute.
    return str(Path(found).expanduser().resolve()) if os.sep in found else found


# ---------------------------------------------------------------------------
# LAMMPS data-file writer
# ---------------------------------------------------------------------------


def _write_lammps_data(
    path: Path,
    positions: np.ndarray,
    types: Sequence[str],
    type_order: list[str],
    box_padding: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a LAMMPS ``read_data`` file with shrink-wrap box.

    Returns (box_lo, box_hi) — the 3-vectors defining the axis-aligned box,
    so the same box can be used for the ``final.data`` partner file.
    """
    positions = np.asarray(positions, dtype=np.float64)
    # Compute box bounds with padding (shrink-wrap boundary requires some
    # space beyond the atoms).
    lo = positions.min(axis=0) - box_padding
    hi = positions.max(axis=0) + box_padding

    n_atoms = len(positions)
    n_types = len(type_order)
    type_idx = {t: i + 1 for i, t in enumerate(type_order)}

    lines: list[str] = [
        "# NEB LAMMPS data file written by Analysis.blender_viewer.neb_lammps",
        "",
        f"{n_atoms} atoms",
        f"{n_types} atom types",
        "",
        f"{lo[0]:.10f} {hi[0]:.10f} xlo xhi",
        f"{lo[1]:.10f} {hi[1]:.10f} ylo yhi",
        f"{lo[2]:.10f} {hi[2]:.10f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    from ase.data import atomic_masses, atomic_numbers  # noqa: PLC0415
    for i, elt in enumerate(type_order, start=1):
        mass = atomic_masses[atomic_numbers[elt]]
        lines.append(f"{i} {mass:.6f}  # {elt}")
    lines.append("")
    lines.append("Atoms  # atomic")
    lines.append("")
    for i, (elt, pos) in enumerate(zip(types, positions, strict=True), start=1):
        t = type_idx[elt]
        lines.append(f"{i} {t} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return lo, hi


def _write_final_coords(path: Path, positions: np.ndarray) -> None:
    """Write the ``final final.coords`` file LAMMPS NEB reads via the `neb`
    command's last argument when file-style=``final``.

    Format (per LAMMPS docs): one line with atom count, then ``atom_id x y z``
    per line for atoms that should be displaced. We write all N atoms.
    """
    positions = np.asarray(positions, dtype=np.float64)
    lines = [f"{len(positions)}"]
    for i, pos in enumerate(positions, start=1):
        lines.append(f"{i} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# LAMMPS input-script writer
# ---------------------------------------------------------------------------


def _write_neb_input(
    path: Path,
    potential_path: Path,
    pair_style: str,
    type_order: list[str],
    mover_position: np.ndarray,
    mover_ball_A: float,
    kspring: float,
    etol: float,
    ftol: float,
    n_iter_stage1: int,
    n_iter_stage2: int,
    dump_every: int,
    final_data_path: Path,
) -> None:
    """Write the LAMMPS NEB input script."""
    coeff_elements = " ".join(type_order)
    mx, my, mz = float(mover_position[0]), float(mover_position[1]), float(mover_position[2])
    script = f"""# NEB for a pyKMC event (neighbour subset, frozen outer shell)
# Generated by Analysis.blender_viewer.neb_lammps

units           metal
atom_style      atomic
atom_modify     map array sort 0 0.0
boundary        s s s

# Initial state (replica 0; LAMMPS auto-interpolates the rest from final coords).
read_data       initial.data

# Pair potential.
pair_style      {pair_style}
pair_coeff      * * {potential_path} {coeff_elements}

# Mover ball: atoms within {mover_ball_A:.2f} A of the mover's init position.
# These are mobile. Everything outside the ball is frozen.
region          mover_ball sphere {mx:.6f} {my:.6f} {mz:.6f} {mover_ball_A:.6f} units box
group           mobile   region   mover_ball
group           frozen   subtract all mobile
fix             freeze   frozen   setforce 0.0 0.0 0.0

# NEB setup.
min_style       quickmin
timestep        0.01

fix             1 mobile neb {kspring:.6f} parallel ideal

# Per-replica dump: variable i is the partition (replica) index.
variable        i equal part
dump            mypath all custom {dump_every} dump.neb.$i id type x y z

# stage 1 minimises MEP; stage 2 applies climbing image for accurate saddle.
neb             {etol:.6e} {ftol:.6e} {n_iter_stage1} {n_iter_stage2} 100 final {final_data_path.name} verbosity verbose
"""
    path.write_text(script)


# ---------------------------------------------------------------------------
# LAMMPS dump parser (extract the last frame of each replica)
# ---------------------------------------------------------------------------


def _parse_dump_last_frame(path: Path) -> np.ndarray:
    """Return the last-frame positions from a LAMMPS custom dump.

    Positions are sorted by atom id so the returned (N, 3) array aligns
    with the input data file's atom ordering.
    """
    text = path.read_text().splitlines()
    # Find all "ITEM: TIMESTEP" positions.
    starts = [i for i, line in enumerate(text) if line.startswith("ITEM: TIMESTEP")]
    if not starts:
        raise RuntimeError(f"no frames in {path}")
    last = starts[-1]

    # Expected block layout: TIMESTEP / N / BOX BOUNDS header / 3 bound lines /
    # ATOMS header / N atom lines.
    n_atoms = int(text[last + 3])
    atoms_header_idx = None
    for j in range(last + 4, len(text)):
        if text[j].startswith("ITEM: ATOMS"):
            atoms_header_idx = j
            break
    if atoms_header_idx is None:
        raise RuntimeError(f"no ATOMS section in last frame of {path}")
    header_cols = text[atoms_header_idx][len("ITEM: ATOMS"):].strip().split()
    col_id = header_cols.index("id")
    col_x = header_cols.index("x")
    col_y = header_cols.index("y")
    col_z = header_cols.index("z")

    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        parts = text[atoms_header_idx + 1 + k].split()
        aid = int(parts[col_id]) - 1
        positions[aid] = (float(parts[col_x]), float(parts[col_y]), float(parts[col_z]))
    return positions


def _parse_neb_barriers(log_path: Path) -> tuple[float, float] | None:
    """Scrape the final NEB log line for (forward, reverse) barriers (eV).

    LAMMPS prints one row per NEB iteration. The last row of the "climb"
    stage lists: Step MaxReplicaForce MaxAtomForce GradV0 GradV1 GradVc
    EBF EBR RDT RD1 PE1 RD2 PE2 ... RDN PEN.
    We want EBF (forward barrier) and EBR (reverse barrier).
    """
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    # Find the last numeric row under a "Step" header.
    last_data: list[str] | None = None
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            [float(p) for p in parts[:8]]
        except ValueError:
            continue
        last_data = parts
    if last_data is None:
        return None
    try:
        ebf = float(last_data[6])
        ebr = float(last_data[7])
        return ebf, ebr
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_neb_for_side(
    side: EventSide,
    out_dir: Path,
    potential: str | os.PathLike,
    n_images: int = 11,
    kspring: float = 1.0,
    mover_ball_A: float = 5.0,
    etol: float = 0.0,
    ftol: float = 0.05,
    n_iter_stage1: int = 2000,
    n_iter_stage2: int = 1000,
    pair_style: str = "eam/alloy",
    mpirun: str | os.PathLike | None = None,
    lmp_binary: str | os.PathLike | None = None,
    lmp_lib_dir: str | os.PathLike | None = None,
    overwrite: bool = True,
) -> dict:
    """Run NEB for one forward-or-backward event side. Returns diagnostics.

    ``potential`` is the interatomic potential file handed to
    ``pair_coeff * * <potential> <elements>`` with ``pair_style``; elements
    are listed alphabetically (the pyKMC type order). ``mpirun``,
    ``lmp_binary`` and ``lmp_lib_dir`` override the environment lookups
    described in the module docstring.
    """
    potential_path = Path(potential).expanduser().resolve()
    if not potential_path.is_file():
        raise FileNotFoundError(f"potential file not found: {potential_path}")
    lmp_exe = _require_tool(
        "lammps", lmp_binary,
        "NEB needs a LAMMPS executable built with the REPLICA package.",
    )
    mpirun_exe = _require_tool(
        "mpirun", mpirun,
        "NEB runs one MPI rank per image; use the launcher LAMMPS was built with.",
    )

    out_dir = Path(out_dir)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build type order (alphabetical — matches pyKMC convention).
    elements = sorted(set(side.types))

    # Write initial / final data files.
    initial_data = out_dir / "initial.data"
    final_data = out_dir / "final.data"
    box_lo, box_hi = _write_lammps_data(
        initial_data, side.init_positions, side.types, elements,
    )
    # final.data just needs to exist for LAMMPS to ignore on partition 0;
    # we use the file-style `final` of the neb command instead for the
    # displacement field — but LAMMPS still insists the file exist.
    _write_lammps_data(final_data, side.final_positions, side.types, elements)

    # Write a plain coords file LAMMPS reads via `neb ... final <file>`.
    final_coords = out_dir / "final.coords"
    _write_final_coords(final_coords, side.final_positions)

    # Mover's initial position (lab frame — same frame as the data file).
    mover_pos = np.asarray(side.init_positions[side.mover_idx], dtype=np.float64)

    neb_in = out_dir / "neb.in"
    _write_neb_input(
        neb_in,
        potential_path=potential_path,
        pair_style=pair_style,
        type_order=elements,
        mover_position=mover_pos,
        mover_ball_A=mover_ball_A,
        kspring=kspring,
        etol=etol,
        ftol=ftol,
        n_iter_stage1=n_iter_stage1,
        n_iter_stage2=n_iter_stage2,
        dump_every=100,
        final_data_path=final_coords,
    )

    # Launch: mpirun -n N lmp -partition Nx1 -in neb.in
    env = os.environ.copy()
    # Put the launcher's own directory first on PATH so Open MPI finds the
    # helper programs from the same installation.
    mpirun_dir = Path(shutil.which(mpirun_exe) or mpirun_exe).parent
    if mpirun_dir.is_absolute():
        env["PATH"] = f"{mpirun_dir}{os.pathsep}{env.get('PATH', '')}"
    lib_dir = lmp_lib_dir or os.environ.get("PYKMC_ANALYSIS_DYLD_LIBRARY_PATH")
    if sys.platform == "darwin" and lib_dir:
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env.get('DYLD_LIBRARY_PATH', '')}"
    # --map-by :OVERSUBSCRIBE lets us run N replicas on a machine with
    # fewer physical cores than N (e.g. 11 images on an 8-core laptop).
    cmd = [
        mpirun_exe, "-n", str(n_images),
        "--map-by", ":OVERSUBSCRIBE",
        lmp_exe, "-partition", f"{n_images}x1",
        "-in", neb_in.name,
        "-log", "log.lammps",
        "-screen", "none",
    ]
    result = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(
            f"NEB failed (exit {result.returncode}) in {out_dir}:\n{tail}"
        )

    # Parse the 11 per-replica dumps.
    positions = np.stack([
        _parse_dump_last_frame(out_dir / f"dump.neb.{i}") for i in range(n_images)
    ])
    # LAMMPS NEB may have moved the outer atoms via partition-0's initial
    # minimise; we still align by atom id so the returned ordering matches
    # the input ``side.init_positions`` ordering.

    barriers = _parse_neb_barriers(out_dir / "log.lammps.0")
    if barriers is None:
        barriers = _parse_neb_barriers(out_dir / "log.lammps")
    ebf, ebr = (barriers if barriers is not None else (float("nan"), float("nan")))

    diag = {
        "n_images": int(n_images),
        "n_atoms": int(positions.shape[1]),
        "pickle_ea_forward": float(side.energy_barrier),
        "neb_ea_forward": ebf,
        "neb_ea_reverse": ebr,
        "ea_error_eV": abs(ebf - side.energy_barrier) if not np.isnan(ebf) else None,
        "mover_idx": int(side.mover_idx),
        "box_lo": box_lo.tolist(),
        "box_hi": box_hi.tolist(),
        "potential": str(potential_path),
    }
    (out_dir / "neb_diagnostics.json").write_text(json.dumps(diag, indent=2))

    return {"positions": positions, **diag}


def run_neb_for_pair(
    pair: EventPair,
    out_dir: Path,
    potential: str | os.PathLike,
    n_images: int = 11,
    **kwargs,
) -> dict:
    """Run NEB for just the forward side of a pair (backward plays in reverse).

    Returns {'positions': (n_images, N_atoms, 3), 'types': list[str], ...diagnostics}.
    The returned positions are in the forward event's local atom ordering — align
    with pair.forward.init_positions[i] == positions[0, i]. Extra keyword
    arguments go to :func:`run_neb_for_side`.
    """
    out_dir = Path(out_dir)
    result = run_neb_for_side(
        pair.forward, out_dir, potential=potential, n_images=n_images, **kwargs,
    )
    result["types"] = list(pair.forward.types)
    return result


def save_neb_npz(
    out_path: Path,
    positions: np.ndarray,
    types: Sequence[str],
    diag: dict,
) -> None:
    """Serialise NEB result to a numpy-only npz (Blender-readable)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        positions=np.asarray(positions, dtype=np.float64),
        types=np.asarray(list(types)),
        pickle_ea_forward=np.array(diag.get("pickle_ea_forward", np.nan)),
        neb_ea_forward=np.array(diag.get("neb_ea_forward", np.nan)),
        neb_ea_reverse=np.array(diag.get("neb_ea_reverse", np.nan)),
        mover_idx=np.array(diag.get("mover_idx", -1)),
    )


def load_neb_npz(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Inverse of save_neb_npz — numpy-only, Blender-safe."""
    data = np.load(Path(path), allow_pickle=False)
    positions = np.asarray(data["positions"], dtype=np.float64)
    types = [str(t) for t in data["types"]]
    diag = {
        "pickle_ea_forward": float(data["pickle_ea_forward"]),
        "neb_ea_forward": float(data["neb_ea_forward"]),
        "neb_ea_reverse": float(data["neb_ea_reverse"]),
        "mover_idx": int(data["mover_idx"]),
    }
    return positions, types, diag
