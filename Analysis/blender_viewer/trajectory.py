"""KMC / MD trajectory playback.

Two halves, same pattern as ``lattice.py``:

- **Pure-Python**: load a trajectory from disk (LAMMPS dump or multi-frame
  extxyz via ASE) → produce a ``(T, N, 3)`` numpy positions array + per-atom
  element list + ``(T, 3, 3)`` cell array, or dump it to an npz for Blender.

- **bpy-dependent**: efficient bulk keyframing of atom locations via
  ``fcurve.keyframe_points.foreach_set`` — ~50× faster than
  ``keyframe_insert`` in a loop.

Blender's Python may not have ASE, so convert in the host Python first:

    from Analysis.blender_viewer.trajectory import (
        dump_trajectory_npz, load_extxyz_trajectory,
    )
    positions, types, cell = load_extxyz_trajectory(
        "/path/to/run/trajkmc.xyz", max_frames=50,
    )
    dump_trajectory_npz("traj.npz", positions, types, cell)

then inside Blender:

    from Analysis.blender_viewer.trajectory import play_trajectory_from_npz
    play_trajectory_from_npz('traj.npz', frame_start=1)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Host-venv: load multi-frame trajectory to numpy arrays
# ---------------------------------------------------------------------------


def load_extxyz_trajectory(
    path: str | Path,
    max_frames: int | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a multi-frame extended XYZ file via ASE.

    Returns
    -------
    positions : (T, N, 3) float64
    types : list[str] of length N (assumed constant across frames)
    cell : (T, 3, 3) float64
    """
    from ase.io import read as ase_read  # noqa: PLC0415

    frames = ase_read(str(path), index=":")
    if max_frames is not None:
        frames = frames[:max_frames]

    n_atoms = len(frames[0])
    types = [str(s) for s in frames[0].get_chemical_symbols()]

    positions = np.stack([np.asarray(f.get_positions(), dtype=np.float64) for f in frames])
    cell = np.stack([np.asarray(f.cell[:], dtype=np.float64) for f in frames])

    if any(len(f) != n_atoms for f in frames):
        raise ValueError("variable atom count between frames not supported")

    return positions, types, cell


def load_lammps_dump(
    path: str | Path,
    max_frames: int | None = None,
    element_map: dict[int, str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Parse a LAMMPS dump (text format, ITEM: sections).

    ``element_map``: optional dict mapping LAMMPS integer type → element
    symbol. If omitted, atoms are labelled ``"type{N}"``.

    Only supports the common ``ITEM: ATOMS id type x y z`` layout (plus any
    extra columns, ignored). For ``xs ys zs`` scaled coordinates, extra
    parsing work would be needed — we raise instead to surface the issue.
    """
    path = Path(path)
    text = path.read_text().splitlines()

    frames_pos: list[np.ndarray] = []
    frames_cell: list[np.ndarray] = []
    types: list[str] | None = None
    n_atoms: int | None = None

    i = 0
    while i < len(text) and (max_frames is None or len(frames_pos) < max_frames):
        line = text[i]
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms = int(text[i + 1])
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            # next three lines: xlo xhi [xy], ylo yhi [xz], zlo zhi [yz]
            cell = np.zeros((3, 3), dtype=np.float64)
            for axis in range(3):
                parts = text[i + 1 + axis].split()
                lo, hi = float(parts[0]), float(parts[1])
                cell[axis, axis] = hi - lo
            frames_cell.append(cell)
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            header = line[len("ITEM: ATOMS"):].strip().split()
            if "xs" in header or "ys" in header or "zs" in header:
                raise NotImplementedError(
                    "scaled dump coords not supported; re-run LAMMPS dump with x y z not xs ys zs"
                )
            col = {name: idx for idx, name in enumerate(header)}
            col_type = col.get("type")
            col_x = col["x"]; col_y = col["y"]; col_z = col["z"]
            pos = np.empty((n_atoms, 3), dtype=np.float64)
            frame_types: list[str] = []
            for a in range(n_atoms):
                parts = text[i + 1 + a].split()
                pos[a] = (float(parts[col_x]), float(parts[col_y]), float(parts[col_z]))
                if col_type is not None:
                    t_int = int(parts[col_type])
                    if element_map is not None and t_int in element_map:
                        frame_types.append(element_map[t_int])
                    else:
                        frame_types.append(f"type{t_int}")
            frames_pos.append(pos)
            if types is None:
                types = frame_types
            i += 1 + n_atoms
            continue
        i += 1

    if types is None or not frames_pos:
        raise ValueError(f"no ATOMS section found in {path}")

    positions = np.stack(frames_pos)
    cell = np.stack(frames_cell) if frames_cell else np.broadcast_to(
        np.eye(3), (len(frames_pos), 3, 3)
    ).copy()
    return positions, types, cell


def dump_trajectory_npz(
    path: str | Path,
    positions: np.ndarray,
    types: list[str] | np.ndarray,
    cell: np.ndarray,
) -> None:
    """Serialise a trajectory to a numpy-only npz file for Blender to read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=np.asarray(positions, dtype=np.float64),
        types=np.asarray(types),
        cell=np.asarray(cell, dtype=np.float64),
    )


def load_trajectory_npz(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return (
        np.asarray(data["positions"], dtype=np.float64),
        [str(t) for t in data["types"]],
        np.asarray(data["cell"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Blender-side bulk keyframing
# ---------------------------------------------------------------------------


def bulk_keyframe_foreach_set(
    atom_objects: list["bpy.types.Object"],  # type: ignore[name-defined]
    positions: np.ndarray,
    frame_start: int = 1,
    interpolation: Literal["CONSTANT", "LINEAR", "BEZIER"] = "CONSTANT",
) -> None:
    """Write per-frame location keyframes using ``fcurves.foreach_set``.

    ~50× faster than ``obj.keyframe_insert`` in a Python loop. CONSTANT
    interpolation between frames is the physically correct choice for KMC
    (each frame = one KMC state; the atom
    is stationary at its site until the next event). Switch to BEZIER if
    you want smooth visual motion (common for MD trajectories).

    Parameters
    ----------
    atom_objects : list of bpy objects, length N
    positions : (T, N, 3) float array
    frame_start : first frame (Blender's timeline is 1-indexed by convention)
    interpolation : which keyframe interpolation mode to apply
    """
    import bpy  # noqa: PLC0415

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must be (T, N, 3); got {positions.shape}")
    T, N, _ = positions.shape
    if len(atom_objects) != N:
        raise ValueError(f"len(atom_objects)={len(atom_objects)} != N={N} from positions")

    frames = np.arange(frame_start, frame_start + T, dtype=np.float64)

    for i, obj in enumerate(atom_objects):
        # Seed one keyframe so animation_data + action exist.
        obj.location = tuple(float(x) for x in positions[0, i])
        obj.keyframe_insert(data_path="location", frame=int(frames[0]))
        action = obj.animation_data.action
        # Three separate fcurves, one per axis.
        for axis in range(3):
            fc = action.fcurves.find("location", index=axis)
            # Add the remaining T-1 keyframe points in bulk.
            existing = len(fc.keyframe_points)
            needed = T - existing
            if needed > 0:
                fc.keyframe_points.add(count=needed)

            flat = np.empty(2 * T, dtype=np.float64)
            flat[0::2] = frames
            flat[1::2] = positions[:, i, axis]
            fc.keyframe_points.foreach_set("co", flat)
            fc.update()

            for kp in fc.keyframe_points:
                kp.interpolation = interpolation


def play_trajectory_from_npz(
    npz_path: str | Path,
    name: str = "Trajectory",
    interpolation: Literal["CONSTANT", "LINEAR", "BEZIER"] = "CONSTANT",
    frame_start: int = 1,
    subdivisions: int = 1,
) -> dict:
    """Load an npz trajectory, spawn atoms, and bulk-keyframe positions.

    Uses one atom object per atom (linked-data duplicates of a shared
    icosphere) — at ~10³ atoms this is fine; for ≥10⁴ switch to
    :func:`Analysis.blender_viewer.lattice.spawn_atoms_instanced` and
    animate the point-cloud attribute instead.

    Returns dict of refs for verification.
    """
    import bpy  # noqa: PLC0415

    from Analysis.blender_viewer import materials as mats  # noqa: PLC0415
    from Analysis.blender_viewer.constants import display_radius  # noqa: PLC0415

    positions, types, cell = load_trajectory_npz(npz_path)
    T, N, _ = positions.shape

    # Root collection
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)

    # Shared icosphere mesh keyed by element.
    icos: dict[str, bpy.types.Mesh] = {}

    atom_objects: list[bpy.types.Object] = []
    for i, elt in enumerate(types):
        if elt not in icos:
            mesh = bpy.data.meshes.new(f"Ico_{elt}_{name}")
            import bmesh  # noqa: PLC0415
            bm = bmesh.new()
            bmesh.ops.create_icosphere(
                bm, subdivisions=subdivisions, radius=display_radius(elt),
            )
            bm.to_mesh(mesh); bm.free()
            mesh.materials.append(None)
            icos[elt] = mesh

        obj = bpy.data.objects.new(f"{name}.{i:05d}", icos[elt])
        coll.objects.link(obj)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mats.get_material(elt, mats.ROLE_NEIGHBOR)
        atom_objects.append(obj)

    bulk_keyframe_foreach_set(
        atom_objects, positions,
        frame_start=frame_start, interpolation=interpolation,
    )

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + T - 1
    scene.frame_set(frame_start)

    return {
        "collection": coll,
        "atoms": atom_objects,
        "n_frames": T,
        "n_atoms": N,
        "frame_range": (scene.frame_start, scene.frame_end),
    }
