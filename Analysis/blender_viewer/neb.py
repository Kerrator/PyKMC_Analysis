"""NEB minimum-energy-path image interpolation.

Given a set of NEB image files (e.g. ``neb_0.xyz`` ... ``neb_4.xyz`` from a
VASP or LAMMPS NEB calculation), keyframe the atomic positions at evenly-
spaced frames with AUTO_CLAMPED Bezier handles so the resulting F-curve
traces the true MEP through configuration space.

Host-venv entry (requires ASE) writes an npz intermediate; Blender-side
entry reads the npz and builds the scene. Same pattern as
``trajectory.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Analysis.blender_viewer.trajectory import (
    bulk_keyframe_foreach_set,
    dump_trajectory_npz,
    load_trajectory_npz,
)


# ---------------------------------------------------------------------------
# Host side: collect NEB image files into a single (T, N, 3) array
# ---------------------------------------------------------------------------


def load_neb_images(
    glob_pattern: str | Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read NEB images matching ``glob_pattern`` in sort order.

    Typical usage: ``load_neb_images('neb_dir/neb_*.xyz')``.
    Returns the same shape as :func:`trajectory.load_extxyz_trajectory`.
    """
    from ase.io import read as ase_read  # noqa: PLC0415

    paths = sorted(Path().glob(str(glob_pattern)))
    if not paths:
        raise FileNotFoundError(f"no NEB images matching {glob_pattern!r}")

    positions: list[np.ndarray] = []
    cell: list[np.ndarray] = []
    types: list[str] | None = None

    for p in paths:
        atoms = ase_read(str(p))
        positions.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        cell.append(np.asarray(atoms.cell[:], dtype=np.float64))
        if types is None:
            types = [str(s) for s in atoms.get_chemical_symbols()]
        elif len(atoms) != len(types):
            raise ValueError(f"inconsistent atom count in {p}")

    return np.stack(positions), types or [], np.stack(cell)


def dump_neb_npz(
    path: str | Path,
    positions: np.ndarray,
    types: list[str],
    cell: np.ndarray,
) -> None:
    """Same serialisation as :func:`trajectory.dump_trajectory_npz`."""
    dump_trajectory_npz(path, positions, types, cell)


# ---------------------------------------------------------------------------
# Blender side: load + keyframe NEB images with Bezier AUTO_CLAMPED
# ---------------------------------------------------------------------------


def play_neb_from_npz(
    npz_path: str | Path,
    name: str = "NEB",
    frames_per_image: int = 8,
    frame_start: int = 1,
    subdivisions: int = 2,
) -> dict:
    """Build a scene animating through NEB images at ``frames_per_image`` cadence.

    NEB gives a discrete set of images along the minimum-energy path. To
    animate "through" the MEP, we place keyframes every
    ``frames_per_image`` frames and let Blender's Bezier AUTO_CLAMPED
    interpolator trace a smooth curve between them. With 5 NEB images and
    frames_per_image=8 that's a 33-frame animation (frames 1..33 inclusive).

    Returns refs dict suitable for verification.
    """
    import bpy  # noqa: PLC0415

    from Analysis.blender_viewer import materials as mats  # noqa: PLC0415
    from Analysis.blender_viewer.constants import (  # noqa: PLC0415
        BEZIER_HANDLE_TYPE,
        display_radius,
    )

    positions, types, _cell = load_trajectory_npz(npz_path)
    T_images, N, _ = positions.shape
    total_frames = (T_images - 1) * frames_per_image + 1
    image_frames = np.arange(T_images) * frames_per_image + frame_start

    # Build a (total_frames, N, 3) array where only image frames have data;
    # we'll keyframe just those and let Blender interpolate.
    # Actually, we can keyframe directly at image_frames — no need to
    # hallucinate intermediate positions.

    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)

    # Shared icospheres per element.
    icos: dict[str, bpy.types.Mesh] = {}
    atoms: list[bpy.types.Object] = []
    import bmesh  # noqa: PLC0415

    for i, elt in enumerate(types):
        if elt not in icos:
            mesh = bpy.data.meshes.new(f"Ico_{elt}_{name}")
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
        atoms.append(obj)

    # Keyframe each image frame individually (no bulk foreach_set because
    # we want Bezier handles on the sparse keys, not a dense per-frame grid).
    for i, obj in enumerate(atoms):
        for t, frame in enumerate(image_frames):
            obj.location = tuple(float(x) for x in positions[t, i])
            obj.keyframe_insert(data_path="location", frame=int(frame))
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                if fc.data_path.startswith("location"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "BEZIER"
                        kp.handle_left_type = BEZIER_HANDLE_TYPE
                        kp.handle_right_type = BEZIER_HANDLE_TYPE

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + total_frames - 1
    scene.frame_set(frame_start)

    return {
        "collection": coll,
        "atoms": atoms,
        "n_images": T_images,
        "total_frames": total_frames,
        "image_frames": image_frames.tolist(),
    }
