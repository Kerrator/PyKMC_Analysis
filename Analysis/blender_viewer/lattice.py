"""Lattice construction and atom spawning utilities.

Two halves:

- **Pure-Python** (numpy / scipy / ASE) — supercell construction, vacancy
  detection, position/type array preparation. Callable from the host venv
  or from Blender's bundled Python equally well.

- **bpy-dependent** — spawn atoms into a Blender scene. Two strategies:
  :func:`spawn_atoms_instanced` (Geometry Nodes Instance-on-Points,
  preferred for N ≥ 500) and :func:`spawn_atoms_linked` (linked-data
  duplicates, usable for any N but heavier per-object overhead).

Coordinates are always in Å. 1 Blender unit = 1 Å unless the scene overrides
``scene.unit_settings.scale_length = 1e-10``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

LatticeStyle = Literal["fcc", "bcc"]

FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
])

BCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])


# ---------------------------------------------------------------------------
# Supercell construction (pure Python)
# ---------------------------------------------------------------------------


def build_supercell(
    lattice: LatticeStyle,
    a: float,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """Build an Nx3 array of atomic positions for an ideal FCC or BCC supercell.

    Parameters
    ----------
    lattice : "fcc" or "bcc"
    a : lattice parameter (Å)
    nx, ny, nz : unit-cell repetitions in each direction

    Returns
    -------
    (N, 3) numpy float array in Å. N = basis_size * nx * ny * nz.
    """
    if lattice == "fcc":
        basis = FCC_BASIS
    elif lattice == "bcc":
        basis = BCC_BASIS
    else:
        raise ValueError(f"unknown lattice: {lattice!r}")

    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)
    grid = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1).reshape(-1, 3)
    cells = grid[:, None, :] + basis[None, :, :]
    return (cells.reshape(-1, 3) * a).astype(np.float64)


def build_fcc_supercell(a: float, nx: int, ny: int, nz: int) -> np.ndarray:
    return build_supercell("fcc", a, nx, ny, nz)


def build_bcc_supercell(a: float, nx: int, ny: int, nz: int) -> np.ndarray:
    return build_supercell("bcc", a, nx, ny, nz)


def supercell_cell_vectors(
    lattice: LatticeStyle, a: float, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Return the 3x3 cell vectors (rows) for the supercell, in Å."""
    return np.diag([a * nx, a * ny, a * nz]).astype(np.float64)


# ---------------------------------------------------------------------------
# Vacancy detection
# ---------------------------------------------------------------------------


def detect_vacancies(
    ideal_sites: np.ndarray,
    occupied_positions: np.ndarray,
    tol: float = 0.5,
) -> np.ndarray:
    """Return indices into ``ideal_sites`` that are NOT occupied.

    An ideal site is "occupied" if any position in ``occupied_positions`` lies
    within ``tol`` Å of it (scipy cKDTree nearest-neighbour lookup).

    Parameters
    ----------
    ideal_sites : (N_ideal, 3) array of lattice sites
    occupied_positions : (N_occ, 3) array of actual atom positions
    tol : Å — the largest distance at which an ideal site counts as occupied

    Returns
    -------
    (N_vac,) integer array of indices into ``ideal_sites``.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if len(occupied_positions) == 0:
        return np.arange(len(ideal_sites))

    tree = cKDTree(occupied_positions)
    dists, _ = tree.query(ideal_sites, k=1)
    return np.where(dists > tol)[0]


# ---------------------------------------------------------------------------
# extxyz loading (ASE shim — requires ASE in the active environment)
# ---------------------------------------------------------------------------


def load_extxyz(path: str, index: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (positions, types, cell) for a single frame of an extxyz file.

    Requires ASE. Call from the host venv — Blender's bundled Python usually
    does not ship ASE, so upstream code should ``dump_supercell_npz`` outside
    Blender and have the bpy side read the npz intermediate.
    """
    from ase.io import read as ase_read  # noqa: PLC0415

    atoms = ase_read(path, index=index)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    types = [str(s) for s in atoms.get_chemical_symbols()]
    cell = np.asarray(atoms.cell[:], dtype=np.float64)
    return positions, types, cell


# ---------------------------------------------------------------------------
# npz bridge (pure numpy — Blender-safe)
# ---------------------------------------------------------------------------


def dump_supercell_npz(
    path: str,
    positions: np.ndarray,
    types: list[str] | np.ndarray,
    cell: np.ndarray,
    vacancy_sites: np.ndarray | None = None,
) -> None:
    """Serialise a supercell (+ optional vacancy marker positions) to .npz."""
    payload: dict[str, np.ndarray] = {
        "positions": np.asarray(positions, dtype=np.float64),
        "types": np.asarray(types),
        "cell": np.asarray(cell, dtype=np.float64),
    }
    if vacancy_sites is not None:
        payload["vacancy_sites"] = np.asarray(vacancy_sites, dtype=np.float64)
    np.savez(path, **payload)


def load_supercell_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    out: dict[str, np.ndarray] = {
        "positions": np.asarray(data["positions"], dtype=np.float64),
        "types": np.asarray(data["types"]),
        "cell": np.asarray(data["cell"], dtype=np.float64),
    }
    if "vacancy_sites" in data.files:
        out["vacancy_sites"] = np.asarray(data["vacancy_sites"], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Blender-side atom spawning
# ---------------------------------------------------------------------------


def spawn_atoms_instanced(
    positions: np.ndarray,
    types: Iterable[str],
    name: str = "Atoms",
    parent_collection: str | None = None,
    subdivisions: int = 2,
    radius_scale: float | None = None,
) -> "bpy.types.Object":  # type: ignore[name-defined]
    """Spawn atoms via a Geometry Nodes Instance-on-Points modifier.

    Scales to 10⁴+ atoms without chokepoints because each atom is an instance
    of a single shared icosphere mesh. One object per *element* — each object
    has its own subset of point positions so the per-element material can
    attach cleanly.

    Returns the top-level Empty that parents the per-element point meshes.

    Notes
    -----
    - Requires bpy. Import only succeeds inside Blender.
    - Uses ``scene_builder``'s per-object material override pattern — the
      instanced icosphere's mesh has one empty slot that each per-element
      point object overrides.
    """
    import bpy  # noqa: PLC0415

    from Analysis.blender_viewer import materials as mats  # noqa: PLC0415
    from Analysis.blender_viewer.constants import (  # noqa: PLC0415
        EVENT_RADIUS_SCALE,
        display_radius,
    )

    positions = np.asarray(positions, dtype=np.float64)
    types_list = list(types)
    if len(positions) != len(types_list):
        raise ValueError("positions and types length mismatch")

    scale = EVENT_RADIUS_SCALE if radius_scale is None else radius_scale

    # Root empty to parent everything under.
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)

    # Find or create parent collection.
    if parent_collection is not None:
        coll = bpy.data.collections.get(parent_collection)
        if coll is None:
            coll = bpy.data.collections.new(parent_collection)
            bpy.context.scene.collection.children.link(coll)
    else:
        coll = bpy.context.scene.collection

    # Shared icosphere mesh (one per distinct radius so per-element radius
    # differences are honoured).
    icos: dict[float, bpy.types.Mesh] = {}

    types_array = np.asarray(types_list)
    unique_elements = np.unique(types_array)

    for element in unique_elements:
        mask = types_array == element
        elt_positions = positions[mask]
        if len(elt_positions) == 0:
            continue

        radius = display_radius(str(element), scale)
        if radius not in icos:
            icos[radius] = _make_icosphere_mesh(
                f"IcoTemplate_r{radius:.3f}", subdivisions=subdivisions, radius=radius,
            )
        ico_mesh = icos[radius]

        # Point mesh: N vertices, no edges/polys.
        point_mesh = bpy.data.meshes.new(f"{name}.{element}.points")
        point_mesh.from_pydata(elt_positions.tolist(), [], [])
        # Reserve one material slot on the point mesh so per-object override
        # keeps each element's material separate.
        point_mesh.materials.append(None)

        point_obj = bpy.data.objects.new(f"{name}.{element}", point_mesh)
        point_obj.parent = root
        coll.objects.link(point_obj)
        point_obj.material_slots[0].link = "OBJECT"
        point_obj.material_slots[0].material = mats.get_material(
            str(element), mats.ROLE_NEIGHBOR,
        )

        # Geometry Nodes Instance-on-Points.
        mod = point_obj.modifiers.new(name="InstanceOnPoints", type="NODES")
        node_group = _make_instance_on_points_group(
            group_name=f"GN_Instance_{element}",
            instance_mesh_name=ico_mesh.name,
        )
        mod.node_group = node_group

    return root


def _make_icosphere_mesh(name: str, subdivisions: int, radius: float) -> "bpy.types.Mesh":  # type: ignore[name-defined]
    """Create a standalone icosphere mesh (no object linked to scene)."""
    import bmesh  # noqa: PLC0415
    import bpy  # noqa: PLC0415

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    # Reserve one material slot so downstream point objects inherit a slot.
    mesh.materials.append(None)
    return mesh


def _make_instance_on_points_group(
    group_name: str, instance_mesh_name: str
) -> "bpy.types.NodeTree":  # type: ignore[name-defined]
    """Build a minimal Geometry Nodes group: mesh → Instance on Points (object)."""
    import bpy  # noqa: PLC0415

    existing = bpy.data.node_groups.get(group_name)
    if existing is not None:
        return existing

    # Bring the instance mesh to a Blender object so it can be referenced
    # by an "Object Info" node (GN instance input expects an Object, not a
    # Mesh datablock).
    proto_obj_name = f"{instance_mesh_name}.proto"
    proto_obj = bpy.data.objects.get(proto_obj_name)
    if proto_obj is None:
        proto_obj = bpy.data.objects.new(proto_obj_name, bpy.data.meshes[instance_mesh_name])
        bpy.context.scene.collection.objects.link(proto_obj)
        proto_obj.hide_viewport = True
        proto_obj.hide_render = True

    g = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    # Inputs / outputs
    g.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    g.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    grp_in = g.nodes.new("NodeGroupInput");  grp_in.location = (-400, 0)
    grp_out = g.nodes.new("NodeGroupOutput"); grp_out.location = (600, 0)
    iop = g.nodes.new("GeometryNodeInstanceOnPoints"); iop.location = (150, 0)
    obj_info = g.nodes.new("GeometryNodeObjectInfo"); obj_info.location = (-200, -200)
    obj_info.inputs["Object"].default_value = proto_obj

    g.links.new(grp_in.outputs["Geometry"], iop.inputs["Points"])
    g.links.new(obj_info.outputs["Geometry"], iop.inputs["Instance"])
    g.links.new(iop.outputs["Instances"], grp_out.inputs["Geometry"])

    return g


# ---------------------------------------------------------------------------
# Vacancy markers (visual representation of missing lattice sites)
# ---------------------------------------------------------------------------


def spawn_vacancy_markers(
    vacancy_positions: np.ndarray,
    style: Literal["wireframe", "emission", "transparent"] = "wireframe",
    edge_length_A: float = 2.0,
    collection_name: str = "Vacancies",
) -> list["bpy.types.Object"]:  # type: ignore[name-defined]
    """Place markers at vacancy sites.

    Styles (in order of visual dominance):
    - ``wireframe``: cube + Wireframe modifier + yellow emission (crisp pedagogical)
    - ``emission``: emissive sphere (hero frames)
    - ``transparent``: alpha-blended sphere (bulk overview)
    """
    import bpy  # noqa: PLC0415

    vacancy_positions = np.asarray(vacancy_positions, dtype=np.float64)
    if len(vacancy_positions) == 0:
        return []

    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)

    # One shared material.
    mat = bpy.data.materials.get(f"Mat_Vacancy_{style}")
    if mat is None:
        mat = bpy.data.materials.new(f"Mat_Vacancy_{style}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if style == "wireframe":
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 0.0, 1.0)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 0.0, 1.0)
            else:
                bsdf.inputs["Emission"].default_value = (1.0, 1.0, 0.0, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 3.0
        elif style == "emission":
            bsdf.inputs["Base Color"].default_value = (1.0, 0.95, 0.2, 1.0)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.95, 0.2, 1.0)
            else:
                bsdf.inputs["Emission"].default_value = (1.0, 0.95, 0.2, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 8.0
        else:  # transparent
            bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.7, 1.0)
            bsdf.inputs["Alpha"].default_value = 0.20
            bsdf.inputs["Roughness"].default_value = 0.5
            mat.blend_method = "BLEND"

    markers: list[bpy.types.Object] = []
    for i, pos in enumerate(vacancy_positions):
        name = f"vac_{i:03d}"
        if style == "wireframe":
            bpy.ops.mesh.primitive_cube_add(size=edge_length_A, location=tuple(float(x) for x in pos))
            obj = bpy.context.active_object
            obj.name = name
            mod = obj.modifiers.new("Wireframe", type="WIREFRAME")
            mod.thickness = 0.05
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2,
                radius=edge_length_A * 0.45,
                location=tuple(float(x) for x in pos),
            )
            obj = bpy.context.active_object
            obj.name = name

        obj.active_material = mat
        obj.hide_select = False
        # Move into the Vacancies collection.
        for uc in list(obj.users_collection):
            uc.objects.unlink(obj)
        coll.objects.link(obj)
        markers.append(obj)

    return markers
