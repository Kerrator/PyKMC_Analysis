"""Build a Blender scene for one pyKMC event pair.

Consumes an :class:`Analysis.blender_viewer.data.EventPair` (pure data) and
produces a populated Blender scene. Two styles are available:

- ``style="dual"`` (default) — separate Forward_Event and Backward_Event
  collections with visibility-toggled timelines. Shows the true asymmetric
  saddle geometry of both directions.

- ``style="loop"`` — single atom set; the forward event plays forward 1→60
  then in reverse 60→120. Passes through the saddle twice over one cycle
  but loops seamlessly with no visibility flip. Better for looping GIFs/
  previews.

Camera styles are orthogonal (static / orbit / dolly); see ``camera.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import bpy
import numpy as np

from Analysis.blender_viewer import animation, camera, materials
from Analysis.blender_viewer.constants import (
    COOP_DISPLACEMENT_THR_A,
    COVALENT_RADIUS_A,
    EVENT_RADIUS_SCALE as RADIUS_SCALE,
    MOVER_RADIUS_BOOST,
)
from Analysis.blender_viewer.data import EventPair, EventSide

Style = Literal["dual", "loop"]
CameraStyle = Literal["static", "orbit", "dolly"]


@dataclass
class DualFrames:
    """Frame budget for the two-collection 'dual' style."""

    fwd_init: int = 1
    fwd_saddle: int = 30
    fwd_final: int = 60
    hold_end: int = 80
    bwd_saddle: int = 105
    bwd_final: int = 130


@dataclass
class LoopFrames:
    """Frame budget for the single-set 'loop' style (forward then reverse).

    Timeline is 800 frames @ 60 fps ≈ 13.3 s per cycle. Orbit camera
    completes exactly one 360° revolution in sync, so the wrap 800 → 1 is
    seamless both in atom positions (both are at init) and in rotation
    (360° ≡ 0°).
    """

    init: int = 1
    saddle_out: int = 200
    final: int = 400
    saddle_back: int = 600
    init_return: int = 800


@dataclass
class LoopFramesNEB:
    """Frame budget for N-image NEB keyframing in the loop style.

    Forward half: N NEB images evenly spaced over frames [1, total//2 + 1].
    Backward half: reverse of the same N images over [total//2 + 1, total + 1].
    The mover (and every atom) is keyframed at each NEB image position.
    Wrap total → 1 is seamless because both endpoints are at the init image.
    """

    total_frames: int = 800
    n_images: int = 11

    @property
    def forward_frames(self) -> list[int]:
        import numpy as _np  # noqa: PLC0415
        return _np.linspace(1, self.total_frames // 2 + 1, self.n_images, dtype=int).tolist()

    @property
    def backward_frames(self) -> list[int]:
        import numpy as _np  # noqa: PLC0415
        return _np.linspace(
            self.total_frames // 2 + 1, self.total_frames, self.n_images, dtype=int
        ).tolist()


def min_pairwise_distance(pair: EventPair) -> float:
    """Smallest inter-atom distance across all frames of fwd+bwd events.

    Used to size atom spheres for "closest-pack, no overlap" rendering:
    ``uniform_radius = 0.48 * min_pairwise_distance(pair)`` guarantees
    atoms never touch even at the saddle point (4% safety margin).

    Computed via scipy.spatial.cKDTree for O(N log N) per state. Blender's
    bundled Python has no scipy; there a dense numpy distance matrix is used
    instead (fine for event subsets of a few hundred atoms).
    """
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415
    except ImportError:
        cKDTree = None

    min_d = float("inf")
    for side in (pair.forward, pair.backward):
        for positions in (side.init_positions, side.saddle_positions, side.final_positions):
            pos = np.asarray(positions, dtype=np.float64)
            if cKDTree is not None:
                # Query 2nd nearest (1st is self at distance 0).
                dists, _ = cKDTree(pos).query(pos, k=2)
                min_d = min(min_d, float(dists[:, 1].min()))
            else:
                d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                min_d = min(min_d, float(d.min()))
    return min_d


def build_scene(
    pair: EventPair,
    style: Style = "dual",
    camera_style: CameraStyle = "static",
    radius_override: float | None = None,
    mover_boost: float | None = None,
    neb_images: np.ndarray | None = None,
    theme: str = "default",
    flat_shade: bool = False,
    motion_cycles_per_pan: int = 1,
    cooperative_highlight: bool = False,
    coop_displacement_thr: float = COOP_DISPLACEMENT_THR_A,
) -> dict:
    """Build the scene and return refs for verification.

    Parameters
    ----------
    radius_override
        If provided, use this uniform radius (Å) for every atom instead of
        the per-element ``display_radius(element) * EVENT_RADIUS_SCALE``
        rule. Typical "closest-pack" use: compute the min pairwise distance
        across the whole animation and pass ``0.48 * min_dist`` — atoms
        touch but never overlap at the saddle. Use
        :func:`min_pairwise_distance` as a helper.
    mover_boost
        Overrides the default mover-radius multiplier (1.15). Set to 1.0
        in closest-pack mode so the mover doesn't crash into its neighbors
        at the saddle.
    neb_images
        Optional ``(N, N_atoms, 3)`` array of intermediate NEB positions
        aligned to forward's atom ordering. When provided AND style='loop',
        every atom is keyframed at each of the N images (forward half)
        and again in reverse (backward half), yielding a physically
        correct trajectory for the whole neighbour subset.
    cooperative_highlight
        If True, every atom whose ``|final - init|`` exceeds
        ``coop_displacement_thr`` is treated as a mover (ROLE_MOVER) —
        gets the mover boost + emission and the theme's ``mover_color``.
        Use with ``theme='burgundy_gold_coop'`` to light up the 3 atoms
        of a ``1NN_hop_triple`` event in yellow gold against a burgundy
        crystal. The registered ``side.mover_idx`` is always included.
    coop_displacement_thr
        Per-atom displacement threshold (Å) above which an atom counts as
        a cooperative mover. Default 0.5 Å — well above thermal noise,
        well below a real 1NN hop (~2.5 Å). Ignored when
        ``cooperative_highlight=False``.
    """
    _clear_scene()
    materials.reset_cache()
    # Theme must be set on the scene BEFORE _build_icosphere_template /
    # _spawn_side so materials.get_material reads the right palette.
    scene = bpy.context.scene
    scene["__theme__"] = str(theme)
    scene["__flat_shade__"] = bool(flat_shade)
    scene["__motion_cycles__"] = int(motion_cycles_per_pan)
    scene["__coop_highlight__"] = bool(cooperative_highlight)
    scene["__coop_thr__"] = float(coop_displacement_thr)
    template = _build_icosphere_template()
    scene["__radius_override__"] = (
        float(radius_override) if radius_override is not None else -1.0
    )
    scene["__mover_boost__"] = (
        float(mover_boost) if mover_boost is not None else MOVER_RADIUS_BOOST
    )

    root = _ensure_collection(
        f"KMC_Event_{pair.forward.idx_ref}_{pair.backward.idx_ref}"
    )
    _link_to_scene(root)

    if style == "dual":
        if neb_images is not None:
            raise ValueError("neb_images requires style='loop'")
        refs = _build_dual(pair, template, root)
        total_frames = refs["frames"].bwd_final
    elif style == "loop":
        if neb_images is not None:
            refs = _build_loop_neb(pair, template, root, neb_images)
        else:
            refs = _build_loop(pair, template, root)
        total_frames = refs["frames"].total_frames if hasattr(
            refs["frames"], "total_frames"
        ) else refs["frames"].init_return
    else:
        raise ValueError(f"unknown style: {style!r}")

    # Scene frame range. End at total_frames - 1 so that wrap-around from
    # end+1 → 1 lands on the loop-closing keyframe.
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.frame_set(1)

    refs["camera"] = _install_camera(pair, total_frames, style=camera_style)
    refs["style"] = style
    refs["camera_style"] = camera_style
    return refs


# ---------------------------------------------------------------------------
# Style: dual (two collections, visibility-toggled)
# ---------------------------------------------------------------------------


def _build_dual(
    pair: EventPair,
    template: bpy.types.Mesh,
    root: bpy.types.Collection,
) -> dict:
    frames = DualFrames()
    fwd_coll = _ensure_collection("Forward_Event", parent=root)
    bwd_coll = _ensure_collection("Backward_Event", parent=root)

    fwd_objs, fwd_mover = _spawn_side(pair.forward, fwd_coll, template, "fwd")
    bwd_objs, bwd_mover = _spawn_side(pair.backward, bwd_coll, template, "bwd")

    animation.animate_side(
        fwd_objs,
        pair.forward.init_positions,
        pair.forward.saddle_positions,
        pair.forward.final_positions,
        (frames.fwd_init, frames.fwd_saddle, frames.fwd_final),
    )
    animation.animate_side(
        bwd_objs,
        pair.backward.init_positions,
        pair.backward.saddle_positions,
        pair.backward.final_positions,
        (frames.hold_end, frames.bwd_saddle, frames.bwd_final),
    )

    for obj in fwd_objs:
        animation.keyframe_visibility(obj, frame=1, visible=True)
        animation.keyframe_visibility(obj, frame=frames.hold_end, visible=False)
    for obj in bwd_objs:
        animation.keyframe_visibility(obj, frame=1, visible=False)
        animation.keyframe_visibility(obj, frame=frames.hold_end, visible=True)

    return {
        "root": root,
        "fwd_objs": fwd_objs,
        "bwd_objs": bwd_objs,
        "fwd_mover": fwd_mover,
        "bwd_mover": bwd_mover,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Style: loop (single collection, forward-then-reverse for seamless cycle)
# ---------------------------------------------------------------------------


def _build_loop(
    pair: EventPair,
    template: bpy.types.Mesh,
    root: bpy.types.Collection,
) -> dict:
    """Single atom set that plays forward then in reverse for a perfect loop.

    The animation visits: init → saddle → final → saddle → init. Physically
    this passes through the forward saddle on the return trip (which is the
    same atomic configuration as backward's saddle, up to atom ordering), so
    it's physically honest as a "back and forth hop" animation. The key
    benefit over the dual style is that frame 1 and frame N+1 are identical,
    which is required for a clean cycle.
    """
    frames = LoopFrames()
    coll = _ensure_collection("Event_Loop", parent=root)
    atom_objs, mover_obj = _spawn_side(pair.forward, coll, template, "atom")

    # Five keyframes per atom: init / saddle / final / saddle / init.
    f_init, f_sad_out, f_final, f_sad_back, f_init_return = (
        frames.init, frames.saddle_out, frames.final,
        frames.saddle_back, frames.init_return,
    )
    for i, obj in enumerate(atom_objs):
        animation.keyframe_location(obj, f_init, pair.forward.init_positions[i])
        animation.keyframe_location(obj, f_sad_out, pair.forward.saddle_positions[i])
        animation.keyframe_location(obj, f_final, pair.forward.final_positions[i])
        animation.keyframe_location(obj, f_sad_back, pair.forward.saddle_positions[i])
        animation.keyframe_location(obj, f_init_return, pair.forward.init_positions[i])
        animation.convert_to_bezier_auto_clamped(obj)

    return {
        "root": root,
        "atoms": atom_objs,
        "mover": mover_obj,
        "frames": frames,
    }


def _build_loop_neb(
    pair: EventPair,
    template: bpy.types.Mesh,
    root: bpy.types.Collection,
    neb_images: np.ndarray,
) -> dict:
    """Loop style with N NEB images as keyframes (forward + reversed).

    Every atom is keyframed at each of the N NEB images during the forward
    half, then at the same images in reverse during the backward half —
    so every neighbour follows its physical MEP trajectory, not just the
    mover. Pendulum-compatible: first and last frame both land on image 0.
    """
    neb_images = np.asarray(neb_images, dtype=np.float64)
    if neb_images.ndim != 3 or neb_images.shape[2] != 3:
        raise ValueError(f"neb_images must be (N, N_atoms, 3); got {neb_images.shape}")
    if neb_images.shape[1] != len(pair.forward.types):
        raise ValueError(
            f"neb atom count {neb_images.shape[1]} != event atom count "
            f"{len(pair.forward.types)}"
        )

    n_images = int(neb_images.shape[0])
    total_frames = 800
    motion_cycles = int(bpy.context.scene.get("__motion_cycles__", 1))
    frames = LoopFramesNEB(total_frames=total_frames, n_images=n_images)

    coll = _ensure_collection("Event_Loop_NEB", parent=root)
    atom_objs, mover_obj = _spawn_side(pair.forward, coll, template, "atom")

    # Build the waypoint sequence for ``motion_cycles`` full oscillations
    # spread across ``total_frames``. Each full cycle = forward (images 0→N-1)
    # then reverse (images N-1→0), sharing the midpoint image. So M cycles
    # produce 2*M*(N-1) + 1 waypoints. Frame 1 = image 0, frame total_frames
    # = image 0 (loop closes).
    waypoint_images: list[int] = [0]
    for _ in range(motion_cycles):
        waypoint_images.extend(range(1, n_images))             # up to final
        waypoint_images.extend(range(n_images - 2, -1, -1))    # back to init
    # waypoint_images is len = 2*motion_cycles*(n_images - 1) + 1
    n_waypoints = len(waypoint_images)
    waypoint_frames = np.linspace(1, total_frames, n_waypoints, dtype=int).tolist()

    for i, obj in enumerate(atom_objs):
        for img_idx, frame in zip(waypoint_images, waypoint_frames, strict=True):
            animation.keyframe_location(obj, frame, neb_images[img_idx, i])
        animation.convert_to_bezier_auto_clamped(obj)

    return {
        "root": root,
        "atoms": atom_objs,
        "mover": mover_obj,
        "frames": frames,
        "n_neb_images": n_images,
        "motion_cycles": motion_cycles,
        "waypoint_frames": waypoint_frames,
        "waypoint_images": waypoint_images,
    }


# ---------------------------------------------------------------------------
# Atom spawning and materials (shared across styles)
# ---------------------------------------------------------------------------


def _spawn_side(
    side: EventSide,
    collection: bpy.types.Collection,
    template: bpy.types.Mesh,
    name_prefix: str,
) -> tuple[list[bpy.types.Object], bpy.types.Object]:
    atom_objs: list[bpy.types.Object] = []
    mover_obj: bpy.types.Object | None = None

    scene = bpy.context.scene
    radius_override = scene.get("__radius_override__", -1.0)
    mover_boost = scene.get("__mover_boost__", MOVER_RADIUS_BOOST)
    coop_highlight = bool(scene.get("__coop_highlight__", False))
    coop_thr = float(scene.get("__coop_thr__", COOP_DISPLACEMENT_THR_A))

    # In cooperative-highlight mode every atom whose net |final - init|
    # exceeds the threshold is treated as a mover. The registered
    # side.mover_idx is always included even if its displacement is just
    # below threshold (we never demote the canonical mover).
    if coop_highlight:
        disp = np.linalg.norm(
            np.asarray(side.final_positions, dtype=np.float64)
            - np.asarray(side.init_positions, dtype=np.float64),
            axis=1,
        )
        mover_set: set[int] = set(np.where(disp > coop_thr)[0].tolist())
        mover_set.add(int(side.mover_idx))
        print(
            f"[scene_builder] coop highlight: thr={coop_thr:.2f} Å, "
            f"{len(mover_set)} movers (indices: {sorted(mover_set)})"
        )
    else:
        mover_set = {int(side.mover_idx)}

    for i, (elt, pos) in enumerate(zip(side.types, side.init_positions, strict=True)):
        is_mover = i in mover_set
        is_primary_mover = i == side.mover_idx
        role = materials.ROLE_MOVER if is_mover else materials.ROLE_NEIGHBOR
        if radius_override > 0:
            base_radius = float(radius_override)
        else:
            base_radius = COVALENT_RADIUS_A.get(elt, 1.3) * RADIUS_SCALE
        radius = base_radius * (mover_boost if is_mover else 1.0)

        if is_primary_mover:
            obj_name = f"{name_prefix}.mover"
        elif is_mover:
            obj_name = f"{name_prefix}.coopmover.{i:03d}"
        else:
            obj_name = f"{name_prefix}.nei.{i:03d}"
        obj = bpy.data.objects.new(name=obj_name, object_data=template)
        obj.location = (float(pos[0]), float(pos[1]), float(pos[2]))
        obj.scale = (radius, radius, radius)
        # Per-object material override (see README for why).
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials.get_material(elt, role)
        collection.objects.link(obj)
        atom_objs.append(obj)
        if is_primary_mover:
            mover_obj = obj

    if mover_obj is None:
        raise RuntimeError("spawned side with no mover")
    return atom_objs, mover_obj


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def _install_camera(
    pair: EventPair,
    total_frames: int,
    style: CameraStyle,
) -> bpy.types.Object:
    fwd = pair.forward
    # Look-at point: midpoint of mover's hop.
    center = (fwd.init_positions[fwd.mover_idx] + fwd.final_positions[fwd.mover_idx]) / 2.0
    # Bounding radius for the atoms (used to scale camera distance).
    radii = np.linalg.norm(fwd.init_positions - center, axis=1)
    bbox_radius = float(radii.max())

    spec = camera.CameraSpec(center=center, radius=bbox_radius, total_frames=total_frames)
    if style == "static":
        cam = camera.create_static(spec)
    elif style == "orbit":
        cam = camera.create_orbit(spec)
    elif style == "dolly":
        cam = camera.create_dolly_zoom(spec)
    else:
        raise ValueError(f"unknown camera style {style!r}")

    bpy.context.scene.camera = cam

    # Overhead sun lamp. Sun lamps are direction-only (location is cosmetic);
    # rotation (0, 0, 0) points straight down -Z in Blender convention.
    # angle = apparent size of the sun disc — smaller = more distant / more
    # parallel rays / sharper shadows. 0.03 rad ≈ 1.7° ≈ real solar disc size.
    light_data = bpy.data.lights.new("KMC_Sun", type="SUN")
    light_data.energy = 6.0
    light_data.angle = 0.03
    light_obj = bpy.data.objects.new("KMC_Sun", light_data)
    light_obj.location = (float(center[0]), float(center[1]), float(center[2] + 30.0))
    light_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(light_obj)

    # Small fill from +y for depth separation.
    fill_data = bpy.data.lights.new("KMC_Fill", type="SUN")
    fill_data.energy = 1.2
    fill_data.angle = 0.25
    fill_obj = bpy.data.objects.new("KMC_Fill", fill_data)
    fill_obj.location = (float(center[0]), float(center[1] + 15.0), float(center[2] + 10.0))
    # Tilt ~25° from vertical so the fill hits the side of the atoms.
    fill_obj.rotation_euler = (np.deg2rad(65.0), 0.0, 0.0)
    bpy.context.scene.collection.objects.link(fill_obj)

    return cam


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        if coll.name not in {"Collection"}:
            bpy.data.collections.remove(coll)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)


def _ensure_collection(
    name: str, parent: bpy.types.Collection | None = None
) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if parent is not None and coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def _link_to_scene(coll: bpy.types.Collection) -> None:
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)


def _build_icosphere_template() -> bpy.types.Mesh:
    # subdivisions=5 gives 2562 verts / 5120 faces — max-detail sphere.
    # Flat shading per-face (set when scene['__flat_shade__'] is True) makes
    # the 5120 triangular polygons visible as individual panels — detail
    # via facets rather than smooth interpolation.
    flat = bool(bpy.context.scene.get("__flat_shade__", False))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0, location=(0, 0, 0))
    template_obj = bpy.context.active_object
    template_obj.name = "IcoTemplate"
    mesh = template_obj.data
    mesh.name = "IcoTemplateMesh"
    if not mesh.materials:
        mesh.materials.append(None)
    # Explicitly set per-face shading:
    #  - flat=True  → per-polygon lighting (5120 triangular facets visible)
    #  - flat=False → smooth interpolation between vertex normals (true-sphere look)
    for poly in mesh.polygons:
        poly.use_smooth = not flat
    template_obj.hide_viewport = True
    template_obj.hide_render = True
    templates_coll = _ensure_collection("Templates")
    _link_to_scene(templates_coll)
    for user_coll in list(template_obj.users_collection):
        user_coll.objects.unlink(template_obj)
    templates_coll.objects.link(template_obj)
    return mesh
