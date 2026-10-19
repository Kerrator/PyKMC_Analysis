"""Lighting rigs for KMC scenes.

Three styles:
- :func:`three_point_studio` — Birn (2013) key/fill/rim rig.
- :func:`polyhaven_hdri` — image-based lighting from a Poly Haven HDRI that
  is already loaded as the world environment.
- :func:`workbench_matcap` — switches to Workbench engine with a MatCap
  shade for OVITO-like schematic rendering of very large systems.
"""

from __future__ import annotations

from math import cos, radians, sin

import bpy
import numpy as np
from mathutils import Vector

from Analysis.blender_viewer.constants import (
    THREE_POINT_FILL,
    THREE_POINT_KEY,
    THREE_POINT_RIM,
    LightSpec,
)


# ---------------------------------------------------------------------------
# Three-point studio
# ---------------------------------------------------------------------------


def three_point_studio(
    target: np.ndarray,
    target_radius: float = 20.0,
    key: LightSpec = THREE_POINT_KEY,
    fill: LightSpec = THREE_POINT_FILL,
    rim: LightSpec = THREE_POINT_RIM,
    collection_name: str = "Lights",
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """Install a three-point area-light rig aimed at ``target``.

    ``target_radius`` sets the distance each light sits from the target
    (Å). A good rule of thumb is 1.5–3× the supercell bounding radius.
    Returns (key_obj, fill_obj, rim_obj).
    """
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)

    target = np.asarray(target, dtype=float)
    key_obj = _place_area_light(key, target, target_radius, coll)
    fill_obj = _place_area_light(fill, target, target_radius, coll)
    rim_obj = _place_area_light(rim, target, target_radius, coll)
    return key_obj, fill_obj, rim_obj


def _place_area_light(
    spec: LightSpec, target: np.ndarray, r: float, coll: bpy.types.Collection,
) -> bpy.types.Object:
    az = radians(spec.azimuth_deg)
    el = radians(spec.elevation_deg)
    # Position on a sphere of radius r around the target.
    position = target + r * np.array([
        cos(el) * cos(az),
        cos(el) * sin(az),
        sin(el),
    ])

    light_data = bpy.data.lights.new(spec.name, type="AREA")
    light_data.energy = spec.energy_watts
    light_data.size = spec.size_A
    # Blackbody colour mix via the Light's color slot + a driver isn't needed;
    # Blender's "color" is already sRGB — we approximate the Kelvin with a
    # small helper table.
    light_data.color = _kelvin_to_rgb(spec.colour_temperature_k)

    light_obj = bpy.data.objects.new(spec.name, light_data)
    light_obj.location = tuple(float(x) for x in position)
    coll.objects.link(light_obj)

    # Track-to constraint at target.
    _look_at(light_obj, target)
    return light_obj


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """Rough Blackbody -> sRGB mapping.

    For scene-lighting purposes we keep this to a small lookup table; the
    exact colour is less important than hitting the "warm/neutral/cool"
    distinction of key/fill/rim.
    """
    if k < 4000:
        return (1.00, 0.80, 0.60)
    if k < 5000:
        return (1.00, 0.90, 0.75)
    if k < 6000:
        return (1.00, 0.96, 0.90)
    if k < 7000:
        return (1.00, 1.00, 1.00)
    return (0.90, 0.94, 1.00)


def _look_at(obj: bpy.types.Object, target: np.ndarray) -> None:
    direction = np.asarray(target, dtype=float) - np.asarray(obj.location, dtype=float)
    quat = Vector(direction.tolist()).to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


# ---------------------------------------------------------------------------
# Poly Haven HDRI (load the HDRI into the scene first)
# ---------------------------------------------------------------------------


def polyhaven_hdri(
    asset_id: str = "studio_small_03",
    resolution: str = "4k",
    strength: float = 0.5,
    rotation_deg_z: float = 45.0,
) -> None:
    """Install a Poly Haven HDRI as the world environment.

    Load the HDRI BEFORE calling this: download the ``.exr`` for
    ``asset_id`` at ``resolution`` from https://polyhaven.com/hdris and add
    it as the world's Environment Texture (or use a Poly Haven Blender
    add-on that does this). Once the image is attached to the world, this
    function just adjusts strength and rotation (the rotation applies only
    if the world has a Mapping node).

    If no HDRI is attached, the world background is left as it is apart
    from its strength. ``asset_id`` and ``resolution`` are not used here;
    they record which asset the settings were tuned for.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes

    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")

    bg.inputs["Strength"].default_value = strength

    # If a Mapping node exists, set the Z rotation.
    mapping = nodes.get("Mapping")
    if mapping is not None:
        mapping.inputs["Rotation"].default_value[2] = radians(rotation_deg_z)


# ---------------------------------------------------------------------------
# Workbench MatCap (OVITO-like schematic)
# ---------------------------------------------------------------------------


def workbench_matcap(
    matcap: str = "basic_1.exr",
    cavity_ridge: float = 0.25,
    cavity_valley: float = 1.00,
    outline: bool = True,
    outline_opacity: float = 1.0,
) -> None:
    """Switch the scene to Workbench engine with a MatCap shade.

    Fast, deterministic rendering suited to very long KMC animations
    (>10⁴ frames) or supplementary figures. This route produces an
    OVITO-like edge-enhanced look.
    """
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"

    display = scene.display
    display.shading.light = "MATCAP"
    display.shading.studio_light = matcap
    display.shading.show_cavity = True
    display.shading.cavity_type = "BOTH"
    display.shading.cavity_ridge_factor = cavity_ridge
    display.shading.cavity_valley_factor = cavity_valley
    if outline:
        display.shading.show_object_outline = True
        # object_outline_color is the colour, not opacity; workbench doesn't
        # expose an opacity slider — the presence of the outline is the toggle.
