"""Material factory for the Blender viewer (role x element, themed).

Caches the bpy.Material objects so the same material is shared across all
atoms of a given (element, role) pair. Element palette, themes, and
Principled BSDF defaults come from
:mod:`Analysis.blender_viewer.constants` — edit that file to change shared
visual conventions.

Theme override
--------------
The scene-level custom property ``scene['__theme__']`` (set by
``scene_builder.build_scene``) selects between ``'default'`` (Jmol
element palette) and ``'burgundy'`` (uniform wine-red with a slightly
brighter mover). When a non-default theme is active, element-specific
palette is ignored — every atom of non-mover role gets
``theme['neighbor_color']`` and the mover gets ``theme['mover_color']``.
"""

from __future__ import annotations

import bpy

from Analysis.blender_viewer.constants import (
    JMOL_HEX,
    MOVER_COLOR_RGB,
    MOVER_EMISSION_STRENGTH,
    MOVER_ROUGHNESS,
    PRINCIPLED_METALLIC_DEFAULTS,
    THEMES,
    element_color,
)

# Role constants.
ROLE_NEIGHBOR = "neighbor"
ROLE_MOVER = "mover"
ROLE_CONTEXT = "context"

_CACHE: dict[tuple[str, str, str], bpy.types.Material] = {}


def _current_theme() -> dict:
    """Look up the theme config from the active scene's custom property."""
    scene = bpy.context.scene
    name = scene.get("__theme__", "default")
    return THEMES.get(str(name), THEMES["default"])


def get_material(element: str, role: str) -> bpy.types.Material:
    """Return a cached bpy.Material for the given (element, role) pair."""
    theme_cfg = _current_theme()
    theme_name = bpy.context.scene.get("__theme__", "default")

    key = (element, role, str(theme_name))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    name = f"Mat_{theme_name}_{element}_{role}"
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    mover_color = theme_cfg.get("mover_color", MOVER_COLOR_RGB)

    if role == ROLE_MOVER:
        bsdf.inputs["Base Color"].default_value = (*mover_color, 1.0)
        bsdf.inputs["Roughness"].default_value = MOVER_ROUGHNESS
        bsdf.inputs["Metallic"].default_value = PRINCIPLED_METALLIC_DEFAULTS["metallic"]
        _set_emission(bsdf, color=mover_color, strength=MOVER_EMISSION_STRENGTH)
    else:
        if theme_cfg.get("use_element_palette", True):
            color = element_color(element)
        else:
            color = theme_cfg.get("neighbor_color", element_color(element))
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = PRINCIPLED_METALLIC_DEFAULTS["roughness"]
        bsdf.inputs["Metallic"].default_value = PRINCIPLED_METALLIC_DEFAULTS["metallic"]
        _set_emission(bsdf, color=color, strength=0.0)

    _CACHE[key] = mat
    return mat


def _set_emission(
    bsdf: bpy.types.ShaderNodeBsdfPrincipled,
    color: tuple[float, float, float],
    strength: float,
) -> None:
    """Set emission inputs on the Principled BSDF, coping with 4.x node renames."""
    for color_key in ("Emission Color", "Emission"):
        if color_key in bsdf.inputs:
            bsdf.inputs[color_key].default_value = (*color, 1.0)
            break
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength


def reset_cache() -> None:
    """Drop the in-memory cache. Useful if the scene is being rebuilt."""
    _CACHE.clear()


ELEMENT_PALETTE_HEX = JMOL_HEX
