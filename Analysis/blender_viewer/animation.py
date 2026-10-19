"""Keyframing and interpolation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import bpy
import numpy as np


def keyframe_location(obj: bpy.types.Object, frame: int, pos: np.ndarray) -> None:
    """Set ``obj.location`` at ``frame`` and insert a keyframe."""
    obj.location = (float(pos[0]), float(pos[1]), float(pos[2]))
    obj.keyframe_insert(data_path="location", frame=frame)


def convert_to_bezier_auto_clamped(obj: bpy.types.Object) -> None:
    """Force every location keyframe on ``obj`` to Bezier / Auto Clamped.

    Auto Clamped prevents the default Aligned handles from overshooting the
    saddle control point — critical for the animation to pass cleanly through
    all three keyframes.
    """
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    for fcurve in action.fcurves:
        if not fcurve.data_path.startswith("location"):
            continue
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def keyframe_visibility(obj: bpy.types.Object, frame: int, visible: bool) -> None:
    """Keyframe viewport + render visibility with constant interpolation.

    Using constant interpolation ensures the visibility flips cleanly on a
    single frame rather than fading linearly between states.
    """
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)

    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    for fcurve in action.fcurves:
        if fcurve.data_path in ("hide_viewport", "hide_render"):
            for kp in fcurve.keyframe_points:
                kp.interpolation = "CONSTANT"


def animate_side(
    atom_objs: Iterable[bpy.types.Object],
    init_pos: np.ndarray,
    saddle_pos: np.ndarray,
    final_pos: np.ndarray,
    frames: tuple[int, int, int],
) -> None:
    """Keyframe a sequence of atoms at (init, saddle, final) on the given frames."""
    f_init, f_saddle, f_final = frames
    for i, obj in enumerate(atom_objs):
        keyframe_location(obj, f_init, init_pos[i])
        keyframe_location(obj, f_saddle, saddle_pos[i])
        keyframe_location(obj, f_final, final_pos[i])
        convert_to_bezier_auto_clamped(obj)
