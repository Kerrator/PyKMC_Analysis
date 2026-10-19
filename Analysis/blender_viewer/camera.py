"""Camera animation styles for the event scene.

Each style takes the (forward mover path center, atoms bounding radius) and
writes keyframes onto a freshly created camera object so the playback returns
to its starting camera pose at the loop boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Vector

from Analysis.blender_viewer import animation


@dataclass
class CameraSpec:
    """Geometry-agnostic camera placement inputs."""

    center: np.ndarray      # (3,) world-space point the camera looks at
    radius: float           # bounding radius of the atoms from center (Å)
    total_frames: int       # scene end frame (first+last frame look identical)


def create_static(spec: CameraSpec) -> bpy.types.Object:
    """Stationary camera — the original baseline."""
    cam = _new_camera("KMC_Camera_Static")
    dist = max(spec.radius * 2.3, 14.0)
    cam.location = (
        float(spec.center[0]),
        float(spec.center[1] - dist),
        float(spec.center[2] + spec.radius * 0.5 + 3.0),
    )
    _look_at(cam, spec.center)
    return cam


def create_orbit(
    spec: CameraSpec,
    elevation_deg: float = 35.0,
    pan_arc_deg: float = 45.0,
) -> bpy.types.Object:
    """Camera pans ``pan_arc_deg`` degrees forward and back (pendulum).

    Keyframes: frame 1 at 0°, frame total_frames/2 at ``pan_arc_deg``, frame
    total_frames at 0°. Playback wrap ``total_frames`` → 1 is seamless
    because both endpoints are at 0°. Use Bezier Auto-Clamped handles so
    the pendulum eases naturally at the turnaround.

    Set ``pan_arc_deg = 360.0`` to recover the full single-pass orbit
    (final keyframe at 360° ≡ 0°).
    """
    cam = _new_camera("KMC_Camera_Orbit")
    # Wider framing: 2.21× bbox_radius (20% zoom-out from 1.84).
    dist = max(spec.radius * 2.21, 15.0)
    elev_rad = np.deg2rad(elevation_deg)
    z_offset = dist * np.sin(elev_rad)
    r_horiz = dist * np.cos(elev_rad)

    # Parent Empty at the orbit center; rotate the Empty, camera follows.
    pivot = bpy.data.objects.new("KMC_OrbitPivot", None)
    pivot.empty_display_size = 0.5
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.location = tuple(float(x) for x in spec.center)
    bpy.context.scene.collection.objects.link(pivot)

    cam.parent = pivot
    cam.location = (0.0, -r_horiz, z_offset)
    _look_at_local(cam, Vector((0.0, 0.0, 0.0)))

    # Pendulum pan: 0° at start, `pan_arc_deg` at midpoint, back to 0° at end.
    # (For pan_arc_deg = 360° this collapses to a one-pass full orbit.)
    arc = np.deg2rad(pan_arc_deg)
    mid_frame = 1 + (spec.total_frames - 1) // 2
    key_frames = [1, mid_frame, spec.total_frames]
    key_angles = [0.0, arc, 0.0]

    for frame, angle in zip(key_frames, key_angles, strict=True):
        pivot.rotation_euler = (0.0, 0.0, float(angle))
        pivot.keyframe_insert(data_path="rotation_euler", frame=int(frame))

    if pivot.animation_data and pivot.animation_data.action:
        for fcurve in pivot.animation_data.action.fcurves:
            if fcurve.data_path == "rotation_euler":
                for kp in fcurve.keyframe_points:
                    # Bezier + Auto-Clamped eases the turnaround at midpoint
                    # and the seam at frame 1 / total_frames.
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"

    return cam


def create_dolly_zoom(spec: CameraSpec, saddle_frame: int = 30) -> bpy.types.Object:
    """Push in on the saddle, pull back at the end. Loop-safe.

    Frame plan (roughly): pull out at frame 1 → push in at saddle_frame →
    hold through the far end → pull back out to match frame 1 before loop.
    """
    cam = _new_camera("KMC_Camera_Dolly")
    far_dist = max(spec.radius * 2.8, 16.0)
    near_dist = max(spec.radius * 1.3, 7.0)

    def _place(dist: float) -> tuple[float, float, float]:
        return (
            float(spec.center[0]),
            float(spec.center[1] - dist),
            float(spec.center[2] + spec.radius * 0.35 + 3.0),
        )

    # Keyframes: frame 1 far, saddle near, mid back to far, bwd-saddle near-ish,
    # final back to far. This produces a "breathing" zoom that loops.
    end_frame_inclusive = spec.total_frames + 1
    # Bwd saddle frame defaults to ~105 when total is ~130. Pull from passed-in
    # saddle_frame by mirroring around the halfway point.
    bwd_saddle_frame = spec.total_frames - saddle_frame + 30  # symmetric-ish
    plan: list[tuple[int, float]] = [
        (1, far_dist),
        (saddle_frame, near_dist),
        (spec.total_frames // 2, far_dist),
        (bwd_saddle_frame, near_dist),
        (end_frame_inclusive, far_dist),
    ]
    for frame, dist in plan:
        cam.location = _place(dist)
        cam.keyframe_insert(data_path="location", frame=int(frame))

    animation.convert_to_bezier_auto_clamped(cam)
    _look_at(cam, spec.center)
    return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_camera(name: str) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def _look_at(obj: bpy.types.Object, target: np.ndarray) -> None:
    direction = np.asarray(target, dtype=float) - np.asarray(obj.location, dtype=float)
    quat = Vector(direction.tolist()).to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


def _look_at_local(obj: bpy.types.Object, target_local: Vector) -> None:
    """Orient an object to look at a point in its LOCAL parent space.

    Used for orbit cameras where the camera is parented to a pivot Empty; the
    camera should point at its parent's origin regardless of parent rotation.
    """
    direction = target_local - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
