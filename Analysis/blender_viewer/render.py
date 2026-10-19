"""Render configuration presets for stills and animations.

Three entry points:
- :func:`configure_publication_still` — Cycles 512 samples, AgX, transparent
  RGBA, 300–600 DPI output at journal column widths.
- :func:`configure_animation` — Cycles 128 samples at 1080p60 as a PNG
  sequence (resumable on crash); concatenate to H.264 MP4 via
  :func:`png_sequence_to_mp4` afterward.
- :func:`configure_eevee_previs` — fast Eevee Next previews for composition
  checking before committing to Cycles.

All functions modify the current scene in-place. They do not trigger a
render — call :func:`render_still` or :func:`render_animation` (or use
``bpy.ops.render.render``) when ready.

For the **complete final-render preset** (Cycles 32 samples + OIDN +
GPU + 1920×1080 + 60 fps + ``use_neb=True``) used for the curated
showreel, see :data:`constants.FINAL_RENDER_PRESET` and the "Final
render workflow" section in this package's ``README.md``. The :func:`configure_animation`
defaults here (128 samples, OPTIX denoiser) target NVIDIA hardware;
on Apple Silicon override ``samples=32`` and ``denoiser="OPENIMAGEDENOISE"``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

import bpy

from Analysis.blender_viewer.constants import (
    ANIMATION_PRESET,
    DPI_MIN_HALFTONE,
    DPI_PREFERRED,
    EEVEE_PREVIS_PRESET,
    JOURNAL_COLUMN_MM,
    PUBLICATION_PRESET,
)

ColumnKey = Literal["single_narrow", "single_wide", "double", "double_aps", "double_elsevier"]


# ---------------------------------------------------------------------------
# Publication still
# ---------------------------------------------------------------------------


def configure_publication_still(
    column: ColumnKey = "double",
    aspect_ratio: float = 4.0 / 3.0,
    dpi: int = DPI_MIN_HALFTONE,
    samples: int = PUBLICATION_PRESET["samples"],
    transparent: bool = True,
    output_path: str | Path | None = None,
) -> None:
    """Configure Cycles for a journal-ready still.

    Parameters
    ----------
    column : which journal column width to target (see ``JOURNAL_COLUMN_MM``)
    aspect_ratio : width / height. 4:3 is the default; 1:1 for single-atom
        close-ups; 16:9 for video abstracts.
    dpi : 300 min halftone (Nature, Elsevier); 600 preferred for APS.
    samples : Cycles samples. 512 is publication-standard; drop to 256 for
        scenes without HDRI/volumetrics.
    transparent : RGBA output for Inkscape compositing of scale bars & labels.
    output_path : absolute path for the rendered PNG. If None, leave unset.
    """
    scene = bpy.context.scene
    render = scene.render

    render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.adaptive_threshold = PUBLICATION_PRESET["adaptive_threshold"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = PUBLICATION_PRESET["use_denoising"]
    scene.cycles.denoiser = PUBLICATION_PRESET["denoiser"]
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

    # Device selection (GPU when available).
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences
    cycles_prefs.refresh_devices()
    for dev_type in ("OPTIX", "METAL", "HIP", "CUDA", "NONE"):
        try:
            cycles_prefs.compute_device_type = dev_type
        except (TypeError, ValueError):
            continue
        devices = list(cycles_prefs.devices)
        if any(d.type == dev_type for d in devices):
            scene.cycles.device = "GPU"
            for d in devices:
                d.use = (d.type in {dev_type, "CPU"})
            break
    else:
        scene.cycles.device = "CPU"

    # Dimensions: pixels = mm × dpi / 25.4.
    width_mm = JOURNAL_COLUMN_MM[column]
    width_in = width_mm / 25.4
    px_x = int(round(width_in * dpi))
    px_y = int(round(px_x / aspect_ratio))
    render.resolution_x = px_x
    render.resolution_y = px_y
    render.resolution_percentage = 100

    # Film / output.
    render.film_transparent = transparent
    render.image_settings.file_format = PUBLICATION_PRESET["file_format"]
    render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    render.image_settings.color_depth = PUBLICATION_PRESET["color_depth"]

    # Colour management.
    view = scene.view_settings
    view.view_transform = PUBLICATION_PRESET["view_transform"]
    view.look = PUBLICATION_PRESET["look"]

    if output_path is not None:
        render.filepath = str(Path(output_path))


# ---------------------------------------------------------------------------
# Animation (PNG sequence → MP4)
# ---------------------------------------------------------------------------


def configure_animation(
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    fps: int = ANIMATION_PRESET["fps"],
    samples: int = ANIMATION_PRESET["samples"],
    output_dir: str | Path | None = None,
    engine: Literal["CYCLES", "EEVEE"] = "CYCLES",
) -> None:
    """Configure a scene for animation rendering to a PNG sequence.

    Always render to PNG sequence first, then concatenate via
    :func:`png_sequence_to_mp4`. Direct ffmpeg-in-Blender output is fragile
    and non-resumable on a crash.
    """
    scene = bpy.context.scene
    render = scene.render

    if engine == "CYCLES":
        render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = ANIMATION_PRESET["denoiser"]
    else:
        render.engine = "BLENDER_EEVEE_NEXT"

    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = 100
    render.fps = fps

    render.image_settings.file_format = ANIMATION_PRESET["file_format"]
    render.image_settings.color_mode = "RGB"
    render.image_settings.color_depth = ANIMATION_PRESET["color_depth"]
    render.film_transparent = ANIMATION_PRESET["film_transparent"]

    view = scene.view_settings
    view.view_transform = ANIMATION_PRESET["view_transform"]
    view.look = ANIMATION_PRESET["look"]

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Blender's `####.png` pattern turns into 0001.png, 0002.png, ...
        render.filepath = str(out / "frame_####.png")


def png_sequence_to_mp4(
    frames_dir: str | Path,
    out_mp4: str | Path,
    fps: int = 60,
    crf: int = 18,
    pattern: str = "frame_%04d.png",
) -> subprocess.CompletedProcess:
    """Concatenate a PNG sequence into an H.264 MP4 via ffmpeg.

    Uses yuv420p pixel format for universal PowerPoint/Keynote
    compatibility. Requires ffmpeg on PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH; install via brew/apt")
    frames_dir = Path(frames_dir)
    out_mp4 = Path(out_mp4)
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(out_mp4),
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Eevee previs
# ---------------------------------------------------------------------------


def configure_eevee_previs(
    resolution_x: int = 1280,
    resolution_y: int = 720,
    samples: int = EEVEE_PREVIS_PRESET["samples"],
) -> None:
    """Configure Eevee Next for rapid composition checks before Cycles."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    # Ensure PNG output (many default scenes ship with FFMPEG selected for
    # animation; a single still render fails otherwise).
    scene.render.image_settings.file_format = EEVEE_PREVIS_PRESET["file_format"]
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    # Eevee settings are nested under scene.eevee.
    eev = scene.eevee
    eev.taa_render_samples = samples
    if hasattr(eev, "use_raytracing"):
        eev.use_raytracing = EEVEE_PREVIS_PRESET["use_raytracing"]
    if hasattr(eev, "use_ssr"):
        eev.use_ssr = EEVEE_PREVIS_PRESET["use_ssr"]
    if hasattr(eev, "use_bloom"):
        eev.use_bloom = EEVEE_PREVIS_PRESET["use_bloom"]


# ---------------------------------------------------------------------------
# Trigger renders
# ---------------------------------------------------------------------------


def render_still(filepath: str | Path | None = None) -> Path:
    """Render a single frame to ``filepath``. Returns the saved path."""
    scene = bpy.context.scene
    if filepath is not None:
        scene.render.filepath = str(Path(filepath))
    bpy.ops.render.render(write_still=True)
    return Path(scene.render.filepath)


def render_animation(output_dir: str | Path | None = None) -> Path:
    """Render the full frame range to a PNG sequence. Returns the output directory."""
    scene = bpy.context.scene
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out / "frame_####.png")
    bpy.ops.render.render(animation=True)
    return Path(scene.render.filepath).parent
