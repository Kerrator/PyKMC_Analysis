"""Headless render driver for a single showreel .blend.

Used by ``Analysis.blender_viewer.showreel.render_all`` (the ``showreel
render`` CLI step) and ``showreel draft``. One Blender process per blend; the
showreel runner can spawn N of these in parallel.

Invocation pattern::

    blender --background <event_<family>.blend> \\
            --python <this_script> -- \\
            --out-dir /path/to/frames/event_<family> \\
            --engine EEVEE_NEXT --fps 60 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import sys

import bpy

# Args after `--` belong to us.
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

p = argparse.ArgumentParser()
p.add_argument("--out-dir", required=True)
p.add_argument("--engine", default="EEVEE_NEXT",
               help="EEVEE_NEXT (matte, fast) or CYCLES (raytraced, shiny)")
p.add_argument("--fps", type=int, default=60)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--samples", type=int, default=32,
               help="Cycles samples (ignored for Eevee). 32 = showreel default;"
                    " 64 = slightly cleaner but ~50% slower.")
args = p.parse_args(argv)

scene = bpy.context.scene
# Map user-facing engine names to Blender's enum values. Cycles is the
# only one that does NOT take a BLENDER_ prefix.
_ENGINE_ALIASES = {
    "EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
    "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
    "WORKBENCH": "BLENDER_WORKBENCH",
    "BLENDER_WORKBENCH": "BLENDER_WORKBENCH",
    "CYCLES": "CYCLES",
}
scene.render.engine = _ENGINE_ALIASES.get(args.engine, args.engine)
scene.render.resolution_x = args.width
scene.render.resolution_y = args.height
scene.render.resolution_percentage = 100
scene.render.fps = args.fps
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.filepath = f"{args.out_dir}/frame_####.png"

# Engine-specific quality knobs. Cycles gives crisp specular highlights +
# inter-sphere reflections; Eevee Next without raytracing produces a matte
# look. CYCLES is recommended for final renders; EEVEE_NEXT is the fast option.
if scene.render.engine == "CYCLES":
    cy = scene.cycles
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.01
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.device = "GPU"
    # Use the first available GPU backend (Metal, OptiX, CUDA, HIP) and all its devices.
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("METAL", "OPTIX", "CUDA", "HIP"):
        try:
            prefs.compute_device_type = dev_type
            break
        except (TypeError, ValueError):
            continue
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = True
elif scene.render.engine == "BLENDER_EEVEE_NEXT":
    eevee = scene.eevee
    if hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = True
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 64

# Color management — AgX is preferred for the metallic burgundy palette.
scene.view_settings.view_transform = "AgX"

print(
    f"[render_one_blend] {scene.render.engine} {args.width}x{args.height}@{args.fps}fps "
    f"frames {scene.frame_start}..{scene.frame_end} → {args.out_dir}"
)
bpy.ops.render.render(animation=True)
print("[render_one_blend] done")
