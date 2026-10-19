"""Display conventions and materials-science constants for the Blender viewer.

Single source of truth for:
- Element display colours (Jmol palette)
- Metallic and covalent radii (Å)
- Common FCC/BCC lattice parameters
- Nearest-neighbour distance calculations
- Blender material defaults for metallic atoms
- Animation timing conventions

All modules in this package import from here rather than redefining their own
tables — keeps the visual conventions consistent across event viz, lattice
figures, trajectory playback, and publication renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

# ---------------------------------------------------------------------------
# Element display palette (Jmol/VESTA convention)
# ---------------------------------------------------------------------------

# Hex colours from the Jmol palette, mirrored by VESTA and by most
# materials-visualization tools (VMD, OVITO, ASE). Keeping this consistent
# ensures figures render with the same colours as the community expects.
JMOL_HEX: dict[str, str] = {
    "H":  "#FFFFFF",
    "Ni": "#50D050",
    "Fe": "#E06633",
    "Cr": "#8A99C7",
    "Al": "#BFA6A6",
    "Cu": "#C88033",
    "Au": "#FFD123",
    "Ag": "#C0C0C0",
    "Pt": "#D0D0E0",
    "Pd": "#006985",
}


def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) floats in [0, 1] (sRGB, not linear)."""
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def element_color(element: str, default: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> tuple[float, float, float]:
    """Return an sRGB triplet for an element; fall back to neutral grey."""
    if element in JMOL_HEX:
        return hex_to_rgb(JMOL_HEX[element])
    return default


# ---------------------------------------------------------------------------
# Metallic and covalent radii (Å) at 12-coordination (Wells 1984; ASE data)
# ---------------------------------------------------------------------------

METALLIC_RADIUS_A: dict[str, float] = {
    "Ni": 1.25, "Fe": 1.26, "Cr": 1.29, "Al": 1.43,
    "Cu": 1.28, "Au": 1.44, "Ag": 1.44, "Pt": 1.39, "Pd": 1.37,
}

COVALENT_RADIUS_A: dict[str, float] = {
    "H":  0.31,
    "Ni": 1.24, "Fe": 1.32, "Cr": 1.39, "Al": 1.21,
    "Cu": 1.32, "Au": 1.36, "Ag": 1.45, "Pt": 1.36, "Pd": 1.39,
}

# Display scale: sphere radius drawn = metallic_radius * RADIUS_SCALE.
# 0.40 is the VESTA convention (Momma & Izumi 2011) leaving a 60% gap for
# bond visibility; 0.55 is punchier and reads well on projected slides.
# Event viz uses 0.55 as its default; override it for stills.
VESTA_RADIUS_SCALE = 0.40
EVENT_RADIUS_SCALE = 0.55

# Mover atom boost (used by blender_viewer.scene_builder to make the hopping
# atom read as the focus in an animation).
MOVER_RADIUS_BOOST = 1.15


def display_radius(element: str, scale: float = EVENT_RADIUS_SCALE) -> float:
    """Return the Blender-unit sphere radius to use for an element."""
    r = METALLIC_RADIUS_A.get(element) or COVALENT_RADIUS_A.get(element, 1.30)
    return r * scale


# ---------------------------------------------------------------------------
# Lattice parameters (Å) for common structures
# ---------------------------------------------------------------------------

FCC_LATTICE_A: dict[str, float] = {
    "Ni": 3.524, "Cu": 3.615, "Au": 4.078, "Ag": 4.086, "Pt": 3.924,
    "Pd": 3.891, "Al": 4.050,
    # γ-Fe is metastable FCC; 3.58 is the DFT/LAMMPS-common value.
    "Fe": 3.580,
    # FCC Cr is metastable; 3.62 is the DFT-common value.
    "Cr": 3.620,
}

BCC_LATTICE_A: dict[str, float] = {
    "Fe": 2.866, "Cr": 2.910, "W": 3.165, "Mo": 3.147,
}


def fcc_nn_distance(a: float) -> float:
    """FCC nearest-neighbour distance = a / sqrt(2)."""
    return a / sqrt(2)


def bcc_nn_distance(a: float) -> float:
    """BCC nearest-neighbour distance = a * sqrt(3) / 2."""
    return a * sqrt(3) / 2


# Common nearest-neighbour (1NN) distances, precomputed for convenience.
# These are the canonical hop distances for vacancy diffusion.
NN1_DISTANCE_A: dict[str, float] = {
    "Ni_fcc": fcc_nn_distance(FCC_LATTICE_A["Ni"]),  # 2.492
    "Fe_bcc": bcc_nn_distance(BCC_LATTICE_A["Fe"]),  # 2.482
    "Cu_fcc": fcc_nn_distance(FCC_LATTICE_A["Cu"]),  # 2.556
    "Al_fcc": fcc_nn_distance(FCC_LATTICE_A["Al"]),  # 2.864
}


# OVITO's Common Neighbour Analysis (CNA) FCC cutoff (Stukowski 2012).
# Use this when detecting 1NN shells in FCC: r_cut = 0.854 * a0.
def cna_cutoff_fcc(a: float) -> float:
    return 0.854 * a


# ---------------------------------------------------------------------------
# Blender render / material conventions
# ---------------------------------------------------------------------------

# Principled BSDF defaults for metallic atoms. Chosen for readability at
# small screen sizes. Roughness 0.25–0.35 is the practical window — below
# 0.20 gives chrome pinpricks that alias; above 0.40 looks plastic.
PRINCIPLED_METALLIC_DEFAULTS = {
    "metallic": 1.0,
    "roughness": 0.30,
    "coat_weight": 0.0,
    "subsurface": 0.0,
}

# Mover highlight: vivid orange with modest emission. Element-independent.
# (Default theme; overridden when `scene['__theme__']` is set — see materials.py.)
MOVER_COLOR_RGB = (1.00, 0.40, 0.10)
MOVER_EMISSION_STRENGTH = 0.00   # 0 → mover is visually identical to
                                 # neighbours; distinguished only by motion.
                                 # Raise (e.g. 0.8) for a subtle inner-glow cue.
MOVER_ROUGHNESS = 0.30           # match neighbour roughness (was 0.35)

# Context / ghost lattice atoms (static reference in event viz): Alpha-blended.
CONTEXT_ALPHA = 0.70
CONTEXT_ROUGHNESS = 0.40

# ---------------------------------------------------------------------------
# Colour themes (override element palette + mover colour)
# ---------------------------------------------------------------------------

# Burgundy / wine-red theme: all atoms a deep burgundy, mover a vibrant
# slightly-brighter shade in the same hue family. Keeps the hop readable via
# brightness contrast while unifying the crystal as a single colour.
BURGUNDY_RGB = (0.40, 0.08, 0.11)                 # deep burgundy
# Mover shares the neighbour base colour — the only distinction is the
# emission glow (MOVER_EMISSION_STRENGTH). Pure "lit-from-within" cue.
MOVER_VIBRANT_BURGUNDY_RGB = BURGUNDY_RGB

# Yellow gold for cooperative-mover highlights. Used by the
# `burgundy_gold_coop` theme when more than one atom moves significantly
# during the hop (e.g. 1NN_hop_triple, 1NN_hop_cooperative). Picked so
# the highlighted set reads as a unit even when the three atoms straddle
# the centre of frame.
GOLD_RGB = (1.00, 0.84, 0.00)

# Displacement threshold (Å) above which an atom is considered a
# "mover" in cooperative-highlight mode. 0.5 Å is comfortably above the
# vibrational noise floor (~0.05 Å) and well below a 1NN hop (~2.5 Å on
# Ni FCC), so it catches every co-hopping atom without lighting up
# second-shell relaxation.
COOP_DISPLACEMENT_THR_A = 0.5

THEMES: dict[str, dict] = {
    "default": {
        "use_element_palette": True,
        "mover_color": MOVER_COLOR_RGB,
    },
    "burgundy": {
        "use_element_palette": False,
        "neighbor_color": BURGUNDY_RGB,
        "mover_color": MOVER_VIBRANT_BURGUNDY_RGB,
    },
    "burgundy_gold_coop": {
        # Burgundy crystal, all co-hopping atoms (|final - init| above the
        # COOP_DISPLACEMENT_THR_A threshold) lit in yellow gold. Pairs
        # with scene_builder.build_scene(cooperative_highlight=True).
        "use_element_palette": False,
        "neighbor_color": BURGUNDY_RGB,
        "mover_color": GOLD_RGB,
    },
}


# ---------------------------------------------------------------------------
# Lighting preset (three-point studio, Birn 2013)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LightSpec:
    name: str
    azimuth_deg: float      # +x = 0°, +y = 90°; rotation about z axis at target
    elevation_deg: float    # above horizon
    colour_temperature_k: float
    energy_watts: float
    size_A: float           # Area light size (Å)


THREE_POINT_KEY = LightSpec(
    name="Key", azimuth_deg=45, elevation_deg=30,
    colour_temperature_k=5500, energy_watts=800, size_A=20.0,
)
THREE_POINT_FILL = LightSpec(
    name="Fill", azimuth_deg=-30, elevation_deg=15,
    colour_temperature_k=4500, energy_watts=400, size_A=25.0,
)
THREE_POINT_RIM = LightSpec(
    name="Rim", azimuth_deg=180, elevation_deg=45,
    colour_temperature_k=6500, energy_watts=500, size_A=15.0,
)


# ---------------------------------------------------------------------------
# Render presets
# ---------------------------------------------------------------------------

# Publication still: Cycles, AgX, high-DPI with transparent background.
PUBLICATION_PRESET = {
    "engine": "CYCLES",
    "samples": 512,
    "adaptive_threshold": 0.01,
    "use_denoising": True,
    "denoiser": "OPENIMAGEDENOISE",
    "film_transparent": True,
    "color_mode": "RGBA",
    "color_depth": "16",
    "view_transform": "AgX",
    "look": "AgX - Base Contrast",
    "file_format": "PNG",
}

# Presentation animation: smaller res, lower samples, opaque RGB.
ANIMATION_PRESET = {
    "engine": "CYCLES",
    "samples": 128,
    "use_denoising": True,
    "denoiser": "OPTIX",   # fall back to OPENIMAGEDENOISE if no NVIDIA GPU
    "film_transparent": False,
    "color_mode": "RGB",
    "color_depth": "8",
    "view_transform": "AgX",
    "look": "AgX - Base Contrast",
    "file_format": "PNG",  # render to PNG sequence, then ffmpeg → MP4
    "fps": 60,
}

# Eevee previs: for rapid iteration before committing to Cycles.
EEVEE_PREVIS_PRESET = {
    "engine": "BLENDER_EEVEE_NEXT",
    "samples": 16,
    "use_ssr": True,
    "use_bloom": True,
    "use_raytracing": True,
    "file_format": "PNG",
}

# ---------------------------------------------------------------------------
# Three-stage render workflow
# ---------------------------------------------------------------------------
#
# Final-quality renders go through three stages to keep iteration cheap.
# These presets are reference values; nothing applies them automatically.
# Choose theme / resolution / trajectory explicitly for each job.
#
# Stage 1 — DRAFT
#   Fast 3-keyframe Bezier path, single representative event, lower res
#   and detail. Purpose: lock camera angle, radius mode, theme, mover
#   boost. ~1–3 min per attempt.
#
# Stage 2 — NEB_VALIDATION
#   Same single event, same low/mid res, but use_neb=True so the mover
#   follows the LAMMPS-computed minimum energy path. Purpose: confirm
#   the NEB images flow naturally through the saddle (no sudden jumps,
#   no atom overlap, neighbour relaxation looks right). ~5–10 min total
#   (~2 min NEB compute + Bezier-time render).
#
# Stage 3 — FINAL
#   Full curated manifest, target resolution (720p draft / 1080p),
#   Cycles on the GPU (Metal on Apple Silicon), use_neb=True (MANDATORY),
#   theme locked. Render-then-compile loop with ffmpeg drawtext +
#   native-resolution title cards; on macOS wrap it in `caffeinate -i -s`
#   to prevent sleep. ~10–20 h depending on event count + resolution.
#
# Why NEB is required for FINAL: the 3-keyframe Bezier through
# (initial, saddle, final) misses the neighbour relaxation along the
# actual MEP — second-shell atoms move in physically incorrect ways.
# 11-image NEB captures this.

RENDER_STAGE_DRAFT = "draft"
RENDER_STAGE_NEB_VALIDATION = "neb_validation"
RENDER_STAGE_FINAL = "final"

DRAFT_RENDER_PRESET = {
    "stage": RENDER_STAGE_DRAFT,
    "engine": "BLENDER_EEVEE_NEXT",  # fastest; switch to CYCLES if look diverges
    "samples": 16,
    "width": 960, "height": 540,
    "fps": 30,
    "use_neb": False,                # 3-keyframe Bezier path
    "style": "loop", "camera_style": "orbit",
    "embed_title": True,             # quick visual label, no compile step
    "close_pack": True,
    "n_events": 1,                   # one representative event per draft
    "compile_titles": False,
    "expected_runtime": "1-3 min per attempt",
}

NEB_VALIDATION_PRESET = {
    "stage": RENDER_STAGE_NEB_VALIDATION,
    "engine": "BLENDER_EEVEE_NEXT",
    "samples": 16,
    "width": 960, "height": 540,
    "fps": 30,
    "use_neb": True,                 # the point of this stage
    "neb_n_images": 11,              # default NEB image count
    "style": "loop",                 # NEB only works with loop in scene_builder
    "camera_style": "orbit",
    "embed_title": True,
    "close_pack": True,
    "n_events": 1,                   # same single event as draft
    "compile_titles": False,
    "expected_runtime": "5-10 min (2 min NEB + render)",
}

FINAL_RENDER_PRESET = {
    "stage": RENDER_STAGE_FINAL,
    "engine": "CYCLES",
    "samples": 32,
    "use_denoising": True,
    "denoiser": "OPENIMAGEDENOISE",  # OIDN = Metal-GPU friendly on Apple Silicon
    "device": "GPU",                 # METAL on Apple Silicon
    "width": 1920, "height": 1080,   # or 1280x720 — choose per job
    "fps": 60,
    "use_neb": True,                 # MANDATORY for final
    "neb_n_images": 11,
    "style": "loop",
    "camera_style": "orbit",
    "embed_title": False,            # cards/drawtext added at compile step
    "close_pack": True,
    "compile_titles": True,          # native-resolution title cards (1 s each)
    "compile_drawtext": True,        # bottom-left "family_id  Ea=X.XXX eV"
    "render_loop_chunked": True,     # render→ffmpeg→delete frames per event
    "use_caffeinate": True,          # caffeinate -i -s to block macOS sleep
    "stdin_array_workaround": True,  # loop over an array, not `| while read` (Blender eats stdin)
    "expected_runtime": "10-20 h for full curated manifest",
}

# Pre-render checklist. Settle each item before launching any 720p+
# multi-event render (any job that will take >= 30 min wall time). There
# are no implicit defaults for theme or resolution. Nothing reads this
# tuple automatically; it documents the decisions a final render needs.
FINAL_RENDER_QA_QUESTIONS = (
    ("stage",       "Which render stage? (draft / neb_validation / final)"),
    ("resolution",  "Resolution? (540p draft / 720p / 1080p)"),
    ("trajectory",  "Mover trajectory? (bezier_3kf / neb)  [final MUST be neb]"),
    ("theme",       "Theme? (default jmol / burgundy / other)"),
    ("camera",      "Camera style? (orbit / static / dolly)"),
    ("radius_mode", "Atom packing? (close_pack / sparse)"),
    ("scope",       "Scope? (single event / curated manifest / custom subset)"),
    ("output",      "Output directory + final MP4 name?"),
    ("confirm",     "Caffeinate + chunked render-loop + stdin-array confirmed?"),
)

# Journal column widths (mm) — 300 DPI minimum halftone per Nature/APS/Elsevier.
JOURNAL_COLUMN_MM = {
    "single_narrow": 89,     # Nature single column
    "single_wide":   90,     # Acta Materialia single column
    "double":       183,     # Nature double column (also npj Computational Materials)
    "double_aps":   180,     # Physical Review Materials double column
    "double_elsevier": 190,  # Acta Materialia double column
}

DPI_MIN_HALFTONE = 300
DPI_PREFERRED = 600


# ---------------------------------------------------------------------------
# Animation timing conventions (KMC-specific)
# ---------------------------------------------------------------------------

# Frames per vacancy hop at 24 fps: 24 = 1 s, 60 = 2.5 s.
# 24 = dense trajectories (hundreds of events); 60 = pedagogical close-ups.
FRAMES_PER_HOP_DENSE = 24
FRAMES_PER_HOP_PEDAGOGICAL = 60

# Bezier handle type — AUTO_CLAMPED prevents overshoot through the saddle
# control point (Aligned would create phantom motion through the barrier).
BEZIER_HANDLE_TYPE = "AUTO_CLAMPED"

# KMC time→frame mapping modes.
# "proper" preserves stochastic rhythm; "compressed" gives fixed frames/event;
# "log" compresses dynamic range for multi-decade rate spreads.
TIME_MAPPING_MODES = ("proper", "compressed", "log")
