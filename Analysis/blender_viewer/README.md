# blender_viewer

Renders pyKMC events in Blender. The main uses are:

- **One event**: animate a single event (forward hop and its backward
  partner) and save it as a `.blend` file you can open, tweak and render.
- **Showreel**: pick one representative event per event family (or the
  events you flagged in the Event Viewer), build a scene for each, render
  them headless and join them into one MP4 with labels.
- **NEB paths** (optional): run a LAMMPS nudged elastic band calculation per
  event so every atom follows the minimum-energy path, not just the mover.

The package also has helpers for lattice stills (`lattice.py`), trajectory
playback (`trajectory.py`), lighting rigs (`lighting.py`) and render presets
(`render.py`, `constants.py`). The flat 2D counterpart is
[`schematic_viewer`](../schematic_viewer/README.md).

## Prerequisites

| Tool | Needed for | How it is found |
|---|---|---|
| This package (`pip install -e .`) | `prepare`, `select`, `curated`, `compile` | your Python environment |
| Blender 4.2 or newer in the 4.x series (tested with 4.5 LTS) | building scenes and rendering | `--blender`, else `PYKMC_ANALYSIS_BLENDER`, else `blender` on `PATH` (on macOS also `/Applications/Blender.app`) |
| ffmpeg | MP4 output (`compile`, `draft`) | `ffmpeg` on `PATH` |
| LAMMPS built with the REPLICA package, plus MPI | the `neb` step only | `--lammps` / `PYKMC_ANALYSIS_LAMMPS`, `--mpirun` / `PYKMC_ANALYSIS_MPIRUN`; potential file via `--potential` |

Nothing needs to be installed into Blender's own Python: the Blender-side
code uses only numpy, which Blender ships with.

## Why there are two steps

Blender's bundled numpy often cannot unpickle `reference_table.pickle`
(written by pandas under a different numpy). So event data goes through a
small `.npz` file:

```
reference_table.pickle ──▶ prepare.py (your Python) ──▶ event.npz
                                                          │
                                                          ▼
                                      cli.main() inside Blender (numpy only)
                                                          │
                                                          ▼
                                                    event.blend
```

## Quick run: one event

Run from the repository root. `6467` is the `idx_ref` of the forward event,
as shown in the Event Viewer.

```bash
# 1. Your Python environment: dump one event pair to an npz
python -m Analysis.blender_viewer.prepare \
  --sim /path/to/run --idx-ref-fwd 6467 --out event_6467.npz

# 2. Blender, headless: build the scene and save a .blend
blender --background --python-expr "
import sys; sys.path.insert(0, '.')
from Analysis.blender_viewer.cli import main
main(npz_path='event_6467.npz', out='event_6467.blend', style='loop', camera_style='orbit')
"

# 3. Render it to PNG frames, then to an MP4
blender --background event_6467.blend \
  --python Analysis/blender_viewer/render_one_blend.py -- \
  --out-dir frames_6467 --engine EEVEE_NEXT --width 1280 --height 720
ffmpeg -framerate 60 -i frames_6467/frame_%04d.png -c:v libx264 -pix_fmt yuv420p event_6467.mp4
```

Or open `event_6467.blend` in the Blender window and scrub the timeline.
`main()` also takes `close_pack=True` (larger, touching atoms), `theme=`
(`default`, `burgundy`, `burgundy_gold_coop`), `cooperative_highlight=True`
(highlight every atom that moves more than 0.5 Å) and `neb_npz=` (keyframe
along an NEB path, `style='loop'` only). For the basic options there is
also a command line; its arguments go after `--`:

```bash
blender --background --python-expr "import sys; sys.path.insert(0, '.'); from Analysis.blender_viewer.cli import _cli; _cli()" \
  -- --npz event_6467.npz --style loop --camera orbit --out event_6467.blend
```

`render_one_blend.py` accepts `--engine EEVEE_NEXT` (fast, matte),
`CYCLES` (ray traced, uses the GPU when available) or `WORKBENCH` (fastest
preview), plus `--fps`, `--samples` (Cycles only) and `--width/--height`.

## Styles and cameras

| Option | Value | What it does |
|---|---|---|
| `style` | `dual` | Two collections (Forward_Event / Backward_Event), visibility-toggled. Shows the true saddle geometry of both directions. The atom sets differ at the edges, so the loop wrap has a small pop. |
| `style` | `loop` | One atom set; plays init → saddle → final → saddle → init. Loops seamlessly (first and last frames match). Required for NEB keyframes. |
| `camera_style` | `static` | Stationary 3/4 view. |
| `camera_style` | `orbit` | One 360° pan around the mover over the timeline; seamless at the wrap. Linear interpolation for a steady pan. |
| `camera_style` | `dolly` | Zooms in at each saddle and out in between, to emphasise the barrier crossing. |

Useful combinations: `loop` + `orbit` for a looping clip; `dual` + `static`
to compare the forward and backward saddles; `loop` + `dolly` for a
single-event explainer.

### Timelines

**Dual** (130 frames):

| Frame | State |
|---|---|
| 1 | Forward initial (Forward_Event visible) |
| 30 | Forward saddle |
| 60 | Forward final |
| 80 | Backward initial (Forward hidden, Backward visible) |
| 105 | Backward saddle |
| 130 | Backward final (≈ forward initial) |

**Loop** (800 frames, about 13 s at 60 fps; frame 800 ≡ frame 1):

| Frame | State |
|---|---|
| 1 | Initial |
| 200 | Saddle (outbound) |
| 400 | Final |
| 600 | Saddle (return) |
| 800 | Initial (loop boundary) |

With NEB images the loop keyframes every atom at each image instead, spread
evenly over the forward half and mirrored over the backward half. Location
keyframes use Bezier interpolation with Auto Clamped handles, so the path
passes through the saddle without overshoot.

### Scene contents

```
KMC_Event_<fwd>_<bwd>/
├── Forward_Event/        dual: neighbour subset around the forward mover
│   ├── fwd.mover
│   └── fwd.nei.000 … fwd.nei.NNN
├── Backward_Event/       dual: the same for the backward event (bwd.*)
└── Event_Loop/           loop: one atom set (atom.mover, atom.nei.*)
Templates/ (hidden)
└── IcoTemplate           icosphere mesh shared by all atoms
```

Atoms share one mesh and carry per-object materials. Neighbours use the
element colour (Jmol palette), metallic, roughness 0.30. The mover is
orange, 1.15× larger, with no emission by default; it stands out by its
motion. Colours, radii and material values live in `constants.py`.

After building, `cli.main` jumps to key frames and checks the mover against
the stored positions (within 1e-3 Å, or 0.15 Å with NEB keyframes); it
raises `AssertionError` on a mismatch.

## Showreel

`showreel.py` runs the batch pipeline in stages. It reads
`classified_events_with_families.csv` from the output directory
(`PYKMC_ANALYSIS_OUTPUT_DIR`, default `Analysis/lattice_event_classification/`),
so run the classification pipeline first. Relative `sim_path` values in that
CSV are resolved against `PYKMC_ANALYSIS_DATA_ROOT` (or `--data-root`).

```bash
# Your Python environment
python -m Analysis.blender_viewer.showreel select     # one median-barrier event per family
python -m Analysis.blender_viewer.showreel prepare    # npz per event
python -m Analysis.blender_viewer.showreel neb --potential /path/to/potential.eam.alloy   # optional

# Blender, headless
blender --background --python-expr "import sys; sys.path.insert(0, '.'); from Analysis.blender_viewer.showreel import _cli; _cli()" -- build

# Your Python environment (runs Blender and ffmpeg for you)
python -m Analysis.blender_viewer.showreel render
python -m Analysis.blender_viewer.showreel compile
```

Every stage takes `--root DIR` (default `<output dir>/showreel`); pass the
same value to each. Output layout:

```
<root>/
├── manifest.json          selected events
├── npz/                   one EventPair npz per event
├── neb/, neb_npz/         LAMMPS runs and their NEB paths (if run)
├── blend/                 one .blend per event
├── frames/                PNG sequence per event
├── render_logs/           Blender output per event
├── mp4/                   per-event MP4s and title cards
└── showreel_important_events.mp4   (curated_showreel.mp4 for a curated manifest)
```

Selection rule: `assignment_status == "accepted"`, a distinct backward
partner (`idx_ref != idx_backward`), the median `energy_barrier` within each
`family_id`, and a mover displacement of at least 1.5 Å so pure relaxations
are skipped. Families with no such event are skipped with a warning.

**Curated showreel.** Tick "Flag for animation" on events in the Event
Viewer audit panel, then:

```bash
python -m Analysis.blender_viewer.showreel curated    # writes curated_manifest.json
python -m Analysis.blender_viewer.showreel prepare --manifest curated_manifest.json
```

and pass `--manifest curated_manifest.json` to every later stage (for
`build`, after the `--`). `curated` refuses more than 20 flagged events
unless you add `--force`; rendering costs several minutes per event.

**Fast iteration.** Rebuild one event's `.blend` with new settings, then
`python -m Analysis.blender_viewer.showreel draft --family-id <family_id>`
renders just that event to `<root>/draft_<family_id>.mp4`.

`render --parallel N` runs N Blender processes at once. On Apple Silicon two
concurrent Eevee renders have been seen to stall, so the default is 1; add
`--skip-existing` to resume an interrupted batch.

## NEB paths

The three-keyframe path (initial, saddle, final) moves only the mover
smoothly; neighbours just interpolate. `showreel neb` (or
`neb_lammps.run_neb_for_pair`) runs a LAMMPS NEB on the event's neighbour
subset instead:

- atoms more than 5 Å from the mover's initial position are frozen;
- 11 images, spring constant 1.0, `ftol` 0.05 eV/Å, then a climbing-image
  stage;
- one MPI rank per image: `mpirun -n 11 lmp -partition 11x1`, with Open MPI's
  `--map-by :OVERSUBSCRIBE` so it runs on fewer cores.

You must pass the potential file (`--potential`); its elements are listed
alphabetically in `pair_coeff`, and the pair style defaults to `eam/alloy`.
The LAMMPS executable must include REPLICA; the LAMMPS Python module used by
pyKMC often does not, which is why the executable is used. The step prints
the pickle barrier next to the NEB barrier and marks differences above
0.2 eV with `!`. On macOS, if your LAMMPS build finds its shared library
through a relative path, set `PYKMC_ANALYSIS_DYLD_LIBRARY_PATH` to that
library's directory.

`build` picks up `neb_npz/event_<family>.npz` automatically when present.

## Final render workflow

Multi-event renders at 720p or 1080p take hours, so work in three stages.
The settings for each are recorded in `constants.py`:

| Stage | Purpose | Typical cost | Settings |
|---|---|---|---|
| Draft (`DRAFT_RENDER_PRESET`) | Fix camera, theme, atom size | 1–3 min | Eevee 540p, no NEB, one event |
| NEB validation (`NEB_VALIDATION_PRESET`) | Check the motion is physical | 5–10 min | Same, with NEB |
| Final (`FINAL_RENDER_PRESET`) | Production | 10–20 h | Cycles 1080p on the GPU, NEB, full manifest |

Use NEB for the final stage: the three-keyframe path misses the neighbour
relaxation along the minimum-energy path. Nothing applies these presets
automatically; `FINAL_RENDER_QA_QUESTIONS` in `constants.py` lists the
choices to settle (resolution, theme, camera, scope, output) before a long
run.

Practical notes for long runs:

1. On macOS, wrap the job in `caffeinate -i -s` so the machine does not
   sleep.
2. In shell loops, read event ids into an array first. With
   `cmd | while read fid; do blender …; done`, Blender can read the loop's
   stdin and events are silently skipped.
3. `compile` works from the frame directories. If you delete each event's
   frames after encoding it to save disk space, join the per-event MP4s in
   `<root>/mp4/` yourself (see `mp4/concat.txt` for the ffmpeg concat
   format).

## Configuration

| Setting | Environment variable | Flag |
|---|---|---|
| Blender executable | `PYKMC_ANALYSIS_BLENDER` | `--blender` (`render`, `draft`) |
| LAMMPS executable | `PYKMC_ANALYSIS_LAMMPS` | `--lammps` (`neb`) |
| MPI launcher | `PYKMC_ANALYSIS_MPIRUN` | `--mpirun` (`neb`) |
| LAMMPS library directory (macOS only) | `PYKMC_ANALYSIS_DYLD_LIBRARY_PATH` | none |
| Folder of runs | `PYKMC_ANALYSIS_DATA_ROOT` | `--data-root` (`select`, `curated`, `all`) |
| Output directory (showreel uses `<dir>/showreel`) | `PYKMC_ANALYSIS_OUTPUT_DIR` | `--root` sets the showreel directory itself |

`python -m Analysis.blender_viewer.showreel --help` lists every option.

## Modules

| Module | Runs in | Role |
|---|---|---|
| `data.py` | both | `EventPair` data, pickle loader (host) and npz loader (Blender) |
| `prepare.py` | host | CLI: pickle → npz |
| `cli.py` | Blender | build one event scene, save `.blend` |
| `scene_builder.py`, `camera.py`, `animation.py`, `materials.py` | Blender | scene construction |
| `constants.py` | both | colours, radii, lighting and render presets |
| `lighting.py`, `render.py` | Blender | light rigs, render settings, PNG → MP4 |
| `lattice.py`, `trajectory.py`, `neb.py` | both | supercells and vacancies, trajectory playback, NEB image playback |
| `neb_lammps.py` | host | LAMMPS NEB runner |
| `showreel.py` | both | batch pipeline (the `build` stage runs in Blender) |
| `render_one_blend.py` | Blender | headless render driver for one `.blend` |
