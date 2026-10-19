# `schematic_viewer`

Flat 2D triptych renderer for pyKMC events, in the visual language of the
Automatic Process Explorer (APE) figure from Trochet et al. — colour-coded
atoms by role, strong silhouettes, side-by-side **init / saddle / final**
panels with arrows.

This is the sibling of [`blender_viewer/`](../blender_viewer/README.md):

- `blender_viewer/` → photorealistic Blender renders (animations, slides)
- `schematic_viewer/` → flat APE-style 2D figures (papers, theses, static
  slides, supplements)

It needs only this package (matplotlib); Blender is not required.

Both modules read the **same** `.npz` event files produced by
[`blender_viewer/prepare.py`](../blender_viewer/prepare.py). One `prepare`
run feeds both viewers.

## Pipeline

```
reference_table.pickle ──▶ Analysis.blender_viewer.prepare ──▶ event_<...>.npz
                                                                     │
                                            ┌────────────────────────┴───────────┐
                                            ▼                                    ▼
                                Analysis.blender_viewer.cli           Analysis.schematic_viewer.cli
                                (photorealistic .blend / MP4)         (flat .png / .pdf triptych)
```

## Quick run

From the repository root:

```bash
# make the event npz (242 = idx_ref of the forward event)
python -m Analysis.blender_viewer.prepare \
    --sim /path/to/run --idx-ref-fwd 242 --out event_242.npz

# render one event (default view: look down z)
python -m Analysis.schematic_viewer.cli --npz event_242.npz --out-dir schematics

# sweep three viewing axes for an event
for v in x y z; do
    python -m Analysis.schematic_viewer.cli --npz event_242.npz --view "$v" \
        --out-dir schematics
done

# arbitrary view vector (e.g. [110] direction)
python -m Analysis.schematic_viewer.cli --npz event_242.npz --view "1,1,0"

# manually pick the yellow atoms (e.g. the 3 atoms of a cooperative hop)
python -m Analysis.schematic_viewer.cli --npz event_242.npz --active-idx 33,60,63
```

The showreel's `npz/` folder (see the blender_viewer README) also works as
input. `--help` lists all options; `-v` logs the inferred active atoms and the
files written.

## Visual spec

| Role | Colour | Hex | Radius factor |
|---|---|---|---|
| Mover (the hopping atom) | vivid orange | `#FF6633` | 1.15× |
| Active neighbour (undercoordinated, near mover) | yellow | `#F4C430` | 1.0× |
| Regular substrate | light grey | `#CCCCCC` | 1.0× |
| Vacancy outline (optional) | dashed grey | `#666666` | – |
| Flow arrow | dark navy | `#1F3A5F` | – |

- Every atom carries a black silhouette (0.6 pt).
- Atoms are drawn back-to-front by projection-axis depth.
- White background, no axes / ticks / grid.
- Cell box rendering is currently a no-op — the npz pipeline doesn't carry
  cell data. `show_cell=True` will emit a `FutureWarning` to flag it.

## Role inference

The mover is read straight from the npz (`<side>_mover_idx`). The active
("yellow") set is computed automatically by intersecting two rules:

1. **Undercoordinated**: neighbour count `< coord_thr` (default 9 — matches
   the basin module's FCC slab convention).
2. **Near the mover**: within `rnei` Å of the mover (default 3.5 Å —
   one 1NN shell on FCC Ni).

This gives the tight ring around the defect that the APE figure shows,
not the entire slab surface. Override with `--active-idx i,j,k` or tweak
the cutoffs with `--coord-thr N --rnei R`.

## Outputs

Each run writes a PNG (300 dpi) and a PDF (vector) to `--out-dir`, or to
the current directory when it is omitted. Naming convention:
`schematic_<idx_ref>_<sim_label>.{png,pdf}`, e.g.
`schematic_242_100Ni_5vac_T700.png`. `examples/` holds one reference
PNG/PDF pair.

## Verification

`cli.main()` runs three programmatic checks after each render and raises
`AssertionError` on failure:

1. **Shared axis limits** — `xlim` and `ylim` are identical across all
   three panels (so motion reads cleanly).
2. **Mover projection round-trip** — the rendered mover position matches
   the original 3D coordinates re-projected, to within 0.01 Å.
3. **Non-trivial PNG file size** — `> 5 kB` (catches silent matplotlib
   backend failures).

## See also

- [`../blender_viewer/README.md`](../blender_viewer/README.md) — the
  photorealistic sibling.
- [`../blender_viewer/data.py`](../blender_viewer/data.py) — the shared
  npz data contract.
