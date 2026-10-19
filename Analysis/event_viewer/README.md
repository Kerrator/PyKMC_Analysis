# KMC Event Viewer

Interactive 3D browser **and audit UI** for the events of pyKMC runs. Built on
Dash + Plotly, it runs as a standalone web app or inline in a Jupyter
notebook. Audit verdicts are saved to `audit_log.csv` in the output directory
(see [Where files are read and written](#where-files-are-read-and-written)).

## Quick start: one run

To inspect every event of a single pyKMC run you need only this repo and the
run's directory. The viewer reads the run's `reference_table.pickle`; it does
not need a folder of runs or any classifier output.

```bash
git clone https://github.com/Kerrator/PyKMC_Analysis.git
cd PyKMC_Analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m Analysis.event_viewer.app --run /path/to/run
```

Then open <http://127.0.0.1:8050>. `/path/to/run` is the directory pyKMC
wrote `reference_table.pickle` into (Python 3.11 or newer).

- **Pick an event**: the first event is loaded on start. The descriptor
  scatter below the 3D view holds every event in the run, plotted as the
  moving atom's displacement `|d|` against `|dz|`; click a point to load
  it. **← prev / next →** in the card beside the viewer (or the *Event*
  dropdown above it) steps through the run in `idx_ref` order, and the
  ← ↑ → keys or the frame pills switch initial / saddle / final. The
  scatter's *Color by* dropdown also offers `direction` and
  `energy_barrier`.
- **Labels**: events show as `unclassified` unless
  `classified_events_with_families.csv` (or `classified_events.csv`) in the
  output directory has rows for this run. Those files come from the
  classification pipeline (see the repo README) and are not in git.
- **Run identity**: the header shows the run as composition / N vac / T. For
  a run inside a `{composition}/{N}vac/T{temp}/{variant}/` tree, those
  values come from the path. Otherwise the temperature comes from `input.in`
  (`[RateConstant] T`), the composition is the directory name, and N vac
  is 0. Override any of them with `--composition MyAlloy --nvac 2 --temp 500`.
  Audit verdicts are keyed on these values, so set them before you record
  any.
- `--port 8052` (or `EVENT_VIEWER_PORT=8052`) changes the port, and
  `python -m Analysis.event_viewer.app --help` lists all the options.
- **Only open runs you trust.** Loading a `.pickle` file runs Python's
  unpickler, which can execute arbitrary code.

## A folder of runs

To browse many runs at once, keep them in one directory laid out as
`{composition}/{N}vac/T{temp}/{variant}/`, each holding a
`reference_table.pickle`, for example `50Cu_50Ni/2vac/T500/full/` (the
composition folder name must contain a digit). Then either pass the
directory on the command line or set it once in the environment:

```bash
python -m Analysis.event_viewer.app --data-root /path/to/data
# or
export PYKMC_ANALYSIS_DATA_ROOT=/path/to/data
python -m Analysis.event_viewer.app
```

`--run` and `--data-root` cannot be combined. In folder mode the scatter
comes from the classifier output (`classified_events*.csv`, see below), so it
stays empty until the classification pipeline has been run; use `--run` to
see every event of one run without it. The rest of this README covers this
folder mode; everything also works in single-run mode.

## Where files are read and written

| Variable | Used for |
|---|---|
| `PYKMC_ANALYSIS_DATA_ROOT` | Folder of runs when neither `--run` nor `--data-root` is given. |
| `PYKMC_ANALYSIS_OUTPUT_DIR` | Directory holding the classifier CSVs the viewer reads, and the per-user `audit_log.csv` and `view_defaults.json` it writes. Default: `Analysis/lattice_event_classification/` in this checkout (git-ignored). |
| `EVENT_VIEWER_PORT` / `EVENT_VIEWER_RUN` | Defaults for `--port` / `--run` (also used when the app is imported, e.g. from the notebook). |

A fresh clone has no audit log; it is created on the first save.

## The three panels

1. A **3D viewer** of one event's initial → saddle → final configuration
2. An **audit side-panel** where you record a verdict (confirm / reclassify
   / flag / ignore) with an optional `move_type`, **`family_id`, or
   `family_bucket_id`** override and free-text notes
3. A **descriptor scatter** below (defaults to `move_dist` vs `|move_dz|`
   colored by `move_type`) that acts as the primary event picker — any
   numeric column of the classifier CSV (including the ML descriptors) can
   be used as an x/y axis or continuous color, **`family_id` is available
   as a categorical color mode**, and optional **family / bucket
   checklists** narrow the scatter to one curated FCC family and/or bucket.
   Click a point to load it in the 3D viewer, or box/lasso-select a region
   to batch-audit.

### 3D viewer controls

| Control | What it does |
|---|---|
| **Composition / Vacancies / Temperature** | **Multi-select** dropdowns — they filter which events appear in the scatter below. The 3D viewer loads whichever single event is clicked. |
| **Event** | Within-run stepper — populates once you click a point on the scatter. `#idx — barrier (eV) — move_type` |
| **Frame** | ← initial, ↑ saddle, → final (or click the pill buttons) |
| **Color by** | `element` (Ni / Fe / Cr colors) or **CNA** (FCC / HCP / BCC / ICO / OTHER / BOUNDARY) |
| **Render** | `markers` (fast, pixel-fixed Scatter3d sprites) or **spheres** (Mesh3d — true 3D, atoms in the foreground look larger) |
| **Projection** | `perspective` (default) or `orthographic` (isometric feel) |
| **Overlay: vacancies** | Off by default. Toggle on to show open-diamond markers at empty lattice sites in each frame, with faint ghost diamonds showing where vacancies sit in the other two frames |
| **Atom radius (Å)** | 0.3 — 3.0, default 1.25 (Ni metallic radius). Scales marker pixel size in markers mode; actual sphere radius in data units in spheres mode |

The mover atom is always highlighted — in markers mode with an amber outline,
in spheres mode as a lightened shade of its element color. A red polyline
traces its initial → saddle → final path across all frames.

Your camera (rotate / zoom / pan) is preserved across every control change, so
you can set a viewing angle once and explore freely. **Lock view** saves the
current camera for that event in `view_defaults.json`.

### Scatter (displacement + ML descriptor space)

| Control | What it does |
|---|---|
| **Scatter graph** | Default axes: `move_dist` vs `|move_dz|`. Default color: per-`move_type` categorical palette. Reference lines at 1NN (2.49 Å), 2NN (3.52 Å), and the 1 Å layer-change threshold — shown **only** on the default axes. Audited events get a black ring. |
| **Move-type checklist** | Toggle traces on/off. Each label highlights in its scatter color on hover; checked labels get a colored underline matching the trace. |
| **Families checklist** | Narrow the scatter to one or more curated FCC families (`FAMILY_REGISTRY` in pylatkmc, optional extra `families`). Empty = no family filter. Populates from `classified_events_with_families.csv`. |
| **Buckets checklist** | Narrow further by `family_bucket_id` (e.g. `nv1=5`, `li=0_nv1=6`). Options reflect the buckets present in the currently selected families; toggling the Families list re-populates this list automatically. |
| **X axis / Y axis dropdowns** | Pick any numeric column of the classifier CSV as the x or y axis, including the ML descriptors (`disp_top{1-5}`, `Ea_over_kT`, `mover_depth_{initial,final,delta}`, `nn{1,2}_count_{Ni,Fe,Cr}`, `coord_mover_{initial,final,saddle}`, `path_straightness`, `bonds_{broken,formed}_init_to_saddle`, `Ea_gap`, `reverse_barrier`, `kmc_fire_count`, etc.). Search-as-you-type in the dropdown. |
| **Color by dropdown** | `move_type (categorical)` uses the move-type palette. `family_id (categorical)` uses a per-family palette. Any numeric column switches to a continuous Viridis colorscale with a colorbar. |
| **Filter by column + min/max** | Pick any numeric column; enter min and/or max; scatter is filtered to rows in that range. Composes with the composition / vacancies / temperature cascade. |
| **Options: merge up/down** | Collapse directional suffixes (`_up`/`_down`/`_lateral`) into base types via `move_type_base`. |
| **Options: show only un-audited** | Hide events that already have a row in `audit_log.csv`. |
| **Click** | Load the clicked event into the 3D viewer and audit panel. |
| **Box / lasso select** | Drag on the plot to pick a region; the audit panel shows `📦 N events selected`. Double-click to clear. |

The descriptor dropdowns are populated from the scatter frame at app startup
via `numeric_scatter_columns(SCATTER_DF)` (in `data.py`). Add new columns by
extending `_CLASSIFIER_SCATTER_COLS` in the same file — they appear in the
dropdowns automatically.

## Auditing the classification

The audit panel to the right of the 3D viewer records your verdict on the
currently loaded event (and, with box-select, on an entire region at once).

| Field | Values / effect |
|---|---|
| **Verdict** | `confirm` (classifier got it right) · `reclassify` (wrong label, fill an override) · `flag` (suspicious but no override yet) · `ignore (mask out)` (exclude from ML training) · `unset` |
| **Override move_type** | Only meaningful when verdict = `reclassify`. Options cover every value in `ALL_TYPES` and `ALL_TYPES_BASE`. The family-assignment step re-evaluates its seed rules on the patched `move_type`. |
| **Override family_id** | Only meaningful when verdict = `reclassify`. Direct pick from `FAMILY_REGISTRY`. Bypasses seed-rule matching — the family-assignment step uses this value as given. |
| **Override family_bucket_id** | Free-form text (e.g. `nv1=5`, `li=0_nv1=7`). Optional; when empty, the family's `environment_rule` derives the bucket from the row. |
| **Notes** | Free-text — why you chose this verdict, observations about the local environment, etc. |
| **Save audit** | Upserts one row for the current event into `audit_log.csv`. |
| **Apply to selection** | Writes the same verdict / overrides / notes to every event inside the current box or lasso selection. Counts are shown in the status line after save. |

The header above these fields shows the current `move_type`, `family_id` /
`family_bucket_id`, and `assignment_status` so you can see what the family
assignment landed on before you override. Family assignment itself lives in
pylatkmc (`pylatkmc.ingest.family_assignment`), which reads `audit_log.csv`.

Events that detached from the surface (`interlayer_hop_down` with
`|d| > 3 Å`) are relabelled `relaxation` by the classification pipeline
itself (`classify_lattice_events_parts/07_reclassify.py`), so they need no
manual flagging here.

**Persistence.** `audit_log.csv` is keyed by
`(composition, nvac, temp, idx_ref)`. Re-saving the same key upserts — one
row per event. Schema:

| Column | Type |
|---|---|
| `composition` | str |
| `nvac` | int |
| `temp` | float |
| `idx_ref` | int |
| `original_move_type` | str (label at time of audit) |
| `verdict` | str |
| `override_move_type` | str (empty when unused) |
| `override_family_id` | str — curated FCC family, empty when unused |
| `override_bucket_id` | str — e.g. `nv1=5`, empty when unused |
| `override_direction_id` | str — empty when unused |
| `notes` | str |
| `updated_at` | ISO timestamp |
| `render_flag` | `"1"` to include the event in the Blender showreel, else empty |

Older audit logs written before the override columns existed load cleanly —
`audit.py` back-fills the missing columns with empty strings.

Other code can read the log without touching the Dash app:

```python
from Analysis.event_viewer.audit import load_audit, audited_keys
df = load_audit()
ignored = set(audited_keys(df[df["verdict"] == "ignore"]))
```

## Settings panel

Top-right of the header: a `⚙` button opens a small popover.

| Control | What it does |
|---|---|
| **Shut down server** | Pops a `window.confirm` dialog; on yes, the Dash callback calls `os._exit(0)` ~300 ms later so the "Shutting down…" message reaches the browser before the process dies. After that the port is free — re-run `python -m Analysis.event_viewer.app` to bring the viewer back. |

## Launch

### Standalone web app

```bash
python -m Analysis.event_viewer.app --run /path/to/run          # one run (see Quick start)
python -m Analysis.event_viewer.app --data-root /path/to/data   # a folder of runs
```

Browser at <http://127.0.0.1:8050>. Startup prints how many simulations and
classifier rows were loaded.

### Inside a Jupyter notebook

```bash
pip install -e ".[notebooks]"
jupyter lab Analysis/event_viewer/event_viewer.ipynb
```

Set `RUN_DIR` or `DATA_ROOT` in the setup cell, then run the cells
top-to-bottom. The app embeds as an iframe inside the launch cell's output
(Dash's `jupyter_mode="inline"`). **Click into the iframe once** so keyboard
focus is inside it — then the ← ↑ → hotkeys work. The notebook uses port
8051 so it does not collide with a standalone server on 8050.

### As a direct Plotly figure

For a static figure — no Dash, no app — the underlying builder is
importable:

```python
from Analysis.sim_discovery import sim_from_dir
from Analysis.event_viewer.data import load_events, load_classifications
from Analysis.event_viewer.plots import make_event_figure

sim    = sim_from_dir("/path/to/run")
events = load_events(sim, load_classifications())
row    = events.iloc[0]

fig = make_event_figure(row, frame="saddle", color_mode="cna",
                        render_style="spheres", atom_size=0.9,
                        show_vacancies=True)
fig.show()                          # or fig.write_image("event.png") (needs kaleido)
```

## Keyboard shortcuts

Arrow keys switch the Frame — **but only when no input, dropdown, or slider
has keyboard focus**, so typing into dropdowns and sliding the radius slider
with ← / → continue to work normally.

| Key | Frame |
|---|---|
| ← | initial |
| ↑ or ↓ | saddle |
| → | final |

## Data sources

- **Simulations** — found under the data root by
  `Analysis/sim_discovery.py` (`find_simulations()`), which expects the
  layout `{composition}/{N}vac/T{temp}/{variant}/` with a
  `reference_table.pickle` inside. With `--run DIR` discovery is skipped:
  `sim_from_dir()` describes that one directory and the scatter comes from
  `run_scatter_frame()` (its pickle, all events, classifier labels joined
  where they exist) instead of the classifier CSV.
- **Move-type + family labels** — the viewer prefers
  `classified_events_with_families.csv` (written by pylatkmc's family
  assignment), which adds the curated `family_id`, `family_bucket_id`, and
  `assignment_status` columns to the raw classifier output. It falls back to
  `classified_events.csv` when the family-tagged export is absent — the
  audit panel then shows empty family fields. Rows are joined to the event
  catalogue on `(idx_ref, energy_barrier)` with a ±1e-6 eV tolerance (the
  CSV's short `sim_label` is the same for every composition, so a barrier
  match is needed to disambiguate). If the CSV is missing, every event
  renders with `move_type = "unclassified"` and the viewer still works. Old
  CSVs lacking the `composition` column are tolerated — rows default to
  `composition = "100Ni"`.
- **Audit log** — `audit_log.csv` in the output directory. Absent on first
  launch, created on first save from the audit panel.

`reference_table.pickle` files written by older pandas versions are handled
transparently — the loader falls back to `_CompatUnpickler` (in `data.py`)
when the normal unpickler raises `NotImplementedError`.

## File layout

```
Analysis/event_viewer/
├── app.py                    — Dash layout + callbacks + clientside callbacks
├── data.py                   — run discovery, event loading, classifier join, CompatUnpickler, load_scatter_frame(), run_scatter_frame()
├── plots.py                  — make_event_figure() — pure Plotly 3D figure builder
├── scatter.py                — build_scatter_figure() — Plotly displacement scatter
├── scatter_palette.py        — ALL_TYPES, ALL_TYPES_BASE, CUSTOM_COLORS, FAMILY_COLORS, ALL_FAMILY_IDS, NN1_DIST / NN2_DIST / LAYER_CHANGE_DZ
├── audit.py                  — load_audit, save_audit_row, save_audit_bulk, audited_keys (CSV-backed upsert)
├── view_defaults.py          — per-event saved camera ("Lock view"), stored in view_defaults.json
├── spheres.py                — icosphere template + Mesh3d sphere builder for spheres mode
├── analysis.py               — compute_cna(), find_vacancies() (pure SciPy — no heavy deps)
├── event_viewer.ipynb        — Jupyter launcher (inline mode)
├── assets/
│   ├── style.css, tokens.css — layout, pill buttons, audit panel, design tokens
│   ├── scatter_move_types.css — auto-generated from the palettes at app startup; do not edit by hand
│   └── keyboard.js           — ← ↑ → hotkey listener
└── README.md                 — this file
```

`plots.py`, `scatter.py`, `scatter_palette.py`, `audit.py`, `spheres.py`,
and `analysis.py` have **no Dash dependency** — they're pure Plotly /
NumPy / SciPy / pandas, reusable from any notebook.

## Troubleshooting

**"Pick composition, vacancies, temperature." shown indefinitely** — no
simulations were discovered (startup prints `no runs found`). Pass
`--run /path/to/run` for one run, or `--data-root /path/to/data` (or set
`PYKMC_ANALYSIS_DATA_ROOT`) to a folder in the
`{composition}/{N}vac/T{temp}/{variant}/` layout.

**Status bar shows "Loaded 0 events … (pickle missing or unreadable)"** — that
run's `reference_table.pickle` is missing, truncated, or in a format the
compatibility loader does not handle. The terminal shows the exact error.

**Every event is `unclassified`** — the classifier hasn't been run for that
composition/nvac/temp combo, or its rows don't match by barrier. Run
`Analysis/classify_lattice_events.ipynb` first if you want move_type labels.

**Keyboard hotkeys do nothing** — click into the 3D plot area (or anywhere
outside the dropdowns and slider) first to give the page keyboard focus.
When the slider thumb is focused the arrow keys adjust the slider instead,
which is intentional.

**Spheres mode is sluggish** — switching from markers to spheres builds
`n_atoms × 80` triangles in one `Mesh3d` (~10k for a 130-atom cutout).
First render takes 1–2 s. Subsequent frame toggles with the same event
are instant (CNA results are cached per event+frame).

**Atoms at the edge of the cutout look wrong in CNA mode** — they are.
The CNA algorithm needs a full 1st / 2nd NN shell to classify, which
boundary atoms don't have. The viewer labels the outer 15 % shell as
`BOUNDARY` (light grey) instead of misleadingly calling them `OTHER`.

**Port 8050 / 8051 already in use** — pass `--port NNNN` (or change `PORT`
in the notebook). `lsof -i :8050 -P -n` shows what holds the port.

**Scatter is empty for every composition except `100Ni`** — your
`classified_events.csv` was built before the `composition` column existed,
so every row defaults to `composition = "100Ni"`. Re-run
`classify_lattice_events.ipynb` once to rebuild the CSV.

**`Apply to selection` does nothing** — a double-click on the plot clears
the current selection (Plotly default). Drag a fresh box / lasso, then
click the button.

**Audited ring doesn't appear after I save** — the scatter only re-renders
when filters/options change or the internal `audit-refresh` store bumps.
Every successful save bumps it, so either the save itself failed (check
the `audit-status` line under the buttons) or the output directory is not
writable.

## Extending

- **Point at a different folder of runs** — `--data-root` or
  `PYKMC_ANALYSIS_DATA_ROOT`; from Python, `discover_runs()` in `data.py`
  also takes a list of roots.
- **Change the move-type palette** — edit `CUSTOM_COLORS` in
  `scatter_palette.py`. The per-type hover CSS in
  `assets/scatter_move_types.css` is regenerated automatically on next
  app launch.
- **Change the family palette** — edit `FAMILY_COLORS` in
  `scatter_palette.py`. New entries in `FAMILY_REGISTRY` are automatically
  picked up by `ALL_FAMILY_IDS`; if you add a family without a matching
  color, its scatter trace falls back to a neutral grey. The per-family
  hover CSS is regenerated alongside the move-type CSS.
- **Add a new classification rule** — put it in
  `classify_lattice_events_parts/07_reclassify.py`, not in this app. The
  classification pipeline is the canonical source of `move_type`; the
  viewer only displays it. Follow the pattern of the floating-atom rule in
  that file: build a boolean mask from geometric / barrier conditions, then
  reassign `move_type` and `move_type_base`.
- **Add a new ML descriptor** — add the column to
  `_CLASSIFIER_SCATTER_COLS` in `data.py`. It appears automatically in
  the X axis / Y axis / Color by / Filter dropdowns. Upstream: add the
  computation to `compute_ml_descriptors()` in
  `Analysis/lattice_map_events.py`.
- **New CNA color** — edit `CNA_COLORS` in `analysis.py`.
- **New element color** — edit `ELEMENT_COLORS` in `plots.py`.
- **More classifier columns in the dropdown** — edit `event_dropdown_label()`
  in `data.py`.
