# `classify_lattice_events` — editable parts

The notebook `../classify_lattice_events.ipynb` is assembled from the
numbered `.py` files in this folder. Each `NN_label.py` file maps to one
notebook cell, in numeric order. The notebook classifies every event in the
`reference_table.pickle` files of one composition and writes
`classified_events.csv` (see
[`../lattice_event_classification/EVENT_CLASSIFICATION.md`](../lattice_event_classification/EVENT_CLASSIFICATION.md)
for the method).

## Running the classification

All commands run from the repository root, with the package installed
(`pip install -e ".[notebooks]"`) so `Analysis` is importable from the
notebook's directory.

1. Point the notebook at your runs, laid out as
   `{composition}/{N}vac/T{temp}/{variant}/`:

   ```bash
   export PYKMC_ANALYSIS_DATA_ROOT=/path/to/data
   # optional; default is Analysis/lattice_event_classification/
   export PYKMC_ANALYSIS_OUTPUT_DIR=/path/to/output
   ```

   The composition (default `100Ni`), an explicit data root and the
   potential filter (default: runs whose `pair_coeff` contains `NiAlH`;
   `''` keeps every run) are set in the **Parameters** block of
   `01_imports.py`.

2. Run the notebook top to bottom (Jupyter, VS Code, or headless):

   ```bash
   jupyter nbconvert --to notebook --execute \
       --output classify_lattice_events_executed.ipynb \
       Analysis/classify_lattice_events.ipynb
   ```

   `*_executed.ipynb` is git-ignored, so the committed notebook stays
   unexecuted.

3. **Correct the vacancy counts (required).** With the default
   `COMPUTE_LATTICE_ADJACENCY = False` in `04b_ml_descriptors.py`,
   `n_vac_nn1_initial` / `n_vac_nn1_final` are a bulk coordination-shortfall
   proxy that is wrong for surface atoms. Recompute them before any rate
   table is built:

   ```bash
   D=${PYKMC_ANALYSIS_OUTPUT_DIR:-Analysis/lattice_event_classification}
   python -m Analysis.tools.fix_catalogue_nvac \
       --input  $D/classified_events.csv \
       --output $D/classified_events.csv
   ```

   Do this before step 4. See [`../tools/README.md`](../tools/README.md).

4. Family assignment and the family rate table live in
   [pylatkmc](https://github.com/Kerrator/pylatkmc)
   (`pip install -e ".[families]"`):

   ```bash
   python -m pylatkmc.ingest.build_curated_catalogue \
       --events     $D/classified_events.csv \
       --audit      $D/audit_log.csv \
       --out-events $D/classified_events_with_families.csv \
       --out-rate   $D/rate_lookup_table_family.csv \
       --report     $D/family_assignment_report.json
   ```

   (or stage by stage: `python -m pylatkmc.ingest.family_assignment`, then
   `python -m pylatkmc.ingest.build_family_rate_table`). Pass every path
   explicitly: pylatkmc's defaults are relative to its own working
   directory. The audit log is written by the event viewer; a missing audit
   log means "no verdicts yet".

## Editing the notebook

Edit any `NN_*.py` part, then rebuild:

```bash
python Analysis/classify_lattice_events_parts/build_notebook.py
```

This rewrites `../classify_lattice_events.ipynb` from the parts, unexecuted:
no outputs, no widget state, a generic `Python 3` kernel, and cell ids
derived from the part filenames, so rebuilding unchanged parts gives an
identical file.

**Direction of sync**: parts → notebook **only**. Do not hand-edit the
notebook JSON — it will be overwritten on the next build.

## Cell / file format

Parts use the Jupyter "percent" format:

```python
# %% [markdown]
# # Section title
#
# Narrative text here.

# %%
print("this is a code cell")
```

- `# %%` starts a code cell.
- `# %% [markdown]` starts a markdown cell; leading `# ` is stripped on
  build and reinserted on split.
- Ordering is determined by the filename prefix. The regex allows an
  optional lowercase letter suffix (e.g. `04b_`) so insertion files sort
  alphabetically between two numeric siblings without renumbering.

## Part list (19 cells)

| Part | Role |
|---|---|
| `00_title.py` | Notebook title and introduction |
| `01_imports.py` | Imports, the **Parameters** block (`COMPOSITION`, `DATA_ROOT_PATH`, `POTENTIAL_FILTER`) and module-level constants (`kB`, `BARRIER_CUTOFF`, move types) |
| `02_scan.py` | Discovers run directories (via `Analysis.sim_discovery.parse_run_path`) and concatenates their `reference_table.pickle` files into `master` |
| `03_coverage.py` | Coverage heatmap over (nvac, temp) |
| `04_load_events.py` | Classification engine — runs `classify_event_from_row()` over all sub-cutoff events, produces `sub` DataFrame |
| `04b_ml_descriptors.py` | ML descriptor pass (Stage 1f) and lattice-environment fingerprints (Stage 1h). Per-simulation caches (cell, firing counts, optional lattice adjacency) + per-event `compute_ml_descriptors()` call + reverse-barrier self-join. Toggle via `COMPUTE_ML_DESCRIPTORS = True`. |
| `05_saddle_features.py` | Stage 1b visualisation — distributions of saddle-point geometric descriptors |
| `06_gmm_threshold.py` | Stage 1c — GMM-fitted adaptive thresholds |
| `07_reclassify.py` | Stage 1d (enhanced classification) + Stage 1g (z-layer reclassification + floating-atom rule). Relabels subsurface events and `interlayer_hop_down` with `|d| > 3 A` as `relaxation`. |
| `08_species.py` | Stage 1e — species-aware mover/partner extraction |
| `09_displacement.py` | Interactive displacement scatter |
| `10_barriers.py` | Barrier distribution plots |
| `11_arrhenius.py` | Temperature dependence / Arrhenius fits |
| `12_vacancy_trend.py` | Vacancy count dependence |
| `13_pair_analysis.py` | Forward/backward pair analysis (uses `idx_backward`) |
| `14_ase_config.py` | Lattice model parameter extraction |
| `15_diffusion.py` | Diffusion coefficient validation against MD |
| `16_kinetic_coverage.py` | Per-simulation lattice-step fraction |
| `17_export.py` | Writes `classified_events.csv` (including `sim_path` and `sim_relpath`, the run path relative to the data root) and `classified_events_enhanced.csv` |

**Adding a new part:** drop an `NN[letter]_label.py` file in this folder
(e.g. `04c_foo.py`) and rebuild — the builder picks it up automatically. If
you want `--split` mode (notebook → parts) to round-trip cleanly, also add
the new label to the `DEFAULT_LABELS` list in `build_notebook.py`.

## One-shot re-split

To overwrite the part files from the current notebook (e.g. after somebody
edited the `.ipynb` directly):

```bash
python Analysis/classify_lattice_events_parts/build_notebook.py --split
```

This uses the label list in `build_notebook.py` (`DEFAULT_LABELS`). If the
notebook has a different cell count, generic `cellNN` names are used.

## Interactive figures

Cells `09_displacement.py` through `13_pair_analysis.py` render a static
figure (saved to the output directory as `*.png`) **and** an `ipywidgets`
controller below it. The widgets let you toggle move-type visibility,
switch colour scheme, and adjust display options (bins, point size, fit
lines, etc.). Widget state is not saved; controls reset on notebook
reload. The widgets need Jupyter with ipywidgets >= 8 (installed with the
package) and also work in the VS Code notebook viewer.
