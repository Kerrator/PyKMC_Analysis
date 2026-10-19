# `Analysis/tools/` — repair and diagnostic scripts

Each script is a standalone CLI. Run them from the repository root with
`python -m Analysis.tools.<name>` (`--help` lists the options).

## `fix_catalogue_nvac.py` — repair the catalogue's vacancy counts

**What it does.** Recomputes `n_vac_nn1_initial`, `n_vac_nn1_final`, and
`n_vac_nn2_initial` in a classified-events CSV. If the CSV already has
family columns (`classified_events_with_families.csv`), it also re-derives
`family_bucket_id` for the vacancy-bucketed families.

**Why it's needed.** `Analysis/classify_lattice_events_parts/04b_ml_descriptors.py`
runs with `COMPUTE_LATTICE_ADJACENCY = False` by default (the
`build_surface_neighbour_list()` adjacency build is O(surface × sites ×
9 PBC images) in pure Python — minutes per run). With the flag off, the
descriptor falls back to a **coordination-shortfall proxy**:
`n_vac_nn1 = max(0, 12 - coord_mover)`. For surface atoms (coord = 8) this
always reports `n_vac_nn1 = 4`, whatever the actual vacancy environment,
so the column cannot be used as a rate-table key until it is corrected.

The fix uses the **layer-aware formula**, `expected_coord = 8` for the top
layer (`surface_layer_initial == 0`) and `12` everywhere else. The
catalogue already carries `coord_mover_initial` / `coord_mover_final` and
`surface_layer_initial` / `surface_layer_final`, so the correction takes
seconds instead of a full `COMPUTE_LATTICE_ADJACENCY = True` rerun.

**What it changes.**

- `n_vac_nn1_initial` ← `max(0, expected - coord_mover_initial)`
- `n_vac_nn1_final` ← `max(0, expected - coord_mover_final)`
- `n_vac_nn2_initial` ← `max(0, expected_2nn - sum(nn2_count_*))`
  with `expected_2nn = 5` for surface, `6` for bulk
- with family columns present:
  - `family_bucket_id` for `surface_1NN_inplane`, `subsurface_1NN_inplane`,
    `bulk_1NN_inplane` ← 2-axis `nv1=N_nv2=M`
  - `family_bucket_id` for `surface_2NN_diagonal`, `subsurface_2NN_diagonal`,
    `surface_subsurface_exchange_up` / `_down`, `adatom_attachment`
    (and its legacy id `surface_subsurface_exchange_lateral`),
    `adatom_detachment`, `subsurface_interlayer_hop`,
    `subsurface_migration_interlayer` ← 1-axis `nv1=N`

**Usage.** Run it on the notebook's output, before family assignment, so
the families are bucketed on the corrected counts:

```bash
python -m Analysis.tools.fix_catalogue_nvac \
    --input  Analysis/lattice_event_classification/classified_events.csv \
    --output Analysis/lattice_event_classification/classified_events.csv
```

Then assign families and build the family rate table with pylatkmc (see
[`../classify_lattice_events_parts/README.md`](../classify_lattice_events_parts/README.md)).
Run it before family assignment. If you already have
`classified_events_with_families.csv`, fix `classified_events.csv` and then
re-run the pylatkmc family assignment, rather than patching the families
file in place (that would not respect reviewer bucket overrides).

## `pykmc_ea_distribution.py` — the barriers pyKMC actually selected

**What it does.** Parses a run's `pykmc.out` step log and
`reference_table.pickle`, joins each step's `ref_event` to the catalogue
row, classifies the event via `lattice_map_events.classify_event_from_row`,
and aggregates barrier statistics per move type.

**Why it's useful.** The off-lattice pyKMC run is the reference an
on-lattice model built from its catalogue should reproduce. This tool
tells you which catalogue events fired in a run and at what barrier,
directly comparable to the barriers the on-lattice model uses.

**Usage.**

```bash
python -m Analysis.tools.pykmc_ea_distribution \
    --sim /path/to/run \
    --out pykmc_ea_distribution.csv
```

Output: a CSV with one row per move type: `n_steps_fired`,
`n_unique_idx_ref`, `Ea_mean`, `Ea_std`, `Ea_min`, `Ea_max`, `Ea_median`,
`time_total_s`. Stdout has a human-readable summary.

**Best inputs.** `basin = False` runs give the cleanest reference.
`basin = True` runs still give correct *per-event* barriers (basin
acceleration changes the time bridging, not individual event energetics),
so they are usable at low temperature where `basin = False` runs are too
slow.

## Legacy: `patch_cube_cell.py`

Sets one cell's `(rate, Ea)` in a `.kmcrt` rate cube, the binary format
used by earlier pylatkmc versions (current pylatkmc no longer uses it).
Kept only for working with old cube files; make a backup first, as the
file is patched in place:

```bash
python -m Analysis.tools.patch_cube_cell \
    --kmcrt /path/to/model.kmcrt --cell 0,0,0,0,0,0,0,0,0 --Ea 0.6
```

`--cell` is a 9-tuple of axis indices in the order `(site_class, direction,
mover_species, n_vac_nn1, n_Fe_nn1, n_Cr_nn1, n_vac_nn2, n_Fe_nn2,
n_Cr_nn2)`.

## See also

- [`../lattice_event_classification/EVENT_CLASSIFICATION.md`](../lattice_event_classification/EVENT_CLASSIFICATION.md)
  — the classification method, including the ML descriptors and the
  vacancy-count caveat
- [pylatkmc](https://github.com/Kerrator/pylatkmc) — family registry,
  family assignment, and the family rate table (`pylatkmc.ingest`)
