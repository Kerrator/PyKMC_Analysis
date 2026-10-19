# Event Classification: Off-Lattice to On-Lattice KMC Mapping

## 1. Introduction

Off-lattice kinetic Monte Carlo (KMC) simulations discover transition events through saddle point searches (e.g., the Activation-Relaxation Technique, ART nouveau) without assuming a predefined set of mechanisms. The result is a rich event catalogue containing thousands of events with continuous atomic displacements, arbitrary barrier heights, and no inherent categorisation.

To bridge this to an on-lattice KMC model -- where atoms hop between discrete lattice sites with environment-dependent rates -- we need a systematic method to:

1. **Classify** each off-lattice event by its physical mechanism (hop, exchange, concerted motion, etc.)
2. **Project** continuous atomic displacements onto ideal lattice sites to identify which vacancy hop occurred
3. **Assign** a local environment descriptor (e.g., number of neighbouring vacancies) to each hop
4. **Construct** a rate lookup table mapping (environment) --> (barrier, rate) for the on-lattice model

This document describes our classification pipeline and positions it relative to the KMC literature.

---

## 2. Literature Review

### 2.1 Environment and Event Classification in Off-Lattice KMC

Off-lattice KMC codes must classify local atomic environments to decide when a previously discovered event can be reused at a new site. Four main paradigms exist:

#### Topological Classification (k-ART / NAUTY)

The kinetic Activation-Relaxation Technique (k-ART) uses the NAUTY graph-automorphism code to discretise local atomic environments into distinct topological classes. Each atom's neighbourhood is converted into a graph whose canonical labelling defines the topology. All atoms sharing a topology are presumed to have the same accessible mechanisms, enabling catalogue reuse. At least 25 ART nouveau searches are launched per unique topology to ensure rich sampling.

This approach excels at handling disordered and amorphous systems (a-Si, c-Si defects, Fe self-interstitials) because the classification is purely graph-based and does not require a reference lattice.

**References:**
- El-Mellouhi, Mousseau, Lewis, *Phys. Rev. B* 78, 153202 (2008)
- Beland, Brommer, El-Mellouhi, Joly, Mousseau, *Phys. Rev. E* 84, 046704 (2011)

#### Geometric Classification (KDB / EON)

The Kinetic Database (KDB, Henkelman group) stores a minimal geometric representation of kinetic events (reactant, saddle, product configurations). Unlike topological classification, the KDB does not discretise -- geometries that are close but not exact are used to *suggest* saddle point structures, which are then refined by a local saddle point search. As the database grows, suggestions become more accurate and the cost of off-lattice simulation approaches that of standard lattice KMC.

**References:**
- Henkelman, Jonsson, *J. Chem. Phys.* 115, 9657 (2001)
- EON documentation: https://theory.cm.utexas.edu/eon/kdb.html

#### Defect-Centric Classification (PESTO)

PESTO differs from k-ART in that it centres topological analysis on *defects* rather than individual atoms. Clusters are built around identified defects (not arbitrary atoms), and their configurations are compared to stored clusters for that defect topology. This reduces the catalogue size when defects are sparse.

**References:**
- Vernon, Kenny, *Modelling Simul. Mater. Sci. Eng.* (2018)

#### Spatial Discretisation (SLKMC)

The Self-Learning KMC approach (Rahman group) divides the space around a central atom into 3D rectangular boxes. A process is described as the central atom moving to a neighbouring vacant box, accompanied by motion of atoms in surrounding boxes. On the fcc(111) surface, SLKMC-II groups fcc sites, hcp sites, and top-layer substrate atoms into hexagonal rings to identify local environments.

**References:**
- Nandipati, Kara, Shim, Rahman, *J. Comput. Phys.* 231, 3548 (2012)

### 2.2 Shape Matching and Event Reconstruction

The Iterative Rotations and Assignments (IRA) algorithm solves the shape matching problem for atomic structures. In the KMC context, IRA performs two functions: (1) determining whether a local environment matches a known topology from the catalogue, and (2) finding the optimal rotation, translation, and atom permutation to map a reference event onto a new site. IRA uses the Hausdorff distance (a permutation-invariant set-set metric) to evaluate matches.

A "tolerant classification" extension classifies environments with a norm-based tolerance invariant under Euclidean transformations and permutations, bridging the fully discrete (topological) and fully continuous (geometric) approaches.

**References:**
- Gunde, Certain, Beland, Mousseau, Trochet, *J. Chem. Inf. Model.* 61, 5446 (2021)
- Gunde, PhD thesis, Universite de Montreal (2021): https://theses.hal.science/tel-03635139v2

### 2.3 Canonical Event Types on FCC Surfaces

The surface diffusion literature identifies several canonical mechanisms:

| Mechanism | Description | Typical Barrier (Ni) |
|---|---|---|
| **Simple hop (1NN)** | Atom hops to adjacent vacant site on the same layer | ~0.5 eV |
| **Next-nearest hop (2NN)** | Atom hops to second-neighbour vacant site on the same layer | ~0.96 eV |
| **Exchange** | Adatom/vacancy displaces a surface atom vertically, which pops up | ~0.96 eV |
| **Concerted exchange** | Correlated motion of 3+ atoms | ~1.0+ eV |
| **Subsurface exchange** | Atom exchanges between first and second layers | ~1.86 eV |

Exchange mechanisms dominate on fcc(100) for Al, Pt, and Ir. On Ni(100), simple hopping dominates for surface vacancies, with exchange events at higher barriers.

**References:**
- Feibelman, *Phys. Rev. Lett.* 65, 729 (1990)
- Stoltze, *Phys. Rev. B* 60, 2106 (1999)
- Andersen, Panosetti, Reuter, *Front. Chem.* 7, 202 (2019)

### 2.4 Basin Methods and Temporal Coarse-Graining

When systems enter metastable basins (groups of states connected by low barriers), superbasin / coarse-graining approaches group fast-rate states and solve for exit rates via the absorbing Markov chain master equation. This maps many off-lattice micro-events into single macro-events. pyKMC implements this as the basin exploration module.

**References:**
- Novotny, *Phys. Rev. Lett.* 74, 1 (1995)
- Henkelman group, EON documentation: http://theory.cm.utexas.edu/henkelman/code/eon/html-docs/coarse_graining.html

### 2.5 Off-Lattice to On-Lattice Projection

Traditional lattice KMC defines event types and directions *a priori* from the crystal structure, with rates fitted from DFT or empirical potentials. Object KMC (OKMC) treats defects as objects jumping in lattice-specific directions, with parameterisation typically done by hand.

This pipeline instead derives the on-lattice model from the off-lattice catalogue:
1. Classifying off-lattice events by mechanism and projecting their displacement vectors onto lattice directions
2. Extracting environment-dependent rate tables from the off-lattice catalogue and step log
3. Parameterising a reduced lattice-KMC model directly from off-lattice data

---

## 3. Our Classification Scheme

The classification operates in four stages, with optional refinements at Stage 1:

```
Off-lattice event catalogue
         |
    Stage 1a: Mechanism identification (displacement geometry)
         |
    Stage 1b: Saddle-point feature extraction       (optional)
    Stage 1c: GMM adaptive thresholds                (optional)
    Stage 1d: Saddle-assisted enhanced classification (optional)
    Stage 1e: Species-aware classification            (optional)
    Stage 1f: ML descriptors (Tiers 1-4)              (optional)
    Stage 1g: Z-layer + floating-atom reclassification
    Stage 1h: Environment fingerprints (ML Tier 5)    (optional)
         |
    Stage 2: Lattice projection (KDTree site mapping)
         |
    Stage 3: Environment assignment (global trajectory replay)
         |
    Stage 4: Rate table construction
         |
On-lattice rate lookup table
```

Stages 1b--1f and 1h are additive refinements that do not change the base
classifier API. They feed into the same downstream pipeline. Stage 1g is an
automatic post-pass that relabels subsurface events and non-physical
floating-atom artifacts.

Stage 1 runs in the classification notebook
`Analysis/classify_lattice_events.ipynb` (built from
`Analysis/classify_lattice_events_parts/`; see its README for how to run it).
Stage 2 is the per-run diagnostic `python -m Analysis.lattice_map_events
/path/to/run`; Stages 3 and 4 run in `python -m Analysis.build_rate_table`.

### 3.1 Stage 1a: Mechanism Identification

**Source:** `lattice_map_events.py`, function `classify_event_mechanism()`

Each event in the off-lattice catalogue consists of initial, saddle, and final atomic positions for a local cluster. The classification examines the displacement vectors `d = final_pos - initial_pos` for all atoms in the cluster.

#### Constants

| Parameter | Value | Meaning |
|---|---|---|
| `MOVE_THRESHOLD` | 0.5 A | Minimum displacement for an atom to count as "moved" |
| `DIST_TOLERANCE` | 0.5 A | Tolerance for matching displacements to lattice distances |
| `A_NI` | 3.524 A | Ni FCC lattice constant |
| `NN1_DIST` | 2.492 A | 1NN distance on (100) surface = a/sqrt(2) |
| `NN2_DIST` | 3.524 A | 2NN distance on (100) surface = a |
| Layer change threshold | 1.0 A | \|dz\| above this indicates interlayer motion |
| `DEALLOYING_TYPES` | set of 7 | Event types relevant to dealloying: 1NN, 1NN_coop, 2NN, exchange_up/down, interlayer_hop_up/down |

#### Classification Rules

For each event, the pipeline computes:
- `n_moved`: count of atoms with displacement > `MOVE_THRESHOLD`
- `move_dist`: total displacement of the primary mover
- `move_dz`: vertical (z) component of displacement
- `move_dxy`: in-plane component = sqrt(dx^2 + dy^2)
- `layer_change`: whether `|move_dz| > 1.0 A`

**Fallback logic:** If pyKMC's reported `move_atom_idx` has displacement < `MOVE_THRESHOLD`, the code automatically identifies the largest-displacement atom and reclassifies using it as the primary mover.

The event type is assigned by the first matching rule:

| Event Type | Conditions |
|---|---|
| `1NN_hop` | No layer change AND \|move_dist - NN1_DIST\| <= DIST_TOLERANCE |
| `2NN_hop` | No layer change AND \|move_dist - NN2_DIST\| <= DIST_TOLERANCE |
| `exchange` | Layer change AND n_moved >= 2 AND move_dist < NN2_DIST + DIST_TOLERANCE |
| `exchange_lateral` | Layer change AND move_dist >= NN2_DIST + DIST_TOLERANCE |
| `concerted` | n_moved >= 3 |
| `relaxation` | No atom displaced above MOVE_THRESHOLD (null event) |
| `complex` | None of the above |

`1NN_hop`, `1NN_hop_cooperative`, and `2NN_hop` are the event types treated as on-lattice moves (`LATTICE_TYPES`).

#### Threshold Rationale

The thresholds derive from the geometry of the Ni FCC (100) surface:

- **MOVE_THRESHOLD (0.5 A):** Well below the 1NN distance (2.49 A) but large enough to filter thermal noise. Relaxed surfaces show intra-layer z-spread of ~0.3 A.
- **DIST_TOLERANCE (0.5 A):** Provides a +/- 20% tolerance band around lattice distances. The gap between 1NN (2.49 A) and 2NN (3.52 A) is 1.03 A, so a 0.5 A band does not cause overlap.
- **Layer change (1.0 A):** The interlayer spacing for Ni(100) is ~1.76 A (= a/2). A 1.0 A threshold sits between intra-layer noise (~0.3 A) and true interlayer motion.

### 3.1b Stage 1b: Saddle-Point Feature Extraction

**Source:** `lattice_map_events.py`, function `compute_saddle_features()`

Each event in the reference table stores saddle-point atomic positions alongside initial and final positions. Stage 1b extracts six geometric descriptors from the saddle configuration:

| Descriptor | Definition | Physical Meaning |
|---|---|---|
| `saddle_disp_mag` | \|saddle - initial\| for the primary mover | How far the mover travels to reach the saddle |
| `saddle_dz` | Signed z-component of (saddle - initial) | Vertical direction at the saddle |
| `saddle_height_ratio` | mover z at saddle / mean z of displaced neighbours | **Key discriminator**: >1 = hop (mover above neighbours at bridge site); ~1 = exchange (dumbbell) |
| `saddle_n_displaced` | Atoms with \|saddle - initial\| > MOVE_THRESHOLD | Number of atoms participating at the saddle |
| `saddle_asymmetry` | \|d(init->saddle)\| / \|d(init->final)\| | Path symmetry; ~0.5 = symmetric, deviates for asymmetric paths |
| `saddle_max_pair_dz` | max \|dz_i - dz_j\| among displaced atoms at saddle | Large for exchange (atoms at different heights), small for in-plane concerted |

Events where `saddle_positions` are unavailable or identical to `initial_positions` return NaN values, so downstream code can filter gracefully.

Accessed via `classify_event_from_row(row, include_saddle_features=True)`.

### 3.1c Stage 1c: GMM Adaptive Thresholds

**Source:** `lattice_map_events.py`, function `compute_thresholds_gmm()`

Instead of the hardcoded constants in Stage 1a, this stage fits a two-component Gaussian Mixture Model (GMM) to the observed displacement distributions to derive data-driven thresholds. This sits alongside the existing geometric `compute_thresholds()` function.

**Algorithm:**
1. Filter to in-plane events (\|dz\| < 1.0 A) to avoid exchange contamination
2. Fit a 2-component GMM to displacement magnitudes; seed means at a/sqrt(2) and a
3. Set acceptance windows as the tighter of mean +/- 2*sigma and the GMM crossover point
4. Fit a separate 2-component GMM to \|dz\| values for the layer-change boundary
5. Report the Bhattacharyya distance as a confidence metric for shell separation

The fitted thresholds are written to `adaptive_thresholds.json` and compared with the hardcoded ones in `gmm_threshold_comparison.png`.

`classify_event_adaptive(row, thresholds)` uses the same decision-tree logic as `classify_event_mechanism()` but with GMM-derived windows.

### 3.1d Stage 1d: Saddle-Assisted Enhanced Classification

**Source:** `lattice_map_events.py`, function `classify_event_enhanced()`

For events near a classification boundary (displacement within 0.2 A of a threshold edge), Stage 1d consults the saddle-point geometry to resolve ambiguity:

1. Run the standard displacement-based decision tree (Stage 1a, with adaptive thresholds from 1c if available)
2. If the event falls in a boundary zone, check `saddle_height_ratio`:
   - Below 1.15: exchange-like saddle geometry
   - At or above 1.15: hop-like saddle geometry
3. Override the displacement label only when the saddle evidence contradicts it
4. Flag reclassified events with `reclassified=True` and `original_event_type`

**Special case:** Near-zero-barrier `interlayer_hop_down` events (Ea < 0.05 eV) with negligible saddle displacement are reclassified as `relaxation` and flagged with `likely_reverse_relaxation=True`.

The enhanced path only changes labels for events near a threshold boundary, so it makes few or no changes when the displacement shells are well separated. It matters most for alloys, where lattice distortion blurs the shells.

### 3.1e Stage 1e: Species-Aware Classification

**Source:** `lattice_map_events.py`, functions `_extract_species_info()` and `classify_event_from_row(include_species=True)`

For multi-element simulations, Stage 1e tags each event with the chemical species of the atoms involved:

- `mover_species`: species of the primary moving atom
- `partner_species`: species of the second mover in cooperative or exchange events
- `nn_species_counts`: dict of species counts among 1NN neighbours of the mover

This enables species-dependent rate tables via `build_rate_table.py`, which produces `rate_lookup_table_species.csv` grouped by `(mover_species, n_vacant_inplane_nn, n_vacant_2nn)`.

For a single-element system every mover has the same species, so the species table collapses to the extended table.

### 3.1f Stage 1f: ML Descriptors (Tiers 1--4)

**Source:** `lattice_map_events.py`, function `compute_ml_descriptors()`; driver cell `classify_lattice_events_parts/04b_ml_descriptors.py`

Stage 1f augments each event with 31 additional descriptors designed for training barrier-prediction ML models. Organised into four tiers by cost and physical content:

#### Tier 1: Mover-only features (cheap, high signal)

| Descriptor | Definition | Physical Meaning |
|---|---|---|
| `disp_top1` .. `disp_top5` | Sorted top-5 per-atom displacement magnitudes | Hop vs cooperative vs concerted signature |
| `mover_depth_initial` / `_final` / `_delta` | `z_topsurf - z_mover` (Angstroms) | Continuous alternative to surface_layer_{initial,final} |
| `Ea_over_kT` | `energy_barrier / (k_B * T)` | kT-normalised barrier — controls whether the event fires |
| `mover_species_ml` | Primary mover element symbol | Cross-check against Stage 1e `mover_species` (they can differ when the fallback mover is used) |
| `nn1_count_{Ni,Fe,Cr}` | Species histogram in 1NN shell around mover | LAC-style descriptor (Zhao et al. 2023, Messina 2017) |
| `nn2_count_{Ni,Fe,Cr}` | Species histogram in 2NN shell around mover | Extends LAC to second shell; matters for FCC alloy barriers |

#### Tier 2: Neighbourhood geometry (medium cost)

| Descriptor | Definition | Physical Meaning |
|---|---|---|
| `coord_mover_initial` / `_final` | Neighbour count within 1.1 * NN1_DIST | Bulk = 12, surface = 8; undercoordination signals step edges |
| `n_vac_nn1_initial` / `_final` | Count of unoccupied 1NN lattice sites around mover | Vacancy clustering environment |
| `mean_nn1_bond_length_initial` | Mean distance from mover to 1NN neighbours | Strain at mover site (~2.49 A unstrained Ni) |
| `dist_to_nearest_undercoord` | Distance to nearest atom with coordination < 7 | Proximity to step/surface defect |
| `local_rmsd_5A` | RMSD of atoms within 5 A of mover, initial vs final | Locality of perturbation |

#### Tier 3: Saddle-path features (NaN-safe)

| Descriptor | Definition | Physical Meaning |
|---|---|---|
| `path_straightness` | `|final - initial| / (|saddle - initial| + |final - saddle|)` | 1.0 = straight hop; low = curved exchange (S-path) |
| `bonds_broken_init_to_saddle` | 1NN bonds present at init but not at saddle | Typically ~4 for FCC(100) hops |
| `bonds_formed_init_to_saddle` | 1NN bonds present at saddle but not at init | Typically ~4 for FCC(100) hops |
| `coord_mover_saddle` | Neighbour count of mover at saddle | On-top (~4), bridge (~2), hollow (~3) |

NaN when `saddle_positions` is `None` or identical to `initial_positions` (same convention as Stage 1b).

#### Tier 4: Metadata

| Descriptor | Definition | Physical Meaning |
|---|---|---|
| `reverse_barrier` | Ea of the event at `idx_backward` in the same simulation | Forward/backward pair energy |
| `Ea_gap` | `energy_barrier - reverse_barrier` | Deep basin (large gap) vs shallow rattling |
| `saddle_asymmetry_oob` | True if `saddle_asymmetry < 0.1` or `> 1.1` | pARTn convergence sanity flag |
| `kmc_fire_count` | Number of KMC steps that selected this `idx_ref` | Empirical event importance; training weight |

**Pre-loop caches.** `04b_ml_descriptors.py` populates three per-simulation caches before iterating events: (1) cell vectors from `trajkmc.xyz` frame 0 (diagonal `[Lx, Ly, Lz]`), (2) optional ideal lattice + 1NN adjacency from `generate_ideal_sites()` + `build_surface_neighbour_list()` (gated by `COMPUTE_LATTICE_ADJACENCY` flag — disabled by default due to pure-Python O(surface*sites*9) cost), (3) KMC step log counts from `parse_pykmc_out()` on `pykmc.out`.

> **⚠ Caveat: `n_vac_nn1_initial` is a coordination-shortfall proxy when `COMPUTE_LATTICE_ADJACENCY = False`.** With the flag off (the default — the True path takes minutes per simulation), `compute_ml_descriptors()` falls back to `n_vac_nn1 = max(0, 12 − coord_mover)`. This works for bulk atoms (12 expected 1NN) but **always reports `nv1 = 4` for surface atoms** (which have 8 expected 1NN, not 12), regardless of the actual vacancy environment. Family rate tables use `n_vac_nn1_initial` as a bucket key, so the uncorrected column gives wrong environment buckets and wrong rates.
>
> **Fix (post-processor).** Run `python -m Analysis.tools.fix_catalogue_nvac` on `classified_events.csv` after every catalogue rebuild and before family assignment. It recomputes `n_vac_nn1_{initial,final}` and `n_vac_nn2_initial` using the **layer-aware** formula (`expected_coord = 8` for `surface_layer_*_initial == 0`, `12` otherwise; `expected_2nn = 5` surface, `6` bulk); on a table that already has family columns it also re-derives `family_bucket_id` (1-axis or 2-axis, depending on family). Runs in seconds.
>
> **Why not just flip the flag?** Building `surface_neighbour_list()` per simulation is currently O(surface × sites × 9 PBC images) in pure Python and adds minutes per simulation. The post-processor exploits the fact that the catalogue already carries `coord_mover_*` and `nn2_count_*` columns, so the layer-aware formula is exact (no adjacency rebuild needed). Until the neighbour-list build is vectorised, the post-processor is mandatory.
>
> See `Analysis/tools/README.md`.

**Post-loop self-join.** After the per-event loop, `reverse_barrier` and `Ea_gap` are filled by a pandas merge on `(sim_path, idx_backward -> idx_ref)`. `idx_backward` is a per-simulation index, so the composite key is essential. Events without a reverse partner below the barrier cutoff get NaN.

**Column schema:** the `ML_SPECIES_SCHEMA` constant `('Ni', 'Fe', 'Cr')` fixes the histogram column set across compositions. Pure-Ni events get 0 for Fe/Cr counts (not NaN).

**Performance.** About 0.4 ms per event.

**Accessed via:** the notebook cell `04b_ml_descriptors`, which runs with the rest of the classification notebook. The toggle `COMPUTE_ML_DESCRIPTORS = True` at the top of the part file can short-circuit the whole stage for quick iteration.

#### Tier 5: Lattice-environment fingerprints (Stage 1h)

**Source:** `env_fingerprint.py`, called through `lattice_map_events.compute_env_fingerprint()` in the same cell.

| Descriptor | Definition |
|---|---|
| `env_fingerprint_occ` | Occupancy of the mover's 1NN+2NN shell plus the hop vector, canonicalised under the D4 surface symmetry |
| `env_fingerprint_species` | The same with species instead of occupancy |
| `hop_vector_canonical` | The canonical hop vector, in units of (nn1/2, nn1/2, a/2) |
| `env_shell_complete` | True when every shell site lies inside the event cluster (otherwise the fingerprint is left empty) |

Two events with the same local lattice configuration, up to D4 symmetry, get the same fingerprint string.

### 3.1g Stage 1g: Z-Layer + Floating-Atom Reclassification

**Source:** `classify_lattice_events_parts/07_reclassify.py`

After the displacement-based mechanism has been assigned (Stages 1a and 1d), two automatic reclassifications run in order:

#### Z-layer pass (`reclassify_by_zlayer`)

Each event's primary mover is assigned a surface layer index (0 = top layer) in both initial and final configurations via `get_event_surface_layers()`. Events where both endpoints sit below layer 0 are relabelled into a subsurface type:

| Pre-zlayer type | Post-zlayer type (both layers >= 1) |
|---|---|
| `1NN_hop`, `1NN_hop_cooperative`, `2NN_hop` | `subsurface_1NN_hop` |
| `exchange_up`, `exchange_down` | `subsurface_migration` |
| `inplane_complex` | `subsurface_other` |

The surface-layer combinations (from `surface_layer_initial`, `surface_layer_final`) present in the data are summarised by the classification notebook.

#### Floating-atom reclassification (post z-layer)

`interlayer_hop_down` events with primary displacement > 3 A are non-physical artifacts: atoms that detached from the surface entirely, with zero-barrier return paths. These are now reclassified automatically as `relaxation`:

```python
# 07_reclassify.py (end of file)
_float_mask = (
    (sub['move_type'] == 'interlayer_hop_down')
    & (sub['move_dist'] > 3.0)
)
sub.loc[_float_mask, 'move_type'] = 'relaxation'
sub.loc[_float_mask, 'move_type_base'] = 'relaxation'
sub.loc[_float_mask, 'zlayer_reclassified'] = True
```

After this rule, every remaining `interlayer_hop_down` event has `|d| <= 3 A` (the pARTn-valid range).

This rule replaces the previous manual "Auto-flag unphysical" button in the event viewer UI, which was removed because the rule belongs in the classification pipeline, not in the audit tool.

### 3.2 Stage 2: Lattice Projection

**Source:** `lattice_map_events.py`, function `map_event_to_lattice()`

This stage projects the continuous atomic positions onto an ideal FCC lattice to determine *which* lattice site the vacancy moved between.

#### Ideal Lattice Construction

An ideal FCC (100) surface lattice is generated with the same dimensions and number of layers as the simulation cell. Sites are placed at perfect FCC positions using lattice constant `A_NI = 3.524 A`.

#### Layer Detection

The function `group_z_layers()` identifies distinct atomic layers:
1. Sort atoms by z-coordinate
2. Group atoms within `z_tol = 0.5 A` of each other into a single layer
3. Return layer means and atom indices, sorted bottom to top

This handles the ~0.3 A intra-layer z-spread of relaxed surfaces while cleanly separating layers spaced ~1.76 A apart.

#### Site Assignment (KDTree Projection)

For each atom in initial and final configurations:
1. Wrap coordinates to the periodic box (xy-periodic, z free)
2. Expand ideal lattice with 3x3 periodic images in the xy-plane to handle boundary atoms
3. Use a KDTree to find the nearest ideal site for each atom
4. Map back from image site indices to primary cell site indices

#### Vacancy Tracking

Since the atom hops *to* a site, the vacancy moves in the opposite direction:
- `vac_site_from` = the site the atom moved *to* (where the vacancy was before)
- `vac_site_to` = the site the atom moved *from* (where the vacancy ends up)

The vacancy hop distance is computed with periodic boundary handling and validated against the expected 1NN distance.

#### Approximate Local Environment

From the local event cluster, the code counts how many of the vacancy's nearest neighbours are also vacant. This provides an *approximate* environment count. However, the local cluster may not contain all relevant atoms (e.g., a second vacancy just outside the cluster radius). The authoritative environment comes from Stage 3.

> **Legacy (surface-only rate table).** Sections 3.3 and 3.4 below describe
> the 2D surface-only path that remains authoritative only for the legacy
> `rate_lookup_table.csv`. For the 3D-first path — including subsurface,
> exchange, and interlayer events — see Section 3.6.

### 3.3 Stage 3: Environment Assignment (Authoritative)

**Source:** `build_rate_table.py` (`python -m Analysis.build_rate_table --data-root /path/to/data --composition 100Ni`)

The local cluster in Stage 2 gives only an approximate vacancy environment. For the rate table, we need the *true* environment from the full simulation state.

#### Why Local Counts Are Insufficient

The off-lattice event catalogue stores only a local cluster of atoms around the active site. A second vacancy sitting just outside the cluster radius would be invisible to Stage 2's environment count, leading to an incorrect rate assignment.

#### Global Trajectory Replay

For each KMC step that selected a 1NN hop (from the `pykmc.out` step log):

1. Load the trajectory frame at step N-1 (the state *before* the event)
2. Project all atoms onto the ideal lattice using a KDTree
3. Identify all vacancy sites: the `n_vac` ideal sites farthest from any atom
4. Load the trajectory frame at step N (the state *after* the event)
5. Identify all vacancy sites again
6. Determine which vacancy hopped: the site present in vacancies_before but not vacancies_after

#### In-Plane Neighbour Counting

For the hopping vacancy's origin site:
1. Look up its in-plane nearest neighbours from a pre-built adjacency list
2. The adjacency list uses a 2D KDTree with cutoff = `NN1_DIST * 1.1` and 3x3 PBC images
3. Count how many of those neighbours are also in the global vacancy set
4. This gives `n_vacant_inplane_nn`: the true number of vacant in-plane neighbours

The in-plane restriction is deliberate -- the on-lattice model targets a 2D surface KMC, so only in-plane neighbours matter for the environment descriptor.

#### Surface Hop Filter

Only hops where both origin and destination are in the surface layer are included in the rate table. Subsurface exchanges (which cannot be represented by simple lattice hops) are excluded.

### 3.4 Stage 4: Rate Table Construction

**Source:** `build_rate_table.py`

Surface 1NN hops with barriers below `BARRIER_CUTOFF = 1.2 eV` are grouped by `n_vacant_inplane_nn`. For each group:

- Mean, standard deviation, min, max, and median barrier are computed
- Arrhenius rates are calculated at representative temperatures using:
  - Attempt frequency: k_0 = 10 THz (10^13 Hz)
  - k_B = 8.617333 x 10^-5 eV/K
  - rate(T) = k_0 * exp(-E_a / (k_B * T))

#### Output

`rate_lookup_table.csv` has one row per `n_vacant_inplane_nn` value with `n_events`, the barrier statistics above and `rate_{T}K_Hz` columns for T = 300, 400, 500, 600 and 800 K. The script reports the spread of mean barriers across environments; a spread well above the within-bin scatter means per-environment barriers are needed rather than a single global barrier. Directional (`vacancy_pattern`), extended (1NN + 2NN), exchange, exchange_lateral and species variants are written alongside it when the data supports them.

### 3.5 Output Files

The classification notebook writes these files to the output directory (`$PYKMC_ANALYSIS_OUTPUT_DIR`, default `Analysis/lattice_event_classification/`):

| File | Description |
|---|---|
| `classified_events.csv` | Every sub-cutoff event with the base classification, the ML descriptors, the lattice-environment fingerprints and the 3D labels (`motif_family_3d`, `site_class_3d`, `direction_family_3d`). Primary downstream consumers: event viewer, rate table builders, family assignment, ML training. |
| `classified_events_enhanced.csv` | Same row set + saddle features, enhanced type, adaptive type, reclassification flags |
| `saddle_feature_distributions.csv` | Events with saddle-point geometry descriptors |
| `adaptive_thresholds.json` | GMM-fitted thresholds (nn1/nn2 centres, acceptance windows, layer-change boundary, confidence) |
| `saddle_feature_distributions.png` | Box plots and 2D scatter of saddle descriptors by event type |
| `gmm_threshold_comparison.png` | Hardcoded vs GMM-derived thresholds overlaid on displacement histograms |
| `enhanced_reclassification.png` | Scatter showing which events were reclassified by the enhanced saddle-assisted stage |
| `kinetic_coverage.csv`, `lattice_parameters.json` | Per-simulation kinetic coverage (Section 4) and a lattice/parameter summary |

The human audit log (`audit_log.csv`, keyed on `(composition, nvac, temp, idx_ref)`) lives in the same directory but is written by the event viewer, not the notebook.

#### `classified_events.csv` column groups

The columns are grouped for readability:

| Group | Columns |
|---|---|
| Metadata | `composition`, `nvac`, `temp`, `sim_label`, `sim_path`, `sim_relpath`, `idx_ref`, `k`, `dra`, `idx_backward` |
| Classification | `move_type`, `move_type_base`, `move_type_pre_zlayer`, `surface_layer_initial`, `surface_layer_final`, `zlayer_reclassified`, `layer_change`, `n_moved` |
| Geometry (Stage 1a) | `energy_barrier`, `move_dist`, `move_dz`, `move_dz_signed`, `move_dxy` |
| ML Tier 1 | `disp_top{1-5}`, `mover_depth_{initial,final,delta}`, `Ea_over_kT`, `mover_species_ml`, `nn{1,2}_count_{Ni,Fe,Cr}` |
| ML Tier 2 | `coord_mover_{initial,final}`, `n_vac_nn1_{initial,final}`, `mean_nn1_bond_length_initial`, `dist_to_nearest_undercoord`, `local_rmsd_5A` |
| ML Tier 3 | `path_straightness`, `coord_mover_saddle`, `bonds_{broken,formed}_init_to_saddle` |
| ML Tier 4 | `reverse_barrier`, `Ea_gap`, `saddle_asymmetry_oob`, `kmc_fire_count` |
| ML Tier 5 | `env_fingerprint_occ`, `env_fingerprint_species`, `hop_vector_canonical`, `env_shell_complete` |
| 3D labels | `motif_family_3d`, `site_class_3d`, `direction_family_3d` |

`sim_path` is the run directory on the machine that ran the notebook; `sim_relpath` is the same path relative to the data root, so the CSV still finds its runs on another machine (`Analysis._paths.resolve_sim_path` joins it to that machine's `PYKMC_ANALYSIS_DATA_ROOT`).

### 3.6 Stage 3D: 3D occupancy-transition mapping

Extends the surface-only mapping to the full slab. The pure-function helpers
live in `Analysis/lattice_3d.py` (unit-tested in `Analysis/tests/`):

| Helper | Purpose |
|---|---|
| `compute_3d_site_graph` | Full 3D site -> {position, layer_index, 1NN_3d, 2NN_3d} graph |
| `assign_3d_site_classes` | Site -> {surface, subsurface, bulk_like} |
| `classify_3d_motif` | Event row -> motif family (listed below) |
| `extract_3d_occupancy_transition` | Minimal occupancy transition per event |
| `compute_3d_environment_descriptor` | 3D 1NN/2NN vacancy and species counts |
| `assign_direction_family` | DEPRECATED: (di, dj, dz) integer-lattice-index -> FCC(100) direction family. Cannot distinguish 1NN cross-layer hops from 2NN axial hops because both collapse to (di, dj) = (0, 0) in the per-layer in-plane neighbour scheme. |
| `classify_direction_xyz` | Preferred: Cartesian displacement `(dx, dy, dz)` + `nn_dist` -> FCC(100) direction family. Mirrors `classify_direction()` in pylatkmc's `tools/build_initial_config.py` so the training pipeline and the C runtime agree on edge labels. |

**Motif families** (values of column `motif_family_3d`):
`surface_1nn_translation`, `subsurface_1nn_translation`,
`surface_2nn_translation`, `surface_subsurface_exchange`,
`interlayer_translation`, `subsurface_exchange`, `concerted_3d`,
`unresolved_multisite`.

**Direction families** (cubic-frame Miller indices — values of column
`direction_family_3d`): `<110>_inplane` (surface 1NN along cubic <110>),
`<100>_inplane` (in-plane 2NN face-diagonal), `<111>_interlayer` (1NN hop
across a layer), `<001>_exchange` (pure z-exchange, no in-plane shift),
`unresolved`.

**Outputs** (written by `build_rate_table.py`):

- `rate_lookup_table_3d_master.csv` groups the mapped-step rows by
  (`motif_family_3d`, `site_class_3d`, `n_vacant_1nn_3d`, `n_vacant_2nn`,
  `direction_family_3d`) over every row with
  `event_type in ACCEPTED_TYPES_3D` — including the subsurface
  events excluded from the legacy surface gate. Sparse in motifs because
  pyKMC fires mostly `1NN_hop` events at moderate T (the 0.5 eV surface
  1NN barrier dominates 1 eV exchange/interlayer barriers by ~10⁵×). This
  is **the kinetic-weighted view**: it reflects what pyKMC actually sampled.
  For a mechanism-complete alternative (every discovered saddle, regardless
  of firing frequency), see the catalogue table below.
- `rate_lookup_table_3d_catalogue.csv` groups the full event catalogue
  (every `classified_events.csv` row) by (`motif_family_3d`,
  `site_class_3d`, `n_vac_nn1_initial`). Covers every motif family, so it
  suits an on-lattice model that needs barriers across all motifs.
  `n_vac_nn1_initial` is a local-cluster proxy for the 1NN vacancy
  environment (from the Stage 1f ML descriptors; correct it first, see
  Section 3.1f).

The legacy scalar `rate_lookup_table.csv` is still produced, together with
an identical copy named `rate_lookup_table_surface_legacy.csv`.

---

## 4. Kinetic Coverage

Not all event types contribute equally to the simulation kinetics. Exchange and concerted events are present in the catalogue but are rarely *selected* by the KMC algorithm when their barriers are well above the 1NN hop barrier. Kinetic coverage measures how much of an off-lattice run an on-lattice model built from lattice-representable events would reproduce.

**Source:** notebook cell `16_kinetic_coverage.py` (every run; written to `kinetic_coverage.csv`) and `compute_kinetic_coverage()` in `lattice_map_events.py` (the single-run report).

For each simulation, the `pykmc.out` step log is joined to the classified catalogue, and two fractions are reported:

- the percentage of KMC steps that selected a lattice-representable event (`LATTICE_TYPES`: 1NN and 2NN hops), and
- the percentage of simulated time (sum of step `dt`) spent in those steps.

The notebook reports both before (`_any`) and after (`_surface`) the Stage 1g z-layer relabelling, and separately for 1NN hops alone (`pct_*_1nn_*`). High values mean a surface lattice KMC model captures most of the off-lattice kinetics at those conditions.

---

## 5. Comparison with Literature

| Approach | Environment Descriptor | Event Classification | Lattice Projection | Environment-Dependent Rates |
|---|---|---|---|---|
| **k-ART (NAUTY)** | Graph topology | By topology class | No | By topology |
| **KDB (EON)** | Geometric distance | By geometric similarity | No | By geometry |
| **SLKMC** | 3D box pattern | By box occupancy | Implicit (box grid) | By pattern |
| **PESTO** | Defect-centered topology | By defect cluster | No | By defect type |
| **Ours** | Displacement geometry + global trajectory | By mechanism type | KDTree projection onto FCC | By vacant NN count |

### Key Differences

**vs. k-ART:** We classify by displacement geometry (magnitude, direction, layer change), not graph topology. Our approach requires a reference lattice, making it specific to crystalline surfaces. k-ART handles amorphous systems; we do not need to.

**vs. SLKMC:** We project onto the exact ideal lattice using KDTree nearest-neighbour search, rather than discretising space into boxes. This preserves the full crystallographic structure and allows precise identification of which lattice site the vacancy moved to.

**vs. KDB:** We classify by *mechanism type* (1NN hop, exchange, etc.) and *local environment* (vacant neighbour count), rather than by geometric similarity of the full event. This gives a compact, physically interpretable parameterisation for the on-lattice model.

---

## 6. Curated FCC Family Catalogue

The pipeline above emits a per-event classification keyed by (motif_family_3d, site_class_3d, direction_family_3d) tuples. Those keys are correct, but a rate table built from them stays coupled to whatever pyKMC happened to fire in the step log — kinetically weighted and sparse in rare motifs.

The **curated family catalogue** is a parallel, human-auditable output. Family assignment and the family rate table live in [pylatkmc](https://github.com/Kerrator/pylatkmc) (`pylatkmc.ingest`), which reads `classified_events.csv` and the event viewer's audit log from this repository:

- **Registry.** `pylatkmc/ingest/families.py` declares a stable list of `FCCFamily` entries. Each family has:
  - a `family_id` (machine) / `family_name` (human) / `movement_template` (one-liner physical description),
  - a `seed_rule` — a pure predicate over existing classifier columns (`motif_family_3d`, `direction_family_3d`, `site_class_3d`, `move_type_pre_zlayer`, `n_moved`, ...) — that must not overlap any other family's seed at the same `priority` tier,
  - an `environment_rule` — maps a row to a family-specific `family_bucket_id` (e.g. `nv1=1_nv2=0` for surface_1NN_inplane, `li=0_nv1=6` for interlayer hops). Granularity is per family.
  - a `fit_barrier` flag: `False` for multisite / unresolved families that stay visible in the export but are excluded from barrier fitting.

- **Two-axis bucketing.** Grouping events only by `n_vac_nn1` mixes isolated-vacancy environments (low Ea) with clustered configurations (high Ea) in a single bucket and biases the family's weighted-mean barrier. The dominant single-atom hop families — `surface_1NN_inplane`, `subsurface_1NN_inplane` and `bulk_1NN_inplane` — therefore use 2-axis `(nv1, nv2)` buckets with ids `nv1=N_nv2=M` (e.g. `nv1=1_nv2=0` for an isolated vacancy). Other vacancy-bucketed families use `nv1=N`. Both counts come from the corrected `n_vac_nn1_initial` / `n_vac_nn2_initial` columns (Section 3.1f).

- **Adatom attachment.** The family seeded by Stage 1a `exchange_lateral` events was first named `surface_subsurface_exchange_lateral`; its events are single-atom adatom re-insertions into a surface vacancy, so it is now `adatom_attachment`. The old id is kept as a legacy alias (`LEGACY_FAMILY_ALIASES`) so older audit logs and CSVs still resolve. The reverse leg, `adatom_detachment` (seeded on `move_dz_signed > 0`), is declared for completeness. The Stage 1a move type keeps the name `exchange_lateral`, which is a geometric bucket (layer change + long traverse), not a mechanism.

- **Audit integration.** Family assignment reads the audit log (keyed by `(composition, nvac, temp, idx_ref)`) before running the seed predicates:
  - verdicts `ignore` / `unset` / flagged-bad → `assignment_status = excluded`,
  - verdict `reclassify` with `override_family_id` → that family is assigned directly (an optional `override_bucket_id` replaces the environment bucket),
  - verdict `reclassify` with only `override_move_type` → the override is applied to a row copy, then seed predicates run as normal,
  - no audit entry or `confirm` → seed predicates run as normal.

- **Outputs.**
  - `classified_events_with_families.csv`: the classified catalogue plus 7 columns (`family_id`, `family_name`, `family_bucket_id`, `family_bucket_name`, `assignment_status`, `assignment_note`, `audit_excluded`). Every row has `assignment_status ∈ {accepted, excluded, unresolved}`.
  - `rate_lookup_table_family.csv`: one row per (family_id, family_bucket_id) with `n_events`, `Ea_mean_eV`, `Ea_std_eV`, min/max/median, `source_filter`, and a JSON list of up to 5 `representative_row_indices` for reviewer drill-down. Declared-but-empty buckets and `fit_barrier=False` families are retained as visibility rows. This is the rate input pylatkmc builds its on-lattice models from.
  - `family_assignment_report.json`: coverage stats and an `unresolved_signatures` block listing the top (motif × class × direction) signatures that matched no seed rule — the actionable output for iterating on the registry.
  - `family_vs_current_comparison.csv` (from `compare_family_vs_current.py` in this repository): maps legacy (motif × class × direction) keys to curated families so a reader can see which legacy rows collapse into which curated bucket.

- **Scope.** The step-log-derived master CSV (`rate_lookup_table_3d_master.csv`) stays as a diagnostic.

- **Running** (from the repository root; `D` is the output directory):
  ```bash
  D=Analysis/lattice_event_classification

  # 1. Classify: run Analysis/classify_lattice_events.ipynb   -> $D/classified_events.csv
  # 2. Correct the vacancy counts (required, Section 3.1f)
  python -m Analysis.tools.fix_catalogue_nvac --input $D/classified_events.csv --output $D/classified_events.csv
  # 3. Families + family rate table (pylatkmc)
  python -m pylatkmc.ingest.family_assignment \
      --events $D/classified_events.csv --audit $D/audit_log.csv \
      --output $D/classified_events_with_families.csv --report $D/family_assignment_report.json
  python -m pylatkmc.ingest.build_family_rate_table \
      --input $D/classified_events_with_families.csv --output $D/rate_lookup_table_family.csv
  # 4. Step-log tables and the comparison
  python -m Analysis.build_rate_table --data-root /path/to/data --composition 100Ni
  python -m Analysis.compare_family_vs_current
  ```
  `python -m pylatkmc.ingest.build_curated_catalogue` runs step 3 in one pass. Tests for this repository: `python -m pytest Analysis/tests`.

---

## 7. References

1. El-Mellouhi, Mousseau, Lewis, "Kinetic activation-relaxation technique: An off-lattice self-learning kinetic Monte Carlo algorithm," *Phys. Rev. B* 78, 153202 (2008)
2. Beland, Brommer, El-Mellouhi, Joly, Mousseau, "Kinetic activation-relaxation technique," *Phys. Rev. E* 84, 046704 (2011)
3. Trochet, Mousseau, Beland, Henkelman, "Off-Lattice Kinetic Monte Carlo Methods," *Handbook of Materials Modeling* (2019)
4. Henkelman, Jonsson, "Long time scale kinetic Monte Carlo simulations without lattice approximation and predefined event table," *J. Chem. Phys.* 115, 9657 (2001)
5. Nandipati, Kara, Shim, Rahman, "Off-lattice pattern recognition scheme for kinetic Monte Carlo simulations," *J. Comput. Phys.* 231, 3548 (2012)
6. Gunde, Certain, Beland, Mousseau, Trochet, "IRA: A shape matching approach for recognition and comparison of generic atomic patterns," *J. Chem. Inf. Model.* 61, 5446 (2021)
7. Andersen, Panosetti, Reuter, "A practical guide to surface kinetic Monte Carlo simulations," *Front. Chem.* 7, 202 (2019)
8. Stoltze, "Models for adatom diffusion on fcc(001) metal surfaces," *Phys. Rev. B* 60, 2106 (1999)
9. Feibelman, "Diffusion path for an Al adatom on Al(001)," *Phys. Rev. Lett.* 65, 729 (1990)
10. Novotny, "Monte Carlo algorithms with absorbing Markov chains: Fast local algorithms for slow dynamics," *Phys. Rev. Lett.* 74, 1 (1995)
11. Xu, Osetsky, Stoller, "Self-Evolving Atomistic Kinetic Monte Carlo: Fundamentals and Applications," *J. Phys.: Condens. Matter* 24, 375402 (2012)
