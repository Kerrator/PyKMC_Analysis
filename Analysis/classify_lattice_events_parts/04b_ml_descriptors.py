# %%
# Cell 4b: ML Descriptors
#
# Adds a block of features designed for training barrier-prediction ML models:
#   Tier 1  mover-only      (disp_top_k, mover depth, species hist 1NN+2NN, Ea/kT)
#   Tier 2  neighborhood    (coordination, nearby vacancies, bond lengths, RMSD)
#   Tier 3  saddle-path     (path straightness, bond breaking, coord at saddle)
#   Tier 4  metadata        (reverse barrier, convergence flags, KMC fire count)
#
# All descriptors are computed at notebook-load time from data already in the
# per-event row (positions, species, move_atom_idx, energy_barrier) plus
# per-simulation cell vectors loaded once from trajkmc.xyz frame 0 and step
# logs parsed once from pykmc.out. No reference_table.pickle is regenerated.

from collections import Counter

from ase.io import read as ase_read

from Analysis.lattice_map_events import (
    compute_env_fingerprint,
    compute_ml_descriptors,
    parse_pykmc_out,
)

COMPUTE_ML_DESCRIPTORS = True
# Lattice adjacency (build_surface_neighbour_list) is expensive
# (~minutes per sim due to its pure-Python O(surface × sites × 9PBC) loop).
# Required for Tier 2 features `n_vac_nn1_initial/final`. With the flag off
# they fall back to a bulk coordination-shortfall proxy (wrong for surface
# atoms) — correct the exported CSV with `python -m Analysis.tools.fix_catalogue_nvac`
# (see Analysis/tools/README.md). The Stage 1h environment fingerprint runs
# from cluster geometry alone — no lattice adjacency needed — so it always fires.
COMPUTE_LATTICE_ADJACENCY = False

if COMPUTE_ML_DESCRIPTORS:
    # ── Per-simulation caches: load cells, lattices, step logs ──
    sim_cells = {}
    sim_lattices = {}   # sim_path -> (ideal_sites, neighbours_dict)
    sim_firing = {}     # sim_path -> Counter[ref_event] -> #selections

    for sp in sub['sim_path'].unique():
        sp_path = Path(sp)

        # Cell vectors from trajectory frame 0.
        # generate_ideal_sites and build_surface_neighbour_list expect cell as
        # a 3-element scalar array [Lx, Ly, Lz] (the diagonal), not the full 3x3
        # matrix (same convention as build_lattice_infrastructure in
        # Analysis/build_rate_table.py).
        try:
            atoms0 = ase_read(sp_path / 'trajkmc.xyz', index=0)
            sim_cells[sp] = list(atoms0.cell.array.diagonal())
        except Exception as e:
            print(f"[ml] cell load failed for {sp_path.name}: {e}")
            sim_cells[sp] = None

        # Ideal lattice + 1NN adjacency dict (expensive; gated by flag)
        if COMPUTE_LATTICE_ADJACENCY and sim_cells[sp] is not None:
            try:
                nvac_sp = int(sub.loc[sub['sim_path'] == sp, 'nvac'].iloc[0])
                cell_diag = sim_cells[sp]
                ideal = generate_ideal_sites(
                    atoms0.get_positions(), cell_diag, nvac_sp
                )
                neighbours, _, _, _, _ = build_surface_neighbour_list(
                    ideal, cell_diag, NN1_DIST
                )
                sim_lattices[sp] = (ideal, neighbours)
            except Exception as e:
                print(f"[ml] lattice load failed for {sp_path.name}: {e}")
                sim_lattices[sp] = (None, None)
        else:
            sim_lattices[sp] = (None, None)

        # KMC fire count from step log
        pykmc_path = sp_path / 'pykmc.out'
        if pykmc_path.exists():
            try:
                df_steps = parse_pykmc_out(pykmc_path)
                if len(df_steps) > 0:
                    sim_firing[sp] = Counter(df_steps['ref_event'].astype(int).tolist())
                else:
                    sim_firing[sp] = Counter()
            except Exception as e:
                print(f"[ml] firing-log parse failed for {sp_path.name}: {e}")
                sim_firing[sp] = Counter()
        else:
            sim_firing[sp] = Counter()

    print(f"[ml] cached {len(sim_cells)} sims; "
          f"{sum(1 for v in sim_cells.values() if v is not None)} with valid cell, "
          f"{sum(1 for v in sim_firing.values() if v)} with non-empty firing log"
          + (f", {sum(1 for v in sim_lattices.values() if v[0] is not None)} with lattice"
             if COMPUTE_LATTICE_ADJACENCY else ""))

    # ── Per-event loop: compute ML descriptors + Stage 1h fingerprint ──
    ml_rows = []
    n_fp_complete = 0
    for idx, row in sub.iterrows():
        sp = row['sim_path']
        cell = sim_cells.get(sp)
        ideal, adj = sim_lattices.get(sp, (None, None))
        fire = sim_firing.get(sp, Counter())

        features = compute_ml_descriptors(
            row, cell=cell, ideal_sites=ideal, nn_adjacency=adj
        )
        features['master_idx'] = idx
        features['kmc_fire_count'] = int(fire.get(int(row['idx_ref']), 0))

        # Stage 1h: lattice-environment fingerprint (cluster-based, no global lattice)
        fp = compute_env_fingerprint(row)
        features.update(fp)
        if fp['env_shell_complete']:
            n_fp_complete += 1

        ml_rows.append(features)

    ml_df = pd.DataFrame(ml_rows).set_index('master_idx')
    print(f"[fp] complete shells: {n_fp_complete}/{len(sub)} "
          f"({100*n_fp_complete/len(sub):.1f}%)")

    # ── Reverse-barrier self-join on (sim_path, idx_backward → idx_ref) ──
    # idx_backward is per-sim (not global master index), so key on both.
    rev_lookup = sub[['sim_path', 'idx_ref', 'energy_barrier']].rename(
        columns={'idx_ref': 'idx_backward', 'energy_barrier': 'reverse_barrier'}
    )
    pairs = sub[['sim_path', 'idx_backward']].merge(
        rev_lookup, on=['sim_path', 'idx_backward'], how='left'
    )
    ml_df['reverse_barrier'] = pairs['reverse_barrier'].values
    ml_df['Ea_gap'] = sub['energy_barrier'].values - ml_df['reverse_barrier'].values
    n_paired = int(ml_df['reverse_barrier'].notna().sum())
    print(f"[ml] reverse-barrier paired {n_paired}/{len(ml_df)} events "
          f"({100*n_paired/len(ml_df):.1f}%)")

    # ── Merge into sub ──
    sub = sub.join(ml_df, rsuffix='_ml')
    print(f"[ml] added {ml_df.shape[1]} ML columns to sub ({len(sub)} rows)")
