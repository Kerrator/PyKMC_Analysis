# %%
# Cell 11: Export

# Derive composition from sim_path ({composition}/{N}vac/T{temp}/{variant},
# parsed by parse_run_path) so the event_viewer audit UI can filter across
# compositions even though one notebook run targets a single COMPOSITION.
# sim_relpath is sim_path relative to the data root, so the CSV still finds
# its runs on another machine (Analysis._paths.resolve_sim_path joins it to
# that machine's data root).
if 'sim_path' in sub.columns:
    sub = sub.copy()
    if 'composition' not in sub.columns:
        def _composition(p):
            parsed = parse_run_path(Path(p)) if p else None
            return parsed[0] if parsed else COMPOSITION
        sub['composition'] = sub['sim_path'].map(_composition)

    def _relpath(p):
        try:
            return Path(p).relative_to(DATA_ROOT.parent).as_posix()
        except (TypeError, ValueError):
            return ''
    sub['sim_relpath'] = sub['sim_path'].map(_relpath)

# 1. Classified events CSV (original + ML descriptors)
export_cols = [
    'composition', 'nvac', 'temp', 'sim_label', 'sim_path', 'sim_relpath',
    'idx_ref', 'energy_barrier', 'k',
    'move_type', 'move_type_base', 'move_type_pre_zlayer',
    'surface_layer_initial', 'surface_layer_final', 'zlayer_reclassified',
    'move_dist', 'move_dz', 'move_dz_signed', 'move_dxy', 'n_moved',
    'layer_change', 'idx_backward', 'dra',
    # --- ML Tier 1: mover-only ---
    'disp_top1', 'disp_top2', 'disp_top3', 'disp_top4', 'disp_top5',
    'mover_depth_initial', 'mover_depth_final', 'mover_depth_delta',
    'Ea_over_kT', 'mover_species_ml',
    'nn1_count_Ni', 'nn1_count_Fe', 'nn1_count_Cr',
    'nn2_count_Ni', 'nn2_count_Fe', 'nn2_count_Cr',
    # --- ML Tier 2: neighborhood geometry ---
    'coord_mover_initial', 'coord_mover_final',
    'n_vac_nn1_initial', 'n_vac_nn1_final',
    'mean_nn1_bond_length_initial', 'dist_to_nearest_undercoord',
    'local_rmsd_5A',
    # --- ML Tier 3: saddle-path ---
    'path_straightness', 'coord_mover_saddle',
    'bonds_broken_init_to_saddle', 'bonds_formed_init_to_saddle',
    # --- ML Tier 4: metadata ---
    'reverse_barrier', 'Ea_gap', 'saddle_asymmetry_oob', 'kmc_fire_count',
    # --- ML Tier 5: lattice-environment fingerprints ---
    'env_fingerprint_occ', 'env_fingerprint_species',
    'hop_vector_canonical', 'env_shell_complete',
    # --- 3D pipeline labels ---
    'motif_family_3d', 'site_class_3d', 'direction_family_3d',
]

# Compute 3D labels post-hoc from already-classified rows. The authoritative
# direction_family_3d lives in rate_lookup_table_3d_master.csv (built by
# build_rate_table.py with site-graph access); here we use a coarse fallback
# based on layer_change + displacement magnitudes.
from Analysis.lattice_map_events import classify_3d_motif
if 'motif_family_3d' not in sub.columns:
    sub['motif_family_3d'] = sub.apply(
        lambda r: classify_3d_motif({
            'move_type': r['move_type'],
            'n_moved': int(r.get('n_moved', 0) or 0),
            'surface_layer_initial': int(r.get('surface_layer_initial', -1) or -1),
        }),
        axis=1,
    )
if 'site_class_3d' not in sub.columns:
    def _site_class(li):
        try:
            li_int = int(li)
        except (TypeError, ValueError):
            return 'bulk_like'
        if li_int == 0:
            return 'surface'
        if li_int == 1:
            return 'subsurface'
        return 'bulk_like'
    sub['site_class_3d'] = sub['surface_layer_initial'].map(_site_class)
if 'direction_family_3d' not in sub.columns:
    # Canonical Cartesian classifier (same rule used by the runtime rate-table
    # builder and by the pylatkmc edge labeller). `classify_direction_xyz` only
    # uses np.hypot(dx, dy) and |dz|, so passing the in-plane magnitude as dx
    # with dy=0.0 is equivalent to passing the full Cartesian vector.
    from Analysis.lattice_3d import classify_direction_xyz
    def _vec_dir(r):
        dxy = float(r.get('move_dxy', 0.0) or 0.0)
        dz = float(r.get('move_dz_signed', r.get('move_dz', 0.0)) or 0.0)
        return classify_direction_xyz(dxy, 0.0, dz, NN1_DIST)
    sub['direction_family_3d'] = sub.apply(_vec_dir, axis=1)

# Subset to columns actually present (backward-compatible for older runs)
export_cols_present = [c for c in export_cols if c in sub.columns]
export_df = sub[export_cols_present].copy()
export_df.to_csv(OUT_DIR / 'classified_events.csv', index=False)
print(f'Saved classified_events.csv ({len(export_df)} events, {len(export_cols_present)} cols)')

# 1b. Enhanced classified events CSV (includes saddle features + enhanced classification)
enhanced_cols = export_cols + [
    'saddle_disp_mag', 'saddle_dz', 'saddle_height_ratio',
    'saddle_n_displaced', 'saddle_asymmetry', 'saddle_max_pair_dz',
    'enhanced_type', 'reclassified', 'original_event_type',
    'likely_reverse_relaxation', 'adaptive_type',
]
enhanced_present = [c for c in enhanced_cols if c in sub.columns]
enhanced_export = sub[enhanced_present].copy()
enhanced_export.to_csv(OUT_DIR / 'classified_events_enhanced.csv', index=False)
print(f'Saved classified_events_enhanced.csv ({len(enhanced_export)} events, {len(enhanced_present)} cols)')

# 2. Parameter summary JSON
param_summary = {
    'lattice': {
        'type': 'fcc_100_surface',
        'a_lat_A': A_NI,
        'nn1_dist_A': float(NN1_DIST),
        'nn2_dist_A': float(NN2_DIST),
    },
    'barrier_cutoff_eV': BARRIER_CUTOFF,
    'n_simulations': len(sims),
    'n_events_total': len(master),
    'n_events_classified': len(sub),
    'events_per_type': params,
    'enhanced_reclassified': int(sub['reclassified'].sum()) if 'reclassified' in sub.columns else 0,
    'zlayer_reclassified': int(sub['zlayer_reclassified'].sum()) if 'zlayer_reclassified' in sub.columns else 0,
}

# Add kinetic coverage
if len(cov_df) > 0:
    param_summary['kinetic_coverage'] = {
        'mean_pct_steps_lattice': float(cov_df['pct_steps_lattice'].mean()),
        'mean_pct_time_lattice': float(cov_df['pct_time_lattice'].mean()),
    }

with open(OUT_DIR / 'lattice_parameters.json', 'w') as f:
    json.dump(param_summary, f, indent=2, default=str)
print(f'Saved lattice_parameters.json')

# 3. Coverage CSV
if len(cov_df) > 0:
    cov_df.to_csv(OUT_DIR / 'kinetic_coverage.csv', index=False)
    print(f'Saved kinetic_coverage.csv')

print(f'\nAll outputs saved to {OUT_DIR}/')
print('Files:')
for p in sorted(OUT_DIR.iterdir()):
    print(f'  {p.name} ({p.stat().st_size / 1024:.1f} KB)')
