# %%
# Cell 10: Kinetic Coverage — per-simulation lattice step fraction
#
# For each simulation, compute what fraction of KMC steps are captured
# by on-lattice moves (1NN + 2NN hops).
#
# This cell uses the classifier path only: it reclassifies catalogue rows
# with classify_event_from_row() and then checks which selected ref_event
# values are typed as lattice moves. It does not read any event-level
# lattice projection export such as the deprecated mapped_1nn_events.csv.

coverage_rows = []

for sim in sims:
    sim_path = sim['path']
    pykmc_path = sim_path / 'pykmc.out'
    ref_path = sim_path / 'reference_table.pickle'

    if not pykmc_path.exists() or not ref_path.exists():
        continue

    df_steps = parse_pykmc_out(pykmc_path)
    if len(df_steps) == 0:
        continue

    catalog = pd.read_pickle(ref_path)

    # Reclassify each catalogue event with the shared displacement-based
    # classifier, then apply the z-layer pass so we mirror the semantics
    # of classify_lattice_events cell 07. We keep two accounting views:
    #   * "_any"     : geometry-only (pre z-layer) — old semantics, kept
    #                  for backward compatibility with downstream consumers
    #   * "_surface" : post z-layer — only true surface 1NN/2NN hops
    on_lattice_any = set()        # pre z-layer LATTICE_TYPES
    on_lattice_surface = set()    # post z-layer LATTICE_TYPES
    nn_hop_any = set()            # pre z-layer 1NN family
    nn_hop_surface = set()        # post z-layer 1NN family
    for i, row in catalog.iterrows():
        if row['energy_barrier'] >= BARRIER_CUTOFF:
            continue
        cls = classify_event_from_row(row)
        pre_t = cls['event_type']
        # z-layer relabel
        sl_i, sl_f = get_event_surface_layers(
            row['initial_positions'], row['final_positions'],
            int(row['move_atom_idx']))
        post_t = reclassify_by_zlayer(pre_t, sl_i, sl_f)
        if pre_t in LATTICE_TYPES:
            on_lattice_any.add(i)
        if post_t in LATTICE_TYPES:
            on_lattice_surface.add(i)
        if pre_t in {TYPE_1NN, TYPE_1NN_COOP}:
            nn_hop_any.add(i)
        if post_t in {TYPE_1NN, TYPE_1NN_COOP}:
            nn_hop_surface.add(i)

    total_steps = len(df_steps)
    total_time = df_steps['time'].iloc[-1] if len(df_steps) > 0 else 0

    def _pct(mask_set, col):
        if col == 'steps':
            n = df_steps['ref_event'].isin(mask_set).sum()
            return 100 * n / total_steps if total_steps > 0 else 0
        t = df_steps.loc[df_steps['ref_event'].isin(mask_set), 'dt'].sum()
        return 100 * t / total_time if total_time > 0 else 0

    coverage_rows.append({
        'nvac': sim['nvac'],
        'temp': sim['temp'],
        'total_steps': total_steps,
        'total_time': total_time,
        # Backward-compatible alias: _lattice = _lattice_any (pre z-layer)
        'pct_steps_lattice': _pct(on_lattice_any, 'steps'),
        'pct_steps_lattice_any': _pct(on_lattice_any, 'steps'),
        'pct_steps_lattice_surface': _pct(on_lattice_surface, 'steps'),
        'pct_steps_1nn': _pct(nn_hop_any, 'steps'),
        'pct_steps_1nn_any': _pct(nn_hop_any, 'steps'),
        'pct_steps_1nn_surface': _pct(nn_hop_surface, 'steps'),
        'pct_time_lattice': _pct(on_lattice_any, 'time'),
        'pct_time_lattice_any': _pct(on_lattice_any, 'time'),
        'pct_time_lattice_surface': _pct(on_lattice_surface, 'time'),
        'pct_time_1nn': _pct(nn_hop_any, 'time'),
        'pct_time_1nn_any': _pct(nn_hop_any, 'time'),
        'pct_time_1nn_surface': _pct(nn_hop_surface, 'time'),
    })

cov_df = pd.DataFrame(coverage_rows)
print('Kinetic coverage summary:')
print(cov_df.to_string(index=False, float_format='%.1f'))

if len(cov_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, (col_any, col_surface, ylabel, title) in enumerate([
        ('pct_steps_lattice_any', 'pct_steps_lattice_surface',
         'Steps captured (%)', 'Kinetic coverage by steps (1NN + 2NN)'),
        ('pct_time_lattice_any', 'pct_time_lattice_surface',
         'Sim time captured (%)', 'Kinetic coverage by sim time (1NN + 2NN)'),
    ]):
        for row_idx, (col, suffix) in enumerate([(col_any, 'any (pre z-layer)'),
                                                 (col_surface, 'surface (post z-layer)')]):
            ax = axes[row_idx][col_idx]
            for nv in sorted(cov_df['nvac'].unique()):
                nv_data = cov_df[cov_df['nvac'] == nv].sort_values('temp')
                ax.plot(nv_data['temp'], nv_data[col],
                        'o-', label=f'{nv}vac', markersize=6)
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} — {suffix}')
            ax.legend(fontsize=7, ncol=2)
            ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'kinetic_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()
