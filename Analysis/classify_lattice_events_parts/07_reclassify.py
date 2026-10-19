# %%
# Cell 2d: Enhanced Classification (Saddle-Assisted)
#
# Uses classify_event_enhanced to resolve ambiguous boundary events
# using saddle-point geometry.

print('Running enhanced classification (saddle-assisted)...')
enhanced_results = []
for idx, row in sub.iterrows():
    cls = classify_event_enhanced(row, thresholds=thresholds_gmm)
    cls['master_idx'] = idx
    enhanced_results.append(cls)

enh_df = pd.DataFrame(enhanced_results).set_index('master_idx')
sub['enhanced_type'] = enh_df['event_type']
sub['reclassified'] = enh_df['reclassified']
sub['original_event_type'] = enh_df['original_event_type']
sub['likely_reverse_relaxation'] = enh_df['likely_reverse_relaxation']

n_reclass = sub['reclassified'].sum()
print(f'\nEnhanced reclassification: {n_reclass} / {len(sub)} events ({100*n_reclass/len(sub):.2f}%)')

if n_reclass > 0:
    print('\nReclassification summary:')
    reclass = sub[sub['reclassified']]
    for _, row in reclass.groupby(['original_event_type', 'enhanced_type']).size().reset_index(name='count').iterrows():
        print(f'  {row["original_event_type"]} -> {row["enhanced_type"]}: {row["count"]}')

    # Barrier distributions of reclassified events
    print(f'\nReclassified event barriers:')
    print(f'  Mean Ea: {reclass["energy_barrier"].mean():.3f} eV')
    print(f'  Std Ea:  {reclass["energy_barrier"].std():.3f} eV')
    print(f'  Range:   [{reclass["energy_barrier"].min():.3f}, {reclass["energy_barrier"].max():.3f}] eV')

    # Displacement scatter with reclassified events highlighted
    fig, ax = plt.subplots(figsize=(8, 6))
    not_reclass = sub[~sub['reclassified']]
    ax.scatter(not_reclass['move_dist'], not_reclass['move_dz'],
               alpha=0.2, s=5, c='grey', label='unchanged')
    ax.scatter(reclass['move_dist'], reclass['move_dz'],
               alpha=0.8, s=20, c='red', marker='x', label=f'reclassified (n={n_reclass})')
    ax.axhline(1.0, color='blue', ls='--', alpha=0.5, label='layer_change=1.0 A')
    ax.axvline(NN1_DIST, color='green', ls=':', alpha=0.5, label=f'1NN={NN1_DIST:.2f}')
    ax.axvline(NN2_DIST, color='orange', ls=':', alpha=0.5, label=f'2NN={NN2_DIST:.2f}')
    ax.set_xlabel('|displacement| (A)')
    ax.set_ylabel('|dz| (A)')
    ax.set_title('Enhanced classification: reclassified events near boundaries')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'enhanced_reclassification.png', dpi=150, bbox_inches='tight')
    plt.show()

# Report reverse-relaxation events
n_rev_relax = sub['likely_reverse_relaxation'].sum()
if n_rev_relax > 0:
    print(f'\nNear-zero barrier interlayer_hop_down -> relaxation: {n_rev_relax} events')


# ── Z-layer pass ───────────────────────────────────────────────────────────
# Relabel events whose primary moving atom sits below the top surface layer.
# This catches subsurface vacancy migrations that displacement-only rules
# misclassify as exchange_up/down (mover sits in layer 1+ but has big |dz|).

print('\nRunning z-layer reclassification...')
_layer_pairs = [
    get_event_surface_layers(row['initial_positions'], row['final_positions'],
                             int(row['move_atom_idx']))
    for _, row in sub.iterrows()
]
sub['surface_layer_initial'] = [p[0] for p in _layer_pairs]
sub['surface_layer_final'] = [p[1] for p in _layer_pairs]
# Keep `surface_layer` as an alias of the initial layer for convenience /
# backward compatibility.
sub['surface_layer'] = sub['surface_layer_initial']
sub['move_type_pre_zlayer'] = sub['move_type']
sub['move_type'] = [
    reclassify_by_zlayer(mt, li, lf)
    for mt, li, lf in zip(sub['move_type'],
                          sub['surface_layer_initial'],
                          sub['surface_layer_final'])
]
sub['zlayer_reclassified'] = sub['move_type'] != sub['move_type_pre_zlayer']
# Refresh the directionless alias introduced in 04_load_events so downstream
# cells and the interactive widgets immediately pick up subsurface labels.
sub['move_type_base'] = sub['move_type'].map(_base_move_type)

n_zre = int(sub['zlayer_reclassified'].sum())
print(f'Z-layer reclassification: {n_zre}/{len(sub)} events '
      f'({100*n_zre/len(sub):.3f}%)')
if n_zre:
    print('  by (from -> to):')
    grp = (sub[sub['zlayer_reclassified']]
             .groupby(['move_type_pre_zlayer', 'move_type'])
             .size()
             .sort_values(ascending=False))
    for (old, new), n in grp.items():
        print(f'    {old:22s} -> {new:24s} n={n}')

print('\nEvents by (layer_initial, layer_final)  [0 = top surface]:')
print(sub.groupby(['surface_layer_initial', 'surface_layer_final'])
         .size().sort_index().to_string())

# ── Floating-atom reclassification (post z-layer) ─────────────────────────
# interlayer_hop_down with |d| > 3 Å are atoms that detached from the surface
# entirely — pARTn found the saddle but the event is not a physical hop.
# These have zero-barrier return paths and are reclassified as relaxation.
_float_mask = (
    (sub['move_type'] == 'interlayer_hop_down')
    & (sub['move_dist'] > 3.0)
)
_n_float = int(_float_mask.sum())
if _n_float > 0:
    sub.loc[_float_mask, 'move_type'] = 'relaxation'
    sub.loc[_float_mask, 'move_type_base'] = 'relaxation'
    sub.loc[_float_mask, 'zlayer_reclassified'] = True
    print(f'\nFloating-atom reclassification: {_n_float} interlayer_hop_down '
          f'with |d|>3 A -> relaxation')
else:
    print(f'\nNo floating-atom events found (interlayer_hop_down |d|>3 A)')
