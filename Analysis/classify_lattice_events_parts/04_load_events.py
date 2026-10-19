# %%
# Cell 2: Classification Engine
#
# Uses classify_event_from_row from lattice_map_events.py — shared with
# the per-simulation script so classification is consistent.
# Now includes saddle-point features for downstream analysis.

# Apply classification to all sub-cutoff events
sub = master[master['energy_barrier'] < BARRIER_CUTOFF].copy()
print(f'Classifying {len(sub)} events under {BARRIER_CUTOFF} eV (with saddle features)...')

classifications = []
for idx, row in sub.iterrows():
    cls = classify_event_from_row(row, include_saddle_features=True)
    cls['master_idx'] = idx
    classifications.append(cls)

cls_df = pd.DataFrame(classifications).set_index('master_idx')

# Merge back
sub = sub.join(cls_df)

# Rename for consistency with rest of notebook
sub.rename(columns={'event_type': 'move_type'}, inplace=True)


def _base_move_type(mt: str) -> str:
    """Collapse forward/backward direction suffixes.

    Up and down variants of the same saddle are the forward/backward
    labels of a single event; for visualisation and rate aggregation the
    directionless base type is usually what you want. Direction is
    preserved in ``move_dz_signed`` so the merge is lossless.
    """
    if mt.endswith('_up') or mt.endswith('_down'):
        return mt.rsplit('_', 1)[0]
    return mt


sub['move_type_base'] = sub['move_type'].map(_base_move_type)

print(f'\nClassification results:')
type_counts = sub['move_type'].value_counts()
for mtype, count in type_counts.items():
    pct = 100 * count / len(sub)
    ea_mean = sub.loc[sub['move_type'] == mtype, 'energy_barrier'].mean()
    ea_std = sub.loc[sub['move_type'] == mtype, 'energy_barrier'].std()
    on_lat = ' [ON-LATTICE]' if mtype in LATTICE_TYPES else ''
    print(f'  {mtype:20s}: {count:5d} ({pct:5.1f}%)  '
          f'Ea = {ea_mean:.3f} \u00b1 {ea_std:.3f} eV{on_lat}')

# Flag complex events for review
n_complex = (sub['move_type'] == TYPE_COMPLEX).sum()
if n_complex > 0:
    print(f'\n{n_complex} events classified as "{TYPE_COMPLEX}" \u2014 review needed:')
    for _, row in sub[sub['move_type'] == TYPE_COMPLEX].head(10).iterrows():
        print(f'  {row["sim_label"]} event {row.name}: '
              f'Ea={row["energy_barrier"]:.3f} eV, '
              f'd={row["move_dist"]:.3f} A, dz={row["move_dz"]:.3f} A, '
              f'n_moved={row["n_moved"]}')
