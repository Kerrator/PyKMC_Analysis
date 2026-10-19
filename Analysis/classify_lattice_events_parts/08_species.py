# %%
# Cell 2e: Species Breakdown
#
# For multi-element simulations: barrier distributions by mover species.
# For pure Ni, this confirms all movers are type 1 (Ni).

# Check if species info is available
has_types = False
for idx, row in sub.head(5).iterrows():
    if 'initial_types' in row.index and row['initial_types'] is not None:
        has_types = True
        break

if has_types:
    print('Extracting species information...')
    species_info = []
    for idx, row in sub.iterrows():
        sp = classify_event_from_row(row, include_species=True)
        species_info.append({
            'master_idx': idx,
            'mover_species': sp.get('mover_species'),
            'partner_species': sp.get('partner_species'),
        })
    sp_df = pd.DataFrame(species_info).set_index('master_idx')
    sub['mover_species'] = sp_df['mover_species']
    sub['partner_species'] = sp_df['partner_species']

    unique_species = sub['mover_species'].dropna().unique()
    print(f'Unique mover species: {unique_species}')
    print(f'\nSpecies breakdown by move type:')
    species_summary = sub.groupby(['move_type', 'mover_species']).agg(
        count=('energy_barrier', 'size'),
        Ea_mean=('energy_barrier', 'mean'),
        Ea_std=('energy_barrier', 'std'),
    ).reset_index()
    print(species_summary.to_string(index=False))

    # If multiple species, plot barrier distributions by species
    if len(unique_species) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        for sp in sorted(unique_species):
            sp_data = sub[sub['mover_species'] == sp]
            ax.hist(sp_data['energy_barrier'], bins=50, alpha=0.5, label=f'Species {sp}')
        ax.set_xlabel('Energy barrier (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Barrier distributions by mover species')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'species_barrier_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
else:
    print('No species information available in reference tables.')
    print('Species-dependent analysis skipped.')
