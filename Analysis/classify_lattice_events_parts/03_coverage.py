# %%
# Cell 1: Coverage Heatmap

# Build coverage matrix
nvac_vals = sorted(master['nvac'].unique())
temp_vals = sorted(master['temp'].unique())

n_total_grid = np.full((len(nvac_vals), len(temp_vals)), np.nan)
n_sub_grid = np.full((len(nvac_vals), len(temp_vals)), np.nan)

for i, nv in enumerate(nvac_vals):
    for j, t in enumerate(temp_vals):
        mask = (master['nvac'] == nv) & (master['temp'] == t)
        if mask.any():
            n_total_grid[i, j] = mask.sum()
            n_sub_grid[i, j] = (master.loc[mask, 'energy_barrier'] < BARRIER_CUTOFF).sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, grid, title in [
    (axes[0], n_total_grid, 'Total events in catalogue'),
    (axes[1], n_sub_grid, f'Events under {BARRIER_CUTOFF} eV'),
]:
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(temp_vals)))
    ax.set_xticklabels([f'T{t}' for t in temp_vals], rotation=45)
    ax.set_yticks(range(len(nvac_vals)))
    ax.set_yticklabels([f'{n}vac' for n in nvac_vals])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Vacancy count')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='N events')

    # Annotate cells
    for i in range(len(nvac_vals)):
        for j in range(len(temp_vals)):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='grey')
            else:
                ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=8,
                        color='white' if val > grid[~np.isnan(grid)].max() * 0.6 else 'black')

plt.suptitle('NiAlH_jea.eam — Ni(100) Event Catalogue Coverage', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'coverage_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Report missing cells
all_temps = [300, 400, 500, 600, 700, 800]
all_nvacs = list(range(1, 11))
print('\nMissing simulations (T300-T800 grid):')
for nv in all_nvacs:
    missing = [t for t in all_temps
               if not ((master['nvac'] == nv) & (master['temp'] == t)).any()]
    if missing:
        print(f'  {nv}vac: {[", ".join(f"T{t}" for t in missing)]}')
