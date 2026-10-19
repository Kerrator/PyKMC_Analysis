# %%
# Cell 2b: Saddle Feature Distributions
#
# Visualises saddle-point geometry features to validate the
# hop-vs-exchange discriminator (saddle_height_ratio).
#
# NOTE: this cell runs BEFORE the z-layer pass (cell 07), so
# `move_type` here still carries the pre-pass displacement-based
# labels (exchange_up/down, etc.). Post-pass subsurface labels only
# appear from cell 07 onward.

saddle_cols = ['saddle_disp_mag', 'saddle_dz', 'saddle_height_ratio',
               'saddle_n_displaced', 'saddle_asymmetry', 'saddle_max_pair_dz']

has_saddle = sub['saddle_height_ratio'].notna()
print(f'Events with saddle features: {has_saddle.sum()} / {len(sub)}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Box plot: saddle_height_ratio by move_type
ax = axes[0]
types_for_box = [t for t in ALL_TYPES
                 if t in sub.loc[has_saddle, 'move_type'].unique()]
data = [sub.loc[has_saddle & (sub['move_type'] == t), 'saddle_height_ratio'].dropna().values
        for t in types_for_box]
data = [d for d, t in zip(data, types_for_box) if len(d) > 0]
types_for_box = [t for t, d_orig in zip(types_for_box,
    [sub.loc[has_saddle & (sub['move_type'] == t), 'saddle_height_ratio'].dropna().values
     for t in types_for_box]) if len(d_orig) > 0]
if data:
    ax.boxplot(data, tick_labels=[t.replace('_', '\n') for t in types_for_box],
               patch_artist=True, showfliers=False, whis=[5, 95])
    ax.axhline(1.15, color='red', ls='--', alpha=0.7, label='exchange threshold (1.15)')
    ax.set_ylabel('saddle_height_ratio')
    ax.set_title('Saddle height ratio by event type')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

# 2. Scatter: saddle_height_ratio vs saddle_max_pair_dz, colored by move_type
ax = axes[1]
for t in types_for_box:
    mask = has_saddle & (sub['move_type'] == t)
    ax.scatter(sub.loc[mask, 'saddle_max_pair_dz'],
               sub.loc[mask, 'saddle_height_ratio'],
               label=t, alpha=0.4, s=10)
ax.axhline(1.15, color='red', ls='--', alpha=0.7)
ax.set_xlabel('saddle_max_pair_dz (\u00c5)')
ax.set_ylabel('saddle_height_ratio')
ax.set_title('Saddle geometry: height ratio vs max pair dz')
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(OUT_DIR / 'saddle_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Export saddle feature distributions
saddle_export = sub.loc[has_saddle, ['sim_label', 'idx_ref', 'move_type',
                                     'energy_barrier'] + saddle_cols].copy()
saddle_export.to_csv(OUT_DIR / 'saddle_feature_distributions.csv', index=False)
print(f'Saved saddle_feature_distributions.csv ({len(saddle_export)} events)')
