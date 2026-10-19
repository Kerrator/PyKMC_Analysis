# %%
# Cell 2c: GMM Adaptive Threshold Comparison
#
# Fits a Gaussian Mixture Model to displacement distributions to derive
# adaptive thresholds, and compares them to the hardcoded values.
#
# NOTE: this cell runs BEFORE the z-layer pass (cell 07); `move_type`
# still carries the pre-pass labels. Reclassification tallies here
# refer to GMM-threshold vs hard-coded threshold changes, not the
# z-layer pass.

# Compute GMM thresholds from the classified events
displacements = sub['move_dist'].values
dz_values = sub['move_dz'].values

thresholds_gmm = compute_thresholds_gmm(displacements, dz_values, lattice_constant=A_NI)

print('GMM-derived thresholds vs hardcoded:')
print(f'  1NN centre:  {thresholds_gmm["nn1_center"]:.4f} A  (hardcoded: {NN1_DIST:.4f} A)')
print(f'  2NN centre:  {thresholds_gmm["nn2_center"]:.4f} A  (hardcoded: {NN2_DIST:.4f} A)')
print(f'  1NN range:   [{thresholds_gmm["nn1_range"][0]:.3f}, {thresholds_gmm["nn1_range"][1]:.3f}] A  '
      f'(hardcoded: [{NN1_DIST - DIST_TOLERANCE:.3f}, {NN1_DIST + DIST_TOLERANCE:.3f}] A)')
print(f'  2NN range:   [{thresholds_gmm["nn2_range"][0]:.3f}, {thresholds_gmm["nn2_range"][1]:.3f}] A  '
      f'(hardcoded: [{NN2_DIST - DIST_TOLERANCE:.3f}, {NN2_DIST + DIST_TOLERANCE:.3f}] A)')
print(f'  Layer change: {thresholds_gmm["layer_change_threshold"]:.3f} A  (hardcoded: 1.000 A)')
print(f'  GMM confidence (Bhattacharyya): {thresholds_gmm["confidence"]:.2f}')

# Plot GMM components overlaid on displacement histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Displacement GMM
ax = axes[0]
inplane = sub[sub['move_dz'].abs() < 1.0]
d_vals = inplane.loc[inplane['move_dist'] > MOVE_THRESHOLD, 'move_dist'].values
ax.hist(d_vals, bins=80, density=True, alpha=0.5, color='steelblue', label='data')
x = np.linspace(0, 5, 500)
gmm = thresholds_gmm['gmm_displacement']
for i, (w, m, v) in enumerate(zip(gmm.weights_, gmm.means_.ravel(),
                                    gmm.covariances_.ravel())):
    ax.plot(x, w * np.exp(-0.5 * (x - m)**2 / v) / np.sqrt(2 * np.pi * v),
            '--', lw=2, label=f'Component {i+1}: \u03bc={m:.3f}')
ax.axvline(NN1_DIST, color='green', ls=':', alpha=0.8, label=f'1NN={NN1_DIST:.3f}')
ax.axvline(NN2_DIST, color='red', ls=':', alpha=0.8, label=f'2NN={NN2_DIST:.3f}')
ax.set_xlabel('Displacement (\u00c5)')
ax.set_ylabel('Density')
ax.set_title('Displacement GMM (in-plane events)')
ax.legend(fontsize=7)

# dz GMM
ax = axes[1]
abs_dz = sub['move_dz'].abs().values
abs_dz_plot = abs_dz[abs_dz > 0.01]
ax.hist(abs_dz_plot, bins=80, density=True, alpha=0.5, color='steelblue', label='data')
gmm_dz = thresholds_gmm['gmm_dz']
for i, (w, m, v) in enumerate(zip(gmm_dz.weights_, gmm_dz.means_.ravel(),
                                    gmm_dz.covariances_.ravel())):
    ax.plot(x, w * np.exp(-0.5 * (x - m)**2 / v) / np.sqrt(2 * np.pi * v),
            '--', lw=2, label=f'Component {i+1}: \u03bc={m:.3f}')
ax.axvline(thresholds_gmm['layer_change_threshold'], color='red', ls='--',
           label=f'layer thr={thresholds_gmm["layer_change_threshold"]:.3f}')
ax.axvline(1.0, color='green', ls=':', label='hardcoded thr=1.0')
ax.set_xlabel('|dz| (\u00c5)')
ax.set_ylabel('Density')
ax.set_title('|dz| GMM (layer change detection)')
ax.legend(fontsize=7)
ax.set_xlim(0, 3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'gmm_threshold_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Reclassify with adaptive thresholds and report differences
adaptive_types = []
for idx, row in sub.iterrows():
    cls = classify_event_adaptive(row, thresholds_gmm)
    adaptive_types.append(cls['event_type'])

sub['adaptive_type'] = adaptive_types
changed = sub['move_type'] != sub['adaptive_type']
print(f'\nAdaptive reclassification: {changed.sum()} / {len(sub)} events changed ({100*changed.mean():.2f}%)')
if changed.any():
    print('Changes by type:')
    for old_t in sub.loc[changed, 'move_type'].unique():
        new_types = sub.loc[changed & (sub['move_type'] == old_t), 'adaptive_type'].value_counts()
        for new_t, n in new_types.items():
            print(f'  {old_t} -> {new_t}: {n}')

# Save adaptive thresholds
import json as json_mod
thr_export = {k: v for k, v in thresholds_gmm.items()
              if k not in ('gmm_displacement', 'gmm_dz')}
# Convert tuples to lists for JSON
thr_export['nn1_range'] = list(thr_export['nn1_range'])
thr_export['nn2_range'] = list(thr_export['nn2_range'])
with open(OUT_DIR / 'adaptive_thresholds.json', 'w') as f:
    json_mod.dump(thr_export, f, indent=2)
print(f'Saved adaptive_thresholds.json')
