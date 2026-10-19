# %%
# Cell 7: Forward/Backward Pair Analysis
# Interactive: toggle scatter / type-mismatch panels, layers (diagonal
# reference, coloured by type), and colour scheme.

pair_rows = []
for sim_label in sub['sim_label'].unique():
    sim_events = sub[sub['sim_label'] == sim_label]
    for idx, row in sim_events.iterrows():
        bw_idx = row.get('idx_backward', -1)
        if bw_idx < 0:
            continue
        bw_match = sim_events[sim_events['idx_ref'] == bw_idx]
        if len(bw_match) == 0:
            continue
        bw_row = bw_match.iloc[0]
        pair_rows.append({
            'sim_label': sim_label,
            'nvac': row['nvac'],
            'temp': row['temp'],
            'fwd_idx': row['idx_ref'],
            'bwd_idx': bw_idx,
            'fwd_Ea': row['energy_barrier'],
            'bwd_Ea': bw_row['energy_barrier'],
            'fwd_type': row.get('move_type', 'unknown'),
            'bwd_type': bw_row.get('move_type', 'unknown'),
            'fwd_type_pre_zlayer': row.get('move_type_pre_zlayer', 'unknown'),
            'bwd_type_pre_zlayer': bw_row.get('move_type_pre_zlayer', 'unknown'),
            'fwd_reclass': bool(row.get('zlayer_reclassified', False)),
            'bwd_reclass': bool(bw_row.get('zlayer_reclassified', False)),
            'dE': row['energy_barrier'] - bw_row['energy_barrier'],
        })

pairs_df = pd.DataFrame(pair_rows)
print(f'Forward/backward pairs found: {len(pairs_df)}')


def _plot_pairs(pairs_df, cmap_name, colour_by_type, show_diagonal,
                show_scatter, show_mismatch, show_zlayer=True, save=False):
    if len(pairs_df) == 0:
        return
    mts = sorted(pairs_df['fwd_type'].unique())
    palette = _palette(cmap_name, mts) or colors

    n_panels = int(show_scatter) + int(show_mismatch) + int(show_zlayer)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             squeeze=False)
    ax_iter = iter(axes[0])

    if show_scatter:
        ax = next(ax_iter)
        if colour_by_type:
            for mt in mts:
                mask = pairs_df['fwd_type'] == mt
                ax.scatter(pairs_df.loc[mask, 'fwd_Ea'],
                           pairs_df.loc[mask, 'bwd_Ea'],
                           color=palette.get(mt, '#999'), alpha=0.4, s=14,
                           label=f'{mt} ({mask.sum()})', edgecolors='none')
            ax.legend(fontsize=7, loc='upper left')
        else:
            ax.scatter(pairs_df['fwd_Ea'], pairs_df['bwd_Ea'],
                       alpha=0.3, s=10, c='steelblue')
        if show_diagonal:
            lim = max(pairs_df['fwd_Ea'].max(), pairs_df['bwd_Ea'].max()) + 0.1
            ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='symmetric')
        ax.set_xlabel('Forward Ea (eV)')
        ax.set_ylabel('Backward Ea (eV)')
        ax.set_title('Forward vs backward barriers')

    if show_mismatch:
        ax = next(ax_iter)
        same_type = (pairs_df['fwd_type'] == pairs_df['bwd_type']).sum()
        diff_type = len(pairs_df) - same_type
        ax.bar(['Same type', 'Different type'], [same_type, diff_type],
               color=['#4DAF4A', '#E41A1C'])
        ax.set_ylabel('Count')
        ax.set_title('Forward/backward type consistency (post z-layer)')
        ax.text(0, same_type + 5, f'{same_type}', ha='center', fontsize=12)
        ax.text(1, diff_type + 5, f'{diff_type}', ha='center', fontsize=12)

    if show_zlayer and 'fwd_reclass' in pairs_df.columns:
        ax = next(ax_iter)
        # Four mutually exclusive buckets describing whether the forward
        # and backward halves of a pair were touched by the z-layer pass.
        f, b = pairs_df['fwd_reclass'], pairs_df['bwd_reclass']
        n_neither = (~f & ~b).sum()
        n_both = (f & b).sum()
        n_fwd_only = (f & ~b).sum()
        n_bwd_only = (~f & b).sum()
        cats = ['Neither\nreclassified', 'Both\nreclassified',
                'Fwd only', 'Bwd only']
        vals = [int(n_neither), int(n_both), int(n_fwd_only), int(n_bwd_only)]
        bar_colors = ['#4DAF4A', '#6A3D9A', '#FFA500', '#FF7F00']
        ax.bar(cats, vals, color=bar_colors)
        ax.set_ylabel('Pair count')
        ax.set_title('Z-layer pass effect on pairs')
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, v + max(vals) * 0.01, f'{v}', ha='center', fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / 'pair_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if len(pairs_df) > 0:
    # Canonical static save (includes z-layer audit panel)
    _plot_pairs(pairs_df, 'custom', False, True, True, True,
                show_zlayer=True, save=True)

    # Report asymmetric pairs (unchanged)
    asym = pairs_df[pairs_df['fwd_type'] != pairs_df['bwd_type']]
    if len(asym) > 0:
        print(f'\nAsymmetric type pairs ({len(asym)} total):')
        type_combos = asym.groupby(['fwd_type', 'bwd_type']).size().sort_values(ascending=False)
        for (ft, bt), n in type_combos.items():
            print(f'  {ft} ↔ {bt}: {n}')

    # Report z-layer-split pairs (fwd and bwd on different sides of the
    # surface layer boundary — potential classifier edge cases).
    if 'fwd_reclass' in pairs_df.columns:
        split = pairs_df[pairs_df['fwd_reclass'] != pairs_df['bwd_reclass']]
        if len(split) > 0:
            print(f'\nZ-layer-split pairs ({len(split)} pairs where only '
                  f'one side was reclassified):')
            flip_combos = (split.groupby(['fwd_type_pre_zlayer', 'fwd_type',
                                          'bwd_type_pre_zlayer', 'bwd_type'])
                                .size().sort_values(ascending=False).head(10))
            for (fpre, fpost, bpre, bpost), n in flip_combos.items():
                print(f'  fwd {fpre}->{fpost}  ↔  bwd {bpre}->{bpost}: {n}')

    # Interactive controller
    _pair_cmap = widgets.Dropdown(options=CMAP_CHOICES, value='custom',
                                  description='Colours:')
    _pair_by_type = widgets.Checkbox(value=False, description='Colour scatter by type')
    _pair_diag = widgets.Checkbox(value=True, description='Show y=x diagonal')
    _pair_scat = widgets.Checkbox(value=True, description='Show scatter panel')
    _pair_mism = widgets.Checkbox(value=True, description='Show mismatch panel')
    _pair_zlayer = widgets.Checkbox(value=True, description='Show z-layer audit panel')

    def _go_pairs(**kw):
        _plot_pairs(pairs_df, kw['cmap'], kw['by_type'], kw['diag'],
                    kw['scat'], kw['mism'], show_zlayer=kw['zlayer'])

    _pair_controls = {'cmap': _pair_cmap, 'by_type': _pair_by_type,
                      'diag': _pair_diag, 'scat': _pair_scat,
                      'mism': _pair_mism, 'zlayer': _pair_zlayer}
    _pair_ui = widgets.VBox([
        widgets.HTML('<b>Interactive view</b> — toggle panels, colour scheme:'),
        widgets.HBox([_pair_cmap, _pair_by_type, _pair_diag]),
        widgets.HBox([_pair_scat, _pair_mism, _pair_zlayer]),
    ])
    display(_pair_ui, widgets.interactive_output(_go_pairs, _pair_controls))
