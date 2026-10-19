# %%
# Cell 4: Barrier Distributions by Type
# Interactive: toggle move types, switch colour scheme, change bin count, log-y.

_BARRIER_MTS_SPLIT = ['1NN_hop', '1NN_hop_cooperative', '2NN_hop',
                      'exchange_up', 'exchange_down', 'exchange_lateral',
                      'interlayer_hop_up', 'interlayer_hop_down',
                      'inplane_complex', 'complex',
                      'subsurface_migration', 'subsurface_hop',
                      'subsurface_1NN_hop', 'subsurface_other']
_BARRIER_MTS_BASE = ['1NN_hop', '1NN_hop_cooperative', '2NN_hop',
                     'exchange', 'exchange_lateral',
                     'interlayer_hop', 'inplane_complex', 'complex',
                     'subsurface_migration', 'subsurface_hop',
                     'subsurface_1NN_hop', 'subsurface_other']
# Back-compat alias for any code that still references _BARRIER_MTS
_BARRIER_MTS = _BARRIER_MTS_BASE


def _plot_barriers(sub, active_types, cmap_name, bins, log_y,
                   show_boxplot, merge_updown=True, save=False):
    type_col = 'move_type_base' if merge_updown else 'move_type'
    type_order = _BARRIER_MTS_BASE if merge_updown else _BARRIER_MTS_SPLIT
    present = [mt for mt in type_order if (sub[type_col] == mt).any()]
    palette = _palette(cmap_name, type_order) or colors
    mts = [mt for mt in present if mt in active_types]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Box plot by type
    ax = axes[0]
    if show_boxplot and mts:
        data_by_type = [sub.loc[sub[type_col] == mt, 'energy_barrier'].values
                        for mt in mts]
        bp = ax.boxplot(data_by_type, tick_labels=mts, patch_artist=True,
                        showfliers=True, whis=[5, 95])
        for patch, mt in zip(bp['boxes'], mts):
            patch.set_facecolor(palette.get(mt, '#999'))
            patch.set_alpha(0.7)
    ax.set_ylabel('Barrier Ea (eV)')
    ax.set_title('Barrier distribution by move type')
    ax.tick_params(axis='x', rotation=30)

    # Histogram overlay
    ax = axes[1]
    for mt in mts:
        vals = sub.loc[sub[type_col] == mt, 'energy_barrier']
        ax.hist(vals, bins=bins, alpha=0.5, color=palette.get(mt, '#999'),
                label=f'{mt} (n={len(vals)})', density=True)
    ax.set_xlabel('Barrier Ea (eV)')
    ax.set_ylabel('Density')
    if log_y:
        ax.set_yscale('log')
    ax.set_title('Barrier density by type')
    ax.legend(fontsize=8)

    # Box plot by nvac for 1NN hops (independent of merge toggle — 1NN_hop
    # has no up/down split)
    ax = axes[2]
    nn1_data = sub[sub['move_type'] == '1NN_hop']
    if len(nn1_data) > 0 and '1NN_hop' in active_types:
        nvac_groups = sorted(nn1_data['nvac'].unique())
        data_by_nvac = [nn1_data.loc[nn1_data['nvac'] == nv, 'energy_barrier'].values
                        for nv in nvac_groups]
        bp2 = ax.boxplot(data_by_nvac,
                         tick_labels=[f'{nv}vac' for nv in nvac_groups],
                         patch_artist=True, showfliers=True, whis=[5, 95])
        for patch in bp2['boxes']:
            patch.set_facecolor(palette.get('1NN_hop', '#4DAF4A'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Barrier Ea (eV)')
        ax.set_title('1NN hop barriers by vacancy count')

    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / 'barrier_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


# Canonical static save (merged up/down view — matches what we now
# consider the physically meaningful grouping)
_plot_barriers(sub, set(_BARRIER_MTS_BASE), 'custom', 40, False, True,
               merge_updown=True, save=True)

# Summary table
print('\nBarrier summary by move type (merged up/down):')
print(sub.groupby('move_type_base')['energy_barrier'].describe().round(4).to_string())

# Interactive controller
_bar_split_toggles = {mt: widgets.Checkbox(value=True, description=mt, indent=False,
                                           layout=widgets.Layout(width='170px'))
                      for mt in _BARRIER_MTS_SPLIT}
_bar_base_toggles = {mt: widgets.Checkbox(value=True, description=mt, indent=False,
                                          layout=widgets.Layout(width='170px'))
                     for mt in _BARRIER_MTS_BASE}
_bar_cmap = widgets.Dropdown(options=CMAP_CHOICES, value='custom', description='Colours:')
_bar_bins = widgets.IntSlider(value=40, min=10, max=120, step=5, description='Bins:')
_bar_logy = widgets.Checkbox(value=False, description='log y (hist)')
_bar_box = widgets.Checkbox(value=True, description='Show box plot')
_bar_merge = widgets.Checkbox(value=True, description='Merge up/down')


def _go_bar(**kw):
    merge = kw['merge']
    if merge:
        active = {mt for mt in _BARRIER_MTS_BASE if kw['b_' + mt]}
    else:
        active = {mt for mt in _BARRIER_MTS_SPLIT if kw['s_' + mt]}
    _plot_barriers(sub, active, kw['cmap'], kw['bins'], kw['logy'], kw['box'],
                   merge_updown=merge)


_bar_controls = {'cmap': _bar_cmap, 'bins': _bar_bins, 'logy': _bar_logy,
                 'box': _bar_box, 'merge': _bar_merge,
                 **{'s_' + mt: w for mt, w in _bar_split_toggles.items()},
                 **{'b_' + mt: w for mt, w in _bar_base_toggles.items()}}
_bar_ui = widgets.VBox([
    widgets.HTML('<b>Interactive view</b> — toggle types, colour scheme, bins:'),
    widgets.HBox([_bar_cmap, _bar_bins, _bar_logy, _bar_box, _bar_merge]),
    widgets.HTML('<i>base types (Merge on):</i>'),
    widgets.HBox(list(_bar_base_toggles.values())),
    widgets.HTML('<i>full split types (Merge off):</i>'),
    widgets.HBox(list(_bar_split_toggles.values())),
])
display(_bar_ui, widgets.interactive_output(_go_bar, _bar_controls))
