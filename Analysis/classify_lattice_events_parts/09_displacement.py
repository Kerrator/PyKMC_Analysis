# %%
# Cell 3: Displacement Scatter — validates classification boundaries
# Interactive: toggle each move-type on/off, change colour scheme,
# adjust point size. The static PNG (saved to disk) uses the defaults.

custom_colors = {
    TYPE_1NN: '#4DAF4A',
    TYPE_1NN_COOP: '#8DD35F',
    TYPE_1NN_TRIPLE: '#006400',
    TYPE_2NN: '#377EB8',
    TYPE_EXCHANGE_UP: '#E41A1C',
    TYPE_EXCHANGE_DOWN: '#FF6666',
    TYPE_EXCHANGE_LAT: '#FF7F00',
    TYPE_CONCERTED: '#984EA3',
    TYPE_INTERLAYER_HOP_UP: '#F781BF',
    TYPE_INTERLAYER_HOP_DOWN: '#A65628',
    TYPE_INPLANE_COMPLEX: '#BBBBBB',
    TYPE_RELAXATION: '#DDDDDD',
    TYPE_COMPLEX: '#999999',
    # Subsurface (z-layer reclassified) — distinct purple/blue palette so
    # they pop visually against the surface-event cloud.
    TYPE_SUBSURFACE_MIGRATION: '#6A3D9A',
    TYPE_SUBSURFACE_HOP: '#CAB2D6',
    TYPE_SUBSURFACE_1NN_HOP: '#1F78B4',
    TYPE_SUBSURFACE_OTHER: '#666666',
    # Merged (up/down collapsed) — used when the "Merge up/down" widget
    # toggle is on. Pick the 'up' colour as canonical for each family.
    'exchange': '#E41A1C',
    'interlayer_hop': '#F781BF',
}
# keep a module-level default palette so cell 10 etc. can still reference `colors`
colors = custom_colors

# Directionless base types (used when 'Merge up/down' toggle is active).
ALL_TYPES_BASE = []
for _t in ALL_TYPES:
    _b = _base_move_type(_t)
    if _b not in ALL_TYPES_BASE:
        ALL_TYPES_BASE.append(_b)


def _plot_displacement(sub, active_types, cmap_name, point_size,
                       show_thresholds, merge_updown=False, save=False):
    type_col = 'move_type_base' if merge_updown else 'move_type'
    type_order = ALL_TYPES_BASE if merge_updown else ALL_TYPES
    palette = _palette(cmap_name, type_order) or custom_colors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: |d| vs |dz|
    ax = axes[0]
    for mtype in type_order:
        if mtype not in active_types:
            continue
        mask = sub[type_col] == mtype
        if mask.any():
            ax.scatter(sub.loc[mask, 'move_dist'], sub.loc[mask, 'move_dz'],
                       color=palette.get(mtype, '#999'),
                       label=f'{mtype} ({mask.sum()})',
                       alpha=0.4, s=point_size, edgecolors='none')
    if show_thresholds:
        ax.axhline(1.0, color='grey', ls=':', lw=1, label='layer change threshold')
        ax.axvline(NN1_DIST, color='green', ls='--', lw=1, alpha=0.5,
                   label=f'1NN = {NN1_DIST:.2f} A')
        ax.axvline(NN2_DIST, color='blue', ls='--', lw=1, alpha=0.5,
                   label=f'2NN = {NN2_DIST:.2f} A')
    ax.set_xlabel('Primary atom displacement |d| (A)')
    ax.set_ylabel('Out-of-plane |dz| (A)')
    ax.set_title('Classification by displacement')
    ax.legend(fontsize=8, loc='upper right')

    # Plot 2: histogram
    ax = axes[1]
    for mtype in type_order:
        if mtype not in active_types:
            continue
        mask = sub[type_col] == mtype
        if mask.any():
            ax.hist(sub.loc[mask, 'move_dist'], bins=60, alpha=0.6,
                    color=palette.get(mtype, '#999'), label=mtype)
    if show_thresholds:
        ax.axvline(NN1_DIST, color='green', ls='--', lw=2,
                   label=f'1NN = {NN1_DIST:.2f} A')
        ax.axvline(NN2_DIST, color='blue', ls='--', lw=2,
                   label=f'2NN = {NN2_DIST:.2f} A')
    ax.set_xlabel('Primary atom displacement |d| (A)')
    ax.set_ylabel('Count')
    ax.set_title('Displacement distribution by move type')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / 'displacement_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()


# Save the canonical static version first (so PNG output matches prior runs).
_plot_displacement(sub, set(ALL_TYPES), 'custom', 15, True, save=True)

# Interactive controller
# Two parallel checkbox grids — one for split types, one for base types —
# so the widget layout remains stable when 'Merge up/down' is toggled.
_split_toggles = {mt: widgets.Checkbox(value=True, description=mt,
                                       indent=False,
                                       layout=widgets.Layout(width='180px'))
                  for mt in ALL_TYPES}
_base_toggles = {mt: widgets.Checkbox(value=True, description=mt,
                                      indent=False,
                                      layout=widgets.Layout(width='180px'))
                 for mt in ALL_TYPES_BASE}
_cmap_dd = widgets.Dropdown(options=CMAP_CHOICES, value='custom',
                            description='Colours:')
_psize = widgets.IntSlider(value=15, min=2, max=40, step=1, description='Point size:')
_thresh = widgets.Checkbox(value=True, description='Show thresholds')
_merge = widgets.Checkbox(value=True, description='Merge up/down')


def _go(**kw):
    merge = kw['merge']
    if merge:
        active = {mt for mt in ALL_TYPES_BASE if kw['b_' + mt]}
    else:
        active = {mt for mt in ALL_TYPES if kw['s_' + mt]}
    _plot_displacement(sub, active, kw['cmap'], kw['psize'], kw['thresh'],
                       merge_updown=merge)


_controls = {'cmap': _cmap_dd, 'psize': _psize, 'thresh': _thresh,
             'merge': _merge,
             **{'s_' + mt: w for mt, w in _split_toggles.items()},
             **{'b_' + mt: w for mt, w in _base_toggles.items()}}
_ui = widgets.VBox([
    widgets.HTML('<b>Interactive view</b> — toggle types, colour scheme, point size:'),
    widgets.HBox([_cmap_dd, _psize, _thresh, _merge]),
    widgets.HTML('<i>base types (when Merge up/down is on):</i>'),
    widgets.GridBox(list(_base_toggles.values()),
                    layout=widgets.Layout(grid_template_columns='repeat(4, 180px)')),
    widgets.HTML('<i>full split types (when Merge up/down is off):</i>'),
    widgets.GridBox(list(_split_toggles.values()),
                    layout=widgets.Layout(grid_template_columns='repeat(4, 180px)')),
])
_out = widgets.interactive_output(_go, _controls)
display(_ui, _out)
