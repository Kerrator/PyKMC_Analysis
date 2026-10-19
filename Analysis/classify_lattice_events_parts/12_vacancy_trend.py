# %%
# Cell 6: Vacancy Count Dependence
# Interactive: toggle which move types appear in both panels, colour scheme,
# stacked/grouped bar style.

_VAC_MTS = ['1NN_hop', '2NN_hop', 'exchange', 'exchange_lateral',
            'concerted', 'complex']


def _plot_vacancy(sub, active_types, cmap_name, bar_mode,
                  show_fraction, save=False):
    palette = _palette(cmap_name, _VAC_MTS) or colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nvac_vals_present = sorted(sub['nvac'].unique())

    mts = [mt for mt in _VAC_MTS if mt in active_types]

    # Left: stacked or grouped bars of event counts by nvac
    ax = axes[0]
    x = np.arange(len(nvac_vals_present))
    if bar_mode == 'stacked':
        bottoms = np.zeros(len(nvac_vals_present))
        for mt in mts:
            counts = np.array([((sub['nvac'] == nv) & (sub['move_type_base'] == mt)).sum()
                               for nv in nvac_vals_present])
            if counts.sum() > 0:
                ax.bar(x, counts, bottom=bottoms,
                       color=palette.get(mt, '#999'), label=mt, alpha=0.85)
                bottoms += counts
    else:  # grouped
        n = max(len(mts), 1)
        w = 0.8 / n
        for i, mt in enumerate(mts):
            counts = np.array([((sub['nvac'] == nv) & (sub['move_type_base'] == mt)).sum()
                               for nv in nvac_vals_present])
            if counts.sum() > 0:
                ax.bar(x + (i - (n - 1) / 2) * w, counts, w,
                       color=palette.get(mt, '#999'), label=mt, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nv}vac' for nv in nvac_vals_present])
    ax.set_ylabel('Event count')
    ax.set_title(f'Event types by vacancy count (Ea < {BARRIER_CUTOFF} eV)')
    ax.legend(fontsize=8, loc='upper left')

    # Right: fraction of each type vs nvac (if requested, else secondary count view)
    ax = axes[1]
    if show_fraction:
        for mt in mts:
            fracs = []
            for nv in nvac_vals_present:
                total = (sub['nvac'] == nv).sum()
                n_mt = ((sub['nvac'] == nv) & (sub['move_type_base'] == mt)).sum()
                fracs.append(100 * n_mt / total if total > 0 else 0)
            if max(fracs) > 0:
                ax.plot(nvac_vals_present, fracs, 'o-', color=palette.get(mt, '#999'),
                        label=mt, lw=2, markersize=6)
        ax.set_xlabel('Vacancy count')
        ax.set_ylabel('Fraction (%)')
        ax.set_title('Move type fraction vs vacancy count')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
    else:
        for mt in mts:
            counts = [((sub['nvac'] == nv) & (sub['move_type_base'] == mt)).sum()
                      for nv in nvac_vals_present]
            if sum(counts) > 0:
                ax.plot(nvac_vals_present, counts, 'o-',
                        color=palette.get(mt, '#999'), label=mt, lw=2, markersize=6)
        ax.set_xlabel('Vacancy count')
        ax.set_ylabel('Event count')
        ax.set_title('Event counts vs vacancy count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / 'vacancy_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()


# Canonical static save (stacked + fraction — matches the prior behaviour)
_plot_vacancy(sub, set(_VAC_MTS), 'custom', 'stacked', True, save=True)

# Interactive controller
_vac_toggles = {mt: widgets.Checkbox(value=True, description=mt, indent=False,
                                     layout=widgets.Layout(width='170px'))
                for mt in _VAC_MTS}
_vac_cmap = widgets.Dropdown(options=CMAP_CHOICES, value='custom', description='Colours:')
_vac_mode = widgets.ToggleButtons(options=['stacked', 'grouped'], value='stacked',
                                  description='Bars:')
_vac_frac = widgets.Checkbox(value=True, description='Right panel: fraction (%)')


def _go_vac(**kw):
    active = {mt for mt in _VAC_MTS if kw[mt]}
    _plot_vacancy(sub, active, kw['cmap'], kw['mode'], kw['frac'])


_vac_controls = {'cmap': _vac_cmap, 'mode': _vac_mode, 'frac': _vac_frac,
                 **_vac_toggles}
_vac_ui = widgets.VBox([
    widgets.HTML('<b>Interactive view</b> — toggle types, bar mode, fraction:'),
    widgets.HBox([_vac_cmap, _vac_mode, _vac_frac]),
    widgets.HBox(list(_vac_toggles.values())),
])
display(_vac_ui, widgets.interactive_output(_go_vac, _vac_controls))
