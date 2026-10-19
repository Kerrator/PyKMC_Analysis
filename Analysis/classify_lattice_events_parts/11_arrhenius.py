# %%
# Cell 5: Temperature Dependence
# Interactive: toggle which move types appear, colour scheme, error-bar
# display, and Arrhenius fit-line display.
# (kB is defined in 01_imports.py)

_ARR_MTS = ['1NN_hop', '2NN_hop', 'exchange', 'exchange_lateral']


def _plot_arrhenius(sub, active_types, cmap_name, show_err, show_fit, save=False):
    palette = _palette(cmap_name, _ARR_MTS) or colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean barrier vs temperature per type
    ax = axes[0]
    for mt in _ARR_MTS:
        if mt not in active_types:
            continue
        mt_data = sub[sub['move_type_base'] == mt]
        if len(mt_data) == 0:
            continue
        grouped = mt_data.groupby('temp')['energy_barrier'].agg(['mean', 'std', 'count'])
        grouped = grouped[grouped['count'] >= 3]
        if len(grouped) < 2:
            continue
        yerr = grouped['std'] if show_err else None
        ax.errorbar(grouped.index, grouped['mean'], yerr=yerr,
                    marker='o', label=mt, color=palette.get(mt, '#999'),
                    capsize=3, capthick=1)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Mean barrier Ea (eV)')
    ax.set_title('Mean barrier vs temperature')
    ax.legend()

    # Arrhenius plot
    ax = axes[1]
    if '1NN_hop' in active_types:
        nn1_temps = sub[sub['move_type_base'] == '1NN_hop'].groupby('temp')
        inv_t, ln_k = [], []
        for temp, grp in nn1_temps:
            if len(grp) < 3:
                continue
            k0 = 10.0
            ea_mean = grp['energy_barrier'].mean()
            k_mean = k0 * np.exp(-ea_mean / (kB * temp))
            inv_t.append(1000.0 / temp)
            ln_k.append(np.log(k_mean))
        if len(inv_t) >= 2:
            ax.plot(inv_t, ln_k, 'o-',
                    color=palette.get('1NN_hop', '#4DAF4A'), lw=2, markersize=8)
            if show_fit:
                coeffs = np.polyfit(inv_t, ln_k, 1)
                ea_eff = -coeffs[0] * kB * 1000
                ax.plot(inv_t, np.polyval(coeffs, inv_t), '--', color='grey',
                        label=f'Fit: Ea_eff = {ea_eff:.3f} eV')
                ax.legend()
    ax.set_xlabel('1000/T (1/K)')
    ax.set_ylabel('ln(k) [k in ps⁻¹]')
    ax.set_title('Arrhenius plot — 1NN hop rate')

    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / 'temperature_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()


# Canonical static save
_plot_arrhenius(sub, set(_ARR_MTS), 'custom', True, True, save=True)

# Interactive controller
_arr_toggles = {mt: widgets.Checkbox(value=True, description=mt, indent=False,
                                     layout=widgets.Layout(width='180px'))
                for mt in _ARR_MTS}
_arr_cmap = widgets.Dropdown(options=CMAP_CHOICES, value='custom', description='Colours:')
_arr_err = widgets.Checkbox(value=True, description='Error bars')
_arr_fit = widgets.Checkbox(value=True, description='Show fit')


def _go_arr(**kw):
    active = {mt for mt in _ARR_MTS if kw[mt]}
    _plot_arrhenius(sub, active, kw['cmap'], kw['err'], kw['fit'])


_arr_controls = {'cmap': _arr_cmap, 'err': _arr_err, 'fit': _arr_fit, **_arr_toggles}
_arr_ui = widgets.VBox([
    widgets.HTML('<b>Interactive view</b> — toggle types, errors, fit:'),
    widgets.HBox([_arr_cmap, _arr_err, _arr_fit]),
    widgets.HBox(list(_arr_toggles.values())),
])
display(_arr_ui, widgets.interactive_output(_go_arr, _arr_controls))
