# %%
# Cell 9: Validation — Compare analytical D to off-lattice MSD
#
# For 1vac simulations: D_lattice = f * Gamma * d^2 / 4
# where f = 0.4669 is the correlation factor for tracer diffusion via
# single vacancy on a 2D square lattice.
#
# NOTE: post-z-layer pass, `move_type == TYPE_1NN` now means a true
# surface 1NN hop (mover on top layer, both endpoints). Any bulk
# `subsurface_1NN_hop` events are excluded from this surface-diffusion
# validation, which is the physically correct behaviour.

from ase.io import read as ase_read

# Correlation factor for 2D square lattice tracer diffusion
f_corr = 0.4669

# Single-parameter model: use mean 1NN barrier across all 1vac simulations
ea_mean_all_1vac = sub[(sub['nvac'] == 1) & (sub['move_type'] == TYPE_1NN)]['energy_barrier'].mean()
print(f'Mean 1NN barrier across all 1vac sims: {ea_mean_all_1vac:.4f} eV')
print(f'Correlation factor: f = {f_corr}')
print(f'\nDiffusion coefficient validation (1vac only):')
print('-' * 70)
print(f'{"T(K)":>6s} {"Ea_1NN":>8s} {"D_MSD":>12s} {"D_uncorr":>12s} '
      f'{"D_corr":>12s} {"ratio_unc":>10s} {"ratio_corr":>10s}')

validation_rows = []
for sim in sims:
    if sim['nvac'] != 1:
        continue
    temp = sim['temp']
    sim_path = sim['path']

    # Load summary.json for MSD
    summary_path = sim_path / 'analysis' / 'summary.json'
    if not summary_path.exists():
        continue
    with open(summary_path) as f:
        sj = json.load(f)

    msd_final = sj.get('msd_final', [0])[0]
    total_time = sj.get('total_sim_time', 0)
    if total_time <= 0 or msd_final <= 0:
        continue

    # D from MSD: D = MSD / (2*d*t) where d=2 for 2D surface diffusion
    D_msd = msd_final / (4 * total_time)

    # D from per-simulation 1NN barrier
    sim_events = sub[(sub['nvac'] == 1) & (sub['temp'] == temp) & (sub['move_type'] == TYPE_1NN)]
    if len(sim_events) == 0:
        continue
    ea_1nn = sim_events['energy_barrier'].mean()

    rate = k0_Hz * np.exp(-ea_1nn / (kB * temp))
    D_uncorr = rate * NN1_DIST**2 / 4
    D_corr = f_corr * D_uncorr

    # Single-parameter model using mean barrier
    rate_mean = k0_Hz * np.exp(-ea_mean_all_1vac / (kB * temp))
    D_mean_uncorr = rate_mean * NN1_DIST**2 / 4
    D_mean_corr = f_corr * D_mean_uncorr

    ratio_uncorr = D_uncorr / D_msd if D_msd > 0 else float('inf')
    ratio_corr = D_corr / D_msd if D_msd > 0 else float('inf')

    validation_rows.append({
        'temp': temp,
        'Ea_1NN': ea_1nn,
        'D_msd': D_msd,
        'D_uncorr': D_uncorr,
        'D_corr': D_corr,
        'D_mean_uncorr': D_mean_uncorr,
        'D_mean_corr': D_mean_corr,
        'ratio_uncorr': ratio_uncorr,
        'ratio_corr': ratio_corr,
    })

    print(f'{temp:6d} {ea_1nn:8.4f} {D_msd:12.3e} {D_uncorr:12.3e} '
          f'{D_corr:12.3e} {ratio_uncorr:10.2f} {ratio_corr:10.2f}')

if validation_rows:
    vdf = pd.DataFrame(validation_rows).sort_values('temp')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Temperature plot
    ax = axes[0]
    ax.semilogy(vdf['temp'], vdf['D_msd'], 'ko-', label='D from MSD (off-lattice)', lw=2, markersize=8)
    ax.semilogy(vdf['temp'], vdf['D_uncorr'], 's--', color='#E41A1C',
                label='D lattice (uncorrected)', lw=1.5, markersize=6)
    ax.semilogy(vdf['temp'], vdf['D_corr'], '^-', color='#4DAF4A',
                label=f'D lattice (f={f_corr})', lw=2, markersize=8)
    ax.semilogy(vdf['temp'], vdf['D_mean_corr'], 'v:', color='#377EB8',
                label=f'D single-Ea model (Ea={ea_mean_all_1vac:.3f}, f={f_corr})', lw=1.5, markersize=6)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('D (A^2/s)')
    ax.set_title('Surface 1NN hop diffusion validation (1vac, z-layer filtered)')
    ax.legend(fontsize=8)

    # Arrhenius plot
    ax = axes[1]
    inv_t = 1000.0 / vdf['temp']
    ax.plot(inv_t, np.log(vdf['D_msd']), 'ko-', label='D_MSD', lw=2, markersize=8)
    ax.plot(inv_t, np.log(vdf['D_corr']), '^-', color='#4DAF4A',
            label=f'D_lattice (f={f_corr})', lw=2, markersize=8)
    ax.plot(inv_t, np.log(vdf['D_mean_corr']), 'v:', color='#377EB8',
            label=f'Single-Ea model', lw=1.5, markersize=6)
    ax.set_xlabel('1000/T (1/K)')
    ax.set_ylabel('ln(D)')
    ax.set_title('Arrhenius plot')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'diffusion_validation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\nSummary: corrected ratios range from '
          f'{vdf["ratio_corr"].min():.2f} to {vdf["ratio_corr"].max():.2f}')
