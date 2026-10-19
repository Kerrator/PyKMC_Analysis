# %%
# Cell 8: Lattice Model Parameter Extraction
#
# For each move type, extract the parameters needed for an on-lattice KMC.

k0_THz = 10.0  # pyKMC attempt frequency in THz (= ps^-1)
k0_Hz = k0_THz * 1e12  # in Hz

print('=' * 70)
print('LATTICE MODEL PARAMETERS')
print('=' * 70)

params = {}
# Iterate over every known type (including z-layer-aware subsurface
# variants). ALL_TYPES is defined in 01_imports.py and stays in sync
# when new types are added.
for mt in ALL_TYPES:
    mt_data = sub[sub['move_type'] == mt]
    if len(mt_data) == 0:
        continue

    barriers = mt_data['energy_barrier']
    displacements = mt_data['move_dist']

    p = {
        'count': int(len(mt_data)),
        'Ea_mean_eV': float(barriers.mean()),
        'Ea_std_eV': float(barriers.std()),
        'Ea_min_eV': float(barriers.min()),
        'Ea_max_eV': float(barriers.max()),
        'displacement_mean_A': float(displacements.mean()),
        'displacement_std_A': float(displacements.std()),
        'prefactor_THz': k0_THz,
        'prefactor_Hz': k0_Hz,
        'nvac_range': [int(mt_data['nvac'].min()), int(mt_data['nvac'].max())],
        'temp_range': [int(mt_data['temp'].min()), int(mt_data['temp'].max())],
    }

    # Environment dependence: barrier vs nvac
    env_dep = mt_data.groupby('nvac')['energy_barrier'].agg(['mean', 'std', 'count'])
    p['barrier_by_nvac'] = env_dep.to_dict('index')

    params[mt] = p

    print(f'\n{mt}:')
    print(f'  Count: {p["count"]}')
    print(f'  Ea: {p["Ea_mean_eV"]:.4f} ± {p["Ea_std_eV"]:.4f} eV '
          f'(range: {p["Ea_min_eV"]:.4f} – {p["Ea_max_eV"]:.4f})')
    print(f'  Displacement: {p["displacement_mean_A"]:.3f} ± {p["displacement_std_A"]:.3f} A')
    print(f'  Prefactor: {k0_THz} THz')
    print(f'  Barrier by nvac:')
    for nv, stats in sorted(p['barrier_by_nvac'].items()):
        print(f'    {nv}vac: {stats["mean"]:.4f} ± {stats["std"]:.4f} eV (n={int(stats["count"])})')

# Lattice displacement vectors for FCC (100)
print(f'\n{"=" * 70}')
print('LATTICE DISPLACEMENT VECTORS (FCC 100 surface)')
print(f'{"=" * 70}')
print(f'\n1NN hop vectors (4 in-plane, a/sqrt(2) = {NN1_DIST:.3f} A):')
for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    v = np.array([dx, dy, 0]) * A_NI / 2
    print(f'  [{dx:+d},{dy:+d}, 0] x a/2 = [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] A')

print(f'\n2NN hop vectors (4 in-plane, a = {NN2_DIST:.3f} A):')
for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
    v = np.array([dx, dy, 0]) * A_NI
    print(f'  [{dx:+d},{dy:+d}, 0] x a = [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] A')
