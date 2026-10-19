# %%
# Cell 0b: Scan and load the runs of COMPOSITION that pass POTENTIAL_FILTER

def scan_simulations(root, potential_filter=POTENTIAL_FILTER):
    """Find run directories under root ({N}vac/T{temp}/{variant}/).

    Keeps runs with an input.in whose pair_coeff contains potential_filter
    ('' keeps every run). The layout is parsed by parse_run_path.
    """
    sims = []
    for variant_dir in sorted(root.glob('*/*/*')):
        if not variant_dir.is_dir():
            continue
        parsed = parse_run_path(variant_dir)
        if parsed is None:
            continue
        if not (variant_dir / 'input.in').exists():
            continue
        _, nvac, temp, variant = parsed

        sim_config = read_sim_config(variant_dir)
        if not check_potential(sim_config, potential_filter):
            continue

        sims.append({
            'nvac': nvac,
            'temp': int(temp),
            'variant': variant,
            'path': variant_dir,
            'potential': sim_config['potential'],
        })
    return sims


sims = scan_simulations(DATA_ROOT)
print(f'Found {len(sims)} simulations under {DATA_ROOT}')

# Load all reference tables
all_events = []
load_failures = []

for sim in sims:
    ref_path = sim['path'] / 'reference_table.pickle'
    if not ref_path.exists():
        load_failures.append((sim, 'no pickle'))
        continue
    try:
        df = pd.read_pickle(ref_path)
        df = df.copy()
        df['nvac'] = sim['nvac']
        df['temp'] = sim['temp']
        df['sim_path'] = str(sim['path'])
        df['sim_label'] = f"{sim['nvac']}vac_T{sim['temp']}"
        all_events.append(df)
    except Exception as e:
        load_failures.append((sim, str(e)))

if load_failures:
    print(f'\nLoad failures ({len(load_failures)}):')
    for sim, err in load_failures:
        print(f'  {sim["nvac"]}vac T{sim["temp"]} {sim["variant"]}: {err}')

# Combine into master DataFrame
master = pd.concat(all_events, ignore_index=True)
print(f'\nTotal events loaded: {len(master)}')
print(f'Events under {BARRIER_CUTOFF} eV: {(master["energy_barrier"] < BARRIER_CUTOFF).sum()}')
print(f'\nPer-simulation summary:')
summary = master.groupby(['nvac', 'temp']).agg(
    n_total=('energy_barrier', 'count'),
    n_sub_cutoff=('energy_barrier', lambda x: (x < BARRIER_CUTOFF).sum()),
    ea_min=('energy_barrier', 'min'),
    ea_max=('energy_barrier', 'max'),
).reset_index()
print(summary.to_string(index=False))
