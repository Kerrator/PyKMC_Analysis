# %%
# Cell 0: Configuration & Data Loading
#
# Run from an environment where the package is installed (pip install -e .
# from the repository root) so `Analysis` is importable from this directory.
# Set the parameters in the "Parameters" block below.
import pickle
import configparser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Interactive widgets for figure toggles (visibility + colour scheme).
# If running this notebook headless (e.g. nbconvert --execute), the widgets
# simply render as static output of their default state.
import ipywidgets as widgets
from IPython.display import display

# Available matplotlib qualitative colormaps for widget dropdowns.
CMAP_CHOICES = ['custom', 'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'viridis']


def _palette(cmap_name, keys):
    """Return a {key: hex_color} dict for the given cmap name and ordered keys."""
    if cmap_name == 'custom':
        return None  # caller falls back to its hand-picked dict
    cmap = cm.get_cmap(cmap_name, max(len(keys), 2))
    return {k: cmap(i) for i, k in enumerate(keys)}

# Classification API from Analysis/lattice_map_events.py
from Analysis import _paths
from Analysis._paths import BARRIER_CUTOFF, CLASSIFICATION_DIR
from Analysis.sim_discovery import parse_run_path
from Analysis.lattice_map_events import (
    # Classification
    classify_event_mechanism,
    classify_event_from_row,
    compute_saddle_features,
    compute_thresholds_gmm,
    classify_event_adaptive,
    classify_event_enhanced,
    TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE,
    TYPE_2NN, TYPE_EXCHANGE, TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN,
    TYPE_EXCHANGE_LAT,
    TYPE_CONCERTED, TYPE_COMPLEX, TYPE_RELAXATION, LATTICE_TYPES,
    TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN, TYPE_INPLANE_COMPLEX,
    TYPE_SUBSURFACE_MIGRATION, TYPE_SUBSURFACE_HOP,
    TYPE_SUBSURFACE_1NN_HOP, TYPE_SUBSURFACE_OTHER,
    DEALLOYING_TYPES,
    # Z-layer-aware reclassification helpers
    get_surface_layer, get_event_surface_layers, reclassify_by_zlayer,
    coarsen_event_type,
    MOVE_THRESHOLD, DIST_TOLERANCE,
    NN1_DIST, NN2_DIST, A_NI,
    # Config helpers
    read_sim_config, check_potential, REQUIRED_POTENTIAL,
    # Lattice helpers
    group_z_layers,
    generate_ideal_sites,
    find_vacancy_sites,
    build_surface_neighbour_list,
    map_event_to_lattice,
    # Step log
    parse_pykmc_out,
    compute_kinetic_coverage,
    build_rate_lookup,
)

# ── Parameters ──
# COMPOSITION      directory under the data root to classify
#                  (runs laid out as {composition}/{N}vac/T{temp}/{variant}/).
# DATA_ROOT_PATH   None uses $PYKMC_ANALYSIS_DATA_ROOT; or set '/path/to/data'.
# POTENTIAL_FILTER keep runs whose input.in pair_coeff contains this string;
#                  '' keeps every run.
# Outputs go to $PYKMC_ANALYSIS_OUTPUT_DIR, else Analysis/lattice_event_classification/.
COMPOSITION = '100Ni'
DATA_ROOT_PATH = None
POTENTIAL_FILTER = REQUIRED_POTENTIAL

DATA_ROOT = _paths.data_root(DATA_ROOT_PATH) / COMPOSITION
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f'composition directory not found: {DATA_ROOT} (check COMPOSITION)')
OUT_DIR = CLASSIFICATION_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# BARRIER_CUTOFF imported from Analysis._paths (KMCConfig) — single source.
kB = 8.617333e-5      # eV/K — Boltzmann constant (used by ml_descriptors and arrhenius)

# All event types for iteration
ALL_TYPES = [TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN,
             TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN, TYPE_EXCHANGE_LAT,
             TYPE_CONCERTED, TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN,
             TYPE_INPLANE_COMPLEX, TYPE_COMPLEX, TYPE_RELAXATION,
             TYPE_SUBSURFACE_MIGRATION, TYPE_SUBSURFACE_HOP,
             TYPE_SUBSURFACE_1NN_HOP, TYPE_SUBSURFACE_OTHER]

print(f'Data root: {DATA_ROOT}')
print(f'Output:    {OUT_DIR}')
print(f'Barrier cutoff: {BARRIER_CUTOFF} eV')
print(f'Potential filter: {POTENTIAL_FILTER or "(none)"}')
print(f'1NN distance: {NN1_DIST:.3f} A, 2NN distance: {NN2_DIST:.3f} A')
