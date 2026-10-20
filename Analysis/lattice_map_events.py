#!/usr/bin/env python3
"""
lattice_map_events.py — Diagnostic per-simulation lattice projection for FCC (100)

Reads a pyKMC simulation directory containing:
  - trajkmc.xyz       (trajectory, frame 0 used as lattice reference)
  - reference_table.pickle  (event catalogue)
  - pykmc.out          (step log with selected events)
  - input.in           (simulation parameters)

By default only processes runs whose ``pair_coeff`` ([Lammps] section of
input.in) contains REQUIRED_POTENTIAL ("NiAlH"); other runs are skipped.
Pass ``--required-potential ''`` to accept any potential.

Produces diagnostic outputs for a single simulation:
  1. Per-event lattice projection: vacancy site_from → site_to, local environment
  2. Approximate environment-dependent barrier table from local event clusters
  3. Kinetic coverage report: what fraction of steps/time the lattice model captures
  4. Diagnostic plots

This script is useful for inspection and single-simulation diagnostics. The
authoritative cross-simulation lattice KMC inputs live in the output
directory (default Analysis/lattice_event_classification/) and are built from
global trajectory states by build_rate_table.py, not from the local-cluster
projections here.

Usage (from the repository root):
    python -m Analysis.lattice_map_events /path/to/run
    python -m Analysis.lattice_map_events /path/to/run --n-vac 5 --required-potential ''

The vacancy count is read from the ``{composition}/{N}vac/T{temp}/{variant}``
layout when the run sits in one; otherwise pass ``--n-vac`` (default 1).

Output (written to <run>/analysis/):
    analysis/lattice_mapping.csv         — diagnostic per-event lattice projection
    analysis/rate_lookup_table.csv       — approximate local-cluster barrier table
    analysis/kinetic_coverage.txt        — summary report
    analysis/lattice_mapping_plots.png   — diagnostic plots
"""

import argparse
import configparser
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# 3D lattice helpers (Analysis/lattice_3d.py), re-exported so downstream code
# can import them from here alongside the classification API.
from Analysis.lattice_3d import (  # noqa: E402,F401
    assign_3d_site_classes,
    assign_direction_family,
    classify_3d_motif,
    compute_3d_environment_descriptor,
    compute_3d_site_graph,
    extract_3d_occupancy_transition,
)


def _install_pandas_pickle_compat():
    """Handle older StringArray pickle state written by older pandas versions.

    Some reference tables pickle pandas StringArray objects with
    state encoded as ``(StringDtype, ndarray)``. pandas 2.3.x no longer accepts
    that tuple directly during unpickle, so install a narrow compatibility shim
    before any ``pd.read_pickle`` calls happen in notebooks or scripts.
    """
    from pandas._libs.arrays import NDArrayBacked
    from pandas.core.arrays.string_ import StringArray, StringDtype

    if getattr(StringArray, "_pykmc_pickle_compat_installed", False):
        return

    original_setstate = StringArray.__setstate__

    def compat_setstate(self, state):
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], StringDtype)
        ):
            dtype, values = state
            NDArrayBacked.__init__(self, values, dtype)
            return
        return original_setstate(self, state)

    StringArray.__setstate__ = compat_setstate
    StringArray._pykmc_pickle_compat_installed = True


_install_pandas_pickle_compat()

# ── Constants ────────────────────────────────────────────────────────────────

MOVE_THRESHOLD = 0.5   # Å — atom considered "moved" if displaced more than this
DIST_TOLERANCE = 0.5   # Å — tolerance for matching displacement to lattice distances

# Physical/analysis constants — single source in Analysis/_paths.py (KMCConfig).
# A_NI is a default; trajectory detection may override the lattice constant.
from Analysis._paths import (  # noqa: E402
    A_NI,
    NN1_DIST,
    NN2_DIST,
    REQUIRED_POTENTIAL,
)
from Analysis.sim_discovery import parse_run_path  # noqa: E402
from Analysis.reference_table import event_types  # noqa: E402

# Event type names (used consistently across lattice_map_events and notebooks)
TYPE_1NN = "1NN_hop"
TYPE_1NN_COOP = "1NN_hop_cooperative"
TYPE_1NN_TRIPLE = "1NN_hop_triple"
TYPE_2NN = "2NN_hop"
TYPE_EXCHANGE = "exchange"
TYPE_EXCHANGE_UP = "exchange_up"
TYPE_EXCHANGE_DOWN = "exchange_down"
TYPE_EXCHANGE_LAT = "exchange_lateral"
TYPE_CONCERTED = "concerted"
TYPE_RELAXATION = "relaxation"  # sub-threshold displacement, null event on lattice
TYPE_COMPLEX = "complex"
TYPE_INTERLAYER_HOP_UP = "interlayer_hop_up"
TYPE_INTERLAYER_HOP_DOWN = "interlayer_hop_down"
TYPE_INPLANE_COMPLEX = "inplane_complex"
# Subsurface (bulk-like) variants — applied as a post-pass once the moving
# atom's z-layer is known. Displacement geometry alone can't distinguish
# these from surface events.
TYPE_SUBSURFACE_MIGRATION = "subsurface_migration"  # was exchange_* with mover below top
TYPE_SUBSURFACE_HOP = "subsurface_hop"              # was interlayer_hop_* below top
TYPE_SUBSURFACE_1NN_HOP = "subsurface_1NN_hop"      # was 1NN_hop*/2NN_hop below top
TYPE_SUBSURFACE_OTHER = "subsurface_other"          # catch-all for other types below top

LATTICE_TYPES = {TYPE_1NN, TYPE_2NN, TYPE_1NN_COOP}  # types that map to lattice moves

# Event types relevant to dealloying (surface diffusion + interlayer transport)
DEALLOYING_TYPES = {
    TYPE_1NN, TYPE_1NN_COOP, TYPE_2NN,
    TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN,
    TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN,
}

# Map subdivided types back to the original 7-type scheme for backward compatibility
COARSE_TYPE_MAP = {
    TYPE_1NN_COOP: TYPE_1NN,
    TYPE_1NN_TRIPLE: TYPE_1NN,
    TYPE_EXCHANGE_UP: TYPE_EXCHANGE,
    TYPE_EXCHANGE_DOWN: TYPE_EXCHANGE,
    TYPE_INTERLAYER_HOP_UP: TYPE_COMPLEX,
    TYPE_INTERLAYER_HOP_DOWN: TYPE_COMPLEX,
    TYPE_INPLANE_COMPLEX: TYPE_COMPLEX,
}


def coarsen_event_type(event_type):
    """Map subdivided event types back to original 7-type scheme."""
    return COARSE_TYPE_MAP.get(event_type, event_type)


# ── Lattice helpers (adapted from analyse_trajectory.ipynb) ──────────────────

def group_z_layers(positions, tol=0.5):
    """Group atoms into z-layers by clustering z-coordinates.

    Tolerance of 0.5 Å handles intra-layer z-spread (~0.3 Å for relaxed
    surface) while staying well below inter-layer spacing (~1.76 Å).
    """
    z = positions[:, 2]
    order = np.argsort(z)
    layers = []
    used = np.zeros(len(z), dtype=bool)
    for idx in order:
        if used[idx]:
            continue
        mask = (~used) & (np.abs(z - z[idx]) < tol)
        layer_idx = np.where(mask)[0]
        layers.append((np.mean(z[layer_idx]), layer_idx))
        used[mask] = True
    layers.sort(key=lambda x: x[0])
    return layers


def generate_ideal_sites(positions, cell, n_vac):
    """Generate ideal FCC lattice sites from atom positions + inferred vacancy sites.

    For layers with fewer atoms than the maximum (vacancy present), the missing
    site is found by comparison to a full layer of the same sublattice type.
    """
    Lx, Ly = cell[0], cell[1]
    layers = group_z_layers(positions)
    atoms_per_layer = max(len(idx) for _, idx in layers)

    full_layers = [(z, idx) for z, idx in layers if len(idx) == atoms_per_layer]
    short_layers = [(z, idx) for z, idx in layers if len(idx) < atoms_per_layer]

    all_sites = list(positions)  # every atom IS at a site

    offsets_2d = np.array([[dx * Lx, dy * Ly]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]])

    for z_short, short_idx in short_layers:
        n_missing = atoms_per_layer - len(short_idx)
        short_pos_2d = positions[short_idx, :2].copy()

        # Find best-matching full layer (same sublattice)
        best_ref_idx = None
        best_score = np.inf
        for z_full, full_idx in full_layers:
            ref_2d = positions[full_idx, :2].copy()
            short_rep = np.vstack([short_pos_2d + o for o in offsets_2d])
            tree = cKDTree(short_rep)
            dists, _ = tree.query(ref_2d)
            score = np.mean(np.sort(dists)[:len(short_idx)])
            if score < best_score:
                best_score = score
                best_ref_idx = full_idx

        ref_2d = positions[best_ref_idx, :2].copy()
        short_rep = np.vstack([short_pos_2d + o for o in offsets_2d])
        tree = cKDTree(short_rep)
        dists, _ = tree.query(ref_2d)

        nn_dist = np.median(dists[dists < 1.0]) if np.any(dists < 1.0) else 1.0
        missing_mask = dists > max(nn_dist * 2, 1.0)
        missing_2d = ref_2d[missing_mask][:n_missing]

        for m2d in missing_2d:
            vac_pos = np.array([m2d[0] % Lx, m2d[1] % Ly, z_short])
            all_sites.append(vac_pos)

    return np.array(all_sites)


def find_vacancy_sites(positions, ideal_sites, cell, n_vac):
    """Identify vacancy sites by finding ideal sites farthest from any atom.

    Returns (vacancy_site_indices, per_atom_site_map, per_atom_distances).
    """
    Lx, Ly = cell[0], cell[1]
    n_sites = len(ideal_sites)

    pos_wrapped = positions.copy()
    pos_wrapped[:, 0] %= Lx
    pos_wrapped[:, 1] %= Ly

    offsets = np.array([[dx * Lx, dy * Ly, 0.0]
                        for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    atom_rep = np.vstack([pos_wrapped + o for o in offsets])
    atom_tree = cKDTree(atom_rep)

    site_dists, _ = atom_tree.query(ideal_sites)
    vacancy_indices = np.argsort(site_dists)[-n_vac:]

    site_rep = np.vstack([ideal_sites + o for o in offsets])
    site_tree = cKDTree(site_rep)
    atom_dists, atom_site_idx = site_tree.query(pos_wrapped)
    atom_site_idx = atom_site_idx % n_sites

    return vacancy_indices, atom_site_idx, atom_dists


# ── Config helpers ────────────────────────────────────────────────────────────

def read_sim_config(sim_dir):
    """Read input.in and extract simulation parameters.

    Returns dict with temperature, potential, n_vac, or None if unreadable.
    """
    sim_dir = Path(sim_dir)
    config = configparser.ConfigParser()
    config.read(sim_dir / 'input.in')

    result = {}

    # Temperature from [RateConstant] section
    try:
        result['temperature'] = float(config.get('RateConstant', 'T'))
    except (configparser.NoSectionError, configparser.NoOptionError):
        result['temperature'] = None

    # Potential from [Lammps] section
    try:
        result['pair_coeff'] = config.get('Lammps', 'pair_coeff')
    except (configparser.NoSectionError, configparser.NoOptionError):
        result['pair_coeff'] = ''

    result['potential'] = 'unknown'
    if 'NiAlH' in result['pair_coeff']:
        result['potential'] = 'NiAlH_jea'
    elif 'Bonny' in result['pair_coeff']:
        result['potential'] = 'Bonny'
    elif 'EDDE' in result['pair_coeff']:
        result['potential'] = 'EDDE'

    # n_vac from the {composition}/{N}vac/T{temp}/{variant} layout (default 1)
    parsed = parse_run_path(sim_dir)
    result['n_vac'] = parsed[1] if parsed is not None else 1

    return result


def check_potential(sim_config, required=REQUIRED_POTENTIAL):
    """Return True if the run's ``pair_coeff`` contains ``required``.

    An empty ``required`` disables the filter (every run passes).
    """
    if not required:
        return True
    return required in sim_config.get('pair_coeff', '')


# ── Event classification ─────────────────────────────────────────────────────

def classify_event_mechanism(initial_pos, final_pos, saddle_pos, move_atom_idx,
                             nn_dist=NN1_DIST, nn2_dist=NN2_DIST, tol=DIST_TOLERANCE):
    """Classify an event into lattice move types by displacement analysis.

    Uses the primary moving atom's displacement vector to distinguish:
    - 1NN_hop:           in-plane, |d| ≈ a/√2
    - 2NN_hop:           in-plane, |d| ≈ a
    - exchange:          layer change, 2+ atoms moved, short displacement
    - exchange_lateral:  layer change, long displacement
    - concerted:         3+ atoms significantly displaced
    - complex:           none of the above (flagged for review)

    Parameters
    ----------
    initial_pos : ndarray (N, 3)
    final_pos : ndarray (N, 3)
    saddle_pos : ndarray (N, 3)
    move_atom_idx : int
        Index of the primary moving atom in the local environment.
    nn_dist : float
        1NN distance (a/√2 for FCC).
    nn2_dist : float
        2NN distance (a for FCC).
    tol : float
        Tolerance for distance matching.

    Returns
    -------
    dict with keys: event_type, is_on_lattice, n_moved, move_dist,
    move_dz, move_dxy, max_displacement, moved_indices, layer_change.
    """
    disp = final_pos - initial_pos
    disp_mag = np.linalg.norm(disp, axis=1)
    moved_mask = disp_mag > MOVE_THRESHOLD
    n_moved = int(np.sum(moved_mask))
    max_disp = float(np.max(disp_mag)) if len(disp_mag) > 0 else 0.0
    moved_indices = np.where(moved_mask)[0]

    # Primary mover displacement
    mi = move_atom_idx
    move_dist = float(disp_mag[mi])
    move_dz = float(abs(disp[mi, 2]))
    move_dz_signed = float(disp[mi, 2])  # positive = up (toward vacuum)
    move_dxy = float(np.sqrt(disp[mi, 0]**2 + disp[mi, 1]**2))

    # If primary mover barely moved, check whether another atom is the real
    # hopper — pyKMC's move_atom_idx can be wrong for sub-threshold events.
    alt_mover = None
    if move_dist < MOVE_THRESHOLD:
        # Find the atom with the largest displacement
        best_idx = int(np.argmax(disp_mag))
        best_dist = float(disp_mag[best_idx])
        if abs(best_dist - nn_dist) <= tol:
            # Another atom moved ~1NN distance — reclassify with it as the mover
            alt_mover = best_idx
            mi = best_idx
            move_dist = best_dist
            move_dz = float(abs(disp[mi, 2]))
            move_dz_signed = float(disp[mi, 2])
            move_dxy = float(np.sqrt(disp[mi, 0]**2 + disp[mi, 1]**2))
        elif best_dist < MOVE_THRESHOLD:
            # No atom moved significantly — this is a relaxation (null event)
            return {
                "event_type": TYPE_RELAXATION,
                "is_on_lattice": False,
                "n_moved": n_moved,
                "move_dist": float(disp_mag[move_atom_idx]),
                "move_dz": float(abs(disp[move_atom_idx, 2])),
                "move_dz_signed": float(disp[move_atom_idx, 2]),
                "move_dxy": float(np.sqrt(disp[move_atom_idx, 0]**2
                                          + disp[move_atom_idx, 1]**2)),
                "max_displacement": max_disp,
                "moved_indices": moved_indices,
                "layer_change": False,
                "alt_mover": None,
            }

    # Layer change: inter-layer spacing for Ni(100) is a/2 ≈ 1.76 Å
    layer_change = move_dz > 1.0

    # Classification
    if not layer_change and abs(move_dist - nn_dist) <= tol:
        # 1NN hop — subdivide by number of atoms moved
        if n_moved >= 3:
            event_type = TYPE_1NN_TRIPLE
        elif n_moved == 2:
            event_type = TYPE_1NN_COOP
        else:
            event_type = TYPE_1NN
    elif not layer_change and abs(move_dist - nn2_dist) <= tol:
        event_type = TYPE_2NN
    elif layer_change and n_moved >= 2 and move_dist < nn2_dist + tol:
        # Exchange — subdivide by direction
        if move_dz_signed > 0:
            event_type = TYPE_EXCHANGE_UP
        else:
            event_type = TYPE_EXCHANGE_DOWN
    elif layer_change and move_dist >= nn2_dist + tol:
        event_type = TYPE_EXCHANGE_LAT
    elif layer_change and n_moved == 1:
        # Single-atom interlayer hop (was hidden in complex)
        if move_dz_signed > 0:
            event_type = TYPE_INTERLAYER_HOP_UP
        else:
            event_type = TYPE_INTERLAYER_HOP_DOWN
    elif n_moved >= 3:
        event_type = TYPE_CONCERTED
    elif not layer_change:
        event_type = TYPE_INPLANE_COMPLEX
    else:
        event_type = TYPE_COMPLEX

    return {
        "event_type": event_type,
        "is_on_lattice": event_type in LATTICE_TYPES,
        "n_moved": n_moved,
        "move_dist": move_dist,
        "move_dz": move_dz,
        "move_dz_signed": move_dz_signed,
        "move_dxy": move_dxy,
        "max_displacement": max_disp,
        "moved_indices": moved_indices,
        "layer_change": layer_change,
        "alt_mover": alt_mover,
    }


def compute_saddle_features(initial_positions, saddle_positions, final_positions,
                            move_atom_idx):
    """Compute saddle-point descriptors for a single event.

    Extracts geometric features from the saddle-point configuration that
    discriminate hop-like vs exchange-like mechanisms. The key descriptor is
    ``saddle_height_ratio``: for hops the mover sits above its neighbours at
    the bridge site (ratio > 1), while for exchange two atoms form a
    dumbbell at roughly equal height (ratio ≈ 1).

    Parameters
    ----------
    initial_positions : ndarray, shape (n_atoms, 3)
        Atomic positions at the initial minimum.
    saddle_positions : ndarray, shape (n_atoms, 3)
        Atomic positions at the saddle point.
    final_positions : ndarray, shape (n_atoms, 3)
        Atomic positions at the final minimum.
    move_atom_idx : int
        Index of the primary moving atom.

    Returns
    -------
    dict
        saddle_disp_mag : float
            |saddle − initial| for the primary mover.
        saddle_dz : float
            Signed z-component of (saddle − initial) for the mover.
        saddle_height_ratio : float
            mover_z_saddle / mean_z_of_displaced_neighbours.
            >1 means the mover sits above neighbours (hop-like);
            ≈1 means a dumbbell geometry (exchange-like).
        saddle_n_displaced : int
            Number of atoms with |saddle − initial| > MOVE_THRESHOLD.
        saddle_asymmetry : float
            |d(init→saddle)| / |d(init→final)| for the mover.
            ≈0.5 for a symmetric path; deviates for asymmetric paths.
        saddle_max_pair_dz : float
            Max |dz_i − dz_j| among displaced atoms at the saddle
            (large for exchange, small for concerted in-plane moves).
    """
    nan_result = {
        "saddle_disp_mag": np.nan,
        "saddle_dz": np.nan,
        "saddle_height_ratio": np.nan,
        "saddle_n_displaced": 0,
        "saddle_asymmetry": np.nan,
        "saddle_max_pair_dz": np.nan,
    }

    if saddle_positions is None or np.array_equal(saddle_positions, initial_positions):
        return nan_result

    # Displacements from initial to saddle and initial to final
    saddle_disp = saddle_positions - initial_positions
    saddle_disp_mag = np.linalg.norm(saddle_disp, axis=1)
    final_disp = final_positions - initial_positions
    final_disp_mag = np.linalg.norm(final_disp, axis=1)

    mi = move_atom_idx
    mover_saddle_disp = float(saddle_disp_mag[mi])
    mover_saddle_dz = float(saddle_disp[mi, 2])
    mover_final_disp = float(final_disp_mag[mi])

    # Asymmetry: how far along the path the saddle sits (0.5 = symmetric)
    if mover_final_disp > 1e-10:
        saddle_asymmetry = mover_saddle_disp / mover_final_disp
    else:
        saddle_asymmetry = np.nan

    # Displaced atoms at saddle (excluding the primary mover)
    displaced_mask = saddle_disp_mag > MOVE_THRESHOLD
    saddle_n_displaced = int(np.sum(displaced_mask))
    displaced_indices = np.where(displaced_mask)[0]
    neighbour_indices = displaced_indices[displaced_indices != mi]

    # Height ratio: mover z at saddle vs mean z of displaced neighbours
    mover_z_saddle = saddle_positions[mi, 2]
    if len(neighbour_indices) > 0:
        mean_neighbour_z = float(np.mean(saddle_positions[neighbour_indices, 2]))
        if abs(mean_neighbour_z) > 1e-10:
            saddle_height_ratio = mover_z_saddle / mean_neighbour_z
        else:
            saddle_height_ratio = np.nan
    else:
        saddle_height_ratio = np.nan

    # Max pair dz among displaced atoms at saddle
    if len(displaced_indices) >= 2:
        dz_vals = saddle_disp[displaced_indices, 2]
        saddle_max_pair_dz = float(np.max(dz_vals) - np.min(dz_vals))
    else:
        saddle_max_pair_dz = 0.0

    return {
        "saddle_disp_mag": mover_saddle_disp,
        "saddle_dz": mover_saddle_dz,
        "saddle_height_ratio": saddle_height_ratio,
        "saddle_n_displaced": saddle_n_displaced,
        "saddle_asymmetry": saddle_asymmetry,
        "saddle_max_pair_dz": saddle_max_pair_dz,
    }


def classify_event_from_row(row, nn_dist=NN1_DIST, nn2_dist=NN2_DIST, tol=DIST_TOLERANCE,
                            include_saddle_features=False, include_species=False):
    """Convenience wrapper: classify a reference_table row (pandas Series).

    Extracts initial_positions, final_positions, saddle_positions,
    and move_atom_idx from the row and calls classify_event_mechanism.

    Parameters
    ----------
    row : pandas.Series
        A row from the reference_table DataFrame.
    nn_dist, nn2_dist, tol : float
        Classification thresholds.
    include_saddle_features : bool, default False
        If True, compute and include saddle-point descriptors in the result.
    include_species : bool, default False
        If True, include mover/partner species and NN species counts.
    """
    initial_pos = row['initial_positions']
    final_pos = row['final_positions']
    saddle_pos = row.get('saddle_positions', initial_pos)
    move_idx = row['move_atom_idx']

    result = classify_event_mechanism(
        initial_pos, final_pos, saddle_pos, move_idx,
        nn_dist=nn_dist, nn2_dist=nn2_dist, tol=tol,
    )

    if include_saddle_features:
        sf = compute_saddle_features(initial_pos, saddle_pos, final_pos, move_idx)
        result.update(sf)

    if include_species:
        species = event_types(row)
        result.update(_extract_species_info(
            species, move_idx, result.get('moved_indices', np.array([])),
            initial_pos, nn_dist,
        ))

    return result


def _extract_species_info(species, move_idx, moved_indices, initial_pos, nn_dist):
    """Extract species of mover, partner, and NN neighbours.

    Parameters
    ----------
    species : array-like or None
        Per-atom species labels (e.g., [1, 1, 2] for LAMMPS types or
        ['Ni', 'Ni', 'Fe'] for chemical symbols).
    move_idx : int
        Primary mover index.
    moved_indices : ndarray
        Indices of all atoms that moved significantly.
    initial_pos : ndarray (N, 3)
        Atomic positions used to find 1NN neighbours.
    nn_dist : float
        1NN cutoff distance.

    Returns
    -------
    dict with mover_species, partner_species, nn_species_counts.
    """
    if species is None:
        return {
            "mover_species": None,
            "partner_species": None,
            "nn_species_counts": {},
        }

    species = list(species)
    mover_species = species[move_idx]

    # Partner: second moved atom in cooperative/exchange events
    partner_species = None
    partner_candidates = [i for i in moved_indices if i != move_idx]
    if partner_candidates:
        partner_species = species[partner_candidates[0]]

    # 1NN species counts around the mover
    tree = cKDTree(initial_pos)
    nn_indices = tree.query_ball_point(initial_pos[move_idx], nn_dist * 1.1)
    nn_indices = [i for i in nn_indices if i != move_idx]
    nn_species_counts = {}
    for i in nn_indices:
        s = species[i]
        nn_species_counts[s] = nn_species_counts.get(s, 0) + 1

    return {
        "mover_species": mover_species,
        "partner_species": partner_species,
        "nn_species_counts": nn_species_counts,
    }


# ── ML descriptors for barrier-prediction models ────────────────────────────
#
# Computed at notebook-load time from per-event row data plus per-simulation
# cell vectors and lattice adjacency. Tier 1-4 scalars + species-histogram
# columns (see classify_lattice_events_parts/04b_ml_descriptors.py).

ML_SPECIES_SCHEMA = ('Ni', 'Fe', 'Cr')  # fixed columns for nn{1,2}_count_*; absent species = 0


def _species_histogram_shells(positions, types, move_idx):
    """Count atoms by species in the 1NN and 2NN shells around the mover.

    Returns (h1, h2) where each is a dict ``{species_label: int}``.
    Uses NN1_DIST and NN2_DIST module constants with a 10% tolerance.
    Returns ({}, {}) when types is None or mismatched.
    """
    if types is None or len(types) != len(positions):
        return {}, {}
    tol = 1.1
    tree = cKDTree(positions)
    r1 = NN1_DIST * tol
    r2 = NN2_DIST * tol
    idx_r1 = set(tree.query_ball_point(positions[move_idx], r1))
    idx_r2 = set(tree.query_ball_point(positions[move_idx], r2))
    nn1 = idx_r1 - {move_idx}            # within 1NN shell
    nn2 = idx_r2 - idx_r1 - {move_idx}   # between 1NN and 2NN (exclusive)

    h1, h2 = {}, {}
    for i in nn1:
        s = str(types[i])
        h1[s] = h1.get(s, 0) + 1
    for i in nn2:
        s = str(types[i])
        h2[s] = h2.get(s, 0) + 1
    return h1, h2


def _mean_1nn_bond_length(positions, move_idx, cutoff):
    """Mean distance from mover to all atoms within `cutoff`."""
    tree = cKDTree(positions)
    nn = tree.query_ball_point(positions[move_idx], cutoff)
    nn = [i for i in nn if i != move_idx]
    if not nn:
        return np.nan
    dists = np.linalg.norm(positions[nn] - positions[move_idx], axis=1)
    return float(dists.mean())


def _local_rmsd(init, final, move_idx, radius=5.0):
    """RMSD of atoms within `radius` (Å) of mover's initial position."""
    tree = cKDTree(init)
    idx = tree.query_ball_point(init[move_idx], radius)
    if not idx:
        return np.nan
    d = final[idx] - init[idx]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def _dist_to_nearest_undercoord(positions, move_idx, cutoff, threshold):
    """Distance from mover to nearest atom with coordination < threshold.

    Coordination = # neighbors within `cutoff` (excluding self).
    On FCC(100), bulk = 12, surface = 8; threshold = 7 catches step edges.
    """
    tree = cKDTree(positions)
    # Vectorized: query all atoms' neighbor counts in one batch
    nn_lists = tree.query_ball_point(positions, cutoff)
    coords = np.array([len(nn) - 1 for nn in nn_lists])  # -1 for self
    under = np.where(coords < threshold)[0]
    under = under[under != move_idx]
    if len(under) == 0:
        return np.nan
    dists = np.linalg.norm(positions[under] - positions[move_idx], axis=1)
    return float(dists.min())


def _count_vacant_1nn(site_idx, positions, ideal_sites, nn_adjacency, occ_tol=0.6):
    """Count unoccupied 1NN sites of `site_idx` in the ideal lattice.

    A site is occupied if at least one atom is within `occ_tol` (Å) of it.
    """
    if site_idx < 0 or nn_adjacency is None:
        return np.nan
    neighbours = nn_adjacency.get(site_idx, [])
    if not neighbours:
        return np.nan
    atom_tree = cKDTree(positions)
    vacant = 0
    for s in neighbours:
        d, _ = atom_tree.query(ideal_sites[s])
        if d > occ_tol:
            vacant += 1
    return int(vacant)


def _nearest_lattice_site(pos, ideal_sites):
    """Return index of ideal site nearest to `pos` (3-vector)."""
    if ideal_sites is None or len(ideal_sites) == 0:
        return -1
    d2 = np.sum((ideal_sites - pos) ** 2, axis=1)
    return int(np.argmin(d2))


def compute_ml_descriptors(row, cell=None, ideal_sites=None, nn_adjacency=None):
    """Compute ML descriptors for a single event row.

    Returns a dict of scalar features (Tiers 1-4) designed for training
    barrier-prediction ML models. Species histogram columns (nn{1,2}_count_*)
    are included via _species_histogram_shells().

    Parameters
    ----------
    row : pandas.Series
        One row of the event DataFrame (must contain initial_positions,
        final_positions, saddle_positions, move_atom_idx, energy_barrier,
        temp, idx_backward, initial_types, sim_path).
    cell : list [Lx, Ly, Lz] or None
        Simulation cell diagonal loaded from trajkmc.xyz frame 0.
    ideal_sites : ndarray (N, 3) or None
        Ideal lattice sites for this simulation, from generate_ideal_sites().
    nn_adjacency : dict {site_idx: [1NN site indices]} or None
        Top-layer 1NN adjacency, from build_surface_neighbour_list()[0].
    """
    out = {}

    init = np.asarray(row['initial_positions'])
    final = np.asarray(row['final_positions'])
    saddle = row.get('saddle_positions')
    if saddle is not None:
        saddle = np.asarray(saddle)
    mi = int(row['move_atom_idx'])

    # ── Tier 1: top-k per-atom displacement magnitudes ──────────────────
    disp_mag = np.linalg.norm(final - init, axis=1)
    top5 = np.sort(disp_mag)[::-1][:5]
    for k in range(5):
        out[f'disp_top{k + 1}'] = float(top5[k]) if k < len(top5) else 0.0

    # ── Tier 1: continuous mover depth (Å) ──────────────────────────────
    # depth = z_topsurf - z_mover; positive = mover below surface.
    try:
        layers_init = group_z_layers(init, tol=0.5)
        z_top_init = layers_init[-1][0]  # highest z_mean
        out['mover_depth_initial'] = float(z_top_init - init[mi, 2])

        layers_final = group_z_layers(final, tol=0.5)
        z_top_final = layers_final[-1][0]
        out['mover_depth_final'] = float(z_top_final - final[mi, 2])
        out['mover_depth_delta'] = out['mover_depth_final'] - out['mover_depth_initial']
    except Exception:
        out['mover_depth_initial'] = np.nan
        out['mover_depth_final'] = np.nan
        out['mover_depth_delta'] = np.nan

    # ── Tier 1: kT-normalized barrier ───────────────────────────────────
    _kB = 8.617333e-5  # eV/K
    T = float(row.get('temp', np.nan))
    Ea = float(row.get('energy_barrier', np.nan))
    if T > 0 and not np.isnan(T) and not np.isnan(Ea):
        out['Ea_over_kT'] = Ea / (_kB * T)
    else:
        out['Ea_over_kT'] = np.nan

    # ── Tier 1: species histograms 1NN + 2NN (fixed schema) ────────────
    types = event_types(row)
    h1, h2 = _species_histogram_shells(init, types, mi)
    for sp in ML_SPECIES_SCHEMA:
        out[f'nn1_count_{sp}'] = int(h1.get(sp, 0))
        out[f'nn2_count_{sp}'] = int(h2.get(sp, 0))
    # Mover species (emitted alongside existing mover_species from
    # 08_species.py as a cross-check; they differ on ~1% alt_mover events).
    if types is not None and len(types) > mi:
        out['mover_species_ml'] = str(types[mi])
    else:
        out['mover_species_ml'] = None

    # ── Tier 2: neighborhood geometry ───────────────────────────────────
    tol = 1.1
    r1 = NN1_DIST * tol

    # Coordination: # atoms within 1NN cutoff (excluding self)
    tree_init = cKDTree(init)
    tree_final = cKDTree(final)
    out['coord_mover_initial'] = len(tree_init.query_ball_point(init[mi], r1)) - 1
    out['coord_mover_final'] = len(tree_final.query_ball_point(final[mi], r1)) - 1

    # Vacancy count in 1NN shell — use lattice adjacency if available,
    # otherwise approximate as (expected_bulk_coord - actual_coord).
    if ideal_sites is not None and nn_adjacency is not None:
        site_i = _nearest_lattice_site(init[mi], ideal_sites)
        site_f = _nearest_lattice_site(final[mi], ideal_sites)
        out['n_vac_nn1_initial'] = _count_vacant_1nn(
            site_i, init, ideal_sites, nn_adjacency
        )
        out['n_vac_nn1_final'] = _count_vacant_1nn(
            site_f, final, ideal_sites, nn_adjacency
        )
    else:
        # Approximation: for FCC bulk, expected 1NN = 12. Shortfall = vacancies.
        # For surface atoms this over-counts (missing = vacuum, not vacancy),
        # but it's a reasonable proxy until lattice adjacency is enabled.
        out['n_vac_nn1_initial'] = max(0, 12 - out['coord_mover_initial'])
        out['n_vac_nn1_final'] = max(0, 12 - out['coord_mover_final'])

    # Mean 1NN bond length (picks up strained sites)
    out['mean_nn1_bond_length_initial'] = _mean_1nn_bond_length(init, mi, r1)

    # Distance to nearest under-coordinated neighbour (step-edge proxy)
    # Threshold 7: FCC(100) surface has coord=8, step edges have coord<=6
    out['dist_to_nearest_undercoord'] = _dist_to_nearest_undercoord(
        init, mi, r1, threshold=7
    )

    # Local RMSD within 5 Å of mover (locality of perturbation)
    out['local_rmsd_5A'] = _local_rmsd(init, final, mi, radius=5.0)

    # ── Tier 3: saddle-path descriptors ─────────────────────────────────
    # NaN when saddle is missing or identical to initial (pARTn didn't
    # produce a separate saddle — same pattern as compute_saddle_features).
    if saddle is None or np.allclose(saddle, init):
        out['path_straightness'] = np.nan
        out['bonds_broken_init_to_saddle'] = np.nan
        out['bonds_formed_init_to_saddle'] = np.nan
        out['coord_mover_saddle'] = np.nan
    else:
        # Straightness: direct mover displacement / path-via-saddle
        d_direct = np.linalg.norm(final[mi] - init[mi])
        d_via = (np.linalg.norm(saddle[mi] - init[mi])
                 + np.linalg.norm(final[mi] - saddle[mi]))
        out['path_straightness'] = float(d_direct / d_via) if d_via > 1e-6 else np.nan

        # Coordination at saddle
        tree_saddle = cKDTree(saddle)
        out['coord_mover_saddle'] = len(
            tree_saddle.query_ball_point(saddle[mi], r1)
        ) - 1

        # Bonds broken / formed: 1NN neighbours of mover at initial vs saddle
        init_nn = set(tree_init.query_ball_point(init[mi], r1)) - {mi}
        saddle_nn = set(tree_saddle.query_ball_point(saddle[mi], r1)) - {mi}
        out['bonds_broken_init_to_saddle'] = len(init_nn - saddle_nn)
        out['bonds_formed_init_to_saddle'] = len(saddle_nn - init_nn)

    # ── Tier 4: convergence sanity flag ─────────────────────────────────
    sa = row.get('saddle_asymmetry', np.nan)
    if sa is not None and not (isinstance(sa, float) and np.isnan(sa)):
        sa = float(sa)
        out['saddle_asymmetry_oob'] = bool(sa < 0.1 or sa > 1.1)
    else:
        out['saddle_asymmetry_oob'] = False

    return out


# ── Lattice-environment fingerprint (Stage 1h) ──────────────────────────────
#
# Wraps the env_fingerprint.py module to produce per-event fingerprints
# from a row + per-simulation lattice cache (ideal_sites + site_graph).

def compute_env_fingerprint(row, surface_or_bulk='auto'):
    """Compute lattice-environment fingerprint columns for a single event.

    Uses the per-event cluster's positions directly (mover-relative shell
    template matching). Doesn't depend on a global site_graph or ideal_sites
    grid — works on the centered, "canonical" cluster geometry stored in
    reference_table.pickle.

    Parameters
    ----------
    row : pandas.Series
        Reference-table row with initial_positions, final_positions,
        initial_types, move_atom_idx.
    surface_or_bulk : 'auto', 'surface', or 'bulk'
        Which shell template to use. 'auto' guesses from the cluster's
        z-extent above/below the mover: if the mover is at the top of its
        z-range (within 0.5 A of cluster z_max), use surface; else bulk.

    Returns dict with keys:
        env_fingerprint_occ      : str — D4-canonical (occupancy, hop) fingerprint
        env_fingerprint_species  : str — D4-canonical (species, hop) fingerprint
        hop_vector_canonical     : str — '(dx,dy,dz)' in (nn1/2, nn1/2, a/2) units
        env_shell_complete       : bool — True iff all shell slots are inside cluster reach

    All fields are None / False if the cluster is too small to resolve the shell.
    """
    out = {
        'env_fingerprint_occ': None,
        'env_fingerprint_species': None,
        'hop_vector_canonical': None,
        'env_shell_complete': False,
    }

    # Lazy import to keep core lattice_map_events.py independent of env_fingerprint
    from Analysis.env_fingerprint import (
        SHELL_REF_POSITIONS_SURFACE, SHELL_REF_POSITIONS_BULK,
        PERMS_SURFACE, PERMS_BULK,
        SHELL_SLOT_OUT_OF_RANGE,
        canonicalise, extract_env_vectors_from_cluster, fingerprint_string,
    )

    init = np.asarray(row['initial_positions'])
    final = np.asarray(row['final_positions'])
    types = event_types(row)
    mi = int(row['move_atom_idx'])

    if init is None or len(init) == 0 or mi < 0 or mi >= len(init):
        return out

    # Decide surface vs bulk template
    if surface_or_bulk == 'auto':
        z_mover = init[mi, 2]
        z_max = init[:, 2].max()
        z_min = init[:, 2].min()
        # Surface = mover near top of cluster (within ~half a layer)
        # Layer spacing is a/2 = NN1_DIST/sqrt(2) ~= 1.76 A
        is_surface = (z_max - z_mover) < (NN1_DIST / np.sqrt(2.0)) * 0.5
        # Sanity: also check there's at least one layer below
        if (z_mover - z_min) < (NN1_DIST / np.sqrt(2.0)) * 0.5:
            return out  # cluster too thin to resolve shell
    else:
        is_surface = (surface_or_bulk == 'surface')

    ref_positions = (SHELL_REF_POSITIONS_SURFACE if is_surface
                     else SHELL_REF_POSITIONS_BULK)
    perms = PERMS_SURFACE if is_surface else PERMS_BULK

    # Compute hop vector directly from the mover's pre/post positions in the cluster.
    # The cluster is in pARTn's centered frame, but the mover delta is invariant.
    unit_xy = NN1_DIST / 2.0
    unit_z = NN1_DIST / np.sqrt(2.0)
    delta = final[mi] - init[mi]
    hop_vec = (
        int(round(delta[0] / unit_xy)),
        int(round(delta[1] / unit_xy)),
        int(round(delta[2] / unit_z)),
    )

    # Extract occupancy + species using cluster-relative shell matching
    occ_vec, spec_vec = extract_env_vectors_from_cluster(
        init, types, mi, ref_positions, NN1_DIST,
    )

    # If any slot is out-of-range, fingerprint is incomplete
    if (occ_vec == SHELL_SLOT_OUT_OF_RANGE).any():
        return out

    # Canonicalise under D4 (env + hop together)
    canon_occ, canon_hop_occ = canonicalise(occ_vec, hop_vec, perms)
    canon_spec, canon_hop_spec = canonicalise(spec_vec, hop_vec, perms)

    out['env_fingerprint_occ'] = fingerprint_string(canon_occ, canon_hop_occ, "occ")
    out['env_fingerprint_species'] = fingerprint_string(canon_spec, canon_hop_spec, "spc")
    out['hop_vector_canonical'] = f"({canon_hop_occ[0]},{canon_hop_occ[1]},{canon_hop_occ[2]})"
    out['env_shell_complete'] = True
    return out


# ── Unsupervised event clustering ─────────────────────────────────

def cluster_classify_events(events_df, min_samples=10):
    """Cluster events by displacement features using DBSCAN.

    Discovers natural event type boundaries from data rather than hardcoded
    thresholds. Returns a DataFrame with cluster assignments and diagnostics.

    Parameters
    ----------
    events_df : DataFrame
        Must have columns: move_dist, move_dz, move_dxy, n_moved.
    min_samples : int
        DBSCAN min_samples parameter (minimum cluster size).

    Returns
    -------
    DataFrame with columns: cluster_id, cluster_label, centroid_dist,
        classification_source ('cluster').
    dict with diagnostics: silhouette_score, n_clusters, n_noise,
        cluster_centers, k_distance_eps.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    features = ["move_dist", "move_dz", "move_dxy", "n_moved"]
    X = events_df[features].values.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-tune eps from k-distance plot (k = min_samples)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, -1])
    # Use 90th percentile of k-distances as eps (conservative)
    eps = float(np.percentile(k_dist, 90))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Compute cluster centers (in original feature space)
    cluster_centers = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        cluster_centers[cid] = X[mask].mean(axis=0)

    # Label clusters by matching centroid move_dist to known lattice distances
    cluster_labels = {}
    for cid, center in cluster_centers.items():
        cdist = center[0]  # move_dist is feature 0
        cdz = center[1]    # move_dz is feature 1
        if cdz > 1.0:
            # Interlayer event
            if cdist < NN2_DIST + DIST_TOLERANCE:
                cluster_labels[cid] = TYPE_EXCHANGE
            else:
                cluster_labels[cid] = TYPE_EXCHANGE_LAT
        elif abs(cdist - NN1_DIST) <= DIST_TOLERANCE:
            cluster_labels[cid] = TYPE_1NN
        elif abs(cdist - NN2_DIST) <= DIST_TOLERANCE:
            cluster_labels[cid] = TYPE_2NN
        else:
            cluster_labels[cid] = TYPE_COMPLEX

    # Build result DataFrame
    result = pd.DataFrame(index=events_df.index)
    result["cluster_id"] = labels
    result["cluster_label"] = [
        cluster_labels.get(l, "noise") if l != -1 else "noise"
        for l in labels
    ]
    # Distance to cluster center
    result["centroid_dist"] = 0.0
    for cid, center in cluster_centers.items():
        mask = labels == cid
        center_scaled = scaler.transform(center.reshape(1, -1))
        dists = np.linalg.norm(X_scaled[mask] - center_scaled, axis=1)
        result.loc[mask, "centroid_dist"] = dists

    result["classification_source"] = "cluster"

    # Diagnostics
    diagnostics = {
        "n_clusters": n_clusters,
        "n_noise": int(np.sum(labels == -1)),
        "eps": eps,
        "cluster_centers": {int(k): v.tolist() for k, v in cluster_centers.items()},
        "cluster_labels": {int(k): v for k, v in cluster_labels.items()},
        "cluster_sizes": {
            int(cid): int(np.sum(labels == cid))
            for cid in set(labels) if cid != -1
        },
    }

    # Silhouette score (only if >1 cluster and <50k samples for speed)
    if n_clusters > 1 and len(X_scaled) < 50000:
        from sklearn.metrics import silhouette_score
        non_noise = labels != -1
        if np.sum(non_noise) > n_clusters:
            diagnostics["silhouette_score"] = float(
                silhouette_score(X_scaled[non_noise], labels[non_noise])
            )

    return result, diagnostics


def classify_event_hybrid(initial_pos, final_pos, saddle_pos, move_atom_idx,
                          cluster_result=None, event_index=None,
                          nn_dist=NN1_DIST, nn2_dist=NN2_DIST, tol=DIST_TOLERANCE):
    """Hybrid classifier: rule-based primary, cluster fallback for 'complex'.

    Uses the standard rule-based classifier first. If the result is TYPE_COMPLEX
    and a cluster_result DataFrame is available, falls back to the cluster label.

    Parameters
    ----------
    cluster_result : DataFrame or None
        Output of cluster_classify_events() indexed by event index.
    event_index : hashable or None
        Index into cluster_result for this event.

    Returns
    -------
    dict — same as classify_event_mechanism, plus 'classification_source'.
    """
    result = classify_event_mechanism(
        initial_pos, final_pos, saddle_pos, move_atom_idx,
        nn_dist=nn_dist, nn2_dist=nn2_dist, tol=tol,
    )

    if (result["event_type"] in {TYPE_COMPLEX, TYPE_INPLANE_COMPLEX}
            and cluster_result is not None
            and event_index is not None
            and event_index in cluster_result.index):
        cluster_label = cluster_result.loc[event_index, "cluster_label"]
        if cluster_label != "noise":
            result["event_type"] = cluster_label
            result["is_on_lattice"] = cluster_label in LATTICE_TYPES
            result["classification_source"] = "cluster"
        else:
            result["classification_source"] = "rule_complex"
    else:
        result["classification_source"] = "rule"

    return result


# ── Adaptive threshold computation ─────────────────────────────────

def compute_thresholds(positions, cell, surface_orientation="100"):
    """Derive classification thresholds from detected lattice geometry.

    Instead of hardcoded constants, computes thresholds from the actual
    atomic positions and cell dimensions. Supports FCC (100), (111), and (110).

    Parameters
    ----------
    positions : ndarray (N, 3)
        Atomic positions from the first trajectory frame.
    cell : list of 3 floats
        Cell dimensions [Lx, Ly, Lz].
    surface_orientation : str
        One of "100", "111", "110".

    Returns
    -------
    dict with keys: nn1_dist, nn2_dist, dist_tolerance, move_threshold,
        layer_change_threshold, a_lattice, n_inplane_nn, surface_orientation.
    """
    # Detect NN distance from positions
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    nn_dist = float(np.median(dists[:, 1]))

    # Derive lattice constant (FCC: a = nn * sqrt(2))
    a_lat = nn_dist * np.sqrt(2)

    if surface_orientation == "100":
        nn1 = nn_dist           # a/sqrt(2) — in-plane 1NN
        nn2 = a_lat             # a — in-plane 2NN
        interlayer = a_lat / 2  # a/2
        n_inplane_nn = 4
    elif surface_orientation == "111":
        nn1 = nn_dist           # a/sqrt(2) — in-plane 1NN
        nn2 = a_lat             # a — in-plane 2NN
        interlayer = a_lat / np.sqrt(3)  # a/sqrt(3)
        n_inplane_nn = 6
    elif surface_orientation == "110":
        nn1 = nn_dist           # a/sqrt(2)
        nn2 = a_lat             # a
        interlayer = a_lat / (2 * np.sqrt(2))  # a/(2*sqrt(2))
        n_inplane_nn = 2
    else:
        raise ValueError(f"Unsupported surface orientation: {surface_orientation}")

    # Derive tolerances from shell gap
    shell_gap = nn2 - nn1
    dist_tolerance = shell_gap * 0.4  # 40% of gap between 1NN and 2NN shells
    move_threshold = nn1 * 0.2        # 20% of 1NN distance
    layer_change_thr = interlayer * 0.6  # 60% of interlayer spacing

    return {
        "nn1_dist": nn1,
        "nn2_dist": nn2,
        "dist_tolerance": dist_tolerance,
        "move_threshold": move_threshold,
        "layer_change_threshold": layer_change_thr,
        "a_lattice": a_lat,
        "n_inplane_nn": n_inplane_nn,
        "surface_orientation": surface_orientation,
    }


# ── GMM-based adaptive thresholds ─────────────────────────────────────────

def compute_thresholds_gmm(displacements, dz_values, lattice_constant=None):
    """Determine classification thresholds from displacement distributions via GMM.

    Fits a two-component Gaussian mixture to the in-plane displacement
    magnitudes to locate the 1NN and 2NN shells, and a separate mixture to
    |dz| values to find the in-plane / interlayer boundary.

    Parameters
    ----------
    displacements : array-like
        |d| for all sub-cutoff events.
    dz_values : array-like
        |dz| for all sub-cutoff events.
    lattice_constant : float or None
        If given, seed GMM means at a/sqrt(2) and a for faster convergence.

    Returns
    -------
    dict
        nn1_center : float
            Fitted 1NN displacement centre.
        nn2_center : float
            Fitted 2NN displacement centre.
        nn1_range : tuple of float
            (lower, upper) acceptance window for 1NN.
        nn2_range : tuple of float
            (lower, upper) acceptance window for 2NN.
        layer_change_threshold : float
            dz boundary between in-plane and interlayer events.
        confidence : float
            Bhattacharyya distance between 1NN and 2NN components
            (larger ⇒ better separated).
        gmm_displacement : GaussianMixture
            Fitted model on displacement magnitudes (for plotting).
        gmm_dz : GaussianMixture
            Fitted model on |dz| values (for plotting).
    """
    from sklearn.mixture import GaussianMixture

    displacements = np.asarray(displacements, dtype=float)
    dz_values = np.asarray(dz_values, dtype=float)

    # ── Displacement GMM (1NN vs 2NN) ──
    # Filter to in-plane events to avoid exchange contamination
    inplane_mask = np.abs(dz_values) < 1.0
    d_inplane = displacements[inplane_mask]
    # Only keep events that are plausibly lattice hops (exclude relaxations)
    d_inplane = d_inplane[d_inplane > MOVE_THRESHOLD]

    gmm_d = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    if lattice_constant is not None:
        gmm_d.means_init = np.array([
            [lattice_constant / np.sqrt(2)],
            [lattice_constant],
        ])
    gmm_d.fit(d_inplane.reshape(-1, 1))

    # Sort components by mean (lower = 1NN, higher = 2NN)
    order = np.argsort(gmm_d.means_.ravel())
    means = gmm_d.means_.ravel()[order]
    stds = np.sqrt(gmm_d.covariances_.ravel()[order])

    nn1_center = float(means[0])
    nn2_center = float(means[1])

    # Acceptance windows: tighter of mean±2σ and GMM crossover
    crossover = (nn1_center * stds[1] + nn2_center * stds[0]) / (stds[0] + stds[1])
    nn1_range = (float(nn1_center - 2 * stds[0]), float(min(nn1_center + 2 * stds[0], crossover)))
    nn2_range = (float(max(nn2_center - 2 * stds[1], crossover)), float(nn2_center + 2 * stds[1]))

    # Bhattacharyya distance (separation quality)
    mean_diff = nn2_center - nn1_center
    var_avg = (stds[0]**2 + stds[1]**2) / 2
    bhatt = 0.25 * mean_diff**2 / var_avg + 0.5 * np.log(var_avg / (stds[0] * stds[1]))

    # ── dz GMM (in-plane vs interlayer) ──
    abs_dz = np.abs(dz_values)
    abs_dz = abs_dz[abs_dz > 0.01]  # exclude zero-displacement noise

    gmm_dz = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm_dz.fit(abs_dz.reshape(-1, 1))

    dz_order = np.argsort(gmm_dz.means_.ravel())
    dz_means = gmm_dz.means_.ravel()[dz_order]
    dz_stds = np.sqrt(gmm_dz.covariances_.ravel()[dz_order])
    layer_change_threshold = float(
        (dz_means[0] * dz_stds[1] + dz_means[1] * dz_stds[0]) / (dz_stds[0] + dz_stds[1])
    )

    return {
        "nn1_center": nn1_center,
        "nn2_center": nn2_center,
        "nn1_range": nn1_range,
        "nn2_range": nn2_range,
        "layer_change_threshold": layer_change_threshold,
        "confidence": float(bhatt),
        "gmm_displacement": gmm_d,
        "gmm_dz": gmm_dz,
    }


def classify_event_adaptive(row, thresholds):
    """Classify using adaptive thresholds from :func:`compute_thresholds_gmm`.

    Same decision-tree logic as :func:`classify_event_mechanism` but
    replaces hardcoded ``NN1_DIST ± DIST_TOLERANCE`` with GMM-derived
    acceptance windows.

    Parameters
    ----------
    row : pandas.Series
        A row from the reference_table DataFrame.
    thresholds : dict
        Output of :func:`compute_thresholds_gmm`.

    Returns
    -------
    dict — same keys as :func:`classify_event_mechanism`.
    """
    initial_pos = row['initial_positions']
    final_pos = row['final_positions']
    move_atom_idx = row['move_atom_idx']

    disp = final_pos - initial_pos
    disp_mag = np.linalg.norm(disp, axis=1)
    moved_mask = disp_mag > MOVE_THRESHOLD
    n_moved = int(np.sum(moved_mask))
    max_disp = float(np.max(disp_mag)) if len(disp_mag) > 0 else 0.0
    moved_indices = np.where(moved_mask)[0]

    mi = move_atom_idx
    move_dist = float(disp_mag[mi])
    move_dz = float(abs(disp[mi, 2]))
    move_dz_signed = float(disp[mi, 2])
    move_dxy = float(np.sqrt(disp[mi, 0]**2 + disp[mi, 1]**2))

    # Fallback for misidentified primary mover
    alt_mover = None
    nn1_lo, nn1_hi = thresholds["nn1_range"]
    if move_dist < MOVE_THRESHOLD:
        best_idx = int(np.argmax(disp_mag))
        best_dist = float(disp_mag[best_idx])
        if nn1_lo <= best_dist <= nn1_hi:
            alt_mover = best_idx
            mi = best_idx
            move_dist = best_dist
            move_dz = float(abs(disp[mi, 2]))
            move_dz_signed = float(disp[mi, 2])
            move_dxy = float(np.sqrt(disp[mi, 0]**2 + disp[mi, 1]**2))
        elif best_dist < MOVE_THRESHOLD:
            return {
                "event_type": TYPE_RELAXATION,
                "is_on_lattice": False,
                "n_moved": n_moved,
                "move_dist": float(disp_mag[move_atom_idx]),
                "move_dz": float(abs(disp[move_atom_idx, 2])),
                "move_dz_signed": float(disp[move_atom_idx, 2]),
                "move_dxy": float(np.sqrt(disp[move_atom_idx, 0]**2
                                          + disp[move_atom_idx, 1]**2)),
                "max_displacement": max_disp,
                "moved_indices": moved_indices,
                "layer_change": False,
                "alt_mover": None,
            }

    layer_thr = thresholds.get("layer_change_threshold", 1.0)
    layer_change = move_dz > layer_thr
    nn2_lo, nn2_hi = thresholds["nn2_range"]

    if not layer_change and nn1_lo <= move_dist <= nn1_hi:
        if n_moved >= 3:
            event_type = TYPE_1NN_TRIPLE
        elif n_moved == 2:
            event_type = TYPE_1NN_COOP
        else:
            event_type = TYPE_1NN
    elif not layer_change and nn2_lo <= move_dist <= nn2_hi:
        event_type = TYPE_2NN
    elif layer_change and n_moved >= 2 and move_dist < nn2_hi:
        event_type = TYPE_EXCHANGE_UP if move_dz_signed > 0 else TYPE_EXCHANGE_DOWN
    elif layer_change and move_dist >= nn2_hi:
        event_type = TYPE_EXCHANGE_LAT
    elif layer_change and n_moved == 1:
        event_type = TYPE_INTERLAYER_HOP_UP if move_dz_signed > 0 else TYPE_INTERLAYER_HOP_DOWN
    elif n_moved >= 3:
        event_type = TYPE_CONCERTED
    elif not layer_change:
        event_type = TYPE_INPLANE_COMPLEX
    else:
        event_type = TYPE_COMPLEX

    return {
        "event_type": event_type,
        "is_on_lattice": event_type in LATTICE_TYPES,
        "n_moved": n_moved,
        "move_dist": move_dist,
        "move_dz": move_dz,
        "move_dz_signed": move_dz_signed,
        "move_dxy": move_dxy,
        "max_displacement": max_disp,
        "moved_indices": moved_indices,
        "layer_change": layer_change,
        "alt_mover": alt_mover,
    }


# ── Z-layer (surface vs subsurface) reclassification ─────────────

def _layer_from_top(z_atom, layers, tol=0.5):
    """Map a single z coordinate to its layer index (0 = top)."""
    for i, (z_mean, _) in enumerate(reversed(layers)):
        if abs(z_mean - z_atom) < tol:
            return i
    dists = [abs(zm - z_atom) for zm, _ in reversed(layers)]
    return int(np.argmin(dists))


def get_surface_layer(positions, atom_idx, tol=0.5):
    """Return the layer index of ``atom_idx`` (0 = top surface layer).

    Uses :func:`group_z_layers` to cluster the configuration into layers,
    then returns the layer index counted from the top. Atoms that don't
    fall cleanly into a cluster are assigned to the nearest layer.
    """
    layers = group_z_layers(positions, tol=tol)  # ascending in z
    return _layer_from_top(positions[atom_idx, 2], layers, tol=tol)


def get_event_surface_layers(initial_positions, final_positions, atom_idx,
                             tol=0.5):
    """Layer indices (from top) of the mover at both endpoints.

    Returns ``(layer_initial, layer_final)``. Layer clustering uses the
    initial configuration's atomic positions — the slab geometry doesn't
    migrate between initial and final, only the moving atom does, so the
    initial clustering is stable and both endpoints are placed against
    the same layer basis.
    """
    layers = group_z_layers(initial_positions, tol=tol)
    li = _layer_from_top(initial_positions[atom_idx, 2], layers, tol=tol)
    lf = _layer_from_top(final_positions[atom_idx, 2], layers, tol=tol)
    return li, lf


def reclassify_by_zlayer(event_type, layer_initial, layer_final):
    """Relabel an event based on the mover's endpoints.

    An event is considered *subsurface* only when the moving atom neither
    starts nor ends in the top layer. A surface exchange where a
    subsurface atom promotes to the surface (and a surface atom drops)
    has ``layer_initial=1, layer_final=0`` — that participates in the
    surface and keeps its original label.
    """
    if layer_initial == 0 or layer_final == 0:
        return event_type
    if event_type in (TYPE_EXCHANGE, TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN,
                      TYPE_EXCHANGE_LAT):
        return TYPE_SUBSURFACE_MIGRATION
    if event_type in (TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN):
        return TYPE_SUBSURFACE_HOP
    if event_type in (TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN):
        return TYPE_SUBSURFACE_1NN_HOP
    return TYPE_SUBSURFACE_OTHER


# ── Saddle-assisted enhanced classification ──────────────────────

def classify_event_enhanced(row, thresholds=None, use_saddle=True,
                            saddle_exchange_threshold=1.15, boundary_width=0.2):
    """Enhanced classifier combining displacement and saddle-point features.

    Runs the standard displacement-based decision tree then, for events near
    a threshold boundary, consults saddle-point geometry to resolve ambiguity.

    Parameters
    ----------
    row : pandas.Series
        A row from the reference_table DataFrame.
    thresholds : dict or None
        Output of :func:`compute_thresholds_gmm`.  If *None*, uses the
        hardcoded constants.
    use_saddle : bool, default True
        Whether to consult saddle features for boundary events.
    saddle_exchange_threshold : float, default 1.15
        ``saddle_height_ratio`` below this is classified as exchange-like.
    boundary_width : float, default 0.2
        Events within this distance (Å) of a threshold boundary are
        considered ambiguous and eligible for saddle-based reclassification.

    Returns
    -------
    dict
        All keys from :func:`classify_event_mechanism`, plus:
        - saddle feature keys (from :func:`compute_saddle_features`)
        - ``reclassified`` : bool — True if the saddle override fired
        - ``original_event_type`` : str or None — type before override
        - ``likely_reverse_relaxation`` : bool — True if near-zero barrier
          interlayer_hop_down was reclassified as relaxation
    """
    initial_pos = row['initial_positions']
    final_pos = row['final_positions']
    saddle_pos = row.get('saddle_positions', initial_pos)
    move_idx = row['move_atom_idx']

    # Step 1: displacement-based classification
    if thresholds is not None:
        result = classify_event_adaptive(row, thresholds)
        nn1_lo, nn1_hi = thresholds["nn1_range"]
        nn2_lo, nn2_hi = thresholds["nn2_range"]
    else:
        result = classify_event_mechanism(
            initial_pos, final_pos, saddle_pos, move_idx,
        )
        nn1_lo = NN1_DIST - DIST_TOLERANCE
        nn1_hi = NN1_DIST + DIST_TOLERANCE
        nn2_lo = NN2_DIST - DIST_TOLERANCE
        nn2_hi = NN2_DIST + DIST_TOLERANCE

    # Step 2: compute saddle features
    sf = compute_saddle_features(initial_pos, saddle_pos, final_pos, move_idx)
    result.update(sf)
    result["reclassified"] = False
    result["original_event_type"] = None
    result["likely_reverse_relaxation"] = False

    if not use_saddle or np.isnan(sf.get("saddle_height_ratio", np.nan)):
        return result

    # Step 3: check if event is near a threshold boundary
    move_dist = result["move_dist"]
    near_nn1_boundary = (
        abs(move_dist - nn1_lo) < boundary_width
        or abs(move_dist - nn1_hi) < boundary_width
    )
    near_nn2_boundary = (
        abs(move_dist - nn2_lo) < boundary_width
        or abs(move_dist - nn2_hi) < boundary_width
    )

    if near_nn1_boundary or near_nn2_boundary:
        height_ratio = sf["saddle_height_ratio"]
        original_type = result["event_type"]

        if height_ratio < saddle_exchange_threshold:
            # Exchange-like saddle geometry
            if not result["layer_change"]:
                # Was classified as in-plane but looks like exchange
                pass  # don't reclassify in-plane events as exchange without layer change
            elif result["event_type"] in {TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN}:
                # Was classified as hop but saddle says exchange
                new_type = TYPE_EXCHANGE_UP if result.get("move_dz_signed", 0) > 0 else TYPE_EXCHANGE_DOWN
                result["original_event_type"] = original_type
                result["event_type"] = new_type
                result["is_on_lattice"] = new_type in LATTICE_TYPES
                result["reclassified"] = True
        else:
            # Hop-like saddle geometry
            if result["layer_change"] and result["event_type"] in {TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN}:
                # Was classified as exchange but saddle says hop
                if abs(move_dist - NN1_DIST) < abs(move_dist - NN2_DIST):
                    new_type = TYPE_1NN
                else:
                    new_type = TYPE_2NN
                result["original_event_type"] = original_type
                result["event_type"] = new_type
                result["is_on_lattice"] = new_type in LATTICE_TYPES
                result["reclassified"] = True

    # Step 4: near-zero barrier interlayer_hop_down → relaxation
    energy_barrier = row.get('energy_barrier', np.inf)
    if (result["event_type"] == TYPE_INTERLAYER_HOP_DOWN
            and energy_barrier < 0.05
            and sf["saddle_disp_mag"] < MOVE_THRESHOLD):
        result["original_event_type"] = result["event_type"]
        result["event_type"] = TYPE_RELAXATION
        result["is_on_lattice"] = False
        result["reclassified"] = True
        result["likely_reverse_relaxation"] = True

    return result


# ── Lattice-level event mapping ─────────────────────────────────────────────

def build_surface_neighbour_list_legacy(ideal_sites, cell, nn_dist, z_tol=0.5):
    """LEGACY (surface-only). Build a NN adjacency list for the top-layer lattice sites.

    For (100) FCC surface, each site has 4 in-plane NN at distance a/√2.
    Also identifies second-layer sites for interlayer hops.

    The 3D-first path uses `compute_3d_site_graph` from `lattice_3d.py`.
    This legacy function is kept for backward compatibility with existing
    notebooks and callers.
    """
    Lx, Ly = cell[0], cell[1]
    layers = group_z_layers(ideal_sites)
    top_z = max(z for z, _ in layers)

    # Identify surface-layer and subsurface-layer site indices
    surface_sites = []
    subsurface_sites = []
    for z_mean, idx in layers:
        if abs(z_mean - top_z) < z_tol:
            surface_sites.extend(idx)
        elif abs(z_mean - (top_z - nn_dist * np.sqrt(2) / 2)) < z_tol:
            # Second layer is a/2 below top layer for (100)
            subsurface_sites.extend(idx)

    surface_set = set(surface_sites)
    subsurface_set = set(subsurface_sites)

    # Build adjacency for surface sites (PBC in x,y)
    offsets = np.array([[dx * Lx, dy * Ly, 0.0]
                        for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    site_rep = np.vstack([ideal_sites + o for o in offsets])
    n_sites = len(ideal_sites)

    neighbours = {i: [] for i in surface_sites}         # all NN
    inplane_neighbours = {i: [] for i in surface_sites}  # same-layer only
    interlayer_neighbours = {i: [] for i in surface_sites}  # cross-layer
    cutoff = nn_dist * 1.1  # slight tolerance

    for i in surface_sites:
        pos_i = ideal_sites[i]
        for img_idx in range(len(site_rep)):
            j = img_idx % n_sites
            if j == i:
                continue
            d = np.linalg.norm(site_rep[img_idx] - pos_i)
            if d < cutoff:
                if j not in neighbours[i]:
                    neighbours[i].append(j)
                    if j in surface_set:
                        inplane_neighbours[i].append(j)
                    elif j in subsurface_set:
                        interlayer_neighbours[i].append(j)

    return (neighbours, inplane_neighbours, interlayer_neighbours,
            sorted(surface_sites), sorted(subsurface_sites))


# Backward-compat alias. The 3D-first path uses compute_3d_site_graph
# (see lattice_3d.py). Existing notebooks and classify_lattice_events_parts/
# still import the old name.
build_surface_neighbour_list = build_surface_neighbour_list_legacy


def map_event_to_lattice(event_row, ideal_sites, cell, nn_dist, n_vac,
                         surface_sites_set, neighbours, inplane_neighbours=None):
    """Map a single reference_table event to lattice-level vacancy transition.

    For the event's local cluster (initial_positions → final_positions), we:
    1. Project both initial and final configs onto the ideal lattice
    2. Identify which site(s) become vacant / occupied
    3. Determine the vacancy hop: site_from → site_to
    4. Characterize the local environment (occupied/vacant NN sites)

    The local environment counts returned here are approximate because they are
    inferred from the event's local cluster rather than the full simulation
    state. For authoritative environment-dependent rate extraction, use the
    step-based global mapping in build_rate_table.py.

    Returns dict with lattice mapping info, or None if unmappable.
    """
    init_pos = event_row['initial_positions']
    final_pos = event_row['final_positions']
    barrier = event_row['energy_barrier']
    move_idx = event_row['move_atom_idx']
    n_local = len(init_pos)

    # Classify the mechanism first
    mech = classify_event_mechanism(
        init_pos, final_pos,
        event_row.get('saddle_positions', init_pos),
        move_idx,
        nn_dist=nn_dist,
    )

    Lx, Ly = cell[0], cell[1]
    offsets = np.array([[dx * Lx, dy * Ly, 0.0]
                        for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    n_sites = len(ideal_sites)
    site_rep = np.vstack([ideal_sites + o for o in offsets])
    site_tree = cKDTree(site_rep)

    # Map initial and final atom positions to nearest lattice sites
    init_wrapped = init_pos.copy()
    init_wrapped[:, 0] %= Lx
    init_wrapped[:, 1] %= Ly
    final_wrapped = final_pos.copy()
    final_wrapped[:, 0] %= Lx
    final_wrapped[:, 1] %= Ly

    _, init_site_idx = site_tree.query(init_wrapped)
    init_site_idx = init_site_idx % n_sites
    _, final_site_idx = site_tree.query(final_wrapped)
    final_site_idx = final_site_idx % n_sites

    # Sites occupied before and after
    sites_before = set(init_site_idx)
    sites_after = set(final_site_idx)

    # Vacancy = site that was in "before" but atoms moved away from it
    # (or equivalently, sites that become unoccupied)
    # Since the local cluster doesn't include ALL atoms, we focus on the
    # moving atom's site changes
    move_idx = event_row['move_atom_idx']

    site_from_atom = init_site_idx[move_idx]   # site the moving atom was at
    site_to_atom = final_site_idx[move_idx]     # site the moving atom goes to

    # For a vacancy mechanism: atom moves FROM site_from_atom TO site_to_atom
    # This means the vacancy moves FROM site_to_atom TO site_from_atom
    # (vacancy and atom swap)
    vac_site_from = site_to_atom    # vacancy was here (atom lands here)
    vac_site_to = site_from_atom    # vacancy ends up here (atom left here)

    # Wait — let's think more carefully. In a vacancy hop:
    #   Before: vacancy at site V, atom at site A (neighbour)
    #   After:  atom moves to site V, vacancy now at site A
    # So: atom goes from A → V, vacancy goes from V → A
    # move_atom moves from site_from_atom → site_to_atom
    # Therefore: vacancy was at site_to_atom, moves to site_from_atom
    vac_site_from = site_to_atom
    vac_site_to = site_from_atom

    # Check if this is a surface-layer hop
    is_surface_hop = (vac_site_from in surface_sites_set and
                      vac_site_to in surface_sites_set)
    is_nn_hop_on_lattice = vac_site_from in neighbours and \
                           vac_site_to in neighbours.get(vac_site_from, [])

    # Local environment: how many NN of the vacancy origin are also vacant?
    # This is only a local-cluster approximation. The catalogue event does not
    # contain the full global configuration, so any "vacant neighbour" count
    # here is diagnostic and should not be treated as the authoritative rate
    # lookup input for the lattice model.
    n_vacant_nn = 0
    nn_of_vac = neighbours.get(vac_site_from, [])
    occupied_init = set(init_site_idx)
    for nn_site in nn_of_vac:
        if nn_site not in occupied_init:
            n_vacant_nn += 1

    # In-plane vacant NN (for 2D lattice model)
    n_vacant_inplane_nn = 0
    if inplane_neighbours is not None:
        ip_nn_of_vac = inplane_neighbours.get(vac_site_from, [])
        for nn_site in ip_nn_of_vac:
            if nn_site not in occupied_init:
                n_vacant_inplane_nn += 1

    # Distance between vacancy sites (should be ~NN_DIST for NN hop)
    vac_disp = ideal_sites[vac_site_to] - ideal_sites[vac_site_from]
    vac_disp[0] -= Lx * round(vac_disp[0] / Lx)
    vac_disp[1] -= Ly * round(vac_disp[1] / Ly)
    vac_hop_dist = np.linalg.norm(vac_disp)

    return {
        "event_type": mech["event_type"],
        "is_on_lattice": mech["is_on_lattice"],
        "n_moved": mech["n_moved"],
        "max_displacement": mech["max_displacement"],
        "barrier_eV": barrier,
        "vac_site_from": int(vac_site_from),
        "vac_site_to": int(vac_site_to),
        "vac_hop_dist": float(vac_hop_dist),
        "is_surface_hop": is_surface_hop,
        "is_nn_lattice_hop": is_nn_hop_on_lattice,
        "n_vacant_nn_at_origin": n_vacant_nn,
        "n_vacant_inplane_nn": n_vacant_inplane_nn,
        "n_local_atoms": n_local,
        "move_atom_idx": move_idx,
    }


# ── Rate lookup table builder ────────────────────────────────────────────────

def build_rate_lookup(mapped_events_df, temperature):
    """Build an approximate local-cluster rate lookup table from mapped events.

    Groups 1NN/2NN hops by local vacancy configuration (n_vacant_inplane_nn)
    and computes mean/min/max barriers and Arrhenius rates.

    Uses in-plane vacant NN count as the environment descriptor since this
    is what a 2D surface lattice KMC would see.

    This helper is for per-simulation diagnostics only. The authoritative
    cross-simulation rate table is built by build_rate_table.py from the full
    global trajectory state at each selected step.
    """
    kB = 8.617333e-5  # eV/K
    nu0 = 1e13        # attempt frequency (Hz), typical for metals

    nn_hops = mapped_events_df[
        mapped_events_df["is_on_lattice"] &
        mapped_events_df["is_surface_hop"]
    ].copy()

    if len(nn_hops) == 0:
        return pd.DataFrame()

    groups = nn_hops.groupby("n_vacant_inplane_nn")

    rows = []
    for n_vnn, grp in groups:
        barriers = grp["barrier_eV"]
        mean_ea = barriers.mean()
        rate = nu0 * np.exp(-mean_ea / (kB * temperature))
        rows.append({
            "n_vacant_inplane_nn": int(n_vnn),
            "n_events": len(grp),
            "Ea_mean_eV": float(mean_ea),
            "Ea_min_eV": float(barriers.min()),
            "Ea_max_eV": float(barriers.max()),
            "Ea_std_eV": float(barriers.std()) if len(grp) > 1 else 0.0,
            "rate_Hz": float(rate),
            "log10_rate": float(np.log10(rate)) if rate > 0 else -np.inf,
        })

    return pd.DataFrame(rows).sort_values("n_vacant_inplane_nn").reset_index(drop=True)


# ── Step log parser ──────────────────────────────────────────────────────────

def parse_pykmc_out(path):
    """Parse pykmc.out for step metadata."""
    steps = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit() and int(parts[0]) > 0:
                steps.append({
                    'step': int(parts[0]),
                    'dt': float(parts[1]),
                    'time': float(parts[2]),
                    'ref_event': int(parts[3]),
                    'Ea': float(parts[4]),
                    'k_evt': float(parts[5]),
                    'k_tot': float(parts[6]),
                    'energy': float(parts[7]),
                })
    return pd.DataFrame(steps)


# ── Kinetic coverage analysis ────────────────────────────────────────────────

def compute_kinetic_coverage(df_steps, mapped_events_df, include_dealloying=False):
    """Determine what fraction of selected steps map to on-lattice NN hops.

    Cross-references the step log (which ref_event was selected at each step)
    with the event mapping (which events are on-lattice NN hops).

    Parameters
    ----------
    df_steps : DataFrame
        Step log from :func:`parse_pykmc_out`.
    mapped_events_df : DataFrame
        Mapped events with ``event_type``, ``is_on_lattice``, ``is_surface_hop``.
    include_dealloying : bool, default False
        If True, also compute coverage for :data:`DEALLOYING_TYPES`.
    """
    if len(df_steps) == 0 or len(mapped_events_df) == 0:
        return {}

    # Build set of on-lattice event indices (1NN + 2NN surface hops)
    on_lattice_events = set(
        mapped_events_df[
            mapped_events_df["is_on_lattice"] &
            mapped_events_df["is_surface_hop"]
        ].index
    )

    # 1NN hops specifically (subset of on-lattice)
    nn_hop_events = set(
        mapped_events_df[
            mapped_events_df["event_type"].isin(LATTICE_TYPES)
        ].index
    )

    total_steps = len(df_steps)
    total_time = df_steps['time'].iloc[-1] - df_steps['time'].iloc[0] + df_steps['dt'].iloc[0]

    # Count steps where selected event is on-lattice
    df_steps = df_steps.copy()
    df_steps['is_on_lattice_surface'] = df_steps['ref_event'].isin(on_lattice_events)
    df_steps['is_nn_hop'] = df_steps['ref_event'].isin(nn_hop_events)

    n_on_lattice = df_steps['is_on_lattice_surface'].sum()
    n_nn_hop = df_steps['is_nn_hop'].sum()
    time_on_lattice = df_steps.loc[df_steps['is_on_lattice_surface'], 'dt'].sum()
    time_nn_hop = df_steps.loc[df_steps['is_nn_hop'], 'dt'].sum()

    result = {
        'total_steps': total_steps,
        'total_sim_time': total_time,
        'n_steps_on_lattice_surface': int(n_on_lattice),
        'pct_steps_on_lattice_surface': 100 * n_on_lattice / total_steps,
        'n_steps_nn_hop': int(n_nn_hop),
        'pct_steps_nn_hop': 100 * n_nn_hop / total_steps,
        'pct_time_on_lattice_surface': 100 * time_on_lattice / total_time if total_time > 0 else 0,
        'pct_time_nn_hop': 100 * time_nn_hop / total_time if total_time > 0 else 0,
        'per_event_step_counts': df_steps.groupby('ref_event').agg(
            n_selected=('step', 'count'),
            total_dt=('dt', 'sum'),
            mean_Ea=('Ea', 'mean'),
        ).to_dict('index'),
    }

    if include_dealloying:
        dealloying_events = set(
            mapped_events_df[
                mapped_events_df["event_type"].isin(DEALLOYING_TYPES)
            ].index
        )
        df_steps['is_dealloying'] = df_steps['ref_event'].isin(dealloying_events)
        n_dealloying = df_steps['is_dealloying'].sum()
        time_dealloying = df_steps.loc[df_steps['is_dealloying'], 'dt'].sum()
        result['n_steps_dealloying'] = int(n_dealloying)
        result['pct_steps_dealloying'] = 100 * n_dealloying / total_steps
        result['pct_time_dealloying'] = 100 * time_dealloying / total_time if total_time > 0 else 0

    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_diagnostic_plots(mapped_df, lookup_df, coverage, out_dir, nn_dist):
    """Generate diagnostic plots for the lattice mapping."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    type_colors = {
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
        TYPE_COMPLEX: '#999999',
    }

    # 1. Barrier distribution by type
    ax = axes[0, 0]
    for etype in [TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN,
                  TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN, TYPE_EXCHANGE_LAT,
                  TYPE_CONCERTED, TYPE_INTERLAYER_HOP_UP,
                  TYPE_INTERLAYER_HOP_DOWN, TYPE_INPLANE_COMPLEX, TYPE_COMPLEX]:
        sub = mapped_df[mapped_df['event_type'] == etype]
        if len(sub) > 0:
            ax.hist(sub['barrier_eV'], bins=30, alpha=0.6,
                    color=type_colors.get(etype, '#999'),
                    label=f'{etype} ({len(sub)})')
    ax.set_xlabel('Barrier (eV)')
    ax.set_ylabel('Count')
    ax.set_title('Barrier distribution by event type')
    ax.legend(fontsize=8)

    # 2. Vacancy hop distance distribution
    ax = axes[0, 1]
    ax.hist(mapped_df['vac_hop_dist'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(nn_dist, color='red', ls='--', lw=2, label=f'1NN = {nn_dist:.3f} Å')
    nn2 = nn_dist * np.sqrt(2)
    ax.axvline(nn2, color='blue', ls=':', lw=2, label=f'2NN = {nn2:.3f} Å')
    ax.set_xlabel('Vacancy hop distance (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Lattice-projected vacancy hop distance')
    ax.legend()

    # 3. Barrier vs local environment (for NN hops)
    ax = axes[1, 0]
    nn_hops = mapped_df[mapped_df['is_on_lattice'] & mapped_df['is_surface_hop']]
    if len(nn_hops) > 0:
        data_by_env = []
        env_labels = []
        for n_vnn in sorted(nn_hops['n_vacant_inplane_nn'].unique()):
            sub = nn_hops[nn_hops['n_vacant_inplane_nn'] == n_vnn]
            data_by_env.append(sub['barrier_eV'].values)
            env_labels.append(f'{n_vnn} vac NN\n(n={len(sub)})')
        if data_by_env:
            bp = ax.boxplot(data_by_env, tick_labels=env_labels, patch_artist=True,
                            showfliers=True, whis=[5, 95])
            for patch in bp['boxes']:
                patch.set_facecolor('#4DAF4A')
                patch.set_alpha(0.7)
    ax.set_xlabel('Local vacancy environment')
    ax.set_ylabel('Barrier (eV)')
    ax.set_title('NN hop barriers by local environment')

    # 4. Kinetic coverage pie chart
    ax = axes[1, 1]
    if coverage:
        n_on = coverage['n_steps_on_lattice_surface']
        n_off = coverage['total_steps'] - n_on
        if n_on + n_off > 0:
            ax.pie([n_on, n_off],
                   labels=[f'On-lattice\n({n_on} steps)', f'Off-lattice\n({n_off} steps)'],
                   colors=['#4DAF4A', '#E41A1C'], autopct='%1.1f%%',
                   startangle=90)
            ax.set_title('Kinetic coverage (steps)')
    else:
        ax.text(0.5, 0.5, 'No step data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Kinetic coverage')

    plt.tight_layout()
    plt.savefig(out_dir / 'lattice_mapping_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved lattice_mapping_plots.png')


# ── Main ─────────────────────────────────────────────────────────────────────

def main(sim_dir, required_potential=REQUIRED_POTENTIAL, n_vac=None):
    sim_dir = Path(sim_dir).resolve()
    out_dir = sim_dir / 'analysis'
    out_dir.mkdir(exist_ok=True)

    print(f'{"=" * 70}')
    print(f'Lattice Mapping: {sim_dir}')
    print(f'{"=" * 70}')

    # ── Parse simulation config ──────────────────────────────────────────
    sim_config = read_sim_config(sim_dir)

    if not check_potential(sim_config, required_potential):
        print(f'  SKIPPED: potential is "{sim_config["potential"]}", '
              f'requires {required_potential}')
        return

    temperature = sim_config['temperature']
    if n_vac is None:
        n_vac = sim_config['n_vac']
        if parse_run_path(Path(sim_dir)) is None:
            print('  WARNING: run is not in {composition}/{N}vac/T{temp}/{variant}; '
                  f'assuming {n_vac} vacancy (pass --n-vac)')

    if temperature is None:
        print('  ERROR: could not read temperature from input.in')
        sys.exit(1)

    print(f'  Potential:    {sim_config["potential"]}')
    print(f'  Temperature:  {temperature} K')
    print(f'  N vacancies:  {n_vac}')

    # ── Load reference frame and build lattice ───────────────────────────
    # Use ASE if available, otherwise fall back to manual xyz reader
    try:
        from ase.io import read as ase_read
        ref_frame = ase_read(sim_dir / 'trajkmc.xyz', index=0)
        positions = ref_frame.get_positions()
        cell_diag = [ref_frame.cell[i, i] for i in range(3)]
    except Exception:
        # Manual xyz reader fallback
        with open(sim_dir / 'trajkmc.xyz') as f:
            n_atoms_line = int(f.readline().strip())
            header = f.readline().strip()
            positions = []
            for _ in range(n_atoms_line):
                parts = f.readline().split()
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            positions = np.array(positions)
            # Parse cell from header
            import re as _re
            cell_match = _re.search(r'Lattice="([^"]+)"', header)
            if cell_match:
                vals = list(map(float, cell_match.group(1).split()))
                cell_diag = [vals[0], vals[4], vals[8]]
            else:
                cell_diag = [positions[:, i].ptp() * 1.1 for i in range(3)]

    Lx, Ly, Lz = cell_diag
    cell = [Lx, Ly, Lz]
    n_atoms = len(positions)
    n_sites = n_atoms + n_vac

    # Detect NN distance
    tree = cKDTree(positions)
    dists_nn, _ = tree.query(positions, k=2)
    nn_dist = np.median(dists_nn[:, 1])
    a_lat = nn_dist * np.sqrt(2)

    print(f'  N atoms:      {n_atoms}')
    print(f'  N sites:      {n_sites}')
    print(f'  Cell:         {Lx:.3f} × {Ly:.3f} × {Lz:.3f} Å')
    print(f'  a_lat:        {a_lat:.4f} Å  (NN = {nn_dist:.4f} Å)')

    # Generate ideal lattice sites
    ideal_sites = generate_ideal_sites(positions, cell, n_vac)
    print(f'  Ideal sites:  {len(ideal_sites)} (expected {n_sites})')

    # Build surface neighbour list (legacy 2D path).
    (neighbours, inplane_neighbours, interlayer_neighbours,
     surface_sites, subsurface_sites) = build_surface_neighbour_list_legacy(
        ideal_sites, cell, nn_dist
    )
    surface_set = set(surface_sites)
    print(f'  Surface sites: {len(surface_sites)}, subsurface: {len(subsurface_sites)}')
    nn_counts = [len(neighbours[s]) for s in surface_sites if s in neighbours]
    ip_counts = [len(inplane_neighbours[s]) for s in surface_sites]
    il_counts = [len(interlayer_neighbours[s]) for s in surface_sites]
    if nn_counts:
        print(f'  NN per surface site: {np.mean(nn_counts):.1f} total '
              f'({np.mean(ip_counts):.1f} in-plane + '
              f'{np.mean(il_counts):.1f} interlayer)')

    # ── Load event catalogue ─────────────────────────────────────────────
    ref_table_path = sim_dir / 'reference_table.pickle'
    if not ref_table_path.exists():
        print('\n  ERROR: reference_table.pickle not found. Cannot map events.')
        sys.exit(1)

    catalog = pd.read_pickle(ref_table_path)
    n_events = len(catalog)
    print(f'\n  Event catalogue: {n_events} events')

    # ── Map each event to lattice ────────────────────────────────────────
    print('  Mapping events to lattice...')
    mapped_rows = []
    for i, row in catalog.iterrows():
        result = map_event_to_lattice(
            row, ideal_sites, cell, nn_dist, n_vac,
            surface_set, neighbours, inplane_neighbours
        )
        if result is not None:
            result['event_idx'] = i
            mapped_rows.append(result)

    mapped_df = pd.DataFrame(mapped_rows).set_index('event_idx')
    print(f'  Mapped {len(mapped_df)} / {n_events} events')

    # Summary by type
    print(f'\n  Event type breakdown:')
    for etype in [TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN,
                  TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN, TYPE_EXCHANGE_LAT,
                  TYPE_CONCERTED, TYPE_INTERLAYER_HOP_UP,
                  TYPE_INTERLAYER_HOP_DOWN, TYPE_INPLANE_COMPLEX, TYPE_COMPLEX]:
        sub = mapped_df[mapped_df['event_type'] == etype]
        n = len(sub)
        if n > 0:
            surf = sub['is_surface_hop'].sum()
            mean_ea = sub['barrier_eV'].mean()
            on_lat = ' [ON-LATTICE]' if etype in LATTICE_TYPES else ''
            print(f'    {etype:25s}: {n:4d} events, {surf:4d} surface, '
                  f'Ea = {sub["barrier_eV"].min():.3f}–{sub["barrier_eV"].max():.3f} eV '
                  f'(mean {mean_ea:.3f}){on_lat}')

    # ── Build rate lookup table ──────────────────────────────────────────
    lookup_df = build_rate_lookup(mapped_df, temperature)
    print(f'\n  Rate lookup table ({len(lookup_df)} environment bins):')
    if len(lookup_df) > 0:
        for _, r in lookup_df.iterrows():
            print(f'    {int(r["n_vacant_inplane_nn"])} vacant in-plane NN: '
                  f'Ea = {r["Ea_mean_eV"]:.4f} ± {r["Ea_std_eV"]:.4f} eV, '
                  f'rate = {r["rate_Hz"]:.3e} Hz  '
                  f'({int(r["n_events"])} events)')

    # ── Parse step log and compute kinetic coverage ──────────────────────
    coverage = {}
    pykmc_path = sim_dir / 'pykmc.out'
    if pykmc_path.exists():
        df_steps = parse_pykmc_out(pykmc_path)
        if len(df_steps) > 0:
            coverage = compute_kinetic_coverage(df_steps, mapped_df)
            print(f'\n  Kinetic coverage ({coverage["total_steps"]} steps, '
                  f'{coverage["total_sim_time"]:.3e} s):')
            print(f'    Steps captured by surface NN hops: '
                  f'{coverage["n_steps_on_lattice_surface"]} / '
                  f'{coverage["total_steps"]} '
                  f'({coverage["pct_steps_on_lattice_surface"]:.1f}%)')
            print(f'    Sim time captured:                 '
                  f'{coverage["pct_time_on_lattice_surface"]:.1f}%')

            # Per-event breakdown of selected events
            print(f'\n  Most-selected events:')
            evt_counts = coverage['per_event_step_counts']
            sorted_evts = sorted(evt_counts.items(),
                                 key=lambda x: x[1]['n_selected'], reverse=True)
            for evt_id, info in sorted_evts[:15]:
                n_sel = info['n_selected']
                pct = 100 * n_sel / coverage['total_steps']
                ea = info['mean_Ea']
                if evt_id in mapped_df.index:
                    etype = mapped_df.loc[evt_id, 'event_type']
                    on_lat = 'ON ' if mapped_df.loc[evt_id, 'is_on_lattice'] else 'OFF'
                else:
                    etype = '???'
                    on_lat = '???'
                print(f'      Event {evt_id:4d}: {n_sel:5d} steps ({pct:5.1f}%), '
                      f'Ea={ea:.4f} eV, {on_lat} lattice, {etype}')
        else:
            print('\n  No KMC steps in pykmc.out')
    else:
        print('\n  pykmc.out not found — skipping kinetic coverage')

    # ── Save outputs ─────────────────────────────────────────────────────
    mapped_df.to_csv(out_dir / 'lattice_mapping.csv')
    print(f'\n  Saved lattice_mapping.csv ({len(mapped_df)} events)')

    if len(lookup_df) > 0:
        lookup_df.to_csv(out_dir / 'rate_lookup_table.csv', index=False)
        print(f'  Saved rate_lookup_table.csv ({len(lookup_df)} bins)')

    # Write text report
    with open(out_dir / 'kinetic_coverage.txt', 'w') as f:
        f.write(f'Lattice Mapping Report\n')
        f.write(f'{"=" * 60}\n')
        f.write(f'Simulation: {sim_dir}\n')
        f.write(f'Temperature: {temperature} K\n')
        f.write(f'Vacancies: {n_vac}\n')
        f.write(f'Events in catalogue: {n_events}\n\n')

        f.write(f'Potential: {sim_config["potential"]}\n\n')
        f.write(f'Event classification:\n')
        for etype in [TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE, TYPE_2NN,
                      TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN, TYPE_EXCHANGE_LAT,
                      TYPE_CONCERTED, TYPE_INTERLAYER_HOP_UP,
                      TYPE_INTERLAYER_HOP_DOWN, TYPE_INPLANE_COMPLEX,
                      TYPE_COMPLEX]:
            sub = mapped_df[mapped_df['event_type'] == etype]
            if len(sub) > 0:
                on_lat = ' [ON-LATTICE]' if etype in LATTICE_TYPES else ''
                f.write(f'  {etype:25s}: {len(sub):4d} '
                        f'({100*len(sub)/len(mapped_df):.1f}%){on_lat}\n')

        f.write(f'\nRate lookup table (surface NN hops by in-plane environment):\n')
        if len(lookup_df) > 0:
            for _, r in lookup_df.iterrows():
                f.write(f'  {int(r["n_vacant_inplane_nn"])} vacant in-plane NN: '
                        f'Ea = {r["Ea_mean_eV"]:.4f} ± {r["Ea_std_eV"]:.4f} eV, '
                        f'rate = {r["rate_Hz"]:.3e} Hz ({int(r["n_events"])} events)\n')

        if coverage:
            f.write(f'\nKinetic coverage:\n')
            f.write(f'  Steps on-lattice:   {coverage["pct_steps_on_lattice_surface"]:.1f}%\n')
            f.write(f'  Sim time on-lattice: {coverage["pct_time_on_lattice_surface"]:.1f}%\n')

    print(f'  Saved kinetic_coverage.txt')

    # ── Diagnostic plots ─────────────────────────────────────────────────
    make_diagnostic_plots(mapped_df, lookup_df, coverage, out_dir, nn_dist)

    print(f'\n{"=" * 70}')
    print(f'Done. Output in {out_dir}/')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Map off-lattice pyKMC events onto the FCC (100) surface lattice')
    parser.add_argument('sim_dir', nargs='?', default='.',
                        help='Path to pyKMC simulation directory (default: .)')
    parser.add_argument('--required-potential', default=REQUIRED_POTENTIAL,
                        help="Only process the run if its pair_coeff contains this "
                             "string; '' accepts any potential (default: %(default)s)")
    parser.add_argument('--n-vac', type=int, default=None,
                        help='Number of vacancies (default: from the {N}vac/ '
                             'directory, else 1)')
    args = parser.parse_args()
    main(args.sim_dir, required_potential=args.required_potential, n_vac=args.n_vac)
