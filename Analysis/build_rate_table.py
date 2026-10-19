#!/usr/bin/env python3
"""
build_rate_table.py — Build the authoritative rate lookup table for on-lattice KMC

For each run of one composition that uses the required potential (default
NiAlH), iterates over every KMC step in pykmc.out,
loads the corresponding trajectory frame, projects ALL atoms onto the ideal
FCC(100) surface lattice to identify ALL vacancy positions globally, then
for the selected event (any type in ACCEPTED_TYPES_3D) counts how many of
the hopping vacancy's 1NN/2NN neighbours are also vacant. This gives the
true local environment for each selected event.

Unlike lattice_map_events.py, this script derives vacancy environments from the
full global trajectory state at each selected step. Its outputs are the
authoritative cross-simulation step-log tables for on-lattice KMC and validation.

STEP-LOG VS CATALOGUE
---------------------
There are two parallel rate tables, both valid but representing different
questions:

  * `rate_lookup_table_3d_master.csv` (kinetic-weighted) — built from
    ``process_simulation()`` below, aggregating only events that pyKMC
    actually fired during step logging. At moderate T the 0.5 eV surface
    1NN barrier outruns 1 eV exchange/interlayer barriers by ~10⁵×, so
    this table is sparse in motifs beyond `surface_1nn_translation`. It
    reflects *what happened in the pyKMC simulations*.

  * `rate_lookup_table_3d_catalogue.csv` (mechanism-complete) — built from
    ``build_rate_table_3d_catalogue()`` below, aggregating every discovered
    saddle in ``classified_events.csv`` regardless of whether it fired.
    Uses a local-cluster environment descriptor (``n_vac_nn1_initial``).
    It reflects *what mechanisms pARTn discovered*.

Downstream on-lattice KMC consumers typically read the master table and
treat rare motifs via a preprocessor fallback into the catalogue, not by
merging tables inside this script.

DIRECTION FAMILIES
------------------
Direction classification uses the Cartesian displacement between vac_origin
and vac_dest sites (``classify_direction_xyz`` in lattice_3d.py). The old
integer-index form (``assign_direction_family``) cannot distinguish 1NN
cross-layer hops from 2NN axial exchanges because ``neighbour_directions``
only stores in-plane edges — both collapse to ``(0, 0, ±1)`` there.

CURATED FAMILY CATALOGUE
------------------------
For the human-auditable, audit-aware alternative to the step-log tables,
see pylatkmc's ingest step (``python -m pylatkmc.ingest.family_assignment``
then ``python -m pylatkmc.ingest.build_family_rate_table``, or both at once
with ``python -m pylatkmc.ingest.build_curated_catalogue``). It consumes the
classifier output directly (bypassing this global-replay path) and emits
``rate_lookup_table_family.csv`` keyed by reviewed FCC movement family plus
a family-specific environment bucket. This script's outputs are retained
as diagnostic comparisons; see the curated family catalogue section of
EVENT_CLASSIFICATION.md.

Usage (from the repository root):
    python -m Analysis.build_rate_table --data-root /path/to/data
    python -m Analysis.build_rate_table --composition 95Ni_5Fe --required-potential ''

The data root holds ``{composition}/{N}vac/T{temp}/{variant}/`` run
directories (default: ``$PYKMC_ANALYSIS_DATA_ROOT``). Outputs go to
``--out-dir`` (default: ``$PYKMC_ANALYSIS_OUTPUT_DIR``, else
``Analysis/lattice_event_classification/``).

Output (in the output directory):
    rate_lookup_table.csv
        environment-dependent surface 1NN rate table (plus the directional,
        extended, exchange, species and 3D variants)
    mapped_steps.csv, mapped_1nn_steps.csv
        step-level global vacancy mapping used to build the rate tables
    rate_lookup_plots.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Analysis import _paths
from Analysis._paths import BARRIER_CUTOFF
from Analysis.lattice_map_events import (
    A_NI,
    NN1_DIST,
    REQUIRED_POTENTIAL,
    TYPE_1NN,
    TYPE_1NN_COOP,
    TYPE_1NN_TRIPLE,
    TYPE_2NN,
    TYPE_EXCHANGE_UP,
    TYPE_EXCHANGE_DOWN,
    TYPE_EXCHANGE_LAT,
    TYPE_INTERLAYER_HOP_UP,
    TYPE_INTERLAYER_HOP_DOWN,
    coarsen_event_type,
    classify_event_from_row,
    generate_ideal_sites,
    get_event_surface_layers,
    group_z_layers,
    parse_pykmc_out,
    read_sim_config,
    reclassify_by_zlayer,
    check_potential,
)
from Analysis.lattice_3d import (
    assign_direction_family,        # legacy, kept for callers that still pass integer deltas
    classify_direction_xyz,          # preferred: Cartesian-vector classifier
    classify_3d_motif,
    min_image,
)
from Analysis.sim_discovery import parse_run_path

# Move types accepted for the rate table
ACCEPTED_TYPES = {
    TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE,
    TYPE_2NN,
    TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN,
    TYPE_EXCHANGE_LAT,
    TYPE_INTERLAYER_HOP_UP, TYPE_INTERLAYER_HOP_DOWN,
}

# 3D-master rate table accepts all lattice-projectable events, including the
# subsurface variants excluded from the legacy surface gate. Listed explicitly
# (not union-by-wildcard) so adding a new event type is an intentional choice.
ACCEPTED_TYPES_3D = ACCEPTED_TYPES | {
    "subsurface_1NN_hop",
    "subsurface_migration",
    "subsurface_hop",       # interlayer_hop_* demoted when mover is below top
    "subsurface_other",     # catch-all z-layer demotion; keeps the count honest
}

# 1NN move-type family (promoted to module scope so main() and
# build_lookup_table() both see the same definition).
_1nn_types = {TYPE_1NN, TYPE_1NN_COOP, TYPE_1NN_TRIPLE}

kB = 8.617333e-5  # eV/K
k0_Hz = 10.0e12  # attempt frequency in Hz (10 THz)


def scan_nialh_sims(root, required_potential=REQUIRED_POTENTIAL):
    """Find the run directories under one composition directory.

    ``root`` holds ``{N}vac/T{temp}/{variant}/``. Only runs whose
    ``pair_coeff`` contains ``required_potential`` are kept; an empty string
    keeps every run.
    """
    sims = []
    for variant_dir in sorted(Path(root).glob("*/*/*")):
        if not variant_dir.is_dir():
            continue
        parsed = parse_run_path(variant_dir)
        if parsed is None:
            continue
        _, nvac, temp, _ = parsed
        cfg = read_sim_config(variant_dir)
        if not check_potential(cfg, required_potential):
            continue
        sims.append({
            "nvac": nvac,
            "temp": int(temp),
            "path": variant_dir,
            "temperature": cfg["temperature"],
        })
    return sims


def build_lattice_infrastructure(ref_frame, cell_diag, nvac):
    """Build ideal lattice sites, surface set, and in-plane neighbour list.

    Done once per simulation from trajectory frame 0.
    Returns (ideal_sites, surface_set, inplane_neighbours, second_neighbours,
             neighbour_directions, site_tree, n_sites, cell, nn_dist).
    """
    positions = ref_frame.get_positions()
    Lx, Ly, Lz = cell_diag
    cell = [Lx, Ly, Lz]

    # Detect NN distance
    tree = cKDTree(positions)
    dists_nn, _ = tree.query(positions, k=2)
    nn_dist = float(np.median(dists_nn[:, 1]))

    # Build ideal lattice
    ideal_sites = generate_ideal_sites(positions, cell, nvac)
    n_sites = len(ideal_sites)

    # Identify surface layer via z-layers
    layers = group_z_layers(ideal_sites)
    top_z = max(z for z, _ in layers)
    surface_sites = []
    for z_mean, idx in layers:
        if abs(z_mean - top_z) < 0.5:
            surface_sites.extend(idx)
    surface_set = set(surface_sites)

    # Build in-plane neighbour list using KDTree
    offsets_2d = np.array([[dx * Lx, dy * Ly]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    surf_pos_2d = ideal_sites[np.array(surface_sites), :2]
    n_surf = len(surface_sites)
    surf_rep_2d = np.vstack([surf_pos_2d + o for o in offsets_2d])
    surf_tree_2d = cKDTree(surf_rep_2d)

    cutoff = nn_dist * 1.1
    inplane_neighbours = {}
    for local_i, global_i in enumerate(surface_sites):
        pos_i = ideal_sites[global_i, :2]
        nn_idx = surf_tree_2d.query_ball_point(pos_i, cutoff)
        neighbours_i = set()
        for img_idx in nn_idx:
            j_local = img_idx % n_surf
            j_global = surface_sites[j_local]
            if j_global != global_i:
                neighbours_i.add(j_global)
        inplane_neighbours[global_i] = list(neighbours_i)

    # ── Build second-shell (2NN) neighbour list ──
    # On FCC(100), 2NN are at distance a (= nn_dist * sqrt(2)) along diagonals
    cutoff_2nn = nn_dist * np.sqrt(2) * 1.1
    second_neighbours = {}
    for local_i, global_i in enumerate(surface_sites):
        pos_i = ideal_sites[global_i, :2]
        nn2_idx = surf_tree_2d.query_ball_point(pos_i, cutoff_2nn)
        nn1_set = set(inplane_neighbours[global_i])
        nn2_set = set()
        for img_idx in nn2_idx:
            j_local = img_idx % n_surf
            j_global = surface_sites[j_local]
            if j_global != global_i and j_global not in nn1_set:
                nn2_set.add(j_global)
        second_neighbours[global_i] = list(nn2_set)

    # ── Build site-to-layer mapping ──
    # Map every ideal site to its layer index and z-value
    site_layer = {}      # site_idx -> layer_index (0 = bottom, highest = top)
    layer_sites = {}     # layer_index -> list of site indices
    for layer_idx, (z_mean, site_indices) in enumerate(layers):
        for si in site_indices:
            site_layer[si] = layer_idx
        layer_sites[layer_idx] = list(site_indices)
    top_layer_idx = max(site_layer.values())

    # ── Build interlayer neighbour lists (3D NN) ──
    # On FCC(100), each site has 4 interlayer NN above and 4 below at the same
    # distance as in-plane NN (a/sqrt(2)). Use full 3D KDTree with PBC in x,y.
    all_rep_3d = np.vstack([ideal_sites + o for o in
                            np.array([[dx * Lx, dy * Ly, 0.0]
                                      for dx in [-1, 0, 1] for dy in [-1, 0, 1]])])
    all_tree_3d = cKDTree(all_rep_3d)

    interlayer_above = {}  # site_idx -> list of NN site indices in layer above
    interlayer_below = {}  # site_idx -> list of NN site indices in layer below
    for si in range(n_sites):
        pos_i = ideal_sites[si]
        layer_i = site_layer.get(si)
        if layer_i is None:
            interlayer_above[si] = []
            interlayer_below[si] = []
            continue
        nn_idx = all_tree_3d.query_ball_point(pos_i, cutoff)
        above = []
        below = []
        for img_idx in nn_idx:
            sj = img_idx % n_sites
            if sj == si:
                continue
            layer_j = site_layer.get(sj)
            if layer_j is None:
                continue
            if layer_j == layer_i + 1:
                above.append(sj)
            elif layer_j == layer_i - 1:
                below.append(sj)
        # Deduplicate (PBC images may produce repeats)
        interlayer_above[si] = list(set(above))
        interlayer_below[si] = list(set(below))

    # ── Build neighbour direction map ──
    # For each (site_i, site_j) 1NN pair, compute the lattice direction (di, dj).
    # The displacement vector is mapped to the nearest of the 4 lattice directions.
    neighbour_directions = {}
    for global_i, nn_list in inplane_neighbours.items():
        pos_i = ideal_sites[global_i, :2]
        for global_j in nn_list:
            pos_j = ideal_sites[global_j, :2]
            # Minimum image displacement
            d = pos_j - pos_i
            d[0] -= Lx * round(d[0] / Lx)
            d[1] -= Ly * round(d[1] / Ly)
            # Normalise by NN distance and round to integer lattice direction
            d_norm = d / nn_dist
            di = int(round(d_norm[0]))
            dj = int(round(d_norm[1]))
            neighbour_directions[(global_i, global_j)] = (di, dj)

    # Pre-build site tree for atom-to-site projection (with PBC images)
    offsets_3d = np.array([[dx * Lx, dy * Ly, 0.0]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    site_rep = np.vstack([ideal_sites + o for o in offsets_3d])
    site_tree = cKDTree(site_rep)

    return (ideal_sites, surface_set, inplane_neighbours, second_neighbours,
            neighbour_directions, interlayer_above, interlayer_below,
            site_layer, top_layer_idx, site_tree, n_sites, cell, nn_dist)


def find_vacancy_sites_fast(positions, site_tree, n_sites, cell, nvac):
    """Identify vacancy sites by finding ideal sites farthest from any atom.

    Returns set of vacancy site indices.
    """
    Lx, Ly = cell[0], cell[1]
    pos_wrapped = positions.copy()
    pos_wrapped[:, 0] %= Lx
    pos_wrapped[:, 1] %= Ly

    offsets = np.array([[dx * Lx, dy * Ly, 0.0]
                        for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    atom_rep = np.vstack([pos_wrapped + o for o in offsets])
    atom_tree = cKDTree(atom_rep)

    # For each ideal site, find distance to nearest atom
    # Use the pre-built site positions (first n_sites of site_rep)
    ideal_sites = site_tree.data[:n_sites]
    site_dists, _ = atom_tree.query(ideal_sites)
    vacancy_indices = np.argsort(site_dists)[-nvac:]

    return set(int(v) for v in vacancy_indices)


def project_atom_to_site(positions, site_tree, n_sites, cell):
    """Map each atom to its nearest ideal lattice site.

    Returns array of site indices (one per atom).
    """
    Lx, Ly = cell[0], cell[1]
    pos_wrapped = positions.copy()
    pos_wrapped[:, 0] %= Lx
    pos_wrapped[:, 1] %= Ly
    _, site_idx = site_tree.query(pos_wrapped)
    return site_idx % n_sites


def process_simulation(sim):
    """For each step in a simulation, determine the local vacancy environment
    of the selected event from the full global configuration.

    Returns list of dicts: one per step where the selected event is a 1NN hop.
    This is the authoritative mapping path for local-environment statistics.
    """
    sim_dir = sim["path"]
    nvac = sim["nvac"]
    temp = sim["temp"]

    # Check required files
    traj_path = sim_dir / "trajkmc.xyz"
    pykmc_path = sim_dir / "pykmc.out"
    ref_path = sim_dir / "reference_table.pickle"
    if not all(p.exists() for p in [traj_path, pykmc_path, ref_path]):
        return [], 0

    # Load step log
    df_steps = parse_pykmc_out(pykmc_path)
    if len(df_steps) == 0:
        return [], 0

    # Load event catalogue and classify each event
    catalog = pd.read_pickle(ref_path)

    # Pre-classify all events to know which ref_events are 1NN hops.
    # Apply the z-layer pass inline so the same ACCEPTED_TYPES gate that
    # filters this script already excludes subsurface-relabelled events —
    # keeps this script in sync with classify_lattice_events.ipynb cell 07.
    event_classification = {}
    n_zlayer_relabelled = 0
    for i, row in catalog.iterrows():
        cls = classify_event_from_row(row)
        sl_i, sl_f = get_event_surface_layers(
            row["initial_positions"],
            row["final_positions"],
            int(row["move_atom_idx"]),
        )
        pre_t = cls["event_type"]
        cls["event_type"] = reclassify_by_zlayer(pre_t, sl_i, sl_f)
        cls["event_type_pre_zlayer"] = pre_t
        cls["surface_layer_initial"] = int(sl_i)
        cls["surface_layer_final"] = int(sl_f)
        if cls["event_type"] != pre_t:
            n_zlayer_relabelled += 1
        event_classification[i] = cls

    # Load trajectory (all frames)
    from ase.io import read as ase_read
    try:
        traj = ase_read(traj_path, index=":")
    except Exception as e:
        print(f"  SKIP {nvac}vac T{temp}: cannot read trajectory ({e})")
        return [], 0

    # Build lattice infrastructure from frame 0
    ref_frame = traj[0]
    cell_diag = [ref_frame.cell[i, i] for i in range(3)]
    (ideal_sites, surface_set, inplane_neighbours, second_neighbours,
     neighbour_directions, interlayer_above, interlayer_below,
     site_layer, top_layer_idx, site_tree, n_sites, cell, nn_dist) = (
        build_lattice_infrastructure(ref_frame, cell_diag, nvac)
    )

    Lx, Ly = cell[0], cell[1]

    # Iterate over each step
    rows = []
    for _, step_row in df_steps.iterrows():
        step_num = int(step_row["step"])
        ref_event = int(step_row["ref_event"])
        ea = step_row["Ea"]

        # Check if this event is an accepted type. Use ACCEPTED_TYPES_3D so
        # subsurface variants enter the row stream — legacy aggregation in
        # build_lookup_table() still filters to _1nn_types + is_surface_hop,
        # so legacy CSV outputs are unchanged.
        cls = event_classification.get(ref_event)
        if cls is None or cls["event_type"] not in ACCEPTED_TYPES_3D:
            continue
        if ea >= BARRIER_CUTOFF:
            continue

        # Load the trajectory frame BEFORE this step (the state the event acts on)
        # Frame i corresponds to state after step i; frame (step-1) is state before step
        frame_idx = step_num - 1
        if frame_idx < 0 or frame_idx >= len(traj):
            continue

        frame = traj[frame_idx]
        positions = frame.get_positions()

        # Find ALL vacancy sites in the global configuration
        vacancy_sites = find_vacancy_sites_fast(positions, site_tree, n_sites, cell, nvac)

        # Determine which vacancy is hopping for this event
        # The event's move_atom_idx tells us which atom moves; the vacancy it fills
        # is the site the atom moves TO, so the vacancy origin is the atom's destination site.
        # But we need the event's local cluster mapped to the global frame.
        #
        # Simpler approach: the event with barrier ea was selected at atom site X.
        # From the event catalogue, the move_atom moves from site_A to site_B.
        # In the global frame, we need to identify which vacancy the atom is hopping into.
        #
        # Best approach: find the vacancy site that is closest to one of the event's
        # NN hop destinations. Since we know the event is a 1NN hop, the hopping vacancy
        # is the one that's a NN of the atom that moves.
        #
        # Actually, since we know the vacancy positions from the global frame,
        # and this is a 1NN hop, we just need to find which vacancy is involved.
        # For 1vac this is trivial. For multi-vac, any of the vacancies could be hopping.
        #
        # The pykmc.out doesn't tell us WHICH vacancy hops, only the ref_event index.
        # But the ref_event was applied to a specific atom in the simulation — we don't
        # have that mapping from pykmc.out alone.
        #
        # For environment counting, we can take a different approach: for EACH vacancy
        # in the global config, count its n_vacant_inplane_nn, then associate the
        # selected event's barrier with the environment of the vacancy that actually hopped.
        #
        # Since we don't know which vacancy hopped from pykmc.out, we use the trajectory:
        # compare frame[step-1] and frame[step] to find which vacancy moved.

        if step_num < len(traj):
            frame_after = traj[step_num]
            pos_after = frame_after.get_positions()
            vacancy_sites_after = find_vacancy_sites_fast(
                pos_after, site_tree, n_sites, cell, nvac
            )

            # The vacancy that moved: present in vacancy_sites but not in vacancy_sites_after
            moved_from = vacancy_sites - vacancy_sites_after
            moved_to = vacancy_sites_after - vacancy_sites
            if len(moved_from) != 1 or len(moved_to) != 1:
                # Ambiguous or no change — skip. Typical cause is a basin
                # super-event where several vacancies moved between trajectory
                # frames; we cannot attribute one barrier to one pair. See
                # build_rate_table_3d_catalogue() for a motif-rich table built
                # directly from the event catalogue instead.
                continue
            vac_origin = moved_from.pop()
            vac_dest = moved_to.pop()
        else:
            continue

        # Count in-plane vacant neighbours of the hopping vacancy
        n_vacant_inplane_nn = 0
        ip_nn = inplane_neighbours.get(vac_origin, [])
        for nn_site in ip_nn:
            if nn_site in vacancy_sites:
                n_vacant_inplane_nn += 1

        # ── Count second-shell (2NN) vacant neighbours ──
        n_vacant_2nn = 0
        nn2_sites = second_neighbours.get(vac_origin, [])
        for nn2_site in nn2_sites:
            if nn2_site in vacancy_sites:
                n_vacant_2nn += 1

        # ── Hop direction and directional vacancy pattern ──
        hop_dir = neighbour_directions.get((vac_origin, vac_dest))
        vacancy_pattern = ""
        if hop_dir is not None:
            hop_di, hop_dj = hop_dir
            # Define relative directions: left (CCW 90), right (CW 90), behind (180)
            # Rotation table for (di, dj):
            #   left  = (-dj, di)   [CCW 90]
            #   right = (dj, -di)   [CW 90]
            #   behind = (-di, -dj) [180]
            relative_dirs = {
                "left": (-hop_dj, hop_di),
                "right": (hop_dj, -hop_di),
                "behind": (-hop_di, -hop_dj),
            }
            # For each relative direction, find which NN site it corresponds to,
            # and check if that site is vacant
            pattern_bits = []
            for rel_name in ["left", "right", "behind"]:
                rel_di, rel_dj = relative_dirs[rel_name]
                # Find the NN site in this direction
                nn_site = None
                for candidate in ip_nn:
                    cand_dir = neighbour_directions.get((vac_origin, candidate))
                    if cand_dir == (rel_di, rel_dj):
                        nn_site = candidate
                        break
                if nn_site is not None and nn_site in vacancy_sites:
                    pattern_bits.append("1")
                else:
                    pattern_bits.append("0")
            vacancy_pattern = "".join(pattern_bits)  # e.g. "010"
        else:
            hop_di, hop_dj = 0, 0

        # Is the hopping vacancy on the surface?
        is_surface = vac_origin in surface_set
        move_type = cls["event_type"]

        # ── 3D environment: interlayer vacancy counts ──
        n_vacant_above = 0
        for nn_site in interlayer_above.get(vac_origin, []):
            if nn_site in vacancy_sites:
                n_vacant_above += 1
        n_vacant_below = 0
        for nn_site in interlayer_below.get(vac_origin, []):
            if nn_site in vacancy_sites:
                n_vacant_below += 1

        # Layer info
        layer_origin = site_layer.get(vac_origin, -1)
        layer_dest = site_layer.get(vac_dest, -1)
        dz_sign = 0
        if layer_dest > layer_origin:
            dz_sign = 1   # vacancy moved up (toward vacuum)
        elif layer_dest < layer_origin:
            dz_sign = -1  # vacancy moved down (into bulk)

        # Layer type: surface = top 2 layers, bulk = everything else
        layer_type = "surface" if layer_origin >= top_layer_idx - 1 else "bulk"

        # ── 3D-master enrichment ──
        motif_family_3d = classify_3d_motif({
            "move_type": move_type,
            "n_moved": int(cls.get("n_moved", 0)),
            "surface_layer_initial": int(cls.get("surface_layer_initial", -1)),
        })
        if layer_origin == top_layer_idx:
            site_class_3d = "surface"
        elif layer_origin == top_layer_idx - 1:
            site_class_3d = "subsurface"
        else:
            site_class_3d = "bulk_like"

        # Cartesian-vector classification distinguishes <111>_interlayer
        # (1NN cross-layer) from <001>_exchange (2NN axial); the integer
        # lattice-index form collapses both to (0, 0, ±1).
        orig_xyz = ideal_sites[vac_origin]
        dest_xyz = ideal_sites[vac_dest]
        dx = min_image(dest_xyz[0] - orig_xyz[0], Lx)
        dy = min_image(dest_xyz[1] - orig_xyz[1], Ly)
        dz = dest_xyz[2] - orig_xyz[2]   # slab: no PBC in z
        direction_family_3d = classify_direction_xyz(dx, dy, dz, nn_dist)

        rows.append({
            "nvac": nvac,
            "temp": temp,
            "step": step_num,
            "ref_event": ref_event,
            "barrier_eV": ea,
            "move_type": move_type,
            "move_type_pre_zlayer": cls.get("event_type_pre_zlayer", move_type),
            "surface_layer_initial": cls.get("surface_layer_initial", -1),
            "surface_layer_final": cls.get("surface_layer_final", -1),
            "vac_origin_site": int(vac_origin),
            "vac_dest_site": int(vac_dest),
            "is_surface_hop": is_surface,
            "n_vacant_inplane_nn": n_vacant_inplane_nn,
            "n_vacant_2nn": n_vacant_2nn,
            "n_vacant_above": n_vacant_above,
            "n_vacant_below": n_vacant_below,
            "layer_origin": layer_origin,
            "layer_dest": layer_dest,
            "layer_type": layer_type,
            "dz_sign": dz_sign,
            "hop_di": hop_di,
            "hop_dj": hop_dj,
            "vacancy_pattern": vacancy_pattern,
            # 3D fields
            "motif_family_3d": motif_family_3d,
            "site_class_3d": site_class_3d,
            "direction_family_3d": direction_family_3d,
        })

    return rows, n_zlayer_relabelled


def _barrier_stats(barriers):
    """Compute barrier statistics for a group of events."""
    return {
        "n_events": len(barriers),
        "Ea_mean_eV": float(barriers.mean()),
        "Ea_std_eV": float(barriers.std()) if len(barriers) > 1 else 0.0,
        "Ea_min_eV": float(barriers.min()),
        "Ea_max_eV": float(barriers.max()),
        "Ea_median_eV": float(barriers.median()),
    }


def _add_rates(lookup):
    """Add Arrhenius rates at representative temperatures."""
    for T in [300, 400, 500, 600, 800]:
        lookup[f"rate_{T}K_Hz"] = k0_Hz * np.exp(-lookup["Ea_mean_eV"] / (kB * T))
    return lookup


def build_lookup_table(all_rows):
    """Group mapped steps by move type and local environment.

    Produces rate tables:
    1. Scalar: 1NN surface hops by n_vacant_inplane_nn (backward compatible)
    2. Directional: 1NN surface hops by (n_vacant_inplane_nn, vacancy_pattern)
    3. Extended: 1NN surface hops by (n_vacant_inplane_nn, n_vacant_2nn)
    4. Exchange: exchange events by (n_vacant_inplane_nn, layer_type, dz_sign)
    5. Exchange_lateral: exchange_lateral by (n_vacant_inplane_nn, layer_type, dz_sign)
    """
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        # Return 7 items matching the non-empty branch so main() can unpack:
        # (lookup, lookup_dir, lookup_ext, lookup_exch, lookup_exlat,
        #  lookup_species, df).
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            df,
        )

    # Print move type breakdown
    print(f"\nTotal mapped steps: {len(df)}")
    for mt, grp in df.groupby("move_type"):
        n_surf = grp["is_surface_hop"].sum()
        print(f"  {mt}: {len(grp)} ({n_surf} surface)")

    # LEGACY: the surface-only rate table gates on is_surface_hop and filters
    # to 1NN move types. The 3D-master rate table (build_rate_table_3d_master,
    # added below) does NOT apply this gate — it groups by
    # (motif_family_3d, site_class_3d, n_vacant_1nn_3d, n_vacant_2nn,
    # direction_family_3d) over every row in ACCEPTED_TYPES_3D.
    # _1nn_types is defined at module scope so main() sees the same set.
    surface_1nn = df[(df["is_surface_hop"]) & (df["move_type"].isin(_1nn_types))].copy()
    surface_hops = surface_1nn  # alias for existing code below
    print(f"\nSurface 1NN hops: {len(surface_hops)} / {len(df)} total")

    # ── 1. Scalar table (backward compatible) ──
    rows = []
    for n_vnn, grp in surface_hops.groupby("n_vacant_inplane_nn"):
        row = {"n_vacant_inplane_nn": int(n_vnn)}
        row.update(_barrier_stats(grp["barrier_eV"]))
        rows.append(row)
    lookup = pd.DataFrame(rows).sort_values("n_vacant_inplane_nn").reset_index(drop=True)
    _add_rates(lookup)

    # ── 2. Directional table: grouped by (n_vacant_inplane_nn, vacancy_pattern) ──
    has_pattern = surface_hops["vacancy_pattern"].str.len() > 0
    dir_rows = []
    if has_pattern.any():
        dir_surface = surface_hops[has_pattern]
        for (n_vnn, pat), grp in dir_surface.groupby(
            ["n_vacant_inplane_nn", "vacancy_pattern"]
        ):
            row = {
                "n_vacant_inplane_nn": int(n_vnn),
                "vacancy_pattern": pat,
            }
            row.update(_barrier_stats(grp["barrier_eV"]))
            dir_rows.append(row)
    lookup_dir = pd.DataFrame(dir_rows)
    if len(lookup_dir) > 0:
        lookup_dir = lookup_dir.sort_values(
            ["n_vacant_inplane_nn", "vacancy_pattern"]
        ).reset_index(drop=True)
        _add_rates(lookup_dir)

    # ── 3. Extended table: grouped by (n_vacant_inplane_nn, n_vacant_2nn) ──
    ext_rows = []
    for (n_vnn, n_2nn), grp in surface_hops.groupby(
        ["n_vacant_inplane_nn", "n_vacant_2nn"]
    ):
        row = {
            "n_vacant_inplane_nn": int(n_vnn),
            "n_vacant_2nn": int(n_2nn),
        }
        row.update(_barrier_stats(grp["barrier_eV"]))
        ext_rows.append(row)
    lookup_ext = pd.DataFrame(ext_rows)
    if len(lookup_ext) > 0:
        lookup_ext = lookup_ext.sort_values(
            ["n_vacant_inplane_nn", "n_vacant_2nn"]
        ).reset_index(drop=True)
        _add_rates(lookup_ext)

    # ── 4. Exchange table: by (n_vacant_inplane_nn, layer_type, dz_sign) ──
    _exchange_types = {TYPE_EXCHANGE_UP, TYPE_EXCHANGE_DOWN}
    df_exchange = df[df["move_type"].isin(_exchange_types)].copy()
    exch_rows = []
    if len(df_exchange) > 0:
        for (n_vnn, lt, dz), grp in df_exchange.groupby(
            ["n_vacant_inplane_nn", "layer_type", "dz_sign"]
        ):
            row = {
                "n_vacant_inplane_nn": int(n_vnn),
                "layer_type": lt,
                "dz_sign": int(dz),
            }
            row.update(_barrier_stats(grp["barrier_eV"]))
            exch_rows.append(row)
    lookup_exch = pd.DataFrame(exch_rows)
    if len(lookup_exch) > 0:
        lookup_exch = lookup_exch.sort_values(
            ["n_vacant_inplane_nn", "layer_type", "dz_sign"]
        ).reset_index(drop=True)
        _add_rates(lookup_exch)

    # ── 5. Exchange_lateral table: by (n_vacant_inplane_nn, layer_type, dz_sign) ──
    df_exlat = df[df["move_type"] == TYPE_EXCHANGE_LAT].copy()
    exlat_rows = []
    if len(df_exlat) > 0:
        for (n_vnn, lt, dz), grp in df_exlat.groupby(
            ["n_vacant_inplane_nn", "layer_type", "dz_sign"]
        ):
            row = {
                "n_vacant_inplane_nn": int(n_vnn),
                "layer_type": lt,
                "dz_sign": int(dz),
            }
            row.update(_barrier_stats(grp["barrier_eV"]))
            exlat_rows.append(row)
    lookup_exlat = pd.DataFrame(exlat_rows)
    if len(lookup_exlat) > 0:
        lookup_exlat = lookup_exlat.sort_values(
            ["n_vacant_inplane_nn", "layer_type", "dz_sign"]
        ).reset_index(drop=True)
        _add_rates(lookup_exlat)

    # ── 6. Species table: by (mover_species, n_vacant_inplane_nn, n_vacant_2nn) ──
    # Only meaningful for multi-element simulations; for pure Ni this collapses
    # to the extended table.
    lookup_species = pd.DataFrame()
    if "mover_species" in df.columns and df["mover_species"].notna().any():
        species_rows = []
        for (sp, n_vnn, n_2nn), grp in surface_hops.groupby(
            ["mover_species", "n_vacant_inplane_nn", "n_vacant_2nn"]
        ):
            row = {
                "mover_species": sp,
                "n_vacant_inplane_nn": int(n_vnn),
                "n_vacant_2nn": int(n_2nn),
            }
            row.update(_barrier_stats(grp["barrier_eV"]))
            species_rows.append(row)
        if species_rows:
            lookup_species = pd.DataFrame(species_rows).sort_values(
                ["mover_species", "n_vacant_inplane_nn", "n_vacant_2nn"]
            ).reset_index(drop=True)
            _add_rates(lookup_species)

    return lookup, lookup_dir, lookup_ext, lookup_exch, lookup_exlat, lookup_species, df


def build_rate_table_3d_catalogue(classified_events_csv: Path):
    """3D rate table built directly from the event catalogue, not from fired
    KMC steps.

    Motivation: pyKMC fires almost exclusively `1NN_hop` events at moderate
    temperatures (their ~0.5 eV barrier dominates ~1 eV exchange/interlayer
    rates by ~10^5 at T=500 K). The step-log-based master table therefore
    only sees 1-2 motif families. The catalogue, by contrast, contains every
    discovered saddle point across all motif families.

    This table groups classified_events.csv by (motif_family_3d, site_class_3d,
    n_vac_nn1_initial) and aggregates the catalogue `energy_barrier` column.
    `n_vac_nn1_initial` is a local-cluster 1NN vacancy count (Stage 1f ML
    descriptor), not a full global-state count — it's the best environment
    proxy available without replaying each event against its trajectory frame.

    Returns (lookup_catalogue, df_catalogue). Writes nothing.
    """
    if not classified_events_csv.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(classified_events_csv)
    required = {
        "motif_family_3d",
        "site_class_3d",
        "energy_barrier",
        "n_vac_nn1_initial",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame(), df
    df = df.copy()
    # Coerce env count to nullable int; NaN when Stage 1f wasn't run.
    df["n_vac_nn1_initial"] = (
        pd.to_numeric(df["n_vac_nn1_initial"], errors="coerce").fillna(-1).astype(int)
    )
    grp_cols = ["motif_family_3d", "site_class_3d", "n_vac_nn1_initial"]
    out_rows = []
    for keys, grp in df.groupby(grp_cols, dropna=False):
        row = dict(zip(grp_cols, keys))
        row.update(_barrier_stats(grp["energy_barrier"]))
        out_rows.append(row)
    lookup = pd.DataFrame(out_rows).sort_values(grp_cols).reset_index(drop=True)
    if len(lookup) > 0:
        _add_rates(lookup)
    return lookup, df


def build_rate_table_3d_master(all_rows):
    """3D-first rate table: not gated on is_surface_hop; groups by
    (motif_family_3d, site_class_3d, n_vacant_1nn_3d, n_vacant_2nn,
    direction_family_3d).

    Returns (lookup_3d, df_with_3d).
    """
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return pd.DataFrame(), df
    df = df.copy()
    # n_vacant_1nn_3d combines in-plane, above, and below 1NN vacancies.
    df["n_vacant_1nn_3d"] = (
        df["n_vacant_inplane_nn"].fillna(0).astype(int)
        + df["n_vacant_above"].fillna(0).astype(int)
        + df["n_vacant_below"].fillna(0).astype(int)
    )
    grp_cols = [
        "motif_family_3d",
        "site_class_3d",
        "n_vacant_1nn_3d",
        "n_vacant_2nn",
        "direction_family_3d",
    ]
    out_rows = []
    for keys, grp in df.groupby(grp_cols, dropna=False):
        row = dict(zip(grp_cols, keys))
        row.update(_barrier_stats(grp["barrier_eV"]))
        out_rows.append(row)
    lookup_3d = pd.DataFrame(out_rows).sort_values(grp_cols).reset_index(drop=True)
    if len(lookup_3d) > 0:
        _add_rates(lookup_3d)
    return lookup_3d, df


def make_plots(all_df, lookup_df, out_dir):
    """Diagnostic plots for the rate lookup table."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    surface = all_df[all_df["is_surface_hop"]]

    # 1. Barrier distribution by environment
    ax = axes[0]
    env_vals = sorted(surface["n_vacant_inplane_nn"].unique())
    data = [surface.loc[surface["n_vacant_inplane_nn"] == v, "barrier_eV"].values
            for v in env_vals]
    if data:
        bp = ax.boxplot(data,
                        tick_labels=[f"{v} vac NN\n(n={len(d)})" for v, d in zip(env_vals, data)],
                        patch_artist=True, showfliers=True, whis=[5, 95])
        for patch in bp["boxes"]:
            patch.set_facecolor("#4DAF4A")
            patch.set_alpha(0.7)
    ax.set_ylabel("Barrier Ea (eV)")
    ax.set_title("1NN hop barrier by local environment")

    # 2. Mean barrier vs environment
    ax = axes[1]
    if len(lookup_df) > 0:
        ax.errorbar(lookup_df["n_vacant_inplane_nn"], lookup_df["Ea_mean_eV"],
                    yerr=lookup_df["Ea_std_eV"], fmt="o-", capsize=5, lw=2, markersize=8,
                    color="#4DAF4A")
    ax.set_xlabel("Vacant in-plane NN count")
    ax.set_ylabel("Mean barrier Ea (eV)")
    ax.set_title("Environment dependence of 1NN hop barrier")
    if env_vals:
        ax.set_xticks(env_vals)

    # 3. Rate vs environment at different temperatures
    ax = axes[2]
    if len(lookup_df) > 0:
        for T in [300, 400, 500, 600, 800]:
            col = f"rate_{T}K_Hz"
            if col in lookup_df.columns:
                ax.semilogy(lookup_df["n_vacant_inplane_nn"], lookup_df[col],
                            "o-", label=f"T={T}K", markersize=6)
    ax.set_xlabel("Vacant in-plane NN count")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title("1NN hop rate by environment and temperature")
    ax.legend(fontsize=8)
    if env_vals:
        ax.set_xticks(env_vals)

    plt.tight_layout()
    plt.savefig(out_dir / "rate_lookup_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved rate_lookup_plots.png")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build environment-dependent rate lookup tables from the "
                    "pyKMC runs of one composition (global trajectory replay).")
    ap.add_argument("--data-root",
                    help="Directory holding {composition}/{N}vac/T{temp}/{variant}/ "
                         "runs (default: $PYKMC_ANALYSIS_DATA_ROOT)")
    ap.add_argument("--composition", default="100Ni",
                    help="Composition directory under the data root (default: %(default)s)")
    ap.add_argument("--out-dir",
                    help="Output directory (default: $PYKMC_ANALYSIS_OUTPUT_DIR, else "
                         "Analysis/lattice_event_classification/)")
    ap.add_argument("--required-potential", default=REQUIRED_POTENTIAL,
                    help="Only use runs whose pair_coeff contains this string; "
                         "'' uses every run (default: %(default)s)")
    args = ap.parse_args(argv)

    try:
        data_root = _paths.data_root(args.data_root) / args.composition
    except RuntimeError as e:
        ap.error(str(e))
    if not data_root.is_dir():
        ap.error(f"composition directory not found: {data_root}")
    out_dir = _paths.output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building environment-dependent rate lookup table")
    print("(from global trajectory configurations)")
    print("=" * 70)

    sims = scan_nialh_sims(data_root, args.required_potential)
    print(f"Found {len(sims)} simulations under {data_root}"
          f"{' using ' + args.required_potential if args.required_potential else ''}")

    all_rows = []
    total_zlayer_relabelled = 0
    for sim in sims:
        print(f"  Processing {sim['nvac']}vac T{sim['temp']}...", end=" ", flush=True)
        rows, n_zrel = process_simulation(sim)
        print(f"{len(rows)} 1NN steps mapped"
              f"{' (' + str(n_zrel) + ' catalogue events z-layer-relabelled)' if n_zrel else ''}")
        all_rows.extend(rows)
        total_zlayer_relabelled += n_zrel

    print(f"\nTotal 1NN hop steps mapped: {len(all_rows)}")
    print(f"Total catalogue events z-layer-relabelled (excluded from rate table): "
          f"{total_zlayer_relabelled}")

    if len(all_rows) == 0:
        print("ERROR: no events mapped. Check data paths.")
        return

    lookup, lookup_dir, lookup_ext, lookup_exch, lookup_exlat, lookup_species, all_df = build_lookup_table(all_rows)

    print(f"\nRate lookup table ({len(lookup)} environment bins):")
    print(lookup.to_string(index=False, float_format="%.4f"))

    # Check environment dependence
    if len(lookup) > 1:
        ea_range = lookup["Ea_mean_eV"].max() - lookup["Ea_mean_eV"].min()
        ea_max_std = lookup["Ea_std_eV"].max()
        print(f"\nEnvironment dependence: Ea range = {ea_range:.4f} eV "
              f"(max std within bin = {ea_max_std:.4f} eV)")
        if ea_range < 0.05:
            print("  -> Weak dependence: a single mean barrier may suffice")
        else:
            print("  -> Significant dependence: use per-environment barriers")

    # Save scalar table (backward compatible)
    lookup.to_csv(out_dir / "rate_lookup_table.csv", index=False)
    print(f"\nSaved rate_lookup_table.csv")

    # Save directional table
    if len(lookup_dir) > 0:
        lookup_dir.to_csv(out_dir / "rate_lookup_table_directional.csv", index=False)
        print(f"Saved rate_lookup_table_directional.csv "
              f"({len(lookup_dir)} direction-environment bins)")
        # Report variance reduction
        scalar_std = lookup.set_index("n_vacant_inplane_nn")["Ea_std_eV"]
        dir_mean_std = lookup_dir.groupby("n_vacant_inplane_nn")["Ea_std_eV"].mean()
        for n_vnn in sorted(scalar_std.index):
            if n_vnn in dir_mean_std.index:
                reduction = 1.0 - dir_mean_std[n_vnn] / scalar_std[n_vnn] if scalar_std[n_vnn] > 0 else 0.0
                print(f"  n_vnn={n_vnn}: scalar std={scalar_std[n_vnn]:.4f} -> "
                      f"directional mean std={dir_mean_std[n_vnn]:.4f} "
                      f"({reduction:.1%} reduction)")

    # Save extended table (1NN + 2NN)
    if len(lookup_ext) > 0:
        lookup_ext.to_csv(out_dir / "rate_lookup_table_extended.csv", index=False)
        print(f"Saved rate_lookup_table_extended.csv "
              f"({len(lookup_ext)} 1NN+2NN environment bins)")
        # Report variance reduction
        ext_mean_std = lookup_ext.groupby("n_vacant_inplane_nn")["Ea_std_eV"].mean()
        for n_vnn in sorted(scalar_std.index):
            if n_vnn in ext_mean_std.index:
                reduction = 1.0 - ext_mean_std[n_vnn] / scalar_std[n_vnn] if scalar_std[n_vnn] > 0 else 0.0
                print(f"  n_vnn={n_vnn}: scalar std={scalar_std[n_vnn]:.4f} -> "
                      f"extended mean std={ext_mean_std[n_vnn]:.4f} "
                      f"({reduction:.1%} reduction)")

    # Save exchange table
    if len(lookup_exch) > 0:
        lookup_exch.to_csv(out_dir / "rate_lookup_table_exchange.csv", index=False)
        print(f"Saved rate_lookup_table_exchange.csv "
              f"({len(lookup_exch)} exchange environment bins)")

    # Save exchange_lateral table
    if len(lookup_exlat) > 0:
        lookup_exlat.to_csv(out_dir / "rate_lookup_table_exchange_lateral.csv", index=False)
        print(f"Saved rate_lookup_table_exchange_lateral.csv "
              f"({len(lookup_exlat)} exchange_lateral environment bins)")

    # Save species-dependent table
    if len(lookup_species) > 0:
        lookup_species.to_csv(out_dir / "rate_lookup_table_species.csv", index=False)
        print(f"Saved rate_lookup_table_species.csv "
              f"({len(lookup_species)} species-environment bins)")

    # ── 3D-master rate table ──
    lookup_3d, _ = build_rate_table_3d_master(all_rows)
    if len(lookup_3d) > 0:
        lookup_3d.to_csv(out_dir / "rate_lookup_table_3d_master.csv", index=False)
        print(f"Saved rate_lookup_table_3d_master.csv "
              f"({len(lookup_3d)} 3D environment bins, step-log aggregated)")

    # ── 3D-catalogue rate table — aggregates classified_events.csv directly ──
    # (see build_rate_table_3d_catalogue docstring for motivation)
    cat_csv = out_dir / "classified_events.csv"
    lookup_cat, _ = build_rate_table_3d_catalogue(cat_csv)
    if len(lookup_cat) > 0:
        lookup_cat.to_csv(out_dir / "rate_lookup_table_3d_catalogue.csv", index=False)
        n_motifs = lookup_cat["motif_family_3d"].nunique()
        print(
            f"Saved rate_lookup_table_3d_catalogue.csv "
            f"({len(lookup_cat)} bins, {n_motifs} motif families, catalogue-based)"
        )

    # Legacy convenience copy (not a move — the legacy filename keeps being
    # written above for readers of rate_lookup_table.csv).
    import shutil
    legacy_src = out_dir / "rate_lookup_table.csv"
    legacy_dst = out_dir / "rate_lookup_table_surface_legacy.csv"
    if legacy_src.exists():
        shutil.copyfile(legacy_src, legacy_dst)
        print(f"Copied {legacy_src.name} -> {legacy_dst.name} for naming clarity")

    # Save all mapped steps (all move types) and backward-compatible 1NN-only file
    all_df.to_csv(out_dir / "mapped_steps.csv", index=False)
    print(f"Saved mapped_steps.csv ({len(all_df)} steps, all move types)")

    nn_only = all_df[all_df["move_type"].isin(_1nn_types)]
    nn_only.to_csv(out_dir / "mapped_1nn_steps.csv", index=False)
    print(f"Saved mapped_1nn_steps.csv ({len(nn_only)} steps, all 1NN subtypes)")

    make_plots(all_df, lookup, out_dir)

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
