"""Synthetic-lattice fixtures for unit tests."""
from __future__ import annotations

import numpy as np


def make_tiny_fcc100_slab(
    nx: int = 3,
    ny: int = 3,
    nlayers: int = 2,
    a: float = 3.524,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a small FCC(100) slab for testing 3D helpers.

    Each layer holds two sublattices:
      - A layers (even parity): atoms at (i*a, j*a, z) and (i*a + a/2, j*a + a/2, z)
      - B layers (odd parity):  atoms at (i*a + a/2, j*a, z) and (i*a, j*a + a/2, z)
    Layer z-spacing is a/2. Total site count = 2 * nx * ny * nlayers.

    Returns
    -------
    ideal_sites : np.ndarray, shape (N, 3)
    cell        : np.ndarray, shape (3,) — [Lx, Ly, Lz]
    nn_dist     : float — nearest-neighbor distance (a / sqrt(2))
    """
    sites: list[tuple[float, float, float]] = []
    for layer in range(nlayers):
        z = layer * (a / 2.0)
        parity = layer % 2
        if parity == 0:
            basis = [(0.0, 0.0), (a / 2.0, a / 2.0)]
        else:
            basis = [(a / 2.0, 0.0), (0.0, a / 2.0)]
        for i in range(nx):
            for j in range(ny):
                for bx, by in basis:
                    sites.append((i * a + bx, j * a + by, z))
    ideal_sites = np.array(sites, dtype=float)
    cell = np.array([nx * a, ny * a, nlayers * (a / 2.0)], dtype=float)
    nn_dist = a / np.sqrt(2.0)

    assert ideal_sites.shape == (2 * nx * ny * nlayers, 3)
    return ideal_sites, cell, nn_dist
