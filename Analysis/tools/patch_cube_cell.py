"""Patch a single cell in a .kmcrt rate cube to a target Ea value.

A diagnostic aid for the .kmcrt rate-cube format used by earlier pylatkmc
versions (current pylatkmc no longer uses it): set one cell's Ea (the rate
is recomputed as k0·exp(-Ea/kT) at the cube's baked temperature) to test
whether that cell drives a discrepancy, without rebuilding the cube.

Usage (from the repository root):
    python -m Analysis.tools.patch_cube_cell \
        --kmcrt /path/to/model.kmcrt --cell 0,0,0,0,0,0,0,0,0 --Ea 0.6
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def patch_cell(kmcrt_path: Path, axis_indices: tuple[int, ...], new_Ea_eV: float):
    """Patch one cube cell's (rate, Ea) values in-place.

    `axis_indices` is the 9-tuple (sc, dir, mover, nv1, nFe1, nCr1, nv2, nFe2, nCr2).
    """
    data = bytearray(kmcrt_path.read_bytes())
    header_bytes = struct.unpack('<I', data[8:12])[0]
    hdr_text = bytes(data[12:12+header_bytes]).decode('latin-1')
    i = 0; depth = 0
    while i < len(hdr_text):
        if hdr_text[i] == '{': depth += 1
        elif hdr_text[i] == '}':
            depth -= 1
            if depth == 0: i += 1; break
        i += 1
    hdr = json.loads(hdr_text[:i])
    n = hdr['n_entries']
    strides = hdr['strides']
    k0 = hdr['k0_Hz']
    T = hdr['temperature_K']
    kB = 8.617333e-5

    flat_idx = sum(s * i for s, i in zip(strides, axis_indices))
    if flat_idx < 0 or flat_idx >= n:
        raise IndexError(f'cell {axis_indices} -> flat {flat_idx} out of range [0, {n})')

    # Payload layout: u32 n_entries, f32[n] rate, f32[n] Ea, u32[n] count
    # (as read by ratetable_load in the generated C of earlier pylatkmc versions).
    payload_off = 12 + header_bytes + 4   # +4 to skip the u32 n_entries field
    rate_off = payload_off + 4 * flat_idx
    Ea_off = payload_off + 4 * n + 4 * flat_idx

    # Read current values
    cur_rate = struct.unpack('<f', bytes(data[rate_off:rate_off+4]))[0]
    cur_Ea = struct.unpack('<f', bytes(data[Ea_off:Ea_off+4]))[0]

    # Compute new rate from new Ea at the cube's baked T
    new_rate = float(k0) * float(np.exp(-new_Ea_eV / (kB * T)))

    # Write
    struct.pack_into('<f', data, rate_off, new_rate)
    struct.pack_into('<f', data, Ea_off, new_Ea_eV)

    kmcrt_path.write_bytes(bytes(data))

    print(f'Patched {kmcrt_path.name} cell {axis_indices} (flat={flat_idx}):')
    print(f'  rate: {cur_rate:.4e} -> {new_rate:.4e} Hz')
    print(f'  Ea:   {cur_Ea:.4f} -> {new_Ea_eV:.4f} eV')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kmcrt', required=True, type=Path)
    p.add_argument('--cell', required=True,
                   help='9 comma-separated indices: sc,dir,mover,nv1,nFe1,nCr1,nv2,nFe2,nCr2')
    p.add_argument('--Ea', required=True, type=float, help='new Ea in eV')
    args = p.parse_args()

    indices = tuple(int(x) for x in args.cell.split(','))
    if len(indices) != 9:
        raise ValueError(f'expected 9 indices, got {len(indices)}')

    patch_cell(args.kmcrt, indices, args.Ea)


if __name__ == '__main__':
    main()
