"""Host-Python script: dump an event pair to a numpy .npz for Blender to consume.

Usage (from the repository root, in the environment where pandas and the
pyKMC pickle dependencies are installed):

    python -m Analysis.blender_viewer.prepare \\
        --sim /path/to/run --idx-ref-fwd 6467 --out event_6467.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Analysis.blender_viewer.data import dump_event_pair_npz, load_event_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, type=Path,
                        help="pyKMC simulation directory (containing reference_table.pickle)")
    parser.add_argument("--idx-ref-fwd", type=int, required=True)
    parser.add_argument("--out", required=True, type=Path,
                        help=".npz file to write")
    args = parser.parse_args()

    pair = load_event_pair(args.sim, idx_ref_fwd=args.idx_ref_fwd)
    dump_event_pair_npz(pair, args.out)
    print(
        f"[prepare] wrote {args.out} — sim_label={pair.sim_label} "
        f"fwd.idx_ref={pair.forward.idx_ref} bwd.idx_ref={pair.backward.idx_ref} "
        f"n_fwd_atoms={len(pair.forward.types)} n_bwd_atoms={len(pair.backward.types)}"
    )


if __name__ == "__main__":
    main()
