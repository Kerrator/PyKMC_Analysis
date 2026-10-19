"""Command-line entry point for kmos_export.

Usage:
    python -m Analysis.kmos_export.cli build-pilot \\
        --catalogue path/to/classified_events_with_families.csv \\
        --out-xml pilot.xml \\
        --temperature 500

Subcommands:
    build-pilot   — pilot: emit a kmos project XML from
                    `surface_1NN_inplane` events. The XML is then fed to
                    `kmos export <xml> <build_dir>` to compile.
    info          — print the IR descriptor (no kmos import) for a given
                    family; useful for diff-debugging the schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from Analysis.kmos_export.translator import (
    summarize_family,
    translate_pilot_surface_1NN,
)


def cmd_build_pilot(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.catalogue)
    desc, summary = translate_pilot_surface_1NN(
        df,
        project_name=args.name,
        mover_species=args.mover,
        temperature_K=args.temperature,
        k0_Hz=args.k0,
    )

    print(f"family:                {summary.family_id}")
    print(f"n_events:              {summary.n_events}")
    print(f"n_buckets:             {summary.n_buckets}")
    print(f"Ea_weighted_mean (eV): {summary.Ea_weighted_mean_eV:.4f}")
    print(f"Ea range (eV):         {summary.Ea_min_eV:.4f}  {summary.Ea_max_eV:.4f}")
    print(f"per-bucket means:")
    for b, m in sorted(summary.bucket_means.items()):
        print(f"  {b:<12} Ea={m:.4f} eV   n={summary.bucket_counts[b]:>5}")

    # Emit kmos Project XML.
    from Analysis.kmos_export.emitter import descriptor_to_kmos_project

    pt = descriptor_to_kmos_project(desc)
    pt.validate_model()
    out_xml = Path(args.out_xml).resolve()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    pt.save(str(out_xml))
    print()
    print(f"wrote kmos project XML: {out_xml}")
    print(f"  layers:     {[l.name for l in pt.layer_list]}")
    print(f"  species:    {[s.name for s in pt.species_list]}")
    print(f"  parameters: {[p.name for p in pt.parameter_list]}")
    print(f"  processes:  {len(pt.process_list)} ({', '.join(p.name for p in pt.process_list[:3])}...)")
    print()
    print("Next: compile with `kmos export {xml} {build_dir}`".format(
        xml=out_xml.name, build_dir=args.name + "_build",
    ))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print the IR descriptor JSON without importing kmos. Useful for
    inspecting / diffing the translator output in CI."""
    df = pd.read_csv(args.catalogue)
    desc, summary = translate_pilot_surface_1NN(df)

    out = {
        "summary": {
            "family_id": summary.family_id,
            "n_events": summary.n_events,
            "n_buckets": summary.n_buckets,
            "Ea_weighted_mean_eV": summary.Ea_weighted_mean_eV,
            "bucket_means": summary.bucket_means,
            "bucket_counts": summary.bucket_counts,
        },
        "descriptor": desc.model_dump(),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kmos_export",
        description="Translate the curated FCC event catalogue into kmos projects.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build-pilot",
        help="pilot: emit a kmos project XML for surface_1NN_inplane.",
    )
    p_build.add_argument("--catalogue", required=True,
                         help="Path to classified_events_with_families.csv")
    p_build.add_argument("--out-xml", required=True,
                         help="Where to write the kmos project XML")
    p_build.add_argument("--name", default="fcc100_pilot_surface_1NN",
                         help="Project / model name (kmos identifier)")
    p_build.add_argument("--mover", default="Ni",
                         help="Mover species (default: Ni)")
    p_build.add_argument("--temperature", type=float, default=500.0,
                         help="Temperature in K (default: 500)")
    p_build.add_argument("--k0", type=float, default=1.0e13,
                         help="Attempt frequency in Hz (default: 1e13)")
    p_build.set_defaults(func=cmd_build_pilot)

    p_info = sub.add_parser(
        "info",
        help="Print the IR descriptor JSON (no kmos import).",
    )
    p_info.add_argument("--catalogue", required=True,
                        help="Path to classified_events_with_families.csv")
    p_info.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
