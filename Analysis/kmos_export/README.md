# kmos export (experimental)

Translates the curated FCC event catalogue
(`classified_events_with_families.csv`) into a project for
[kmos](https://github.com/mhoffman/kmos), a lattice kinetic Monte Carlo code.
Only a pilot is implemented: the `surface_1NN_inplane` family on an FCC(100)
two-site lattice, with one event-weighted mean barrier for the whole family.

```
classified_events_with_families.csv
        │  translator.py   catalogue → ProcessDescriptor list
        ▼
intermediate description   schema.py (pydantic, no kmos needed)
        │  emitter.py      description → kmos Project
        ▼
kmos project XML  ──►  kmos export <xml> <build_dir>   (kmos compiles it)
```

## Requirements

- This package (`pip install -e .` from the repository root).
- kmos itself, only for writing the XML and compiling it. kmos is **not** a
  dependency of this package: it is licensed GPL-3 and you install it
  separately, following its README (it is published as `python-kmos`).
  Compiling a kmos model also needs a Fortran compiler.

Everything except the XML step (the `info` command, the schema and the
translator) works without kmos.

## Command line

```bash
# Print the intermediate description as JSON (no kmos needed)
python -m Analysis.kmos_export.cli info \
    --catalogue /path/to/classified_events_with_families.csv

# Write a kmos project XML for the surface_1NN_inplane pilot (needs kmos)
python -m Analysis.kmos_export.cli build-pilot \
    --catalogue /path/to/classified_events_with_families.csv \
    --out-xml pilot.xml --temperature 500

# Compile it with kmos
kmos export pilot.xml fcc100_pilot_surface_1NN_build
```

`python -m Analysis.kmos_export.cli build-pilot --help` lists the other
options (`--name`, `--mover`, `--k0`).

## Tests

```bash
python -m pytest Analysis/kmos_export/tests
```

The tests that need kmos are skipped when it is not installed.
