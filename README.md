# PyKMC Analysis

PyKMC Analysis is a set of Python tools to inspect, classify and render the
events found in pyKMC off-lattice kinetic Monte Carlo runs. It reads the
files a run leaves behind (mainly `reference_table.pickle`) and gives you an
interactive 3D event viewer, an event-classification pipeline, rate tables,
and Blender and 2D figure renderers. It does not run simulations, and you do
not need pyKMC or LAMMPS installed to use the viewer.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/Kerrator/PyKMC_Analysis.git
cd PyKMC_Analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

- `pip install -e ".[notebooks]"` adds JupyterLab, for the notebooks.
- `pip install -e ".[families]"` adds pylatkmc, for curated family labels in the viewer.
- `pip install -e ".[dev]"` adds pytest, for the tests.

## Quick start: view one pyKMC run

```bash
python -m Analysis.event_viewer.app --run /path/to/run
```

Then open <http://127.0.0.1:8050> in a browser, and press Ctrl+C in the
terminal to stop. The run directory must contain `reference_table.pickle`,
which pyKMC writes. `input.in` is optional; when present, the viewer reads
the run temperature from it. Click a point in the scatter plot to show that
event in 3D. `--help` lists the options, and the
[event viewer guide](Analysis/event_viewer/README.md) covers everything else.

Only open runs you trust: loading a `.pickle` file can run arbitrary code.

## Using a folder of runs

The viewer's folder mode, the classification pipeline and the rate tables
work on many runs at once. Keep the runs in one folder laid out as:

```
/path/to/data/
└── {composition}/{N}vac/T{temp}/{variant}/     e.g. 50Cu_50Ni/2vac/T500/full/
    ├── reference_table.pickle
    ├── input.in
    ├── pykmc.out          (classification and rate tables)
    └── trajkmc.xyz        (classification and rate tables)
```

The viewer needs only `reference_table.pickle`.

The composition folder name must contain a digit. Pass the top folder on the
command line, or set it once:

```bash
python -m Analysis.event_viewer.app --data-root /path/to/data
# or
export PYKMC_ANALYSIS_DATA_ROOT=/path/to/data
python -m Analysis.event_viewer.app
```

In this mode the viewer's scatter plot lists classified events, so run the
event classification (below) first.

## Configuration

Everything is set by environment variables. A command-line flag, where a
tool has one, wins over the variable.

| Variable | Sets | Default |
|---|---|---|
| `PYKMC_ANALYSIS_DATA_ROOT` | Folder of runs (layout above) | None; only folder tools need it |
| `PYKMC_ANALYSIS_OUTPUT_DIR` | Where classification outputs and the audit log are read and written | `Analysis/lattice_event_classification/` |
| `PYKMC_ANALYSIS_BLENDER` | Blender executable, for renders | `blender` on `PATH` |
| `PYKMC_ANALYSIS_LAMMPS` | LAMMPS executable, for NEB paths in renders | `lmp`, `lmp_mpi` or `lmp_serial` on `PATH` |
| `PYKMC_ANALYSIS_MPIRUN` | MPI launcher for LAMMPS | `mpirun` or `mpiexec` on `PATH` |

Outputs and the audit log written under `Analysis/lattice_event_classification/`
are git-ignored, so a fresh clone starts with none.

## The other tools

- **Event classification**: labels every event (1NN hop, exchange, ...) and
  computes descriptors, in a notebook built from editable parts.
  See [classify_lattice_events_parts](Analysis/classify_lattice_events_parts/README.md).
- **Rate table**: builds rate lookup tables for on-lattice KMC from a folder
  of runs. See [build_rate_table.py](Analysis/build_rate_table.py)
  (`python -m Analysis.build_rate_table --help`).
- **Blender renders**: animations and stills of single events.
  See [blender_viewer](Analysis/blender_viewer/README.md).
- **Schematics**: flat 2D initial / saddle / final figures.
  See [schematic_viewer](Analysis/schematic_viewer/README.md).
- **kmos export** (experimental): turns the event catalogue into a kmos
  lattice KMC project. See [kmos_export](Analysis/kmos_export/README.md).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

## License

MIT, see [LICENSE](LICENSE).

## Related

- [pyKMC](https://github.com/hugomoison/pyKMC): the off-lattice kinetic Monte
  Carlo code whose runs these tools read.
- [pylatkmc](https://github.com/Kerrator/pylatkmc): an on-lattice KMC engine
  built from the curated event families.
