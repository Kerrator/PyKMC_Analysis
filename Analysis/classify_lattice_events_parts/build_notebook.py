"""Build / split the ``classify_lattice_events.ipynb`` notebook.

This script is the sync bridge between the master notebook
(``Analysis/classify_lattice_events.ipynb``) and the editable part files
living alongside this script.

Usage (from the repository root)
--------------------------------
Rebuild the notebook from the current ``[0-9][0-9]_*.py`` parts::

    python Analysis/classify_lattice_events_parts/build_notebook.py

One-shot split of the existing notebook into part files (overwrites
anything already in this folder with a numeric prefix)::

    python Analysis/classify_lattice_events_parts/build_notebook.py --split

The parts use Jupyter's "percent" format:

    # %% [markdown]
    # Some markdown here.
    #
    # Another line.

    # %%
    print("a code cell")

Leading ``# `` (or ``#``) on markdown lines is stripped when building the
notebook and reintroduced when splitting, so the round-trip is lossless
for cell source text. The rebuilt notebook is written unexecuted: no
outputs, no widget state, a generic ``Python 3`` kernelspec, and cell ids
derived from the part filename, so rebuilding unchanged parts gives a
byte-identical file.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "classify_lattice_events.ipynb"

PART_PATTERN = re.compile(r"^\d{2}[a-z]?_.*\.py$")
CELL_DELIM = re.compile(r"^# %%(?:\s+\[(markdown|raw)\])?\s*$")


# -- rebuild: parts -> notebook --------------------------------------------

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}


def _strip_md_prefix(line: str) -> str:
    """Remove the leading ``# `` (or ``#``) used to comment-out markdown."""
    if line.startswith("# "):
        return line[2:]
    if line == "#":
        return ""
    if line.startswith("#"):
        return line[1:]
    return line


def _parse_part(path: Path) -> list[nbformat.NotebookNode]:
    text = path.read_text()
    lines = text.splitlines()
    cells: list[nbformat.NotebookNode] = []
    cur_kind: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_kind is None:
            return
        # Drop the single trailing blank line that separates this cell
        # from the next # %% delimiter (it's file formatting, not content).
        local = list(cur_lines)
        if local and local[-1] == "":
            local.pop()
        body = "\n".join(local)
        if cur_kind == "markdown":
            body = "\n".join(_strip_md_prefix(ln) for ln in body.splitlines())
            if body.strip():
                cells.append(nbformat.v4.new_markdown_cell(body))
        elif cur_kind == "raw":
            if body.strip():
                cells.append(nbformat.v4.new_raw_cell(body))
        else:
            if body.strip():
                cells.append(nbformat.v4.new_code_cell(body))

    for line in lines:
        m = CELL_DELIM.match(line)
        if m:
            flush()
            cur_kind = m.group(1) or "code"
            cur_lines = []
        else:
            if cur_kind is None:
                # content before any delimiter: treat as code cell
                cur_kind = "code"
            cur_lines.append(line)
    flush()
    return cells


def _cell_id(stem: str, i: int) -> str:
    """Stable cell id from the part filename (``04b_ml_descriptors``, ``04b_ml_descriptors-1``, ...)."""
    base = re.sub(r"[^A-Za-z0-9_-]", "-", stem)[:56]
    return base if i == 0 else f"{base}-{i}"


def rebuild() -> None:
    old_size = NOTEBOOK.stat().st_size if NOTEBOOK.exists() else 0

    parts = sorted(p for p in HERE.iterdir() if PART_PATTERN.match(p.name))
    if not parts:
        sys.exit(f"No part files matching NN_*.py found in {HERE}")

    new_cells: list[nbformat.NotebookNode] = []
    for p in parts:
        for i, cell in enumerate(_parse_part(p)):
            cell.id = _cell_id(p.stem, i)
            new_cells.append(cell)

    # Fresh notebook: no outputs, execution counts or widget state carried over.
    nb = nbformat.v4.new_notebook(
        cells=new_cells,
        metadata={"kernelspec": KERNELSPEC, "language_info": LANGUAGE_INFO},
    )
    # Cell ids need nbformat v4.5+.
    nb.nbformat_minor = max(getattr(nb, "nbformat_minor", 0), 5)
    nbformat.validate(nb)
    tmp = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(nb, tmp)
    tmp.replace(NOTEBOOK)
    new_size = NOTEBOOK.stat().st_size
    print(
        f"Wrote {len(new_cells)} cells from {len(parts)} parts "
        f"-> {NOTEBOOK.name} ({old_size} -> {new_size} bytes)"
    )


# -- split: notebook -> parts ----------------------------------------------

# Short labels for the current 19-cell notebook. Order matches cell order.
DEFAULT_LABELS = [
    "title",
    "imports",
    "scan",
    "coverage",
    "load_events",
    "ml_descriptors",
    "saddle_features",
    "gmm_threshold",
    "reclassify",
    "species",
    "displacement",
    "barriers",
    "arrhenius",
    "vacancy_trend",
    "pair_analysis",
    "ase_config",
    "diffusion",
    "kinetic_coverage",
    "export",
]


def _commentify_md(src: str) -> str:
    return "\n".join(("# " + ln) if ln else "#" for ln in src.splitlines())


def split() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cells = nb.cells
    if len(cells) != len(DEFAULT_LABELS):
        print(
            f"WARN: notebook has {len(cells)} cells, "
            f"expected {len(DEFAULT_LABELS)}; using generic labels"
        )
        labels = [f"cell{i:02d}" for i in range(len(cells))]
    else:
        labels = DEFAULT_LABELS

    for i, (cell, label) in enumerate(zip(cells, labels)):
        fname = HERE / f"{i:02d}_{label}.py"
        if cell.cell_type == "markdown":
            header = "# %% [markdown]\n"
            body = _commentify_md(cell.source)
        elif cell.cell_type == "raw":
            header = "# %% [raw]\n"
            body = cell.source
        else:
            header = "# %%\n"
            body = cell.source
        # Normalize: strip trailing newlines from source, then write
        # header + body + one trailing file newline. On rebuild the final
        # blank line is consumed as the cell separator.
        body = body.rstrip("\n")
        fname.write_text(header + body + "\n")
        print(f"wrote {fname.name} ({cell.cell_type}, {len(cell.source)} chars)")


# -- cli --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        action="store_true",
        help="One-shot: split the master notebook into part files",
    )
    args = ap.parse_args()
    if args.split:
        split()
    else:
        rebuild()


if __name__ == "__main__":
    main()
