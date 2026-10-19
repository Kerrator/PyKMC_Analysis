"""Batch showreel builder: one short animation per event family.

End-to-end pipeline, callable in stages or one-shot via :func:`run_all`:

1. :func:`select_representatives` — pick one median-barrier event per family
   from ``classified_events_with_families.csv``; write ``manifest.json``.
   :func:`select_curated` instead takes the events flagged for animation in
   the Event Viewer audit log and writes ``curated_manifest.json``.
2. :func:`prepare_all` — host Python: load the pickle for each selected event,
   dump the ``EventPair`` to ``npz/event_<family>.npz``. Optionally,
   :func:`run_neb_all` then computes a LAMMPS NEB path per event.
3. :func:`build_all` — inside Blender (headless): build a loop-orbit scene
   per event, optionally embed a family-name + barrier Text overlay, save
   ``.blend``.
4. :func:`render_all` — headless Blender: render a PNG sequence per event.
5. :func:`compile_showreel` — host shell: per-family MP4 with drawtext
   overlay + final concat MP4.

Every stage reads and writes under ``--root``, by default
``<output dir>/showreel`` (the output dir is ``PYKMC_ANALYSIS_OUTPUT_DIR``,
else ``Analysis/lattice_event_classification/``). The command sequence is in
``Analysis/blender_viewer/README.md``.

Final-render conventions
------------------------
A multi-event render at 720p or above takes hours, so it goes through three
stages whose settings are recorded in ``constants.py``:

* :data:`constants.DRAFT_RENDER_PRESET` — Eevee 540p, 3-keyframe
  Bezier, single event. Lock camera angle / theme / radius mode.
* :data:`constants.NEB_VALIDATION_PRESET` — same single event with
  ``use_neb=True``. Confirm the physics looks right.
* :data:`constants.FINAL_RENDER_PRESET` — Cycles 1080p on the GPU,
  ``use_neb=True`` (required), full manifest. Render and compile one event
  at a time, with drawtext labels and native-resolution title cards.

The presets are reference values; nothing applies them automatically, so
choose theme and resolution explicitly for each final render. On macOS, wrap
long jobs in ``caffeinate -i -s`` so the machine does not sleep. In shell
loops, read event ids into an array rather than piping into ``while read``:
Blender can consume the loop's stdin and silently skip events.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Analysis import _paths
from Analysis._paths import AUDIT_CSV, CLASSIFIED_WITH_FAMILIES_CSV

DEFAULT_ROOT = _paths.output_dir() / "showreel"
DEFAULT_CURATED_MANIFEST = "curated_manifest.json"

# Minimum mover displacement for a selected event to be considered visually
# meaningful. Events below this read as pure relaxation in Blender.
MIN_MOVER_DISPLACEMENT_A = 1.5

# Fonts for drawtext, first match wins: macOS system fonts, then common Linux
# DejaVu locations. With none present, drawtext uses ffmpeg's default font
# (needs an ffmpeg built with fontconfig).
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


@dataclass
class SelectedEvent:
    """One chosen event per family_id, enough to drive the whole pipeline."""

    family_id: str
    family_name: str
    composition: str
    nvac: int
    temp: float
    sim_path: str
    sim_label: str
    idx_ref: int
    idx_backward: int
    energy_barrier: float
    reverse_barrier: float
    move_type: str
    mover_disp_A: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Step 1: selection
# ---------------------------------------------------------------------------


def _portable_sim_path(row) -> str:
    """The run directory of a catalogue row: the relative ``sim_relpath`` when the
    CSV has it (portable across machines), else the absolute ``sim_path``."""
    rel = getattr(row, "sim_relpath", None)
    return rel if isinstance(rel, str) and rel else row.sim_path


def select_representatives(
    csv_path: Path | str = CLASSIFIED_WITH_FAMILIES_CSV,
    out_manifest: Path | str | None = None,
    min_disp_A: float = MIN_MOVER_DISPLACEMENT_A,
    data_root: Path | str | None = None,
) -> list[SelectedEvent]:
    """Pick one median-barrier event per accepted family_id; write manifest.

    Median-barrier selection is done within the subset of each family where
    ``idx_ref != idx_backward`` (distinct backward partner) and the mover
    actually moves more than ``min_disp_A``. If the strict-median candidate
    has a tiny displacement, walk outward (median ± k) until a real hop is
    found — otherwise skip the family with a warning.

    Relative ``sim_path`` values in the CSV are resolved against
    ``data_root`` (default: ``PYKMC_ANALYSIS_DATA_ROOT``).
    """
    import pandas as pd  # noqa: PLC0415

    csv_path = Path(csv_path)
    if out_manifest is None:
        out_manifest = DEFAULT_ROOT / "manifest.json"
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "family_id", "family_name", "assignment_status",
        "composition", "nvac", "temp", "sim_path", "sim_label",
        "idx_ref", "energy_barrier", "move_type",
    ]
    df = pd.read_csv(csv_path, usecols=[c for c in cols if c])

    accepted = df[df.assignment_status == "accepted"].copy()
    accepted = accepted[accepted.family_id.notna() & (accepted.family_id != "")]

    selections: list[SelectedEvent] = []
    for fam_id, group in accepted.groupby("family_id", sort=False):
        chosen = _pick_from_family(group, min_disp_A=min_disp_A, data_root=data_root)
        if chosen is None:
            print(f"[select] SKIP {fam_id} — no candidate with displacement ≥ {min_disp_A} Å")
            continue
        selections.append(chosen)
        print(
            f"[select] {fam_id:40s}  Ea={chosen.energy_barrier:.3f} eV  "
            f"disp={chosen.mover_disp_A:.2f} Å  ({chosen.sim_label} #{chosen.idx_ref})"
        )

    out_manifest.write_text(
        json.dumps([s.to_dict() for s in selections], indent=2)
    )
    print(f"[select] wrote manifest with {len(selections)} events → {out_manifest}")
    return selections


def _pick_from_family(
    group, min_disp_A: float, data_root: Path | str | None = None,
) -> SelectedEvent | None:
    """Pick median-barrier event in a family, preferring real hops."""
    import pandas as pd  # noqa: PLC0415

    from Analysis.event_viewer.data import _load_pickle  # noqa: PLC0415

    # Sort by barrier, then walk outward from the median.
    sorted_df = group.sort_values("energy_barrier").reset_index(drop=True)
    n = len(sorted_df)
    if n == 0:
        return None
    median_idx = n // 2

    # Spiral outward from the median so we prefer typical barriers but still
    # find a real hop when the strict-median event is a 0.03 Å relaxation.
    order = []
    for k in range(n):
        if median_idx + k < n:
            order.append(median_idx + k)
        if k > 0 and median_idx - k >= 0:
            order.append(median_idx - k)

    # Cache loaded pickles across candidates (within one family they're
    # usually from the same sim).
    pickle_cache: dict[str, "pd.DataFrame"] = {}

    for idx in order:
        row = sorted_df.iloc[idx]
        sim_path = row.sim_path
        if pd.isna(sim_path) or not sim_path:
            continue
        sim_dir = _paths.resolve_sim_path(_portable_sim_path(row), data_root)
        pickle_path = sim_dir / "reference_table.pickle"
        if not pickle_path.exists():
            continue

        if str(pickle_path) not in pickle_cache:
            try:
                pickle_cache[str(pickle_path)] = _load_pickle(pickle_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[select] WARN could not load {pickle_path}: {exc}")
                continue
        pickle_df = pickle_cache[str(pickle_path)]

        fwd_rows = pickle_df[pickle_df.idx_ref == int(row.idx_ref)]
        if fwd_rows.empty:
            continue
        fwd = fwd_rows.iloc[0]
        idx_bwd = int(fwd.idx_backward)
        if idx_bwd == int(row.idx_ref):
            continue  # skip self-symmetric (no distinct backward)
        bwd_rows = pickle_df[pickle_df.idx_ref == idx_bwd]
        if bwd_rows.empty:
            continue
        bwd = bwd_rows.iloc[0]

        m = int(fwd.move_atom_idx)
        disp = float(np.linalg.norm(
            np.asarray(fwd.final_positions)[m] - np.asarray(fwd.initial_positions)[m]
        ))
        if disp < min_disp_A:
            continue

        return SelectedEvent(
            family_id=str(row.family_id),
            family_name=str(row.family_name) if "family_name" in row and not pd.isna(row.family_name) else str(row.family_id),
            composition=str(row.composition),
            nvac=int(row.nvac),
            temp=float(row.temp),
            sim_path=str(sim_dir.resolve()),
            sim_label=str(row.sim_label),
            idx_ref=int(row.idx_ref),
            idx_backward=idx_bwd,
            energy_barrier=float(row.energy_barrier),
            reverse_barrier=float(bwd.energy_barrier),
            move_type=str(row.move_type),
            mover_disp_A=disp,
        )

    return None


# ---------------------------------------------------------------------------
# Step 1b: curated selection (read render_flag from audit_log.csv)
# ---------------------------------------------------------------------------


def select_curated(
    audit_csv: Path | str = AUDIT_CSV,
    classification_csv: Path | str = CLASSIFIED_WITH_FAMILIES_CSV,
    out_manifest: Path | str | None = None,
    force: bool = False,
    warn_threshold: int = 20,
    data_root: Path | str | None = None,
) -> list[SelectedEvent]:
    """Pick events flagged in the Event Viewer for curated rendering.

    Reads ``render_flag`` from the audit CSV (set via the "Flag for animation"
    checkbox in the audit panel), joins each flagged key against
    ``classified_events_with_families.csv`` to recover the classifier's
    family/sim metadata, loads the pickle for forward/backward partner +
    barrier + displacement, and writes a curated manifest.

    Returns the list of resolved events. The manifest at ``out_manifest``
    has the same schema as ``manifest.json`` (``SelectedEvent`` dicts), so
    every downstream stage (``prepare``/``neb``/``build``/``render``/
    ``compile``) accepts it via ``--manifest <path>``.

    Refuses to write a manifest > ``warn_threshold`` events unless ``force``
    is True (rendering blows up at ~2 hours per 9 events). Relative
    ``sim_path`` values are resolved against ``data_root`` (default:
    ``PYKMC_ANALYSIS_DATA_ROOT``).
    """
    import pandas as pd  # noqa: PLC0415

    from Analysis.event_viewer.audit import flagged_keys, load_audit  # noqa: PLC0415
    from Analysis.event_viewer.data import _load_pickle  # noqa: PLC0415

    audit_csv = Path(audit_csv)
    classification_csv = Path(classification_csv)
    if out_manifest is None:
        out_manifest = DEFAULT_ROOT / DEFAULT_CURATED_MANIFEST
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    audit_df = load_audit() if audit_csv == AUDIT_CSV else pd.read_csv(audit_csv)
    keys = flagged_keys(audit_df)
    if not keys:
        print("[select_curated] no events flagged for animation in audit_log.csv")
        out_manifest.write_text("[]")
        return []

    if len(keys) > warn_threshold and not force:
        print(
            f"[select_curated] WARNING: {len(keys)} events flagged "
            f"(> {warn_threshold}). Rendering this many will take "
            f"~{len(keys) * 4} min for blends + ~{len(keys) * 4} min "
            "rendering. Re-run with --force to proceed anyway."
        )
        return []

    cols = [
        "family_id", "family_name", "assignment_status",
        "composition", "nvac", "temp", "sim_path", "sim_label",
        "idx_ref", "energy_barrier", "move_type",
    ]
    classifier = pd.read_csv(classification_csv, usecols=[c for c in cols if c])

    selections: list[SelectedEvent] = []
    pickle_cache: dict[str, "pd.DataFrame"] = {}

    for comp, nvac, temp, idx_ref in sorted(keys):
        match = classifier[
            (classifier.composition.astype(str) == str(comp))
            & (classifier.nvac.astype(int) == int(nvac))
            & (classifier.temp.astype(float).astype(int) == int(temp))
            & (classifier.idx_ref.astype(int) == int(idx_ref))
        ]
        if match.empty:
            print(
                f"[select_curated] SKIP ({comp},{nvac},{temp},{idx_ref}) "
                "— not in classified_events_with_families.csv"
            )
            continue
        row = match.iloc[0]
        sim_path = row.sim_path
        if pd.isna(sim_path) or not sim_path:
            print(f"[select_curated] SKIP idx_ref={idx_ref} — empty sim_path")
            continue
        sim_dir = _paths.resolve_sim_path(_portable_sim_path(row), data_root)
        pickle_path = sim_dir / "reference_table.pickle"
        if not pickle_path.exists():
            print(f"[select_curated] SKIP idx_ref={idx_ref} — pickle missing at {pickle_path}")
            continue

        if str(pickle_path) not in pickle_cache:
            try:
                pickle_cache[str(pickle_path)] = _load_pickle(pickle_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[select_curated] WARN cannot load {pickle_path}: {exc}")
                continue
        pickle_df = pickle_cache[str(pickle_path)]
        fwd_rows = pickle_df[pickle_df.idx_ref == int(row.idx_ref)]
        if fwd_rows.empty:
            print(f"[select_curated] SKIP idx_ref={idx_ref} — not in pickle")
            continue
        fwd = fwd_rows.iloc[0]
        idx_bwd = int(fwd.idx_backward)
        bwd_rows = pickle_df[pickle_df.idx_ref == idx_bwd]
        if bwd_rows.empty:
            print(
                f"[select_curated] WARN idx_ref={idx_ref}: idx_backward={idx_bwd} "
                "missing from pickle (event still included with self-symmetric flag)"
            )
            bwd = fwd  # fall back to self
        else:
            bwd = bwd_rows.iloc[0]

        m = int(fwd.move_atom_idx)
        disp = float(np.linalg.norm(
            np.asarray(fwd.final_positions)[m] - np.asarray(fwd.initial_positions)[m]
        ))
        if disp < MIN_MOVER_DISPLACEMENT_A:
            print(
                f"[select_curated]  flagged idx_ref={idx_ref} has small "
                f"mover displacement {disp:.2f} Å — included anyway (user-curated)"
            )

        sel = SelectedEvent(
            family_id=str(row.family_id) if pd.notna(row.family_id) else f"flag_{idx_ref}",
            family_name=(
                str(row.family_name) if "family_name" in row and pd.notna(row.family_name)
                else str(row.family_id)
            ),
            composition=str(row.composition),
            nvac=int(row.nvac),
            temp=float(row.temp),
            sim_path=str(sim_dir.resolve()),
            sim_label=str(row.sim_label),
            idx_ref=int(row.idx_ref),
            idx_backward=idx_bwd,
            energy_barrier=float(row.energy_barrier),
            reverse_barrier=float(bwd.energy_barrier),
            move_type=str(row.move_type),
            mover_disp_A=disp,
        )
        selections.append(sel)
        print(
            f"[select_curated] {sel.family_id:40s}  Ea={sel.energy_barrier:.3f} eV  "
            f"disp={sel.mover_disp_A:.2f} Å  ({sel.sim_label} #{sel.idx_ref})"
        )

    # Family-id collision: two flagged events from the same family — disambiguate
    # by appending #idx_ref so output filenames don't overwrite each other.
    seen: dict[str, int] = {}
    for sel in selections:
        seen[sel.family_id] = seen.get(sel.family_id, 0) + 1
    for sel in selections:
        if seen[sel.family_id] > 1:
            sel.family_id = f"{sel.family_id}_idx{sel.idx_ref}"

    out_manifest.write_text(json.dumps([s.to_dict() for s in selections], indent=2))
    print(f"[select_curated] wrote {len(selections)} events → {out_manifest}")
    return selections


# ---------------------------------------------------------------------------
# Step 2: prepare (host-venv, batch npz dump)
# ---------------------------------------------------------------------------


def run_neb_all(
    manifest_path: Path | str,
    potential: Path | str,
    npz_dir: Path | str | None = None,
    neb_root: Path | str | None = None,
    neb_npz_dir: Path | str | None = None,
    n_images: int = 11,
    **kwargs,
) -> list[Path]:
    """Run LAMMPS NEB per selected event; save per-event NEB npz.

    For each event pair in the manifest:
    1. Load EventPair via load_event_pair_npz.
    2. Run NEB via Analysis.blender_viewer.neb_lammps.run_neb_for_pair with
       the interatomic ``potential`` file (extra keyword arguments such as
       ``lmp_binary`` / ``mpirun`` / ``pair_style`` are passed through).
    3. Save (n_images, N_atoms, 3) positions + diagnostics to
       ``neb_npz_dir/<family>.npz``.

    Returns list of npz paths. Skips events with Ea mismatch > 0.2 eV
    (logs a warning but continues — the user can regenerate that event
    specifically after investigation).
    """
    from Analysis.blender_viewer.data import load_event_pair_npz  # noqa: PLC0415
    from Analysis.blender_viewer.neb_lammps import (  # noqa: PLC0415
        run_neb_for_pair,
        save_neb_npz,
    )

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    if npz_dir is None:
        npz_dir = root / "npz"
    npz_dir = Path(npz_dir)
    if neb_root is None:
        neb_root = root / "neb"
    neb_root = Path(neb_root)
    if neb_npz_dir is None:
        neb_npz_dir = root / "neb_npz"
    neb_npz_dir = Path(neb_npz_dir)
    neb_root.mkdir(parents=True, exist_ok=True)
    neb_npz_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    out_paths: list[Path] = []

    for sel in manifest:
        family_safe = _safe(sel["family_id"])
        in_npz = npz_dir / f"event_{family_safe}.npz"
        event_neb_dir = neb_root / f"event_{family_safe}"
        out_npz = neb_npz_dir / f"event_{family_safe}.npz"

        pair = load_event_pair_npz(in_npz)
        try:
            result = run_neb_for_pair(
                pair, event_neb_dir, potential=potential, n_images=n_images, **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[neb] FAIL {sel['family_id']}: {e}")
            continue

        save_neb_npz(out_npz, result["positions"], result["types"], result)
        out_paths.append(out_npz)
        err = result.get("ea_error_eV")
        marker = "!" if (err is not None and err > 0.2) else " "
        print(
            f"[neb]{marker} {sel['family_id']:40s}  "
            f"Ea_pickle={result['pickle_ea_forward']:.3f}  "
            f"Ea_NEB={result['neb_ea_forward']:.3f}  "
            f"|Δ|={err:.3f} eV" if err is not None else
            f"[neb]  {sel['family_id']:40s}  Ea_NEB={result['neb_ea_forward']:.3f}"
        )

    return out_paths


def prepare_all(
    manifest_path: Path | str,
    npz_dir: Path | str | None = None,
) -> list[Path]:
    """Dump an EventPair npz per selected event. Returns list of npz paths."""
    from Analysis.blender_viewer.data import dump_event_pair_npz, load_event_pair  # noqa: PLC0415

    manifest_path = Path(manifest_path)
    if npz_dir is None:
        npz_dir = manifest_path.parent / "npz"
    npz_dir = Path(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    out_paths: list[Path] = []
    skipped: list[tuple[str, str]] = []

    for sel in manifest:
        try:
            pair = load_event_pair(
                Path(sel["sim_path"]), idx_ref_fwd=sel["idx_ref"],
            )
        except Exception as exc:  # noqa: BLE001
            # Symmetric events (idx_backward == idx_ref) and other
            # pickle-resolution failures shouldn't kill the whole batch.
            print(
                f"[prepare] SKIP {sel['family_id']:40s} "
                f"idx_ref={sel['idx_ref']} — {type(exc).__name__}: {exc}"
            )
            skipped.append((sel["family_id"], str(exc)))
            continue
        out = npz_dir / f"event_{_safe(sel['family_id'])}.npz"
        dump_event_pair_npz(pair, out)
        out_paths.append(out)
        print(f"[prepare] {sel['family_id']:40s} → {out.name}")

    if skipped:
        print(
            f"[prepare] {len(out_paths)}/{len(manifest)} prepared; "
            f"{len(skipped)} skipped."
        )
    return out_paths


def _safe(s: str) -> str:
    """Sanitize a family_id for use in a filename."""
    return s.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Step 3: build (Blender, per-event .blend with title overlay)
# ---------------------------------------------------------------------------


def build_all(
    manifest_path: Path | str,
    npz_dir: Path | str | None = None,
    blend_dir: Path | str | None = None,
    neb_npz_dir: Path | str | None = None,
    style: str = "loop",
    camera_style: str = "orbit",
    embed_title: bool = False,
    close_pack: bool = True,
    use_neb: bool = False,
    theme: str = "default",
    flat_shade: bool = False,
    motion_cycles_per_pan: int = 1,
    cooperative_highlight: bool = False,
    coop_displacement_thr: float = 0.5,
) -> list[Path]:
    """Call cli.main for each npz; add a title Text object; save .blend.

    Must run inside Blender's Python (e.g. ``blender --background``). One
    call builds every event; the shared scene-builder ``_clear_scene`` wipes
    the scene between iterations.
    """
    import bpy  # noqa: PLC0415

    from Analysis.blender_viewer.cli import main as cli_main  # noqa: PLC0415

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    if npz_dir is None:
        npz_dir = root / "npz"
    npz_dir = Path(npz_dir)
    if blend_dir is None:
        blend_dir = root / "blend"
    blend_dir = Path(blend_dir)
    blend_dir.mkdir(parents=True, exist_ok=True)
    if neb_npz_dir is None:
        neb_npz_dir = root / "neb_npz"
    neb_npz_dir = Path(neb_npz_dir)

    manifest = json.loads(manifest_path.read_text())
    out_paths: list[Path] = []

    for sel in manifest:
        family_safe = _safe(sel["family_id"])
        npz_path = npz_dir / f"event_{family_safe}.npz"
        blend_path = blend_dir / f"event_{family_safe}.blend"

        neb_npz: Path | None = None
        if use_neb:
            candidate = neb_npz_dir / f"event_{family_safe}.npz"
            if candidate.exists():
                neb_npz = candidate
            else:
                print(f"[build] WARN: use_neb=True but {candidate} missing, skipping NEB for this event")

        # Build the loop+orbit scene (cli.main clears state internally).
        cli_main(
            npz_path=npz_path,
            out=blend_path,
            save=False,
            style=style,
            camera_style=camera_style,
            close_pack=close_pack,
            neb_npz=neb_npz,
            theme=theme,
            flat_shade=flat_shade,
            motion_cycles_per_pan=motion_cycles_per_pan,
            cooperative_highlight=cooperative_highlight,
            coop_displacement_thr=coop_displacement_thr,
        )
        if embed_title:
            title_text = f"{sel['family_id']}\nEa = {sel['energy_barrier']:.3f} eV"
            _add_title_overlay(title_text)

        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        out_paths.append(blend_path)
        print(f"[build] {sel['family_id']:40s} → {blend_path.name}")

    return out_paths


def _add_title_overlay(text: str) -> None:
    """Add a small Text object pinned to the active camera's bottom-left.

    Size 0.15 at local Z = -3 puts a corner label that's readable but
    doesn't dominate the frame. Still better to do label overlays via
    ffmpeg drawtext at compile time for full control — this is here so
    the .blend file is informative if opened directly.
    """
    import bpy  # noqa: PLC0415

    cam = bpy.context.scene.camera
    if cam is None:
        return

    txt_data = bpy.data.curves.new(name="TitleText", type="FONT")
    txt_data.body = text
    txt_data.size = 0.15
    txt_data.align_x = "LEFT"
    txt_data.align_y = "BOTTOM"

    txt_obj = bpy.data.objects.new("Title", txt_data)
    txt_obj.parent = cam
    # In camera local coords, -Z is forward. Place ~3 m in front, bottom-left.
    # With ortho camera scale ~2× bbox_radius, (-1.3, -0.7) sits near the
    # lower-left corner of a 16:9 frame.
    txt_obj.location = (-1.3, -0.7, -3.0)
    bpy.context.scene.collection.objects.link(txt_obj)

    # White emissive material.
    mat = bpy.data.materials.new("TitleMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (1.0, 1.0, 1.0, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 3.0
    txt_obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Step 4: render (Blender headless, PNG sequence)
# ---------------------------------------------------------------------------


_RENDER_DRIVER = (
    Path(__file__).parent / "render_one_blend.py"
)


def _blender_binary(explicit: str | Path | None = None) -> str:
    """Locate Blender (argument, ``PYKMC_ANALYSIS_BLENDER``, ``PATH``) or raise."""
    found = _paths.tool("blender", str(explicit) if explicit else None)
    if found is None:
        raise FileNotFoundError(
            "Blender not found. Install Blender 4.2+ and put `blender` on PATH, "
            "set PYKMC_ANALYSIS_BLENDER=/path/to/blender, or pass --blender."
        )
    return found


def render_all(
    manifest_path: Path | str,
    blend_dir: Path | str | None = None,
    frames_dir: Path | str | None = None,
    engine: str = "EEVEE_NEXT",
    resolution: tuple[int, int] = (1280, 720),
    fps: int = 60,
    parallel: int = 1,
    blender_binary: str | Path | None = None,
    skip_existing: bool = False,
    samples: int = 32,
) -> list[Path]:
    """Render each blend's animation as a PNG sequence via parallel
    ``blender --background`` subprocesses.

    Each blend gets its own Blender process — keyframe state never carries
    across blends. ``parallel=N`` runs N processes simultaneously, each
    writing to its own ``frames/event_<family>/`` directory.

    Initially observed deadlocks at ~frame 134 on both ``--parallel 1``
    and ``--parallel 2`` runs were caused by ``subprocess.PIPE`` buffer
    fill (Blender writes "Saved frame" lines faster than the parent
    drained the 64-KB pipe). That's now fixed — each worker writes
    stdout/stderr to a per-blend log under ``render_logs/`` instead of
    PIPE. Both ``parallel=1`` (sequential, default) and ``parallel=2`` (or
    higher) should work. The default stays at 1 because GPU thermals on
    Apple Silicon usually mean parallel scaling is sub-1.5× and not
    worth the risk of OOM on dense scenes.

    With ``skip_existing=True``, blends whose frames directory already
    contains a full set of PNGs (>= 800) are skipped — useful when
    re-running after a partial failure.

    ``blender_binary`` overrides the ``PYKMC_ANALYSIS_BLENDER`` / ``PATH``
    lookup.
    """
    import time  # noqa: PLC0415
    from subprocess import PIPE, Popen  # noqa: PLC0415

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    if blend_dir is None:
        blend_dir = root / "blend"
    blend_dir = Path(blend_dir)
    if frames_dir is None:
        frames_dir = root / "frames"
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not _RENDER_DRIVER.exists():
        raise FileNotFoundError(
            f"render driver missing at {_RENDER_DRIVER} — "
            "tracked file should be in the package"
        )
    blender = _blender_binary(blender_binary)

    manifest = json.loads(manifest_path.read_text())

    # Build the queue of (family_safe, blend_path, out_dir) entries.
    queue: list[tuple[str, Path, Path]] = []
    for sel in manifest:
        family_safe = _safe(sel["family_id"])
        blend_path = blend_dir / f"event_{family_safe}.blend"
        out_dir = frames_dir / f"event_{family_safe}"
        if not blend_path.exists():
            print(f"[render] SKIP {family_safe} — blend missing")
            continue
        if skip_existing and out_dir.exists() and len(list(out_dir.glob("*.png"))) >= 800:
            print(f"[render] SKIP {family_safe} — already rendered")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any stale partial output so frame numbering is clean.
        for old in out_dir.glob("*.png"):
            old.unlink()
        queue.append((family_safe, blend_path, out_dir))

    print(
        f"[render] queue: {len(queue)} blends, parallel={parallel}, "
        f"engine={engine}, {resolution[0]}x{resolution[1]}@{fps}fps"
    )

    out_dirs: list[Path] = []
    active: list[tuple[str, Popen, Path, float, Path]] = []
    queue_iter = iter(queue)
    t_start = time.monotonic()

    # Log directory for per-blend stdout/stderr. CRITICAL: writing to files
    # (not subprocess.PIPE) avoids the 64-KB pipe-buffer deadlock — Blender
    # writes "Saved frame" lines fast enough that PIPE fills, child blocks
    # on write, parent never reads → both processes appear alive but stuck.
    log_dir = frames_dir.parent / "render_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _spawn_one(family_safe: str, blend_path: Path, out_dir: Path) -> tuple[Popen, float, Path]:
        cmd = [
            blender, "--background", str(blend_path),
            "--python", str(_RENDER_DRIVER), "--",
            "--out-dir", str(out_dir),
            "--engine", engine,
            "--fps", str(fps),
            "--width", str(resolution[0]),
            "--height", str(resolution[1]),
            "--samples", str(samples),
        ]
        log_path = log_dir / f"render_{family_safe}.log"
        log_fh = log_path.open("wb")
        return Popen(cmd, stdout=log_fh, stderr=log_fh), time.monotonic(), log_path

    while True:
        # Fill up to `parallel` concurrent workers.
        while len(active) < parallel:
            try:
                family_safe, blend_path, out_dir = next(queue_iter)
            except StopIteration:
                break
            p, t0, log_path = _spawn_one(family_safe, blend_path, out_dir)
            print(f"[render] start  {family_safe}  (active: {len(active) + 1}/{parallel}, log: {log_path.name})")
            active.append((family_safe, p, out_dir, t0, log_path))

        if not active:
            break

        # Poll workers for completion. ~1 s polling is fine since
        # individual renders take minutes.
        time.sleep(1.0)
        for entry in list(active):
            family_safe, p, out_dir, t0, log_path = entry
            if p.poll() is None:
                continue
            elapsed = time.monotonic() - t0
            rc = p.returncode
            if rc == 0:
                n = len(list(out_dir.glob("*.png")))
                print(f"[render] done   {family_safe}  ({n} frames, {elapsed:.1f}s)")
                out_dirs.append(out_dir)
            else:
                stderr_tail = log_path.read_text(errors="replace").splitlines()[-15:]
                print(
                    f"[render] FAIL  {family_safe}  exit={rc} "
                    f"({elapsed:.1f}s)\n  log: {log_path}\n  " + "\n  ".join(stderr_tail)
                )
            active.remove(entry)

    total = time.monotonic() - t_start
    print(f"[render] all done — {len(out_dirs)}/{len(queue)} succeeded in {total:.1f}s")
    return out_dirs


# ---------------------------------------------------------------------------
# Step 5: compile (ffmpeg per-event MP4 + concat)
# ---------------------------------------------------------------------------


def compile_showreel(
    manifest_path: Path | str,
    frames_dir: Path | str | None = None,
    mp4_dir: Path | str | None = None,
    out_mp4: Path | str | None = None,
    fps: int = 60,
    crf: int = 18,
    title_card_seconds: float = 0.5,
    add_drawtext: bool = True,
) -> Path:
    """Per-family MP4 + final concat MP4 with title cards.

    ``add_drawtext`` overlays family_id + barrier at the bottom of each
    frame via ffmpeg drawtext. Default True — this is the authoritative
    label path. Pair with ``build_all(embed_title=False)``.

    Requires ``ffmpeg`` on PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH; install via brew/apt")

    font = _drawtext_font()

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    if frames_dir is None:
        frames_dir = root / "frames"
    frames_dir = Path(frames_dir)
    if mp4_dir is None:
        mp4_dir = root / "mp4"
    mp4_dir = Path(mp4_dir)
    mp4_dir.mkdir(parents=True, exist_ok=True)
    if out_mp4 is None:
        out_mp4 = root / "showreel_important_events.mp4"
    out_mp4 = Path(out_mp4)

    manifest = json.loads(manifest_path.read_text())

    # Per-family MP4 (drawtext overlay optional). Skip events whose frame
    # directory is missing or empty (e.g. symmetric events filtered out
    # at prepare time).
    per_event_mp4s: list[Path] = []
    rendered_manifest: list[dict] = []
    for sel in manifest:
        family_safe = _safe(sel["family_id"])
        fdir = frames_dir / f"event_{family_safe}"
        if not fdir.exists() or not any(fdir.glob("frame_*.png")):
            print(
                f"[compile] SKIP {sel['family_id']:40s} "
                f"— no frames in {fdir}"
            )
            continue
        out_path = mp4_dir / f"event_{family_safe}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(fdir / "frame_%04d.png"),
        ]
        if add_drawtext:
            label = f"{sel['family_id']}  Ea={sel['energy_barrier']:.3f} eV"
            drawtext = (
                f"drawtext={font}text='{_ffmpeg_escape(label)}'"
                ":x=20:y=h-60:fontsize=28:fontcolor=white"
                ":box=1:boxcolor=black@0.55:boxborderw=8"
            )
            cmd.extend(["-vf", drawtext])
        cmd.extend([
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            str(out_path),
        ])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        per_event_mp4s.append(out_path)
        rendered_manifest.append(sel)
        print(f"[compile] {sel['family_id']:40s} → {out_path.name}")

    # Build title cards (1 per event) + concat list.
    card_dir = mp4_dir / "cards"
    card_dir.mkdir(exist_ok=True)
    concat_entries: list[str] = []
    for sel, mp4 in zip(rendered_manifest, per_event_mp4s, strict=True):
        # Single-line title card — ffmpeg drawtext does not interpret
        # backslash-n as a line break without extra setup, so use a separator.
        card_text = (
            f"{sel['family_id']}  |  {sel['move_type']}  |  "
            f"Ea = {sel['energy_barrier']:.3f} eV  |  {sel['sim_label']}"
        )
        card = _render_title_card(
            text=_ffmpeg_escape(card_text),
            out_path=card_dir / f"card_{_safe(sel['family_id'])}.mp4",
            seconds=title_card_seconds,
            fps=fps,
            font=font,
        )
        concat_entries.append(str(card))
        concat_entries.append(str(mp4))

    concat_txt = mp4_dir / "concat.txt"
    concat_txt.write_text("\n".join(f"file '{p}'" for p in concat_entries))

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"[compile] showreel → {out_mp4}")
    return out_mp4


def _render_title_card(
    text: str, out_path: Path, seconds: float, fps: int, font: str,
) -> Path:
    # fontsize=26 fits the long "family_id | move_type | Ea = X.XXX eV | Nvac_TXXX"
    # labels into 1280 px with margin. For shorter labels, bump to 32-36.
    drawtext = (
        f"drawtext={font}text='{text}'"
        ":x=(w-text_w)/2:y=(h-text_h)/2:fontsize=26:fontcolor=white"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r={fps}:d={seconds}",
        "-vf", drawtext,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def _drawtext_font() -> str:
    """``fontfile=<path>:`` for drawtext, or "" to use ffmpeg's default font."""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return f"fontfile={p}:"
    return ""


def _ffmpeg_escape(s: str) -> str:
    """Escape characters that break ffmpeg's drawtext parser."""
    return s.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


# ---------------------------------------------------------------------------
# Step 6: orchestrator + CLI
# ---------------------------------------------------------------------------


DEFAULT_DRAFT_FAMILY = "surface_1NN_inplane"


def draft_render_one(
    family_id: str = DEFAULT_DRAFT_FAMILY,
    manifest_path: Path | str | None = None,
    out_root: Path | str | None = None,
    engine: str = "EEVEE_NEXT",
    fps: int = 60,
    width: int = 1280,
    height: int = 720,
    blender_binary: str | Path | None = None,
) -> Path:
    """Render ONE event's .blend to a PNG sequence + single MP4.

    Used during iteration on visual settings — cuts the feedback loop from
    30 min (batch) to ~3 min (single event). The .blend must already be
    built by ``build_all`` (or equivalent) with the settings under
    evaluation. Returns the path to the single-event MP4.

    The canonical draft event is ``surface_1NN_inplane`` — a clean 1NN
    hop, moderate barrier, good exemplar for most visual tweaks. Change
    with ``family_id='...'`` if a specific event type reveals a bug
    better. ``blender_binary`` overrides the ``PYKMC_ANALYSIS_BLENDER`` /
    ``PATH`` lookup. Requires ``ffmpeg`` on PATH.
    """
    if out_root is None:
        out_root = DEFAULT_ROOT
    out_root = Path(out_root)
    if manifest_path is None:
        manifest_path = out_root / "manifest.json"
    manifest_path = Path(manifest_path)

    family_safe = _safe(family_id)
    blend_path = out_root / "blend" / f"event_{family_safe}.blend"
    if not blend_path.exists():
        raise FileNotFoundError(f"no .blend for {family_id} at {blend_path}")
    blender = _blender_binary(blender_binary)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH")

    frames_dir = out_root / "frames" / f"event_{family_safe}"
    mp4_dir = out_root / "mp4"
    draft_mp4 = out_root / f"draft_{family_safe}.mp4"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Headless Blender render.
    blender_cmd = [
        blender,
        "--background", str(blend_path),
        "--python", str(_RENDER_DRIVER),
        "--",
        "--out-dir", str(frames_dir),
        "--engine", engine,
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
    ]
    print(f"[draft] rendering {family_id} → {frames_dir}")
    subprocess.run(blender_cmd, check=True)

    # Build single-event MP4 with drawtext overlay.
    font = _drawtext_font()
    manifest = json.loads(manifest_path.read_text())
    sel = next((s for s in manifest if s["family_id"] == family_id), None)
    if sel is None:
        raise KeyError(f"family_id {family_id!r} not in manifest")
    label = f"{family_id}  Ea={sel['energy_barrier']:.3f} eV"
    drawtext = (
        f"drawtext={font}text='{_ffmpeg_escape(label)}'"
        ":x=20:y=h-60:fontsize=28:fontcolor=white"
        ":box=1:boxcolor=black@0.55:boxborderw=8"
    )
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", drawtext,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(draft_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"[draft] MP4 → {draft_mp4}  ({draft_mp4.stat().st_size / 1024 / 1024:.1f} MB)")
    return draft_mp4


def run_all(
    csv_path: Path | str = CLASSIFIED_WITH_FAMILIES_CSV,
    out_root: Path | str = DEFAULT_ROOT,
    engine: str = "EEVEE_NEXT",
    data_root: Path | str | None = None,
) -> Path:
    """Host-Python one-shot: select + prepare (the Blender steps follow separately)."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.json"

    select_representatives(csv_path=csv_path, out_manifest=manifest, data_root=data_root)
    prepare_all(manifest_path=manifest, npz_dir=out_root / "npz")
    print(
        "[run_all] selection + prep complete. "
        "Next: run build_all + render_all inside Blender, "
        "then compile_showreel in the host shell."
    )
    return manifest


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=["select", "curated", "prepare", "neb", "build", "render",
                 "compile", "draft", "all"],
        help="which pipeline stage to run (curated = read render_flag from "
             "audit_log.csv; draft = render ONE event for fast iteration)",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"output directory for every stage (default: {DEFAULT_ROOT})")
    parser.add_argument("--csv", type=Path, default=CLASSIFIED_WITH_FAMILIES_CSV)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="(select / curated / all) directory holding the pyKMC "
                             "runs, for relative sim_path values in the CSV "
                             "(default: PYKMC_ANALYSIS_DATA_ROOT)")
    parser.add_argument("--engine", default="EEVEE_NEXT")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--family-id", default=DEFAULT_DRAFT_FAMILY,
                        help="(draft only) family_id of the event to render")
    parser.add_argument("--manifest", default=None,
                        help="manifest filename relative to --root "
                             "(default: manifest.json; use 'curated_manifest.json' "
                             "after `curated` step)")
    parser.add_argument("--out-mp4", default=None,
                        help="(compile only) override final MP4 filename "
                             "(default: showreel_important_events.mp4)")
    parser.add_argument("--force", action="store_true",
                        help="(curated only) skip the >20-event safety check")
    parser.add_argument("--parallel", type=int, default=1,
                        help="(render only) number of concurrent Blender "
                             "processes")
    parser.add_argument("--samples", type=int, default=32,
                        help="(render, CYCLES only) Cycles samples per frame")
    parser.add_argument("--skip-existing", action="store_true",
                        help="(render only) skip blends whose frames dir "
                             "already has 800+ PNGs (resume partial renders)")
    parser.add_argument("--blender", default=None,
                        help="(render / draft) Blender executable "
                             "(default: PYKMC_ANALYSIS_BLENDER, then PATH)")
    parser.add_argument("--potential", type=Path, default=None,
                        help="(neb, required) interatomic potential file for LAMMPS")
    parser.add_argument("--lammps", default=None,
                        help="(neb) LAMMPS executable built with REPLICA "
                             "(default: PYKMC_ANALYSIS_LAMMPS, then PATH)")
    parser.add_argument("--mpirun", default=None,
                        help="(neb) MPI launcher (default: PYKMC_ANALYSIS_MPIRUN, then PATH)")
    # Under Blender (``blender --background --python-expr ... -- build``) our
    # arguments follow ``--``; under plain Python they are sys.argv[1:].
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    manifest_name = args.manifest or "manifest.json"
    manifest = args.root / manifest_name
    args.root.mkdir(parents=True, exist_ok=True)
    out_mp4_name = args.out_mp4 or (
        "curated_showreel.mp4"
        if "curated" in manifest_name
        else "showreel_important_events.mp4"
    )

    if args.step == "select":
        select_representatives(
            csv_path=args.csv, out_manifest=manifest, data_root=args.data_root,
        )
    elif args.step == "curated":
        # If user didn't override --manifest, default to curated_manifest.json
        # for this step so they don't accidentally overwrite manifest.json.
        if args.manifest is None:
            manifest = args.root / DEFAULT_CURATED_MANIFEST
        select_curated(
            classification_csv=args.csv,
            out_manifest=manifest,
            force=args.force,
            data_root=args.data_root,
        )
    elif args.step == "prepare":
        prepare_all(manifest_path=manifest, npz_dir=args.root / "npz")
    elif args.step == "neb":
        if args.potential is None:
            parser.error("the neb step needs --potential /path/to/potential")
        run_neb_all(
            manifest_path=manifest,
            potential=args.potential,
            npz_dir=args.root / "npz",
            neb_root=args.root / "neb",
            neb_npz_dir=args.root / "neb_npz",
            lmp_binary=args.lammps,
            mpirun=args.mpirun,
        )
    elif args.step == "build":
        try:
            import bpy  # noqa: F401, PLC0415
        except ImportError:
            raise SystemExit(
                "The build step runs inside Blender. From the repository root:\n"
                "  blender --background --python-expr \"import sys; sys.path.insert(0, '.'); "
                "from Analysis.blender_viewer.showreel import _cli; _cli()\" -- build"
            ) from None
        build_all(
            manifest_path=manifest,
            npz_dir=args.root / "npz",
            blend_dir=args.root / "blend",
            neb_npz_dir=args.root / "neb_npz",
            use_neb=True,
        )
    elif args.step == "render":
        render_all(
            manifest_path=manifest,
            blend_dir=args.root / "blend",
            frames_dir=args.root / "frames",
            engine=args.engine,
            fps=args.fps,
            parallel=args.parallel,
            blender_binary=args.blender,
            skip_existing=args.skip_existing,
            samples=args.samples,
        )
    elif args.step == "compile":
        compile_showreel(
            manifest_path=manifest,
            frames_dir=args.root / "frames",
            mp4_dir=args.root / "mp4",
            out_mp4=args.root / out_mp4_name,
            fps=args.fps,
        )
    elif args.step == "draft":
        draft_render_one(
            family_id=args.family_id,
            manifest_path=manifest,
            out_root=args.root,
            engine=args.engine,
            fps=args.fps,
            blender_binary=args.blender,
        )
    elif args.step == "all":
        run_all(
            csv_path=args.csv, out_root=args.root, engine=args.engine,
            data_root=args.data_root,
        )


if __name__ == "__main__":
    _cli()
