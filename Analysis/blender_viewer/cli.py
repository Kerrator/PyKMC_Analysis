"""CLI entry point for building a Blender event-pair scene.

Designed to run inside Blender's Python. The reference_table.pickle is loaded
*outside* Blender (by ``prepare.py``) because Blender's numpy may mismatch
the pickle's original numpy. ``cli.main`` consumes the npz intermediate.

Headless, from the repository root::

    blender --background --python-expr \\
        "import sys; sys.path.insert(0, '.'); from Analysis.blender_viewer.cli import _cli; _cli()" \\
        -- --npz event_6467.npz --style loop --camera orbit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from Analysis.blender_viewer.data import (
    EventPair,
    load_event_pair_npz,
    mover_displacement,
)
from Analysis.blender_viewer.scene_builder import CameraStyle, Style, build_scene


def main(
    npz_path: Path | str,
    out: Path | str | None = None,
    save: bool = True,
    style: Style = "dual",
    camera_style: CameraStyle = "static",
    close_pack: bool = False,
    close_pack_factor: float = 0.55,
    neb_npz: Path | str | None = None,
    theme: str = "default",
    flat_shade: bool = False,
    motion_cycles_per_pan: int = 1,
    cooperative_highlight: bool = False,
    coop_displacement_thr: float = 0.5,
) -> dict:
    """Build the scene from a prepared npz and optionally save the .blend.

    When ``close_pack=True``, the atom radius is set to
    ``close_pack_factor * min_pairwise_distance(pair)``. Mover boost is
    dropped in this mode. Factor ≤ 0.50 guarantees no overlap even at the
    saddle; factor > 0.50 (the default 0.55 produces a visually chunkier
    crystal) accepts brief overlap at the tightest frame in exchange for
    more substantial ball rendering.

    When ``neb_npz`` is provided, uses the N-image NEB positions for
    keyframing instead of the default 3-waypoint init/saddle/final.
    Requires ``style='loop'``.

    When ``cooperative_highlight=True``, every atom whose net displacement
    ``|final - init|`` exceeds ``coop_displacement_thr`` (Å) is treated
    as a mover (boost + emission + theme's ``mover_color``). Pairs with
    ``theme='burgundy_gold_coop'`` for the 3-atom cooperative-hop look.

    When ``save=True`` and ``out`` is None, the .blend is written to the
    current directory as ``event_<fwd>_<bwd>_<sim_label>[_<style>_<camera>].blend``.
    """
    import bpy  # noqa: PLC0415

    from Analysis.blender_viewer.scene_builder import min_pairwise_distance  # noqa: PLC0415

    pair = load_event_pair_npz(Path(npz_path))

    neb_images = None
    if neb_npz is not None:
        from Analysis.blender_viewer.neb_lammps import load_neb_npz  # noqa: PLC0415
        neb_images, _types, neb_diag = load_neb_npz(Path(neb_npz))
        print(
            f"[blender_viewer] NEB images: {neb_images.shape} — "
            f"pickle Ea={neb_diag['pickle_ea_forward']:.3f} eV, "
            f"NEB Ea={neb_diag['neb_ea_forward']:.3f} eV"
        )

    print(
        f"[blender_viewer] loaded pair fwd={pair.forward.idx_ref} "
        f"bwd={pair.backward.idx_ref} for {pair.sim_label} — "
        f"fwd Ea={pair.forward.energy_barrier:.3f} eV "
        f"disp={mover_displacement(pair.forward):.3f} Å | "
        f"style={style} camera={camera_style} close_pack={close_pack} "
        f"neb={'yes' if neb_images is not None else 'no'}"
    )

    radius_override = None
    mover_boost = None
    if close_pack:
        min_d = min_pairwise_distance(pair)
        radius_override = close_pack_factor * min_d
        mover_boost = 1.0
        print(
            f"[blender_viewer] close_pack: min_pairwise_distance={min_d:.3f} Å "
            f"→ uniform radius={radius_override:.3f} Å"
        )

    refs = build_scene(
        pair, style=style, camera_style=camera_style,
        radius_override=radius_override, mover_boost=mover_boost,
        neb_images=neb_images,
        theme=theme, flat_shade=flat_shade,
        motion_cycles_per_pan=motion_cycles_per_pan,
        cooperative_highlight=cooperative_highlight,
        coop_displacement_thr=coop_displacement_thr,
    )
    _verify_animation(pair, refs)

    if save:
        if out is not None:
            out_path = Path(out)
        else:
            suffix = "" if style == "dual" and camera_style == "static" else (
                f"_{style}_{camera_style}"
            )
            out_path = Path.cwd() / (
                f"event_{pair.forward.idx_ref}_{pair.backward.idx_ref}"
                f"_{pair.sim_label}{suffix}.blend"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
        print(f"[blender_viewer] saved {out_path}")

    return refs


def _verify_animation(pair: EventPair, refs: dict) -> None:
    """Assert keyframes land at the stored positions (style-specific)."""
    import bpy  # noqa: PLC0415

    style = refs.get("style")
    frames = refs["frames"]
    fwd = pair.forward

    if style == "dual":
        bpy.context.scene.frame_set(frames.fwd_final)
        _assert_mover_at(refs["fwd_mover"], fwd.final_positions[fwd.mover_idx],
                         label=f"dual/fwd_mover@{frames.fwd_final}")

        bpy.context.scene.frame_set(frames.bwd_final)
        _assert_mover_at(
            refs["bwd_mover"],
            pair.backward.final_positions[pair.backward.mover_idx],
            label=f"dual/bwd_mover@{frames.bwd_final}",
        )
        print("[blender_viewer] dual animation correctness verified")
    elif style == "loop":
        if hasattr(frames, "forward_frames"):
            # NEB loop. With multi-cycle playback the waypoint schedule is
            # arbitrary, so verify at the FIRST and LAST waypoints (both
            # must land on NEB image 0 == init position) rather than the
            # hard-coded forward/backward-frames midpoints.
            wf = refs.get("waypoint_frames") or frames.forward_frames
            wi = refs.get("waypoint_images") or list(range(frames.n_images))
            # Frame 1 → image 0 (init), and last frame → image 0 (loop close).
            bpy.context.scene.frame_set(wf[0])
            assert wi[0] == 0, "first waypoint must be NEB image 0"
            _assert_mover_at(refs["mover"], fwd.init_positions[fwd.mover_idx],
                             label=f"loop_neb/mover@{wf[0]}", tol=0.15)
            bpy.context.scene.frame_set(wf[-1])
            assert wi[-1] == 0, "last waypoint must be NEB image 0 (loop close)"
            _assert_mover_at(refs["mover"], fwd.init_positions[fwd.mover_idx],
                             label=f"loop_neb/mover@{wf[-1]}", tol=0.15)
            print("[blender_viewer] loop+NEB animation correctness verified")
        else:
            bpy.context.scene.frame_set(frames.final)
            _assert_mover_at(refs["mover"], fwd.final_positions[fwd.mover_idx],
                             label=f"loop/mover@{frames.final}")
            bpy.context.scene.frame_set(frames.init_return)
            _assert_mover_at(refs["mover"], fwd.init_positions[fwd.mover_idx],
                             label=f"loop/mover@{frames.init_return}")
            print("[blender_viewer] loop animation correctness verified")


def _assert_mover_at(obj, expected: np.ndarray, label: str, tol: float = 1e-3) -> None:
    actual = np.array(obj.matrix_world.translation)
    err = float(np.linalg.norm(actual - expected))
    if err > tol:
        raise AssertionError(
            f"{label}: expected {expected}, got {actual} (err {err:.3e})"
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build a Blender event-pair scene.")
    parser.add_argument("--npz", required=True, help="prepared EventPair .npz")
    parser.add_argument("--out", default=None,
                        help="output .blend path (default: event_<...>.blend in "
                             "the current directory)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--style", default="dual", choices=["dual", "loop"])
    parser.add_argument("--camera", default="static",
                        choices=["static", "orbit", "dolly"])
    # Under Blender our arguments follow ``--``; under plain Python they are sys.argv[1:].
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    main(
        npz_path=args.npz,
        out=args.out,
        save=not args.no_save,
        style=args.style,
        camera_style=args.camera,
    )


if __name__ == "__main__":
    _cli()
