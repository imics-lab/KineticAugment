"""
Week-4 augmentations for the round-trip demo: every profile entry alone, and whole profiles.

`export_week4` turns one clip's unsmoothed round trip (the `face_pipeline.run_clip` result the
demo already fitted) into a `ClipData`, fits its weak camera (`scene.fit_clip_camera`) and runs
`AugmentationChain`:

  - every entry of the ambitious profile alone, at probability 1 and that profile's ranges. The
    time warp runs once per mode (uniform, varied, segment) and the frame drop once per mode
    (rate, random). Two runs force a draw that shows: the mirror always mirrors (p = 1), and
    hand dropout uses the top of its start-rate range (0.05). Each override is written into the
    bundle (`overrides`, `note`) and shown on the page.
  - each shipped profile whole (default, ambitious, literature), as one augmented sample.

The unaugmented clip goes through the same `build_outputs` and is written as `aug/w4.base/`:
the original the week-4 runs are compared with. It is the round trip with the observed face
carried on the head, where the Tier 1 original (`v/off`) has the face read back off the mesh.

Each run is written as `aug/<name>/{clip.json, data.bin}` in the Tier 1 bundle format of
`export_round_trip_demo.py` (times, mesh_verts, aug_arm, aug_hands, aug_face, img_*), with
`kind: "week4"`, the run's draws, applied entries, fallbacks, ranges and `report.measure`.
`times` is the source frame each augmented frame shows, replayed from the time-warp and
frame-drop draws, so the page can show the matching video frame.
"""

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from kinetic_augment.augmentations.chain import AugmentationChain
from kinetic_augment.augmentations.clip import ChainContext, clip_from_result
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.profile import load_profile, resolved_ranges
from kinetic_augment.augmentations.report import measure
from kinetic_augment.augmentations.scene import fit_clip_camera
from kinetic_augment.lab.bundle import (  # noqa: F401
    ARMS,
    SCALE_I16,
    bundle_arrays,
    jsonable,
    source_frames,
    write_data,
)
from kinetic_augment.lab.runs import RUNS, Run, merge as _merge, run_profile  # noqa: F401

PROFILES = {
    "default": Path("configs/slr_week4_default.yaml"),
    "ambitious": Path("configs/slr_week4_ambitious.yaml"),
    "literature": Path("configs/slr_week4_literature.yaml"),
}
BASE = "w4.base"
SIDE = ("L", "R")


def identity_diff_m(name: str, chain: AugmentationChain, clip, base_out) -> Optional[float]:
    """Largest point difference (m) from the original when `name` applies its identity draw."""
    aug = chain.augmentations[name]
    ctx = ChainContext(
        skeleton=chain.skeleton,
        parents=np.asarray(chain.skeleton.parents),
        pivot=chain._pivot(clip),
        tier1_draw=chain.tier1_draw_for(clip),
        seed=chain.profile.seed,
    )
    if aug.stage == "outputs":
        out = aug.apply(base_out, aug.identity_draw(), ctx)
    else:
        out = build_outputs(aug.apply(clip, aug.identity_draw(), ctx), chain.skeleton)
    worst = 0.0
    for a, b in (
        (out.pose_world, base_out.pose_world),
        (out.hands_world, base_out.hands_world),
        (out.face_world, base_out.face_world),
    ):
        if a.shape != b.shape:
            return None
        diff = np.abs(a - b)
        if np.isfinite(diff).any():
            worst = max(worst, float(np.nanmax(diff)))
    return worst


def _flag(name: str, draw: Dict[str, Any]) -> str:
    if draw.get("gated"):
        return "not drawn"
    if draw.get("fallback"):
        return f"fallback: {draw['fallback']}"
    return ""


def detail(name: str, draws: Dict[str, Any], clip, frames: int) -> str:
    """A short plain description of what `name` drew, for the menu and the panel captions."""
    if name.startswith("combined."):
        on = [k for k, d in draws.items() if not d.get("gated") and not d.get("fallback")]
        return f"{len(on)} entries drawn, {clip.frames} → {frames} frames"
    entry = RUNS[name].entry
    d = draws.get(entry, {})
    flag = _flag(entry, d)
    if flag and entry != "fingers.handshape_jitter":
        return flag
    if entry == "body.joint_coupled_noise":
        joints = sorted(d.get("joints", []), key=lambda j: -abs(j["angle_deg"]))[:3]
        text = ", ".join(
            f"{j['side'][0].upper()} {j['joint']} {j['angle_deg']:+.1f}°" for j in joints
        )
        return f"largest: {text}" if text else "no joint drawn"
    if entry == "body.limb_length_scaling":
        return ", ".join(f"{b.replace('_', ' ')} ×{f:.3f}" for b, f in d["factors"].items())
    if entry == "body.movement_size":
        return f"×{d['factor']:.3f}"
    if entry == "body.location_jitter":
        x, y, z = d["offset_mm"]
        return f"offset ({x:+.1f}, {y:+.1f}, {z:+.1f}) mm, |{math.hypot(x, y, z):.1f}| mm"
    if entry == "fingers.joint_coupled_noise":
        return f"MCP up to {max(abs(v) for v in d['mcp_deg']):.1f}°, PIP up to " + (
            f"{max(abs(v) for v in d['pip_deg']):.1f}°"
        )
    if entry == "fingers.handshape_jitter":
        if flag:
            return f"{flag} (no redraw kept the handshape class)"
        worst = {
            s: max((abs(v) for row in d["offsets_deg"][s] for v in row), default=0.0)
            for s in ("left", "right")
        }
        return f"largest offset L {worst['left']:.1f}°, R {worst['right']:.1f}°"
    if entry == "face.expression_jitter":
        names = clip.blendshape_names[1:]
        top = sorted(zip(d["offsets"], names), key=lambda p: -abs(p[0]))[:3]
        return ", ".join(f"{n} {v:+.3f}" for v, n in top if v)
    if entry == "temporal.time_warp":
        return f"duration ×{d['duration']:.3f}, {clip.frames} → {frames} frames"
    if entry == "temporal.frame_drop":
        return f"kept {frames} of {clip.frames} frames"
    if entry == "temporal.hand_dropout":
        runs = d.get("runs") or []
        return f"{len(runs)} runs, {d.get('masked_hand_frames', 0)} hand-frames missing"
    if entry == "scene.mirror":
        return "mirrored" if d.get("mirror") else "not mirrored"
    if entry == "scene.view":
        return f"yaw {d['yaw_deg']:+.1f}°, pitch {d['pitch_deg']:+.1f}°"
    if entry == "scene.scale":
        return f"×{d['factor']:.3f}"
    if entry == "scene.uncentre":
        dx, dy = d["image_shift"]
        return f"shift ({dx:+.3f}, {dy:+.3f}) of the frame"
    return ""


def _write(out: Path, arrays, meta):
    out.mkdir(parents=True, exist_ok=True)
    manifest, size = write_data(out / "data.bin", arrays)
    meta["arrays"] = manifest
    (out / "clip.json").write_text(
        json.dumps(jsonable(meta), separators=(",", ":"), allow_nan=False)
    )
    return size


def export_week4(clip_id, result, hands, face, out_root: Path, models, names, smoothing="off"):
    """
    Write the week-4 runs `names` (keys of `RUNS`) of one clip, plus the `w4.base` original.

    `result`, `hands` and `face` are the round trip the Tier 1 augmentations start from (the
    `run_clip` result and the clips it was fitted on).
    """
    started = time.time()
    skeleton, mp_indices = models.skeleton, np.asarray(models.corr.mp_indices, dtype=int)
    profiles = {key: load_profile(path) for key, path in PROFILES.items()}
    clip = clip_from_result(clip_id, result, hands, face, skeleton)
    camera = fit_clip_camera(clip, hands, skeleton, frames=result.frames)
    clip = clip.replace(camera=camera)
    base_full = build_outputs(clip, skeleton, keep_vertices=True)
    base_out = build_outputs(clip, skeleton)
    frames = np.asarray(result.frames, dtype=np.float64)
    camera_meta = {"scale_per_m": list(camera.scale), "residual_frame_pct": 100 * camera.residual}

    arrays, ok = bundle_arrays(base_full, base_full.vertices, frames, mp_indices)
    size = _write(
        out_root / BASE,
        arrays,
        {
            "clip": clip_id,
            "augmentation": BASE,
            "kind": "week4-base",
            "label": "week-4 original",
            "description": "the round trip as the week-4 chain sees it, with no augmentation",
            "title": "week-4 original",
            "smoothing": smoothing,
            "n": clip.frames,
            "mirrored": False,
            "scale_i16": SCALE_I16,
            "body_ok": ok.astype(int).tolist(),
            "camera": camera_meta,
        },
    )
    print(f"  w4 {BASE:<28} {clip.frames} frames, {size / 1e6:.1f} MB")

    for name in names:
        run = RUNS[name]
        t0 = time.time()
        profile = run_profile(run, profiles)
        chain = AugmentationChain(profile, skeleton)
        res = chain.run(clip)
        full = build_outputs(res.clip, skeleton, keep_vertices=True)
        outputs = res.outputs  # hand dropout edits these after the clip stages
        times = frames[0] + source_frames(chain, res, clip.frames)
        temporal = any(a.startswith("temporal.time_warp") for a in res.applied) or any(
            a.startswith("temporal.frame_drop") for a in res.applied
        )
        # A mirror swaps the sides, so a frame-by-frame wrist / bone comparison would set the
        # left hand against the right one: those measures need an unmirrored, aligned run.
        aligned = res.clip.frames == clip.frames and not temporal and not res.clip.scene.mirror
        metrics = measure(base_out, outputs, aligned)
        metrics["identity_max_abs_diff_m"] = (
            identity_diff_m(run.entry, chain, clip, base_out) if run.entry else None
        )
        arrays, ok = bundle_arrays(outputs, full.vertices, times, mp_indices)
        text = detail(name, res.draws, clip, res.clip.frames)
        meta = {
            "clip": clip_id,
            "augmentation": name,
            "kind": "week4",
            "base": BASE,
            "group": run.group,
            "entry": run.entry,
            "profile": profile.name,
            "label": run.label,
            "description": run.description,
            "title": f"{run.label}: {text}" if text else run.label,
            "detail": text,
            "note": run.note,
            "overrides": run.overrides,
            "smoothing": smoothing,
            "n": res.clip.frames,
            "fps": res.clip.fps,
            "mirrored": bool(res.clip.scene.mirror),
            "scale_i16": SCALE_I16,
            "body_ok": ok.astype(int).tolist(),
            "draws": res.draws,
            "applied": res.applied,
            "fallbacks": res.fallbacks,
            "ranges": (
                {run.entry: resolved_ranges(profile, run.entry)}
                if run.entry
                else {
                    e.name: resolved_ranges(profile, e.name) for e in profile.entries if e.enabled
                }
            ),
            "camera": camera_meta,
            "metrics": metrics,
        }
        size = _write(out_root / name, arrays, meta)
        print(
            f"  w4 {name:<28} {res.clip.frames} frames, {size / 1e6:.1f} MB, "
            f"{time.time() - t0:.0f} s: {meta['title']}"
        )
    print(f"  week-4 augmentations done ({time.time() - started:.0f} s)")
