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

PROFILES = {
    "default": Path("configs/slr_week4_default.yaml"),
    "ambitious": Path("configs/slr_week4_ambitious.yaml"),
    "literature": Path("configs/slr_week4_literature.yaml"),
}
BASE = "w4.base"
SCALE_I16 = 10000.0  # mesh vertices stored as int16 in 0.1 mm, as the Tier 1 bundles
ARMS = slice(11, 17)  # MediaPipe pose arms, the demo's ARM_MP order
SIDE = ("L", "R")


@dataclass(frozen=True)
class Run:
    """One exported week-4 run: an entry alone (`entry`) or a whole profile (`entry` None)."""

    group: str
    label: str
    description: str
    entry: Optional[str] = None
    profile: str = "ambitious"
    overrides: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


RUNS: Dict[str, Run] = {
    "body.joint_coupled_noise": Run(
        "body",
        "arm joint noise",
        "shoulder, elbow and wrist of both arms turned by one coupled draw per clip; "
        "shoulder/wrist σ 2.5° (≤ 5°), elbow σ 4.75° (≤ 9.5°)",
        "body.joint_coupled_noise",
    ),
    "body.limb_length_scaling": Run(
        "body",
        "limb length",
        "upper arm and forearm lengths scaled 0.93–1.07; the arm is re-solved so each wrist "
        "stays where it was",
        "body.limb_length_scaling",
    ),
    "body.movement_size": Run(
        "body",
        "movement size",
        "each wrist's path scaled about its own centre, 0.8–1.2 (log-uniform); the arm is "
        "re-solved to follow it",
        "body.movement_size",
    ),
    "body.location_jitter": Run(
        "body",
        "location jitter",
        "the whole signing space moved by one 3-D offset in the torso frame, σ 12.5 mm, "
        "≤ 25 mm; the arms are re-solved to follow",
        "body.location_jitter",
    ),
    "fingers.joint_coupled_noise": Run(
        "fingers",
        "finger joint noise",
        "the knuckle (MCP) and middle (PIP) bend of each finger changed by one coupled draw, "
        "σ 3.75°, ≤ 7.5°; the end joint (DIP) follows the PIP",
        "fingers.joint_coupled_noise",
    ),
    "fingers.handshape_jitter": Run(
        "fingers",
        "handshape jitter",
        "per-joint finger jitter that keeps the handshape class: non-selected fingers and thumb "
        "σ 7.5°, selected σ 3.75°, selected flexion σ 2°, all ≤ 2σ",
        "fingers.handshape_jitter",
    ),
    "face.expression_jitter": Run(
        "face",
        "expression jitter",
        "brow, eye and cheek/nose blendshapes offset once per clip (σ 0.0625, 0.0625, 0.0375) "
        "and the face points moved to match; mouth and jaw stay off",
        "face.expression_jitter",
    ),
    "temporal.time_warp.uniform": Run(
        "temporal",
        "time warp: uniform",
        "the whole clip played faster or slower by one factor, 0.6–1.65× (log-uniform)",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 1, "varied": 0, "segment": 0}},
    ),
    "temporal.time_warp.varied": Run(
        "temporal",
        "time warp: varied",
        "the speed changes smoothly over the clip through 4 knots of 0.7–1.43× each, on top of "
        "the uniform factor; the last 20% is left unwarped",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 0, "varied": 1, "segment": 0}},
    ),
    "temporal.time_warp.segment": Run(
        "temporal",
        "time warp: segment",
        "one window of 20–60% of the clip slowed or sped up by 0.6–1.65×, on top of the uniform "
        "factor",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 0, "varied": 0, "segment": 1}},
    ),
    "temporal.frame_drop.rate": Run(
        "temporal",
        "frame drop: rate",
        "frames kept on a lower frame-rate grid of 12–25 fps (never below 8 fps), no gap over "
        "125 ms",
        "temporal.frame_drop",
        overrides={"mode_weights": {"rate": 1, "random": 0}},
    ),
    "temporal.frame_drop.random": Run(
        "temporal",
        "frame drop: random",
        "frames dropped at random with p 0–0.2 per frame, no gap over 125 ms",
        "temporal.frame_drop",
        overrides={"mode_weights": {"rate": 0, "random": 1}},
    ),
    "temporal.hand_dropout": Run(
        "temporal",
        "hand dropout",
        "runs of 1–10 frames where a hand goes missing, as when MediaPipe loses it; start rate "
        "0–0.05 per frame",
        "temporal.hand_dropout",
        overrides={"start_rate": [0.05, 0.05]},
        note="This demo uses the top of the start-rate range (0.05), so that a run shows.",
    ),
    "scene.mirror": Run(
        "scene",
        "mirror (week 4)",
        "left/right reflection with the sides swapped, p 0.5",
        "scene.mirror",
        overrides={"p": 1.0},
        note="This demo always mirrors (p = 1), so that the mirror shows.",
    ),
    "scene.view": Run(
        "scene",
        "view (week 4)",
        "yaw ±25° about the vertical axis through the shoulders, pitch ±6°",
        "scene.view",
    ),
    "scene.scale": Run(
        "scene",
        "scale (week 4)",
        "the whole scene about the shoulders, 0.75–1.25 (camera distance)",
        "scene.scale",
    ),
    "scene.uncentre": Run(
        "scene",
        "uncentre (week 4)",
        "the whole scene shifted up to ±0.125 of the frame, hands kept in it",
        "scene.uncentre",
    ),
    "combined.default": Run(
        "combined",
        "default profile",
        "slr_week4_default.yaml as one sample: mirror, view, scale, uncentre, uniform duration, "
        "arm and finger joint noise, random frame drop",
        profile="default",
    ),
    "combined.ambitious": Run(
        "combined",
        "ambitious profile",
        "slr_week4_ambitious.yaml as one sample: every entry on at the used ranges",
        profile="ambitious",
    ),
    "combined.literature": Run(
        "combined",
        "literature profile",
        "slr_week4_literature.yaml as one sample: the default entries at the proven ranges",
        profile="literature",
    ),
}


def _merge(base: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def run_profile(run: Run, profiles):
    """The profile a run executes: a whole profile, or a copy with only `run.entry` on."""
    profile = copy.deepcopy(profiles[run.profile])
    if run.entry is None:
        return profile
    for entry in profile.entries:
        entry.enabled = entry.name == run.entry
        if entry.enabled:
            entry.probability = 1.0
            entry.ranges = _merge(entry.ranges, run.overrides)
    return profile


def jsonable(value):
    """Plain-JSON copy (numpy to lists and numbers, NaN / inf to None)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def source_frames(chain: AugmentationChain, result, count: int) -> np.ndarray:
    """
    The (fractional) source frame each output frame of `result` shows.

    Replays the time-warp times and the frame-drop selection from their recorded draws, in the
    chain's order; every other entry keeps the frames as they are.
    """
    src = np.arange(count, dtype=np.float64)
    draw = result.draws.get("temporal.time_warp")
    if draw and not draw.get("gated"):
        times = chain.augmentations["temporal.time_warp"].times(len(src), draw)
        if not (len(times) == len(src) and np.array_equal(times, np.arange(len(src)))):
            src = np.interp(times, np.arange(len(src)), src)
    draw = result.draws.get("temporal.frame_drop")
    if draw and not draw.get("gated") and draw.get("keep") is not None:
        keep = np.asarray(draw["keep"], dtype=int)
        if len(keep) != len(src):
            if draw.get("fill", "none") == "none":
                src = src[keep]
            elif draw["fill"] == "hold":
                src = src[keep[np.searchsorted(keep, np.arange(len(src)), side="right") - 1]]
    if len(src) != result.clip.frames:
        raise ValueError(f"replayed {len(src)} source times for {result.clip.frames} frames")
    return src


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


def write_data(path: Path, arrays):
    """{name: (array, dtype)} back to back, 4-byte aligned; returns the manifest and size."""
    manifest, offset = {}, 0
    with open(path, "wb") as fh:
        for name, (a, dtype) in arrays.items():
            a = np.ascontiguousarray(a)
            pad = (-offset) % 4
            fh.write(b"\0" * pad)
            offset += pad
            fh.write(a.tobytes())
            manifest[name] = {"dtype": dtype, "shape": list(a.shape), "offset": offset}
            offset += a.nbytes
    return manifest, offset


def bundle_arrays(outputs, vertices, times, mp_indices):
    """The Tier 1 bundle arrays from week-4 outputs, hands and face masked where not valid."""
    count = len(times)
    hand_mask = outputs.hand_valid[:, :, None, None]
    face_mask = outputs.face_valid[:, None, None]
    size = next((len(v) for v in vertices if v is not None), 0)
    verts = np.zeros((count, size, 3), np.int16)
    ok = np.zeros(count, bool)
    for t, v in enumerate(vertices):
        if v is not None and np.isfinite(v).all():
            verts[t] = np.clip(np.round(v * SCALE_I16), -32767, 32767)
            ok[t] = True

    def f32(points, shape):
        """float32 points, or NaN of `shape` when the clip has no camera."""
        if points is None:
            return np.full(shape, np.nan, np.float32)
        return np.asarray(points, np.float32)

    no_camera = outputs.pose_image is None
    arrays = {
        "times": (np.asarray(times, np.float32), "float32"),
        "mesh_verts": (verts, "int16"),
        "aug_arm": (f32(outputs.pose_world[:, ARMS], None), "float32"),
        "aug_hands": (f32(np.where(hand_mask, outputs.hands_world, np.nan), None), "float32"),
        "aug_face": (
            f32(np.where(face_mask, outputs.face_world, np.nan)[:, mp_indices], None),
            "float32",
        ),
        "img_arm": (
            f32(None if no_camera else outputs.pose_image[:, ARMS], (count, 6, 2)),
            "float32",
        ),
        "img_hands": (
            f32(
                None if no_camera else np.where(hand_mask, outputs.hands_image, np.nan),
                (count, 2, 21, 2),
            ),
            "float32",
        ),
        "img_face": (
            f32(
                None if no_camera else np.where(face_mask, outputs.face_image, np.nan),
                (count, 478, 2),
            )[:, mp_indices],
            "float32",
        ),
    }
    return arrays, ok


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
