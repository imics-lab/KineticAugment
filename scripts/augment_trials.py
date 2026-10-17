#!/usr/bin/env python3
"""
Try a few augmentations on whole clips, through the SMPL-X body + MANO hand round trip.

Every augmentation draws its parameters once per clip and applies them to every frame
(V1 Problem Scope §11): the motion inside the clip is kept, only the performance changes.

    yaw          extrinsic  the figure turned about the vertical axis through the shoulders
    mirror       extrinsic  left/right reflection with the sides swapped (a left-handed signer)
    scale        extrinsic  the whole scene scaled about the shoulders (camera nearer or farther)
    uncentre     extrinsic  the whole scene moved off-centre (subject elsewhere in the frame)
    arm pose     intrinsic  SMPL-X body_pose: elbows bent more, shoulders raised
    body shape   intrinsic  SMPL-X betas[0] shifted: a different body, same joint rotations
    time warp    temporal   smooth monotonic re-timing; the clip keeps its length

Arm pose and body shape edit the SMPL-X parameters (the edits of demo_smplx_augmentation.py)
and rebuild landmarks and mesh from them; each MANO hand turns with its forearm. The others
act on the round trip's outputs (landmarks and mesh vertices), which is equivalent for a rigid
transform or a re-timing.

Each augmentation is also measured: zero strength must reproduce the round trip exactly,
bone lengths, how far the wrists move, whether it adds frame-to-frame jitter, and how far the
pose moves once centred on the shoulders and scaled by shoulder width (the normalisation most
pose classifiers apply, which scale and uncentre should leave untouched).

Usage (from the repo root):
    .venv/bin/python scripts/augment_trials.py                    # 2 random cached clips
    .venv/bin/python scripts/augment_trials.py --clips 05631 63229

Reads <landmark-dir>/<clip>_world.npz and <clip>_hands.npz (evaluate_round_trip.py's cache)
and <video-dir>/<clip>.mp4. Writes <output-dir>/<clip>_augmentations.mp4 and
<output-dir>/augment_trials.json.
"""

import argparse
import dataclasses
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from demo_smplx_augmentation import (  # noqa: E402
    ELBOW, body_shape, elbow_flex, full_rotations, shoulder_raise, world_pose,
)
from view_body_hand_round_trip import (  # noqa: E402
    HAND_BONES, POSE_WRIST, BodyFigure, anchored, draw_bones, draw_mesh, hand_round_trip,
    limits, load_observations, overlay_mediapipe, set_view,
)
from view_round_trip import (  # noqa: E402
    ARM_BONES, CONTEXT_BONES, VideoSink, read_frames, round_trip_clip,
)
from kinetic_augment.body_model.converter import reconstruct_body  # noqa: E402
from kinetic_augment.body_model.hand_model import WRIST  # noqa: E402
from kinetic_augment.body_model.hand_observations import SIDES  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

LANDMARK_DIR = Path("data/round_trip_eval")
VIDEO_DIR = Path.home() / "Downloads" / "raw_videos"
OUTPUT_DIR = Path("data/augment_trials")
SEED = 20261011

YAW_DEG = 25.0           # turn about the vertical axis
SCALE = 0.85             # whole-scene scale: 0.8-1.2 in the literature (GISLR; ST-GCN 0.9-1.1)
UNCENTRE_SW = (0.6, 0.2, 0.0)  # shift in shoulder widths: right and down (MediaPipe y is down)
ELBOW_FLEX_DEG = 15.0    # more elbow bend, in each frame's own bending plane
SHOULDER_RAISE_DEG = 10.0
SHAPE_SIGMAS = 2.0       # added to betas[0]
WARP_STRENGTH = 0.5      # tau'(u) = 1 + k cos(2 pi u) >= 1 - k > 0, so time never reverses

SIDE_KEY = {"left": "L", "right": "R"}
MP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
            (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR = np.arange(33)
for a, b in MP_PAIRS:
    MIRROR[a], MIRROR[b] = b, a
ARM_LENGTHS = {"upper arm": [(11, 13), (12, 14)], "forearm": [(13, 15), (14, 16)]}
WRISTS = [15, 16]


@dataclasses.dataclass
class Variant:
    """One version of a clip: arm landmarks (T, 33, 3), stitched hands, and the mesh."""

    name: str
    title: str
    arms: np.ndarray
    hands: dict
    meshes: list
    faces: np.ndarray


# --------------------------------------------------------------------------- parameter space

def from_params(name, title, fits, hand_fits, hand_recon, skeleton, figure, turns=None):
    """
    Landmarks, stitched hands and mesh rebuilt from (possibly edited) SMPL-X parameters.

    `turns[side][t]` is a world rotation applied to that frame's MANO hand about its wrist:
    the change in the forearm's orientation, so the hand turns with the arm.
    """
    count = len(fits)
    arms = np.full((count, 33, 3), np.nan)
    hands = {side: np.full((count, 21, 3), np.nan) for side in SIDES}
    meshes = []
    for t, fit in enumerate(fits):
        if fit is None:
            meshes.append(None)
            continue
        arms[t] = reconstruct_body(fit, skeleton)
        targets = {}
        for side in SIDES:
            hand = hand_recon[side][t]
            if turns is not None and np.isfinite(hand).all():
                hand = (hand - hand[WRIST]) @ turns[side][t].T + hand[WRIST]
            targets[side] = hand
            hands[side][t] = anchored(hand, arms[t, POSE_WRIST[side]])
        vertices, _, _ = figure.pose(fit, {side: hand_fits[side][t] for side in SIDES}, targets)
        meshes.append(vertices)
    return Variant(name, title, arms, hands, meshes, figure.faces)


def edit_arms(fits, body, flex, raise_):
    """Arm-pose edit per frame, and the forearm's world rotation change per side and frame."""
    edited, turns = [], {side: [] for side in SIDES}
    for fit in fits:
        if fit is None:
            edited.append(None)
            for side in SIDES:
                turns[side].append(np.eye(3))
            continue
        new = elbow_flex(shoulder_raise(fit, body, raise_), body, flex)
        before, _ = world_pose(full_rotations(fit, body.num_joints), body)
        after, _ = world_pose(full_rotations(new, body.num_joints), body)
        for side in SIDES:
            joint = ELBOW[SIDE_KEY[side]]
            turns[side].append(after[joint] @ before[joint].T)
        edited.append(new)
    return edited, turns


def edit_shape(fits, body, sigmas):
    return [None if f is None else body_shape(f, body, sigmas) for f in fits]


# --------------------------------------------------------------------------- output space

def _map_points(variant, fn, name, title, faces=None, swap=False):
    arms = fn(variant.arms)
    hands = {side: fn(h) for side, h in variant.hands.items()}
    if swap:
        arms = arms[:, MIRROR]
        hands = {"left": hands["right"], "right": hands["left"]}
    meshes = [None if m is None else fn(m) for m in variant.meshes]
    return Variant(name, title, arms, hands, meshes,
                   variant.faces if faces is None else faces)


def shoulder_centre(variant):
    return np.nanmedian((variant.arms[:, 11] + variant.arms[:, 12]) / 2, axis=0)


def yaw(variant, degrees):
    pivot = shoulder_centre(variant)
    rot = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    return _map_points(variant, lambda p: (p - pivot) @ rot.T + pivot, "yaw",
                       f"yaw {degrees:+.0f}° (extrinsic)")


def mirror(variant):
    flip_x = 2 * shoulder_centre(variant)[0]

    def fn(p):
        out = p.copy()
        out[..., 0] = flip_x - out[..., 0]
        return out

    # Reflection reverses triangle winding; reversing the faces keeps normals outward.
    return _map_points(variant, fn, "mirror", "mirror, sides swapped (extrinsic)",
                       faces=variant.faces[:, ::-1].copy(), swap=True)


def shoulder_width(variant):
    return float(np.nanmedian(np.linalg.norm(variant.arms[:, 11] - variant.arms[:, 12], axis=-1)))


def scale(variant, factor):
    pivot = shoulder_centre(variant)
    return _map_points(variant, lambda p: (p - pivot) * factor + pivot, "scale",
                       f"scale ×{factor:.2f} (extrinsic)")


def uncentre(variant, shift_sw):
    offset = np.asarray(shift_sw, dtype=np.float64) * shoulder_width(variant)
    return _map_points(variant, lambda p: p + offset, "uncentre",
                       f"off-centre ({shift_sw[0]:+.1f}, {shift_sw[1]:+.1f}) sh.w. (extrinsic)")


def warp_times(count, strength):
    u = np.linspace(0.0, 1.0, count)
    return (count - 1) * (u + strength * np.sin(2 * np.pi * u) / (2 * np.pi))


def _resample(frames, times):
    """Linear interpolation between neighbouring frames; the nearer one if the other is missing."""
    out = []
    for tau in times:
        lo = int(np.floor(tau))
        hi = min(lo + 1, len(frames) - 1)
        w = tau - lo
        a, b = frames[lo], frames[hi]
        a_ok = a is not None and np.isfinite(a).all()
        b_ok = b is not None and np.isfinite(b).all()
        if a_ok and b_ok:
            out.append((1 - w) * a + w * b)
        elif a_ok or b_ok:
            out.append(a if (a_ok and (w < 0.5 or not b_ok)) else b)
        else:
            out.append(a if a is not None else b)
    return out


def _resample_landmarks(seq, times):
    """Per-landmark resampling of (T, N, 3), so a landmark missing in one frame stays local."""
    out = np.full_like(seq, np.nan)
    for j in range(seq.shape[1]):
        out[:, j] = np.stack([np.full(3, np.nan) if p is None else p
                              for p in _resample(list(seq[:, j]), times)])
    return out


def time_warp(variant, strength):
    times = warp_times(len(variant.arms), strength)
    return Variant("time_warp", f"time warp k={strength:.1f} (temporal)",
                   _resample_landmarks(variant.arms, times),
                   {side: np.stack(_resample(list(h), times)) for side, h in variant.hands.items()},
                   _resample(variant.meshes, times), variant.faces), times


# --------------------------------------------------------------------------- measurement

def _max_diff(a, b):
    diffs = [np.nanmax(np.abs(a.arms - b.arms))]
    diffs += [np.nanmax(np.abs(a.hands[s] - b.hands[s])) for s in SIDES
              if np.isfinite(a.hands[s]).any()]
    diffs += [np.max(np.abs(x - y)) for x, y in zip(a.meshes, b.meshes) if x is not None]
    return float(max(diffs))


def _jitter_mm(arms):
    """Median size of the wrists' second difference: frame-to-frame roughness, in mm."""
    values = []
    for w in WRISTS:
        second = arms[2:, w] - 2 * arms[1:-1, w] + arms[:-2, w]
        norms = np.linalg.norm(second, axis=-1)
        values.append(norms[np.isfinite(norms)])
    values = np.concatenate(values)
    return float(1000 * np.median(values)) if len(values) else float("nan")


def _normalised(arms):
    """Arm landmarks centred on each frame's shoulder midpoint, in shoulder widths."""
    centre = (arms[:, 11] + arms[:, 12]) / 2
    width = np.linalg.norm(arms[:, 11] - arms[:, 12], axis=-1)
    return (arms[:, 11:17] - centre[:, None]) / width[:, None, None]


def measure(base, variant, compare_frames=True):
    """Bone lengths, wrist displacement and normalised-pose change from the round trip; jitter."""
    row = {}
    for bone, pairs in ARM_LENGTHS.items():
        lengths = {}
        for v in (base, variant):
            lengths[v.name] = np.nanmedian(np.concatenate(
                [np.linalg.norm(v.arms[:, a] - v.arms[:, b], axis=-1) for a, b in pairs]))
        row[f"{bone}_change_pct"] = float(100 * (lengths[variant.name] / lengths[base.name] - 1))
    if compare_frames:
        moved = np.linalg.norm(variant.arms[:, WRISTS] - base.arms[:, WRISTS], axis=-1)
        row["wrist_moved_mm_median"] = float(1000 * np.nanmedian(moved))
        row["wrist_moved_mm_max"] = float(1000 * np.nanmax(moved))
        shift = np.linalg.norm(_normalised(variant.arms) - _normalised(base.arms), axis=-1)
        row["normalised_pose_change_pct_max"] = float(100 * np.nanmax(shift))
    row["wrist_jitter_mm"] = _jitter_mm(variant.arms)
    row["round_trip_jitter_mm"] = _jitter_mm(base.arms)
    return row


# --------------------------------------------------------------------------- clip

def augment_clip(clip, landmark_dir, skeleton, figure, model_dir):
    pose, ok, hands = load_observations(landmark_dir, clip)
    print(f"Clip {clip}: {len(pose)} frames")
    body, fits, _ = round_trip_clip(pose, ok, skeleton)
    hand_fits, hand_recon, _ = hand_round_trip(hands, model_dir)

    def params(name, title, edited, turns=None):
        return from_params(name, title, edited, hand_fits, hand_recon, skeleton, figure, turns)

    base = params("round_trip", "round trip (no augmentation)", fits)
    arm_fits, turns = edit_arms(fits, body, ELBOW_FLEX_DEG, SHOULDER_RAISE_DEG)
    warped, times = time_warp(base, WARP_STRENGTH)
    variants = [
        yaw(base, YAW_DEG),
        mirror(base),
        scale(base, SCALE),
        uncentre(base, UNCENTRE_SW),
        params("arm_pose", f"elbows +{ELBOW_FLEX_DEG:.0f}°, shoulders +{SHOULDER_RAISE_DEG:.0f}°"
               " (intrinsic)", arm_fits, turns),
        params("body_shape", f"betas[0] {SHAPE_SIGMAS:+.0f} (intrinsic)",
               edit_shape(fits, body, SHAPE_SIGMAS)),
        warped,
    ]

    # Zero strength must give back the round trip exactly; mirroring twice must too.
    zero_arms, zero_turns = edit_arms(fits, body, 0.0, 0.0)
    identity = {
        "yaw": _max_diff(base, yaw(base, 0.0)),
        "mirror": _max_diff(base, mirror(mirror(base))),
        "scale": _max_diff(base, scale(base, 1.0)),
        "uncentre": _max_diff(base, uncentre(base, (0.0, 0.0, 0.0))),
        "arm_pose": _max_diff(base, params("z", "", zero_arms, zero_turns)),
        "body_shape": _max_diff(base, params("z", "", edit_shape(fits, body, 0.0))),
        "time_warp": _max_diff(base, time_warp(base, 0.0)[0]),
    }
    report = {"frames": len(pose), "identity_max_abs_diff_m": identity}
    for v in variants:
        report[v.name] = measure(base, v, compare_frames=v.name != "mirror")
        print(f"  {v.name:<10} zero-strength diff {identity[v.name]:.1e} m  " + ", ".join(
            f"{k} {x:.2f}" for k, x in report[v.name].items()))
    report["time_warp"]["source_frame_range"] = [float(times.min()), float(times.max())]
    return base, variants, report


def render(clip, base, variants, report, landmark_dir, video_path, out_dir):
    _, _, hands_obs = load_observations(landmark_dir, clip)
    count = len(base.arms)
    frames, fps = read_frames(clip, count, video_path)
    everything = [base] + variants
    shoulder_y = np.nanmedian(base.arms[:, [11, 12], 1])
    upper = [m[m[:, 1] < shoulder_y + 0.45] for v in everything for m in v.meshes if m is not None]
    lim = limits([np.concatenate(upper)], "front", pad=1.05)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_augmentations.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(24.0, 10.2), dpi=100)
    text = metrics_text(clip, report)
    for t in range(count):
        fig.clf()
        grid = fig.add_gridspec(2, 5, hspace=0.12, wspace=0.04,
                                left=0.01, right=0.99, top=0.95, bottom=0.01)
        axes = [fig.add_subplot(grid[i // 5, i % 5]) for i in range(10)]
        axes[0].imshow(overlay_mediapipe(frames[t], hands_obs, t))
        axes[0].set_title(f"clip {clip}  frame {t + 1}/{count}  (MediaPipe 2-D)", fontsize=10)
        axes[0].axis("off")
        for ax, v in zip(axes[1:9], everything):
            if v.meshes[t] is not None:
                draw_mesh(ax, v.meshes[t], v.faces)
            if v is not base:  # the unaugmented skeleton, for reference
                draw_bones(ax, base.arms[t], ARM_BONES, "front", "#1f5fbf", 1.4, alpha=0.6)
            draw_bones(ax, v.arms[t], CONTEXT_BONES, "front", "#2ca02c", 1.0, alpha=0.5)
            draw_bones(ax, v.arms[t], ARM_BONES, "front", "#2ca02c", 2.0)
            for side in SIDES:
                draw_bones(ax, v.hands[side][t], HAND_BONES, "front", "#2ca02c", 1.0, size=3)
            set_view(ax, lim, v.title)
        axes[9].axis("off")
        axes[9].text(0.0, 1.0, text, family="monospace", fontsize=8.5, va="top",
                     transform=axes[9].transAxes)
        sink.add(fig)
    sink.close()
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def metrics_text(clip, report):
    lines = [f"clip {clip}: per-clip parameters,", "applied to every frame", "",
             "GREEN = augmented, BLUE = unaugmented", "",
             f"{'':<11}{'upper':>6}{'fore':>6}{'wrist':>6}{'norm':>6}{'jitter':>7}",
             f"{'':<11}{'arm %':>6}{'arm %':>6}{'mm':>6}{'%':>6}{'mm':>7}"]
    for name in ("yaw", "mirror", "scale", "uncentre", "arm_pose", "body_shape", "time_warp"):
        r = report[name]
        moved = r.get("wrist_moved_mm_median", float("nan"))
        norm = r.get("normalised_pose_change_pct_max", float("nan"))
        lines.append(f"{name:<11}{r['upper arm_change_pct']:>6.1f}{r['forearm_change_pct']:>6.1f}"
                     f"{moved:>6.0f}{norm:>6.1f}{r['wrist_jitter_mm']:>7.2f}")
    lines.append(f"{'(no aug)':<11}{0.0:>6.1f}{0.0:>6.1f}{0.0:>6.0f}{0.0:>6.1f}"
                 f"{report['yaw']['round_trip_jitter_mm']:>7.2f}")
    lines += ["", "wrist mm: median distance from the",
              "unaugmented wrist. norm %: max change",
              "of the shoulder-centred, shoulder-width",
              "scaled arm pose. jitter: median size",
              "of the wrists' 2nd difference.", "",
              "zero strength reproduces the round trip:"]
    lines += [f"  {k:<11}max diff {v:.1e} m" for k, v in report["identity_max_abs_diff_m"].items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="*", help="clip ids (default: 2 at random)")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    cached = sorted(p.name[:-len("_world.npz")] for p in args.landmark_dir.glob("*_world.npz")
                    if (args.landmark_dir / p.name.replace("_world", "_hands")).exists()
                    and (args.video_dir / p.name.replace("_world.npz", ".mp4")).exists())
    clips = args.clips or sorted(np.random.default_rng(args.seed).choice(
        cached, size=args.count, replace=False).tolist())
    print(f"{len(cached)} cached clips with video; augmenting {clips} (seed {args.seed})")

    skeleton = SMPLXSkeleton(model_path="models")
    figure = BodyFigure(skeleton, "models")
    out = {"seed": args.seed, "clips": clips, "pool": cached,
           "parameters": {"yaw_deg": YAW_DEG, "scale": SCALE, "uncentre_sw": UNCENTRE_SW,
                          "elbow_flex_deg": ELBOW_FLEX_DEG,
                          "shoulder_raise_deg": SHOULDER_RAISE_DEG,
                          "shape_sigmas_betas0": SHAPE_SIGMAS, "warp_strength": WARP_STRENGTH},
           "results": {}}
    for clip in clips:
        base, variants, report = augment_clip(clip, args.landmark_dir, skeleton, figure, "models")
        out["results"][clip] = report
        if not args.no_render:
            report["video"] = str(render(clip, base, variants, report, args.landmark_dir,
                                         args.video_dir / f"{clip}.mp4", args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "augment_trials.json").write_text(json.dumps(out, indent=2))
    print(f"-> {args.output_dir / 'augment_trials.json'}")


if __name__ == "__main__":
    main()
