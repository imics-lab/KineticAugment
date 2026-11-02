#!/usr/bin/env python3
"""
Try the Tier 1 augmentations on whole clips, through the SMPL-X body + MANO hand round trip.

Tier 1 is the set that is safe for every sign (Augmentation_Grounding.md §3). Every parameter
is drawn once per clip from the evidence-based ranges (`kinetic_augment.augmentations.tier1`)
and applied to every frame (V1 Problem Scope §11):

    view         yaw ±20° about the vertical axis through the shoulders, plus pitch ±5°
    mirror       left/right reflection with the sides swapped (a left-handed signer), p = 0.5
    duration     the clip's duration scaled by one factor, 0.8-1.25×; nothing cropped
    arm joints   shoulder, elbow and wrist of each arm, each with p = 0.3, ±4° (SMPL-X body_pose)
    scale        the whole scene about the shoulders, 0.85-1.15 (camera nearer or farther)
    uncentre     the whole scene shifted by up to ±0.1 of the frame, hands kept in the frame
    combined     all of the above, in that chain: arm joints, mirror (if drawn), view, scale,
                 uncentre, duration

Each panel shows one augmentation at the clip's drawn value, so the panels decompose the
combined sample. The mirror panel always mirrors; the combined sample mirrors only if drawn.

The arm-joint draw edits the SMPL-X parameters and rebuilds landmarks and mesh from them; each
MANO hand turns with its SMPL-X wrist. The others act on the round trip's outputs (landmarks and
mesh vertices), which is equivalent for a rigid transform or a re-timing.

Uncentring needs a frame. A weak-perspective camera is fitted per clip from the reconstructed
arm landmarks to MediaPipe's 2-D pose; the shift is then drawn from the part of ±0.1 that keeps
every hand point inside the frame inside it.

Each augmentation is also measured: zero strength must reproduce the round trip exactly,
bone lengths, how far the wrists move, whether it adds frame-to-frame jitter, how far the pose
moves once centred on the shoulders and scaled by shoulder width (the normalisation most pose
classifiers apply, which scale and uncentre should leave untouched), and how many hand points
stay inside the frame.

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
    ELBOW, MAX_BEND, MIN_BEND, SHOULDER, SPINE3, WRIST as SMPLX_WRIST, _angle, _unit,
    full_rotations, rotate_in_world, world_pose,
)
from view_body_hand_round_trip import (  # noqa: E402
    HAND_BONES, POSE_WRIST, VISIBILITY_THRESHOLD, BodyFigure, anchored, draw_bones, draw_mesh,
    hand_round_trip, limits, load_observations, overlay_mediapipe, set_view,
)
from view_round_trip import (  # noqa: E402
    ARM_BONES, CONTEXT_BONES, VideoSink, read_frames, round_trip_clip,
)
from kinetic_augment.augmentations.tier1 import (  # noqa: E402
    ARM_JOINTS, POSE_MIRROR, TIER1, ArmJointDraw, Tier1Draw, WeakCamera, about, duration_times,
    fit_weak_camera, frame_loss, in_frame, reflect_x, resample, resample_points, sample_tier1,
    uncentre_shift, view_rotation,
)
from kinetic_augment.body_model.converter import reconstruct_body  # noqa: E402
from kinetic_augment.body_model.hand_model import WRIST  # noqa: E402
from kinetic_augment.body_model.hand_observations import SIDES  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

LANDMARK_DIR = Path("data/round_trip_eval")
VIDEO_DIR = Path.home() / "Downloads" / "raw_videos"
OUTPUT_DIR = Path("data/augment_trials")
SEED = 20261011

SIDE_KEY = {"left": "L", "right": "R"}
JOINT_INDEX = {"shoulder": SHOULDER, "elbow": ELBOW, "wrist": SMPLX_WRIST}
ARM_LENGTHS = {"upper arm": [(11, 13), (12, 14)], "forearm": [(13, 15), (14, 16)]}
ARM_MP = list(range(11, 17))
WRISTS = [15, 16]
NAMES = ("view", "mirror", "duration", "arm_joints", "scale", "uncentre", "combined")
# The same-length augmentations that keep each point's identity: compared frame by frame.
ALIGNED = ("view", "arm_joints", "scale", "uncentre")


@dataclasses.dataclass
class Variant:
    """One version of a clip: arm landmarks (T, 33, 3), stitched hands, mesh and camera."""

    name: str
    title: str
    arms: np.ndarray
    hands: dict
    meshes: list
    faces: np.ndarray
    camera: WeakCamera


def hand_points(variant):
    """(T, 44, 3): both hands' 21 landmarks and the two pose wrists, the points kept in frame."""
    return np.concatenate([variant.hands["left"], variant.hands["right"],
                           variant.arms[:, WRISTS]], axis=1)


# --------------------------------------------------------------------------- parameter space

def from_params(name, title, fits, hand_fits, hand_recon, skeleton, figure, camera, turns=None):
    """
    Landmarks, stitched hands and mesh rebuilt from (possibly edited) SMPL-X parameters.

    `turns[side][t]` is a world rotation applied to that frame's MANO hand about its wrist:
    the change in the SMPL-X wrist's orientation, so the hand turns with the arm.
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
    return Variant(name, title, arms, hands, meshes, figure.faces, camera)


def _flex_elbow(rotations, body, key, degrees):
    """Bend one elbow by `degrees` within its current bending plane, kept within 5..150°."""
    rot, pos = world_pose(rotations, body)
    upper = pos[ELBOW[key]] - pos[SHOULDER[key]]
    fore = pos[SMPLX_WRIST[key]] - pos[ELBOW[key]]
    bend = _angle(upper, fore)
    axis = _unit(np.cross(upper, fore))
    if axis is None:  # straight arm: bend toward the body's front
        axis = _unit(np.cross(upper, rot[SPINE3] @ np.array([0.0, 0.0, 1.0])))
    if axis is None:
        return
    step = np.clip(bend + degrees, MIN_BEND, MAX_BEND) - bend
    delta = Rotation.from_rotvec(axis * np.radians(step)).as_matrix()
    rotate_in_world(rotations, rot, ELBOW[key], SHOULDER[key], delta)


def perturb_arm_joints(fits, body, joints):
    """
    The arm-joint draw applied to every frame, and each SMPL-X wrist's world rotation change.

    A shoulder or wrist turns about its drawn axis, fixed in the parent joint's frame, so it
    sits the same few degrees differently relative to its parent throughout the clip.
    """
    edited, turns = [], {side: [] for side in SIDES}
    for fit in fits:
        if fit is None:
            edited.append(None)
            for side in SIDES:
                turns[side].append(np.eye(3))
            continue
        rotations = full_rotations(fit, body.num_joints)
        before, _ = world_pose(rotations, body)
        for j in joints:
            key = SIDE_KEY[j.side]
            if j.joint == "elbow":
                _flex_elbow(rotations, body, key, j.angle_deg)
                continue
            index = JOINT_INDEX[j.joint][key]
            delta = Rotation.from_rotvec(np.asarray(j.axis) * np.radians(j.angle_deg))
            rotations[index] = (delta * Rotation.from_rotvec(rotations[index])).as_rotvec()
        after, _ = world_pose(rotations, body)
        for side in SIDES:
            index = SMPLX_WRIST[SIDE_KEY[side]]
            turns[side].append(after[index] @ before[index].T)
        edited.append(dataclasses.replace(fit, body_pose=rotations[1:22].reshape(63).copy()))
    return edited, turns


def arm_joints_title(joints):
    if not joints:
        return f"arm joints: none drawn (p = {TIER1.arm_joint_p} each)"
    return "arm joints: " + ", ".join(
        f"{j.side[0].upper()} {j.joint} {j.angle_deg:+.1f}°" for j in joints)


# --------------------------------------------------------------------------- output space

def _map_points(variant, fn, name, title, faces=None, swap=False):
    arms = fn(variant.arms)
    hands = {side: fn(h) for side, h in variant.hands.items()}
    if swap:
        arms = arms[:, POSE_MIRROR]
        hands = {"left": hands["right"], "right": hands["left"]}
    meshes = [None if m is None else fn(m) for m in variant.meshes]
    return Variant(name, title, arms, hands, meshes,
                   variant.faces if faces is None else faces, variant.camera)


def shoulder_centre(variant):
    return np.nanmedian((variant.arms[:, 11] + variant.arms[:, 12]) / 2, axis=0)


def view(variant, yaw_deg, pitch_deg, name="view"):
    pivot, rot = shoulder_centre(variant), view_rotation(yaw_deg, pitch_deg)
    return _map_points(variant, lambda p: about(p, pivot, rot), name,
                       f"view: yaw {yaw_deg:+.1f}°, pitch {pitch_deg:+.1f}°")


def mirror(variant, name="mirror", title="mirror, sides swapped"):
    centre_x = shoulder_centre(variant)[0]
    # Reflection reverses triangle winding; reversing the faces keeps normals outward.
    return _map_points(variant, lambda p: reflect_x(p, centre_x), name, title,
                       faces=variant.faces[:, ::-1].copy(), swap=True)


def scale(variant, factor, name="scale"):
    pivot = shoulder_centre(variant)
    return _map_points(variant, lambda p: about(p, pivot, factor * np.eye(3)), name,
                       f"scale ×{factor:.3f}")


def uncentre(variant, u, limit, name="uncentre"):
    """Shift by the draw `u`, within the part of ±limit that keeps the hands in the frame."""
    shift = uncentre_shift(variant.camera.project(hand_points(variant)), u, limit)
    offset = variant.camera.world_shift(shift)
    moved = _map_points(variant, lambda p: p + offset, name,
                        f"off-centre ({shift[0]:+.3f}, {shift[1]:+.3f}) of the frame")
    return moved, shift


def retime(variant, factor, name="duration"):
    times = duration_times(len(variant.arms), factor)
    return Variant(name, f"duration ×{factor:.3f} ({len(times)} frames)",
                   resample_points(variant.arms, times),
                   {side: resample_points(h, times) for side, h in variant.hands.items()},
                   resample(variant.meshes, times), variant.faces,
                   variant.camera.resampled(times))


def combine(arm_variant, draw, limit):
    """The chain of the combined sample, starting from the arm-joint variant."""
    v = mirror(arm_variant) if draw.mirror else arm_variant
    v = view(v, draw.yaw_deg, draw.pitch_deg)
    v = scale(v, draw.scale)
    v, shift = uncentre(v, draw.uncentre_u, limit)
    v = retime(v, draw.duration, name="combined")
    v.title = "combined" + (" (mirrored)" if draw.mirror else " (not mirrored)")
    return v, shift


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


def _in_frame_pct(variant):
    uv = variant.camera.project(hand_points(variant))
    present = np.isfinite(uv).all(-1)
    return float(100 * in_frame(uv).sum() / present.sum()) if present.any() else float("nan")


def measure(base, variant):
    """Bone lengths, wrist displacement and normalised-pose change from the round trip; jitter."""
    row = {"frames": len(variant.arms)}
    for bone, pairs in ARM_LENGTHS.items():
        lengths = {}
        for v in (base, variant):
            lengths[v.name] = np.nanmedian(np.concatenate(
                [np.linalg.norm(v.arms[:, a] - v.arms[:, b], axis=-1) for a, b in pairs]))
        row[f"{bone}_change_pct"] = float(100 * (lengths[variant.name] / lengths[base.name] - 1))
    if variant.name in ALIGNED:
        moved = np.linalg.norm(variant.arms[:, WRISTS] - base.arms[:, WRISTS], axis=-1)
        row["wrist_moved_mm_median"] = float(1000 * np.nanmedian(moved))
        row["wrist_moved_mm_max"] = float(1000 * np.nanmax(moved))
        shift = np.linalg.norm(_normalised(variant.arms) - _normalised(base.arms), axis=-1)
        row["normalised_pose_change_pct_max"] = float(100 * np.nanmax(shift))
        loss = frame_loss(base.camera.project(hand_points(base)),
                          variant.camera.project(hand_points(variant)))
        row["hand_points_leaving_frame_pct"] = None if loss is None else 100 * loss
    row["hand_points_in_frame_pct"] = _in_frame_pct(variant)
    row["wrist_jitter_mm"] = _jitter_mm(variant.arms)
    row["round_trip_jitter_mm"] = _jitter_mm(base.arms)
    return row


# --------------------------------------------------------------------------- clip

def fit_camera(base, hands):
    """Weak-perspective camera from the round trip's arm landmarks to MediaPipe's 2-D pose."""
    image = hands.pose_image[:, ARM_MP, :2].astype(np.float64)
    image[hands.pose_visibility[:, ARM_MP] < VISIBILITY_THRESHOLD] = np.nan
    return fit_weak_camera(base.arms[:, ARM_MP], image)


def augment_clip(clip, landmark_dir, skeleton, figure, model_dir, seed):
    pose, ok, hands = load_observations(landmark_dir, clip)
    print(f"Clip {clip}: {len(pose)} frames")
    body, fits, _ = round_trip_clip(pose, ok, skeleton)
    hand_fits, hand_recon, _ = hand_round_trip(hands, model_dir)
    draw = sample_tier1(np.random.default_rng([seed, int(clip)]))

    def params(name, title, edited, camera, turns=None):
        return from_params(name, title, edited, hand_fits, hand_recon, skeleton, figure, camera,
                           turns)

    base = params("round_trip", "round trip (no augmentation)", fits, None)
    base.camera = fit_camera(base, hands)
    print(f"  camera: scale ({base.camera.scale[0]:+.2f}, {base.camera.scale[1]:+.2f}) per m, "
          f"residual {100 * base.camera.residual:.2f}% of the frame")

    arm_fits, turns = perturb_arm_joints(fits, body, draw.arm_joints)
    arms_title = arm_joints_title(draw.arm_joints)
    arm_variant = params("arm_joints", arms_title, arm_fits, base.camera, turns)
    uncentred, shift = uncentre(base, draw.uncentre_u, TIER1.uncentre)
    combined, combined_shift = combine(arm_variant, draw, TIER1.uncentre)
    variants = [
        view(base, draw.yaw_deg, draw.pitch_deg),
        mirror(base, title="mirror, sides swapped" + (" (drawn)" if draw.mirror else
                                                       " (not drawn)")),
        retime(base, draw.duration),
        arm_variant,
        scale(base, draw.scale),
        uncentred,
        combined,
    ]

    # Zero strength must give back the round trip exactly; mirroring twice must too.
    zero_joints = tuple(ArmJointDraw(side, joint, 0.0, (1.0, 0.0, 0.0))
                        for side in SIDES for joint in ARM_JOINTS)
    zero_fits, zero_turns = perturb_arm_joints(fits, body, zero_joints)
    zero_arms = params("arm_joints", "", zero_fits, base.camera, zero_turns)
    zero_draw = Tier1Draw(False, 0.0, 0.0, 1.0, (), 1.0, (0.5, 0.5))
    identity = {
        "view": _max_diff(base, view(base, 0.0, 0.0)),
        "mirror": _max_diff(base, mirror(mirror(base))),
        "duration": _max_diff(base, retime(base, 1.0)),
        "arm_joints": _max_diff(base, zero_arms),
        "scale": _max_diff(base, scale(base, 1.0)),
        "uncentre": _max_diff(base, uncentre(base, (0.5, 0.5), 0.0)[0]),
        "combined": _max_diff(base, combine(zero_arms, zero_draw, 0.0)[0]),
    }
    report = {"frames": len(pose), "draw": draw.as_dict(),
              "uncentre_shift": [float(x) for x in shift],
              "combined_uncentre_shift": [float(x) for x in combined_shift],
              "camera": {"scale_per_m": list(base.camera.scale),
                         "residual_frame_pct": 100 * base.camera.residual},
              "round_trip_hand_points_in_frame_pct": _in_frame_pct(base),
              "identity_max_abs_diff_m": identity}
    for v in variants:
        report[v.name] = measure(base, v)
        print(f"  {v.name:<10} zero-strength diff {identity[v.name]:.1e} m  " + ", ".join(
            f"{k} {x:.2f}" for k, x in report[v.name].items() if x is not None))
    print(f"  arm joints drawn: {arms_title}")
    return base, variants, report


def render(clip, base, variants, report, landmark_dir, video_path, out_dir):
    _, _, hands_obs = load_observations(landmark_dir, clip)
    source_count = len(base.arms)
    everything = [base] + variants
    count = max(len(v.arms) for v in everything)
    frames, fps = read_frames(clip, source_count, video_path)
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
        s = min(t, source_count - 1)  # the shorter clips hold their last frame
        axes[0].imshow(overlay_mediapipe(frames[s], hands_obs, s))
        axes[0].set_title(f"clip {clip}  frame {s + 1}/{source_count}  (MediaPipe 2-D)",
                          fontsize=10)
        axes[0].axis("off")
        for ax, v in zip(axes[1:9], everything):
            f = min(t, len(v.arms) - 1)
            if v.meshes[f] is not None:
                draw_mesh(ax, v.meshes[f], v.faces)
            if v is not base:  # the unaugmented skeleton, for reference
                draw_bones(ax, base.arms[s], ARM_BONES, "front", "#1f5fbf", 1.4, alpha=0.6)
            draw_bones(ax, v.arms[f], CONTEXT_BONES, "front", "#2ca02c", 1.0, alpha=0.5)
            draw_bones(ax, v.arms[f], ARM_BONES, "front", "#2ca02c", 2.0)
            for side in SIDES:
                draw_bones(ax, v.hands[side][f], HAND_BONES, "front", "#2ca02c", 1.0, size=3)
            set_view(ax, lim, v.title)
        axes[9].axis("off")
        axes[9].text(0.0, 1.0, text, family="monospace", fontsize=8.5, va="top",
                     transform=axes[9].transAxes)
        sink.add(fig)
    sink.close()
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def _cell(row, key, fmt):
    value = row.get(key)
    return format(float("nan") if value is None else value, fmt)


def metrics_text(clip, report):
    lines = [f"clip {clip}: Tier 1, drawn once per clip,", "applied to every frame", "",
             "GREEN = augmented, BLUE = unaugmented", "",
             f"{'':<11}{'upper':>6}{'fore':>6}{'wrist':>6}{'norm':>6}{'jitter':>7}{'frame':>6}",
             f"{'':<11}{'arm %':>6}{'arm %':>6}{'mm':>6}{'%':>6}{'mm':>7}{'in %':>6}"]
    for name in NAMES:
        r = report[name]
        lines.append(f"{name:<11}{_cell(r, 'upper arm_change_pct', '>6.1f')}"
                     f"{_cell(r, 'forearm_change_pct', '>6.1f')}"
                     f"{_cell(r, 'wrist_moved_mm_median', '>6.0f')}"
                     f"{_cell(r, 'normalised_pose_change_pct_max', '>6.1f')}"
                     f"{_cell(r, 'wrist_jitter_mm', '>7.2f')}"
                     f"{_cell(r, 'hand_points_in_frame_pct', '>6.0f')}")
    lines.append(f"{'(no aug)':<11}{0.0:>6.1f}{0.0:>6.1f}{0.0:>6.0f}{0.0:>6.1f}"
                 f"{report['view']['round_trip_jitter_mm']:>7.2f}"
                 f"{report['round_trip_hand_points_in_frame_pct']:>6.0f}")
    lines += ["", "wrist mm: median distance from the",
              "unaugmented wrist. norm %: max change",
              "of the shoulder-centred, shoulder-width",
              "scaled arm pose. jitter: median size",
              "of the wrists' 2nd difference. frame in:",
              "hand points inside the frame (camera fit",
              f"residual {report['camera']['residual_frame_pct']:.1f}% of the frame).", "",
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
           "ranges": dataclasses.asdict(TIER1), "results": {}}
    for clip in clips:
        base, variants, report = augment_clip(clip, args.landmark_dir, skeleton, figure,
                                              "models", args.seed)
        out["results"][clip] = report
        if not args.no_render:
            report["video"] = str(render(clip, base, variants, report, args.landmark_dir,
                                         args.video_dir / f"{clip}.mp4", args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "augment_trials.json").write_text(json.dumps(out, indent=2))
    print(f"-> {args.output_dir / 'augment_trials.json'}")


if __name__ == "__main__":
    main()
