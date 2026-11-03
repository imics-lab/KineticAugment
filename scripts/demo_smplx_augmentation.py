#!/usr/bin/env python3
"""
Demonstrate augmentation *through* SMPL-X with the V1 converter.

Each frame goes through the body model:

    fit_body(landmarks)  ->  SMPL-X parameters  ->  edit the parameters
                          ->  reconstruct_body  ->  augmented landmarks

Three edits are shown, each defined anatomically on the fitted pose rather than
on raw local axes (the repo's joint-limit table has the SMPL-X elbow axes wrong,
so "rotate local x" would not mean "flex"):

  elbow_flex      bend each elbow by N degrees within its current bending plane
                  (positive = more bent), keeping the bend within 5..150 degrees.
  shoulder_raise  raise each upper arm N degrees toward the torso's up direction,
                  stopping 10 degrees short of straight up.
  body_shape      shift the first SMPL-X shape coefficient (betas[0], mostly
                  overall size) by N standard deviations; the pose is unchanged.

For every edit the script measures what actually changed on the reconstructed
landmarks (elbow bend, arm elevation, arm lengths), so you can check the
augmentation does what it claims, and renders a video:

    source video  |  front view  |  side view
    BLUE  = MediaPipe input   GREEN = plain round trip   RED = augmented

Usage (from the repo root, which must hold the V1 converter, data/ and models/):
    .venv/bin/python scripts/demo_smplx_augmentation.py --clips 01986
    .venv/bin/python scripts/demo_smplx_augmentation.py --elbow-flex -20 --shoulder-raise 25 --shape 2

Output: data/test_output/videos/<clip>_smplx_<edit>.mp4
"""

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from view_round_trip import (
    ARM_BONES, CONTEXT_BONES, LANDMARK_DIR, OUTPUT_DIR, VideoSink, draw_skeleton, load_clip,
    read_frames, round_trip_clip, set_limits, view_limits,
)
from kinetic_augment.body_model.converter import reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

# SMPL-X joint indices.
SPINE3 = 9
COLLAR = {"L": 13, "R": 14}
SHOULDER = {"L": 16, "R": 17}
ELBOW = {"L": 18, "R": 19}
WRIST = {"L": 20, "R": 21}
# MediaPipe landmark indices (shoulder, elbow, wrist) per side.
MP_ARM = {"L": (11, 13, 15), "R": (12, 14, 16)}

MIN_BEND, MAX_BEND = 5.0, 150.0      # degrees; elbow bend kept anatomical
MAX_ELEVATION = 170.0                # degrees from "hanging down"; never straight overhead


# --------------------------------------------------------------------------- kinematics

def full_rotations(fit, num_joints):
    rotations = np.zeros((num_joints, 3))
    rotations[0] = fit.global_orient
    rotations[1:22] = np.asarray(fit.body_pose).reshape(21, 3)
    return rotations


def world_pose(rotations, body):
    """World rotation matrices and joint positions (same chain as forward_kinematics)."""
    local = Rotation.from_rotvec(rotations).as_matrix()
    rest, parents = np.asarray(body.rest_joints), np.asarray(body.parents)
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    rot = np.zeros_like(local)
    pos = np.zeros_like(rest)
    for j in range(body.num_joints):
        p = parents[j]
        if p < 0:
            rot[j], pos[j] = local[j], offsets[j]
        else:
            rot[j] = rot[p] @ local[j]
            pos[j] = rot[p] @ offsets[j] + pos[p]
    return rot, pos


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b),
                                        -1.0, 1.0)))


def rotate_in_world(rotations, world_rot, joint, parent, delta):
    """Rotate `joint`'s child chain by world rotation `delta` about the joint itself."""
    local = Rotation.from_rotvec(rotations[joint]).as_matrix()
    new_local = world_rot[parent].T @ delta @ world_rot[parent] @ local
    rotations[joint] = Rotation.from_matrix(new_local).as_rotvec()


# --------------------------------------------------------------------------- edits

def elbow_flex(fit, body, degrees):
    rotations = full_rotations(fit, body.num_joints)
    rot, pos = world_pose(rotations, body)
    for side in ("L", "R"):
        upper = pos[ELBOW[side]] - pos[SHOULDER[side]]
        fore = pos[WRIST[side]] - pos[ELBOW[side]]
        bend = _angle(upper, fore)
        axis = _unit(np.cross(upper, fore))
        if axis is None:  # straight arm: bend toward the body's front
            axis = _unit(np.cross(upper, rot[SPINE3] @ np.array([0.0, 0.0, 1.0])))
        if axis is None:
            continue
        # A bend already outside the range is left there, so zero strength is the identity.
        step = np.clip(bend + degrees, min(MIN_BEND, bend), max(MAX_BEND, bend)) - bend
        delta = Rotation.from_rotvec(axis * np.radians(step)).as_matrix()
        rotate_in_world(rotations, rot, ELBOW[side], SHOULDER[side], delta)
    return dataclasses.replace(fit, body_pose=rotations[1:22].reshape(63).copy())


def shoulder_raise(fit, body, degrees):
    rotations = full_rotations(fit, body.num_joints)
    rot, pos = world_pose(rotations, body)
    down = -(rot[SPINE3] @ np.array([0.0, 1.0, 0.0]))   # SMPL-X is y-up at rest
    for side in ("L", "R"):
        upper = pos[ELBOW[side]] - pos[SHOULDER[side]]
        elevation = _angle(upper, down)
        axis = _unit(np.cross(down, upper))              # rotating about it lifts the arm
        if axis is None:  # arm exactly along the body: lift sideways
            axis = _unit(np.cross(down, pos[SHOULDER[side]] - pos[SPINE3]))
        if axis is None:
            continue
        step = np.clip(elevation + degrees, 0.0, max(MAX_ELEVATION, elevation)) - elevation
        delta = Rotation.from_rotvec(axis * np.radians(step)).as_matrix()
        rotate_in_world(rotations, rot, SHOULDER[side], COLLAR[side], delta)
    return dataclasses.replace(fit, body_pose=rotations[1:22].reshape(63).copy())


def body_shape(fit, body, sigmas):
    betas = np.array(fit.betas, dtype=np.float64).copy()
    betas[0] += sigmas
    return dataclasses.replace(fit, betas=betas)


EDITS = {
    "elbow_flex": (elbow_flex, "elbow bend", "°"),
    "shoulder_raise": (shoulder_raise, "arm elevation", "°"),
    "body_shape": (body_shape, "arm length", " mm"),
}


# --------------------------------------------------------------------------- measurement

def measure(points, kind):
    """The quantity each edit is supposed to change, per side, on MediaPipe landmarks."""
    out = {}
    for side, (s, e, w) in MP_ARM.items():
        if not np.isfinite(points[[s, e, w]]).all():
            continue
        if kind == "elbow_flex":
            out[side] = _angle(points[e] - points[s], points[w] - points[e])
        elif kind == "shoulder_raise":
            down = np.array([0.0, 1.0, 0.0])             # MediaPipe world is y-down
            out[side] = _angle(points[e] - points[s], down)
        else:
            out[side] = 1000 * (np.linalg.norm(points[e] - points[s])
                                + np.linalg.norm(points[w] - points[e]))
    return out


# --------------------------------------------------------------------------- rendering

def render(clip, kind, amount, pose, ok, frames, fps, fits, recon, aug, out_dir):
    _, label, unit = EDITS[kind]
    before = [measure(r, kind) for r in recon]
    after = [measure(a, kind) for a in aug]
    deltas = np.array([after[t][s] - before[t][s] for t in range(len(pose))
                       for s in before[t] if s in after[t]])

    centre, half = view_limits(pose, recon, aug)
    out_path = out_dir / f"{clip}_smplx_{kind}.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(14, 5.4), dpi=100)
    title = f"{kind} {amount:+g}{'σ (betas[0])' if kind == 'body_shape' else '°'}"

    for t in range(len(pose)):
        fig.clf()
        axes = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.08))
        axes[0].imshow(frames[t])
        axes[0].set_title(f"clip {clip}  frame {t + 1}/{len(pose)}", fontsize=10)
        axes[0].axis("off")
        valid_r = np.isfinite(recon[t]).all(axis=1)
        valid_a = np.isfinite(aug[t]).all(axis=1)
        for ax, view in ((axes[1], "front"), (axes[2], "side")):
            draw_skeleton(ax, pose[t], ok[t], view, "#7f9cc4", bones=CONTEXT_BONES, lw=1.2,
                          alpha=0.6, joints=False)
            draw_skeleton(ax, pose[t], ok[t], view, "#1f5fbf", lw=1.5, alpha=0.5,
                          label="MediaPipe input")
            if fits[t] is not None:
                draw_skeleton(ax, recon[t], valid_r, view, "#2ca02c", lw=2.0,
                              label="round trip (no edit)")
                draw_skeleton(ax, aug[t], valid_a, view, "#d62728", lw=2.5,
                              bones=ARM_BONES, label=f"augmented: {title}")
            set_limits(ax, centre, half, view,
                       "front view" if view == "front" else "side view (depth →)")
        axes[1].legend(loc="lower left", fontsize=8, framealpha=0.85)
        if before[t]:
            text = "\n".join(f"{side} {label}: {before[t][side]:6.1f} → {after[t][side]:6.1f}{unit}"
                             for side in before[t] if side in after[t])
            axes[2].text(0.98, 0.02, text, transform=axes[2].transAxes, ha="right",
                         va="bottom", fontsize=8, family="monospace",
                         bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        sink.add(fig)

    sink.close()
    plt.close(fig)
    if deltas.size:
        print(f"  {clip} {kind:<15} requested {amount:+g}  measured Δ{label}: "
              f"median {np.median(deltas):+.1f}{unit}  [p5 {np.percentile(deltas, 5):+.1f}, "
              f"p95 {np.percentile(deltas, 95):+.1f}]  -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="*", help="clip ids (default: all cached)")
    parser.add_argument("--elbow-flex", type=float, default=20.0, help="degrees; + bends more")
    parser.add_argument("--shoulder-raise", type=float, default=20.0, help="degrees; + raises")
    parser.add_argument("--shape", type=float, default=2.0, help="std devs added to betas[0]")
    parser.add_argument("--only", choices=list(EDITS), nargs="*", help="subset of edits")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    amounts = {"elbow_flex": args.elbow_flex, "shoulder_raise": args.shoulder_raise,
               "body_shape": args.shape}
    kinds = args.only or list(EDITS)
    clips = args.clips or sorted(p.name.split("_")[0] for p in LANDMARK_DIR.glob("*_world.npz"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skeleton = SMPLXSkeleton(model_path="models")

    for clip in clips:
        pose, ok = load_clip(clip)
        frames, fps = read_frames(clip, len(pose))
        body, fits, recon = round_trip_clip(pose, ok, skeleton)
        for kind in kinds:
            edit = EDITS[kind][0]
            aug = np.full_like(recon, np.nan)
            for t, fit in enumerate(fits):
                if fit is not None:
                    aug[t] = reconstruct_body(edit(fit, body, amounts[kind]), skeleton)
            render(clip, kind, amounts[kind], pose, ok, frames, fps, fits, recon, aug,
                   args.output_dir)


if __name__ == "__main__":
    main()
