#!/usr/bin/env python3
"""
Measure what the V1 MediaPipe -> SMPL-X -> MediaPipe round trip costs.

Runs the converter's identity round trip with no augmentation in between:

    fit_shape(clip)                         ->  betas, once per clip
    fit_body(landmarks, validity, body)     ->  SMPL-X parameters, per frame
    reconstruct_body(parameters, template)  ->  landmarks

Reconstruction sees only the parameters -- the body shape reaches it through
`betas` -- so this is the error a consumer of the converter actually gets.

Four error figures are reported and must not be conflated:

  direct   reconstruct_body output vs. input, with the clip's fitted shape. The
           converter's own transform is rigid, because both spaces are metric.
           This is the real number.
  template the same round trip with SMPL-X's template body (betas = 0): the
           converter before Step 10.
  floor    the same round trip through a skeleton whose arm bone lengths and
           shoulder width are set to each clip's observed medians, bypassing
           betas. No plausible body is required, so it bounds what any
           constant-shape skeleton does with exactly fitted bone directions. The
           floor is the V1 gate: what lies below it is MediaPipe's own
           frame-to-frame non-rigidity, which no rigid body can reproduce.
  aligned  the direct output after an *extra* free similarity (scale included)
           is fitted at evaluation time. Diagnostic only.

Shape is fitted to the clip it reconstructs. The held-out figures fit it on one
contiguous half of each clip and score the other, which is what a fitted shape
claims to have learned about the clip rather than about the frames it saw.

Plausibility is checked separately, because no arm metric can see it: rotation
about the shoulder axis moves no arm landmark, so an upside-down body scores the
same as an upright one. Each frame is fitted with the nose and with the nose
masked, and the body must stand upright in MediaPipe's y-down world both ways.

Part A uses synthetic ground truth, where the answer is known exactly. It reports
joint *positions*: twist is unobservable from positions, so rotation recovery is
not claimed.

The temporal-jitter noise floor is printed as a reference scale only. It measures
a different statistic from reconstruction error and is not a valid threshold; the
rigid-skeleton floor replaced it. See docs/V1_Problem_Scope.md, Step 9.

Usage:
    python scripts/extract_world_landmarks.py   # once
    python scripts/measure_round_trip.py
"""

import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.converter import (
    ARM_CORRESPONDENCE,
    fit_body,
    reconstruct_body,
)
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.shape import fit_shape
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

NAMES = {11: "L shoulder", 12: "R shoulder", 13: "L elbow",
         14: "R elbow", 15: "L wrist", 16: "R wrist"}
BONES = [(11, 13, "L upper arm"), (13, 15, "L forearm"),
         (12, 14, "R upper arm"), (14, 16, "R forearm")]
L_SHOULDER, R_SHOULDER = 11, 12
NOSE = 0
HEAD_SMPLX = 15
L_SHOULDER_SMPLX, R_SHOULDER_SMPLX = 16, 17
VISIBILITY_THRESHOLD = 0.5
NOISE_FLOOR_REFERENCE_PCT = 0.18


def with_arm_geometry(skeleton, lengths, width):
    """
    Copy of `skeleton` with its four arm bone lengths and shoulder width forced to
    the given values, carrying every descendant joint along.

    Used only for the rigid-skeleton floor. It bypasses `betas`, so it may reach
    proportions SMPL-X's shape space cannot: an upper bound on any shape fit.
    """
    rest = skeleton.rest_joints.copy()
    parents = np.asarray(skeleton.parents)

    def shift(joint, delta):
        stack = [joint]
        while stack:
            j = stack.pop()
            rest[j] += delta
            stack.extend(np.flatnonzero(parents == j))

    axis = rest[R_SHOULDER_SMPLX] - rest[L_SHOULDER_SMPLX]
    current = np.linalg.norm(axis)
    if np.isfinite(width):
        half = 0.5 * (width - current) * axis / current
        shift(L_SHOULDER_SMPLX, -half)
        shift(R_SHOULDER_SMPLX, half)
    for (parent, child, _, _), target in zip(SMPLX_ARM_BONES, lengths):
        if np.isfinite(target):
            bone = rest[child] - rest[parent]
            norm = np.linalg.norm(bone)
            shift(child, bone / norm * (target - norm))
    return SimpleNamespace(rest_joints=rest, parents=parents,
                           num_joints=skeleton.num_joints, rest_nose=skeleton.rest_nose)


def clip_median_lengths(pose, ok_all):
    """Each arm bone's median observed length over the frames where it is seen."""
    out = []
    for _, _, mp_parent, mp_child in SMPLX_ARM_BONES:
        seen = ok_all[:, mp_parent] & ok_all[:, mp_child]
        out.append(np.median(np.linalg.norm(pose[seen, mp_child] - pose[seen, mp_parent],
                                            axis=-1)) if seen.any() else np.nan)
    return out


def synthetic_landmarks(skeleton, rotations):
    joints = forward_kinematics(rotations, skeleton)
    landmarks = np.full((33, 3), np.nan)
    for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
        landmarks[mp_idx] = joints[smplx_idx]
    return landmarks, np.isfinite(landmarks).all(axis=1)


def run_synthetic(skeleton, root_deg, trials=200):
    rng = np.random.default_rng(42)
    worst = 0.0
    for _ in range(trials):
        truth = np.zeros((skeleton.num_joints, 3))
        if root_deg:
            axis = rng.normal(size=3)
            truth[0] = axis / np.linalg.norm(axis) * np.deg2rad(root_deg)
        for joint, _, _, _ in SMPLX_ARM_BONES:
            truth[joint] = rng.normal(0, 0.4, 3)
        landmarks, validity = synthetic_landmarks(skeleton, truth)
        out = reconstruct_body(fit_body(landmarks, validity, skeleton), skeleton)
        for _, mp_idx in ARM_CORRESPONDENCE:
            worst = max(worst, np.linalg.norm(out[mp_idx] - landmarks[mp_idx]))
    return worst


def load_clip(path):
    data = np.load(path)
    pose = data["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (data["visibility"] >= VISIBILITY_THRESHOLD)
    return pose, ok


def floor_skeleton(skeleton, pose, ok_all):
    """The clip's median arm geometry, bypassing betas."""
    both = ok_all[:, L_SHOULDER] & ok_all[:, R_SHOULDER]
    width = float(np.median(np.linalg.norm(pose[both, L_SHOULDER] - pose[both, R_SHOULDER],
                                           axis=-1)))
    return with_arm_geometry(skeleton, clip_median_lengths(pose, ok_all), width)


def errors(pose, ok_all, fit_with, reconstruct_with, width):
    """Per-landmark round-trip errors, as fractions of `width`; skipped frame count."""
    out = {m: [] for m in NAMES}
    skipped = 0
    for frame, ok in zip(pose, ok_all):
        try:
            rebuilt = reconstruct_body(fit_body(frame, ok, fit_with), reconstruct_with)
        except ValueError:
            skipped += 1
            continue
        for m in NAMES:
            if ok[m] and np.isfinite(rebuilt[m]).all():   # unplaced landmarks come back NaN
                out[m].append(np.linalg.norm(rebuilt[m] - frame[m]) / width)
    return out, skipped


def is_upright(fit, body):
    """Head joint above the shoulder midpoint, in MediaPipe's y-down world."""
    rotations = np.zeros((body.num_joints, 3))
    rotations[0] = fit.global_orient
    rotations[1:22] = fit.body_pose.reshape(21, 3)
    joints = forward_kinematics(rotations, body)
    return joints[HEAD_SMPLX, 1] < 0.5 * (joints[L_SHOULDER_SMPLX, 1] + joints[R_SHOULDER_SMPLX, 1])


def pooled(per_landmark):
    return np.concatenate([np.asarray(v, dtype=float) for v in per_landmark.values()])


def run_real(skeleton):
    files = sorted(glob.glob("data/test_output/*_world.npz"))
    if not files:
        raise SystemExit("No world clips. Run scripts/extract_world_landmarks.py first.")

    keys = ("direct", "template", "floor", "aligned")
    results = {k: {m: [] for m in NAMES} for k in keys}
    held = {k: [] for k in ("direct", "floor")}
    coverage = {m: [0, 0] for m in NAMES}
    length = {name: [] for *_, name in BONES}
    per_clip, skipped, frames = {}, 0, 0
    posture = {"nose": 0, "masked": 0, "fitted": 0, "angles": []}

    for path in files:
        pose, ok_all = load_clip(path)
        both = ok_all[:, L_SHOULDER] & ok_all[:, R_SHOULDER]
        width = float(np.median(np.linalg.norm(
            pose[both, L_SHOULDER] - pose[both, R_SHOULDER], axis=-1)))
        betas = fit_shape(pose, ok_all, skeleton)
        body = skeleton.shaped(betas)
        floor = floor_skeleton(skeleton, pose, ok_all)

        frames += len(pose)
        for ok in ok_all:
            for m in NAMES:
                coverage[m][1] += 1
                coverage[m][0] += int(ok[m])

        clip = {}
        for key, fit_with, reconstruct_with in (("direct", body, skeleton),
                                                ("template", skeleton, skeleton),
                                                ("floor", floor, floor)):
            found, missed = errors(pose, ok_all, fit_with, reconstruct_with, width)
            for m in NAMES:
                results[key][m] += found[m]
            clip[key] = np.median(pooled(found))
        skipped += missed

        for frame, ok in zip(pose, ok_all):
            if not (ok[L_SHOULDER] and ok[R_SHOULDER]):
                continue
            fit = fit_body(frame, ok, body)
            out = reconstruct_body(fit, skeleton)

            masked = ok.copy()
            masked[NOSE] = False
            fallback = fit_body(frame, masked, body)
            posture["fitted"] += 1
            posture["nose"] += int(is_upright(fit, body))
            posture["masked"] += int(is_upright(fallback, body))
            posture["angles"].append(np.degrees((Rotation.from_rotvec(fit.global_orient).inv()
                                                 * Rotation.from_rotvec(fallback.global_orient))
                                                .magnitude()))

            present = [m for m in NAMES if ok[m] and np.isfinite(out[m]).all()]
            src = np.array([out[m] for m in present])
            tgt = np.array([frame[m] for m in present])
            realigned = apply_similarity(src, *similarity_transform(src, tgt))
            for m, r, t in zip(present, realigned, tgt):
                results["aligned"][m].append(np.linalg.norm(r - t) / width)
            for a, b, name in BONES:
                if ok[a] and ok[b] and np.isfinite(out[[a, b]]).all():
                    length[name].append(abs(np.linalg.norm(out[b] - out[a])
                                            - np.linalg.norm(frame[b] - frame[a])) / width)

        # Held out: shape from one contiguous half, scored on the other.
        half = len(pose) // 2
        for train, test in ((slice(0, half), slice(half, None)),
                            (slice(half, None), slice(0, half))):
            trained = skeleton.shaped(fit_shape(pose[train], ok_all[train], skeleton))
            found, _ = errors(pose[test], ok_all[test], trained, skeleton, width)
            held["direct"].append(pooled(found))
            trained_floor = floor_skeleton(skeleton, pose[train], ok_all[train])
            found, _ = errors(pose[test], ok_all[test], trained_floor, trained_floor, width)
            held["floor"].append(pooled(found))

        clip["norm"] = np.linalg.norm(betas)
        per_clip[Path(path).stem.replace("_world", "")] = clip

    held = {k: np.concatenate(v) for k, v in held.items()}
    return results, held, coverage, length, per_clip, skipped, frames, posture


def pct(values, q=50):
    return np.percentile(values, q) * 100 if len(values) else float("nan")


def main():
    skeleton = SMPLXSkeleton(model_path="models")

    print("=" * 70)
    print("PART A - synthetic ground truth (joint positions; twist not claimed)")
    print("=" * 70)
    for deg in (0, 30):
        worst = run_synthetic(skeleton, deg)
        status = "exact" if worst < 1e-9 else "NOT exact -- regression in root handling"
        print(f"  root rotated {deg:>2} deg   worst error {worst:.3e} m   {status}")

    results, held, coverage, length, per_clip, skipped, frames, posture = run_real(skeleton)
    direct, template = results["direct"], results["template"]
    floor, aligned = results["floor"], results["aligned"]

    print("\n" + "=" * 70)
    print("PART B - real clips, world landmarks, identity round trip")
    print("=" * 70)
    print(f"  frames: {frames}   skipped (a shoulder invalid, so no torso frame): {skipped}")
    print(f"  shape: {skeleton.num_betas} betas per clip, inside SMPL-X's plausible region\n")

    print(f"  {'joint':<12}{'coverage':>9}{'direct':>9}{'p95':>8}{'template':>10}"
          f"{'floor':>8}{'aligned':>9}")
    print("  " + "-" * 63)
    for m, name in NAMES.items():
        cov = 100 * coverage[m][0] / coverage[m][1]
        print(f"  {name:<12}{cov:>8.0f}%{pct(direct[m]):>8.2f}%{pct(direct[m], 95):>7.2f}%"
              f"{pct(template[m]):>9.2f}%{pct(floor[m]):>7.2f}%{pct(aligned[m]):>8.2f}%")

    print(f"\n  {'clip':<8}{'direct':>9}{'template':>10}{'floor':>8}{'|betas|':>9}")
    for clip, row in per_clip.items():
        print(f"  {clip:<8}{row['direct'] * 100:>8.2f}%{row['template'] * 100:>9.2f}%"
              f"{row['floor'] * 100:>7.2f}%{row['norm']:>9.2f}")

    print(f"\n  {'bone':<14}{'length error':>14}")
    for *_, name in BONES:
        print(f"  {name:<14}{pct(length[name]):>13.2f}%")

    d, t = pct(pooled(direct)), pct(pooled(template))
    fl, a = pct(pooled(floor)), pct(pooled(aligned))
    hd, hf = pct(held["direct"]), pct(held["floor"])

    print("\n" + "=" * 70)
    print(f"  DIRECT  round-trip error : {d:.2f}% of shoulder width   <- the real number")
    print(f"    template body (betas = 0)       : {t:.2f}%")
    print(f"  RIGID-SKELETON FLOOR     : {fl:.2f}%   <- the V1 gate (best constant shape)")
    print(f"    shape recovers {t - d:.2f} of the {t - fl:.2f}-pt gap "
          f"({100 * (t - d) / (t - fl):.0f}%); {d - fl:.2f} pts remain")
    print(f"  HELD OUT (shape from the other half): direct {hd:.2f}%, floor {hf:.2f}% "
          f"-> {100 * (t - hd) / (t - hf):.0f}% of that gap")
    print("    below the floor: MediaPipe frame-to-frame non-rigidity, which no rigid")
    print("    body can reproduce")
    print(f"  ALIGNED round-trip error : {a:.2f}%   (diagnostic; an evaluation-time "
          f"similarity hides {d - a:.2f} pts)")
    print(f"  jitter reference         : {NOISE_FLOOR_REFERENCE_PCT:.2f}%   (different "
          f"statistic; NOT a threshold)")
    n, angles = posture["fitted"], posture["angles"]
    print(f"  UPRIGHT BODY             : {posture['nose']}/{n} frames with the nose, "
          f"{posture['masked']}/{n} with it masked")
    print(f"    masked root differs from the nose root by median {np.median(angles):.1f} deg, "
          f"max {np.max(angles):.1f} deg (pitch only; arms unaffected)")
    print("=" * 70)


if __name__ == "__main__":
    main()
