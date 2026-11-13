#!/usr/bin/env python3
"""
Step 9 of the V1 scope: diagnose the round-trip error left below the
rigid-skeleton floor that measure_round_trip.py reports.

Each section tests one explanation and prints the evidence for or against it.

  E1  non-rigidity   MediaPipe's bone lengths and shoulder width vary across the
                     frames of a single clip, which no real body does.
  E2  completeness   a skeleton re-shaped to every frame reproduces the landmarks
                     exactly, so nothing but that variation remains below the floor.
  E4  mechanism      does the variation follow joint angles (an anatomical
                     joint-centre effect) or each bone's angle to the camera (depth
                     ambiguity)? Each is scored against a null built by regressing on
                     another clip's signals, because slowly varying series correlate
                     by chance.
  E5  depth scale    does rescaling one axis per clip make the skeleton rigid? Run on
                     x and y as controls: a depth effect should be z-specific.
  E7  offsets        does a constant joint-centre offset, fixed in the frame of the
                     segment each landmark rides on, explain the residual?
  E6  smoothing      (--unsmoothed) re-extracts the videos with MediaPipe's temporal
                     smoothing off, to test whether the variation is per-frame noise
                     the filter stretched into slow drift. Slow: runs MediaPipe.

Usage:
    python scripts/extract_world_landmarks.py   # once
    python scripts/diagnose_residual.py [--unsmoothed]
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_round_trip import load_clip, with_arm_geometry  # noqa: E402

from kinetic_augment.body_model.converter import fit_body, reconstruct_body  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES  # noqa: E402

BONE_NAMES = ["L upper", "L fore", "R upper", "R fore"]
MP_BONES = [(mp, mc) for _, _, mp, mc in SMPLX_ARM_BONES]
JOINTS = {11: "L shoulder", 12: "R shoulder", 13: "L elbow",
          14: "R elbow", 15: "L wrist", 16: "R wrist"}
# MediaPipe landmark -> SMPL-X joint whose world rotation orients the segment it rides on
SEGMENT_FRAME = {11: 13, 12: 14, 13: 16, 14: 17, 15: 18, 16: 19}
SIDES = {"L": (11, 12, 13, 15), "R": (12, 11, 14, 16)}   # shoulder, other, elbow, wrist


def clips():
    files = sorted(glob.glob("data/test_output/*_world.npz"))
    if not files:
        raise SystemExit("No world clips. Run scripts/extract_world_landmarks.py first.")
    return [(os.path.basename(f)[:5], *load_clip(f)) for f in files]


def frame_geometry(frame, ok):
    lengths = [np.linalg.norm(frame[mc] - frame[mp]) if ok[mp] and ok[mc] else np.nan
               for mp, mc in MP_BONES]
    width = np.linalg.norm(frame[12] - frame[11]) if ok[11] and ok[12] else np.nan
    return lengths, width


def shoulder_width(pose, ok):
    both = ok[:, 11] & ok[:, 12]
    return float(np.median(np.linalg.norm(pose[both, 12] - pose[both, 11], axis=-1)))


def r2(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ coef).var() / y.var(), coef


def e1_nonrigidity(data):
    print("E1  non-rigidity -- coefficient of variation across frames, within each clip")
    print(f"  {'clip':<7}" + "".join(f"{n:>10}" for n in BONE_NAMES) + f"{'sh width':>10}")
    for clip, pose, ok in data:
        geo = [frame_geometry(f, o) for f, o in zip(pose, ok)]
        lengths = np.array([g[0] for g in geo])
        widths = np.array([g[1] for g in geo])
        cv = np.nanstd(lengths, axis=0) / np.nanmedian(lengths, axis=0)
        wcv = np.nanstd(widths) / np.nanmedian(widths)
        print(f"  {clip:<7}" + "".join(f"{100 * c:>9.1f}%" for c in cv) + f"{100 * wcv:>9.1f}%")


def e2_completeness(data, skeleton):
    errors = []
    for _, pose, ok_all in data:
        width = shoulder_width(pose, ok_all)
        for frame, ok in zip(pose, ok_all):
            if not (ok[11] and ok[12]):
                continue
            lengths, w = frame_geometry(frame, ok)
            sk = with_arm_geometry(skeleton, lengths, w)
            out = reconstruct_body(fit_body(frame, ok, sk), sk)
            errors += [np.linalg.norm(out[m] - frame[m]) / width for m in JOINTS
                       if ok[m] and np.isfinite(out[m]).all()]
    print("\nE2  completeness -- skeleton re-shaped to every frame")
    print(f"  pooled median error {100 * np.median(errors):.2e}%  "
          f"(max {100 * np.max(errors):.2e}%): below the floor, only non-rigidity remains")


def angle(a, b):
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def e4_mechanism(data):
    series = {}
    for clip, pose, ok_all in data:
        for side, (sh, other, el, wr) in SIDES.items():
            ok = ok_all[:, sh] & ok_all[:, other] & ok_all[:, el] & ok_all[:, wr]
            upper, fore = pose[:, el] - pose[:, sh], pose[:, wr] - pose[:, el]
            anat = np.column_stack([angle(upper, fore), angle(upper, pose[:, sh] - pose[:, other])])
            for bone, vec in (("upper", upper), ("fore", fore)):
                n = np.linalg.norm(vec, axis=-1)
                depth = (np.abs(vec[:, 2]) / n)[:, None]
                series[(clip, f"{side} {bone}")] = (n[ok], anat[ok], depth[ok])

    rows = []
    for (clip, bone), (y, anat, depth) in series.items():
        ra, _ = r2(y, anat)
        rd, coef = r2(y, depth)
        na, nd = [], []
        for (oc, ob), (_, oa, odp) in series.items():
            if oc != clip and ob == bone:
                m = min(len(y), len(oa))
                na.append(r2(y[:m], oa[:m])[0])
                nd.append(r2(y[:m], odp[:m])[0])
        rows.append((ra, np.median(na), rd, np.median(nd), coef[1]))
    r = np.array(rows)
    print("\nE4  mechanism -- R^2 of bone length, against a mismatched-clip null")
    print(f"  joint angles  R^2 {np.median(r[:, 0]):.2f} vs null {np.median(r[:, 1]):.2f}   "
          f"(beats null in {np.sum(r[:, 0] > r[:, 1])}/{len(r)} bones)")
    print(f"  depth angle   R^2 {np.median(r[:, 2]):.2f} vs null {np.median(r[:, 3]):.2f}   "
          f"(beats null in {np.sum(r[:, 2] > r[:, 3])}/{len(r)} bones)")
    print(f"  depth slope sign: {np.sum(r[:, 4] > 0)} positive, {np.sum(r[:, 4] < 0)} negative "
          f"-- consistent within a clip, not across clips")


def e5_depth_scale(data):
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 241))

    def rigidity(pose, ok, axis, k):
        s = np.ones(3)
        s[axis] = k
        cvs = []
        for mp, mc in MP_BONES:
            seen = ok[:, mp] & ok[:, mc]
            n = np.linalg.norm((pose[seen, mc] - pose[seen, mp]) * s, axis=1)
            cvs.append(np.std(n) / np.mean(n))
        return np.mean(cvs)

    print("\nE5  per-clip single-axis rescaling -- best factor and resulting mean bone CV")
    print(f"  {'clip':<7}{'CV k=1':>8}" + "".join(f"{'k_' + a:>8}{'CV':>7}" for a in "xyz"))
    gains = {a: [] for a in "xyz"}
    for clip, pose, ok in data:
        base = rigidity(pose, ok, 2, 1.0)
        line = f"  {clip:<7}{100 * base:>7.2f}%"
        for axis, name in enumerate("xyz"):
            scores = [rigidity(pose, ok, axis, k) for k in grid]
            i = int(np.argmin(scores))
            gains[name].append(1 - scores[i] / base)
            edge = "*" if i in (0, len(grid) - 1) else " "
            line += f"{grid[i]:>7.2f}{edge}{100 * scores[i]:>6.2f}%"
        print(line)
    reductions = "   ".join(f"{a} {100 * np.mean(g):.1f}%" for a, g in gains.items())
    print(f"  mean CV reduction: {reductions}"
          "   (* = factor at the search boundary: a degenerate optimum)")


def world_rotations(fit, parents):
    local = np.zeros((22, 3))
    local[0] = fit.global_orient
    local[1:22] = np.asarray(fit.body_pose).reshape(21, 3)
    rot = Rotation.from_rotvec(local).as_matrix()
    world = np.zeros_like(rot)
    for j in range(22):
        world[j] = rot[j] if parents[j] < 0 else world[parents[j]] @ rot[j]
    return world


def e7_offsets(data, skeleton):
    before, after = [], []
    for _, pose, ok_all in data:
        width = shoulder_width(pose, ok_all)
        lengths = [np.median(np.linalg.norm(pose[ok_all[:, mp] & ok_all[:, mc], mc]
                                            - pose[ok_all[:, mp] & ok_all[:, mc], mp], axis=-1))
                   for mp, mc in MP_BONES]
        sk = with_arm_geometry(skeleton, lengths, width)
        recs = []
        for frame, ok in zip(pose, ok_all):
            if ok[11] and ok[12]:
                fit = fit_body(frame, ok, sk)
                rotations = world_rotations(fit, sk.parents)
                recs.append((frame, ok, reconstruct_body(fit, sk), rotations))
        for m, j in SEGMENT_FRAME.items():
            rows = [(f, rec, W) for f, ok, rec, W in recs if ok[m] and np.isfinite(rec[m]).all()]
            offset = np.mean([W[j].T @ (f[m] - rec[m]) for f, rec, W in rows], axis=0)
            for f, rec, W in rows:
                before.append(np.linalg.norm(f[m] - rec[m]) / width)
                after.append(np.linalg.norm(f[m] - (rec[m] + W[j] @ offset)) / width)
    rb, ra = np.sqrt(np.mean(np.square(before))), np.sqrt(np.mean(np.square(after)))
    print("\nE7  constant joint-centre offsets, fixed in each landmark's segment frame")
    print(f"  median {100 * np.median(before):.2f}% -> {100 * np.median(after):.2f}%   "
          f"RMS {100 * rb:.2f}% -> {100 * ra:.2f}%   (explains {100 * (1 - ra / rb):.1f}% by RMS)")


def e6_smoothing(data):
    import cv2
    import mediapipe as mp

    def bone_stats(pose, ok):
        out = []
        for mp_, mc in MP_BONES:
            seen = ok[:, mp_] & ok[:, mc]
            n = np.linalg.norm(pose[:, mc] - pose[:, mp_], axis=1)
            cons = seen[1:] & seen[:-1]
            jitter = np.std(np.diff(n)[cons]) / np.sqrt(2)
            out.append((np.std(n[seen]) / np.mean(n[seen]), np.std(n[seen]) / jitter))
        return out

    smooth, raw = [], []
    for clip, pose, ok in data:
        smooth += bone_stats(pose, ok)
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                                  smooth_landmarks=False)
        cap = cv2.VideoCapture(f"data/sign_videos/{clip}.mp4")
        L, V = [], []
        while True:
            okf, img = cap.read()
            if not okf:
                break
            w = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).pose_world_landmarks
            L.append(np.full((33, 3), np.nan) if w is None
                     else [[p.x, p.y, p.z] for p in w.landmark])
            V.append(np.zeros(33) if w is None else [p.visibility for p in w.landmark])
        cap.release()
        holistic.close()
        L, V = np.array(L, float), np.array(V)
        raw += bone_stats(L, np.isfinite(L).all(-1) & (V >= 0.5))
    s, r = np.array(smooth), np.array(raw)
    print("\nE6  smoothing on vs off (median over bones)")
    print(f"  bone CV {100 * np.median(s[:, 0]):.1f}% -> {100 * np.median(r[:, 0]):.1f}%   "
          f"slowness {np.median(s[:, 1]):.1f} -> {np.median(r[:, 1]):.1f}   "
          f"(independent per-frame noise would give ~1)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unsmoothed", action="store_true",
                        help="also re-extract with MediaPipe smoothing off (slow)")
    args = parser.parse_args()

    data = clips()
    skeleton = SMPLXSkeleton(model_path="models")
    e1_nonrigidity(data)
    e2_completeness(data, skeleton)
    e4_mechanism(data)
    e5_depth_scale(data)
    e7_offsets(data, skeleton)
    if args.unsmoothed:
        e6_smoothing(data)


if __name__ == "__main__":
    main()
