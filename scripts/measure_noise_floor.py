#!/usr/bin/env python3
"""
Step 1 of the V1 round-trip scope: measure MediaPipe's own noise floor.

Originally this was to set V1's success gate: "round-trip error at or below
MediaPipe's frame-to-frame noise." It no longer does. Jitter is a different
statistic from round-trip error -- a constant offset has none -- so the result is
a reference scale only; the gate became the rigid-skeleton floor reported by
measure_round_trip.py (docs/V1_Problem_Scope.md, Section 2 and Step 9).

Two independent estimators are used, because each fails in a different way and
agreement between them is the evidence that the number is real:

  Method A - low-percentile frame-to-frame displacement.
      On frames where a landmark is nearly stationary, observed displacement is
      mostly detector jitter. The 10th percentile of ||p[t+1] - p[t]|| approximates
      that. Biased UPWARD by any residual real motion, so it is a conservative bound.

  Method B - second-difference high-frequency estimator.
      For p[t] = s[t] + n[t] with s smooth and n iid noise of deviation sigma,
      the second difference D[t] = p[t-1] - 2p[t] + p[t+1] annihilates constant
      velocity, leaving Var(D) = 6*sigma^2. So sigma ~= std(D)/sqrt(6).
      Independent of any "is it moving?" judgement.

Results are normalised by each clip's median shoulder width, making them
scale-invariant and directly comparable to SMPL-X's 0.3159 m shoulder width.

x/y and z are reported separately: MediaPipe's z is inferred pseudo-depth and is
expected to be substantially noisier than the image-plane coordinates.

Usage:
    python scripts/measure_noise_floor.py
"""

import glob
from pathlib import Path

import numpy as np

# MediaPipe pose landmark indices -- the 12 reliably corresponded body landmarks
# that V1 fits against. Face, hands and feet are out of scope.
IN_SCOPE = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

VISIBILITY_THRESHOLD = 0.5
STATIONARY_PERCENTILE = 10
MOTION_GATE_PERCENTILE = 25
SECOND_DIFF_SCALE = np.sqrt(6.0)
MIN_OBSERVED_FRACTION = 0.5  # landmarks observed less often than this are reported, not pooled


def load_clip(path: str):
    """
    Load one clip's world landmarks and a mask of which are actually observed.

    World landmarks are metric and hip-centred, so the origin is a legitimate
    position and cannot mark missing data; undetected frames are NaN. MediaPipe's
    own per-landmark visibility supplies the rest: in upper-body-framed video it
    rates knees at ~0.02 and ankles at ~0.00, which is the detector stating
    plainly that those coordinates are prior, not observation.
    """
    data = np.load(path)
    pose = data["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (data["visibility"] >= VISIBILITY_THRESHOLD)
    return pose, ok


def shoulder_width(pose: np.ndarray, ok_all: np.ndarray) -> float:
    """Median 3-D shoulder separation over the clip. Median resists per-frame jitter."""
    ok = ok_all[:, LEFT_SHOULDER] & ok_all[:, RIGHT_SHOULDER]
    if ok.sum() == 0:
        return np.nan
    d = pose[ok, LEFT_SHOULDER] - pose[ok, RIGHT_SHOULDER]
    return float(np.median(np.linalg.norm(d, axis=-1)))


def _valid_runs(track: np.ndarray, ok: np.ndarray, width: int):
    """Yield (start, window) for contiguous runs of `width` observed frames."""
    for start in range(len(track) - width + 1):
        if ok[start:start + width].all():
            yield start, track[start:start + width]


def _speed(track: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Per-frame speed, NaN where either endpoint is unobserved."""
    sp = np.full(len(track), np.nan)
    for start, w in _valid_runs(track, ok, 2):
        sp[start] = np.linalg.norm(w[1] - w[0])
    return sp


def method_a(track: np.ndarray, ok: np.ndarray, dims) -> float:
    """Low-percentile frame-to-frame displacement."""
    steps = [np.linalg.norm(w[1][dims] - w[0][dims]) for _, w in _valid_runs(track, ok, 2)]
    if len(steps) < 5:
        return np.nan
    return float(np.percentile(steps, STATIONARY_PERCENTILE))


def method_b(track: np.ndarray, ok: np.ndarray, dims) -> float:
    """
    Second-difference estimator restricted to low-motion windows.

    sigma ~= std(p[t-1] - 2p[t] + p[t+1]) / sqrt(6), valid only where the signal
    is locally linear. Rapid direction changes violate that and inflate the
    estimate, so windows are gated to the slowest MOTION_GATE_PERCENTILE of
    frames for this landmark. Without the gate the estimator tracks movement
    speed rather than detector noise.
    """
    sp = _speed(track, ok)
    if np.all(np.isnan(sp)):
        return np.nan
    gate = np.nanpercentile(sp, MOTION_GATE_PERCENTILE)

    diffs = [w[0][dims] - 2.0 * w[1][dims] + w[2][dims]
             for start, w in _valid_runs(track, ok, 3)
             if np.nanmax(sp[start:start + 2]) <= gate]
    if len(diffs) < 5:
        return np.nan
    per_axis = np.std(np.asarray(diffs), axis=0)
    return float(np.linalg.norm(per_axis) / SECOND_DIFF_SCALE)


def main() -> None:
    files = sorted(glob.glob("data/test_output/*_world.npz"))
    if not files:
        raise SystemExit("No world clips found. Run scripts/extract_world_landmarks.py first.")

    bands = {"xy": [0, 1], "z": [2], "xyz": [0, 1, 2]}
    # rows[band][method] -> list of per-(clip, landmark) values, normalised
    rows = {b: {"A": [], "B": []} for b in bands}
    per_landmark = {name: [] for name in IN_SCOPE}
    observed = {name: [] for name in IN_SCOPE}

    print(f"Clips: {len(files)}\n")
    print(f"{'clip':<10}{'frames':>7}{'shoulderW':>11}"
          f"{'A_xy':>9}{'B_xy':>9}{'A_z':>9}{'B_z':>9}")
    print("-" * 64)

    for path in files:
        pose, ok_all = load_clip(path)
        sw = shoulder_width(pose, ok_all)
        if not np.isfinite(sw) or sw <= 0:
            print(f"{Path(path).stem:<10} skipped (no shoulder reference)")
            continue

        clip_vals = {b: {"A": [], "B": []} for b in bands}
        for name, idx in IN_SCOPE.items():
            track, ok = pose[:, idx], ok_all[:, idx]
            frac = ok.mean()
            observed[name].append(frac)
            usable = frac >= MIN_OBSERVED_FRACTION

            for band, dims in bands.items():
                a = method_a(track, ok, dims) / sw
                b = method_b(track, ok, dims) / sw
                for key, val in (("A", a), ("B", b)):
                    if np.isfinite(val) and usable:
                        clip_vals[band][key].append(val)
                        rows[band][key].append(val)
            bxyz = method_b(track, ok, bands["xyz"]) / sw
            if np.isfinite(bxyz) and usable:
                per_landmark[name].append(bxyz)

        def med(band, key):
            v = clip_vals[band][key]
            return np.median(v) if v else np.nan

        print(f"{Path(path).stem.replace('_world',''):<10}{len(pose):>7}{sw:>11.4f}"
              f"{med('xy','A')*100:>8.2f}%{med('xy','B')*100:>8.2f}%"
              f"{med('z','A')*100:>8.2f}%{med('z','B')*100:>8.2f}%")

    print("\n" + "=" * 64)
    print("NOISE FLOOR  (as % of shoulder width, pooled over all clips/landmarks)")
    print("=" * 64)
    print(f"{'band':<8}{'method A (p10 step)':>24}{'method B (2nd diff)':>24}")
    print("-" * 64)
    summary = {}
    for band in ("xy", "z", "xyz"):
        a = np.median(rows[band]["A"]) * 100
        b = np.median(rows[band]["B"]) * 100
        summary[band] = (a, b)
        print(f"{band:<8}{a:>23.2f}%{b:>23.2f}%")

    print("\nPer-landmark (method B, 3-D, median across clips):")
    print(f"  {'landmark':<16}{'observed':>10}{'noise':>9}")
    excluded = []
    for name in IN_SCOPE:
        frac = np.mean(observed[name]) if observed[name] else 0.0
        if frac < MIN_OBSERVED_FRACTION:
            excluded.append((name, frac))
            print(f"  {name:<16}{frac * 100:>9.0f}%{'--':>9}   EXCLUDED (low visibility)")
        else:
            v = per_landmark[name]
            val = f"{np.median(v) * 100:.2f}%" if v else "--"
            print(f"  {name:<16}{frac * 100:>9.0f}%{val:>9}")

    if excluded:
        print(f"\n  {len(excluded)} landmark(s) excluded: MediaPipe reports visibility below "
              f"{VISIBILITY_THRESHOLD} in over\n  {1 - MIN_OBSERVED_FRACTION:.0%} of frames. The "
              f"detector is stating that these coordinates are\n  its prior rather than an "
              f"observation.")

    a_xyz, b_xyz = summary["xyz"]
    budget = max(a_xyz, b_xyz)
    ratio = summary["z"][1] / summary["xy"][1] if summary["xy"][1] else float("nan")
    print("\n" + "=" * 64)
    print(f"z is {ratio:.1f}x noisier than x/y")
    print(f"JITTER REFERENCE: {budget:.2f}% of shoulder width "
          f"(~{budget / 100 * 0.3159 * 1000:.1f} mm at SMPL-X scale)")
    print("  A reference scale, not a pass/fail gate: jitter is a different statistic from")
    print("  round-trip error. The V1 gate is the rigid-skeleton floor (measure_round_trip.py).")
    print("=" * 64)


if __name__ == "__main__":
    main()
