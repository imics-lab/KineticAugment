#!/usr/bin/env python3
"""
How much does MediaPipe disagree with itself? Context for the V1 round-trip error.

Re-extracts every clip with one setting changed at a time -- each a reasonable way
to run the same detector on the same video -- and compares the arm landmarks with
the stored extraction (full model, smoothing on, video mode). The statistic is the
round-trip harness's own: median per-landmark distance as a percentage of the
clip's shoulder width, over landmarks valid in both runs. World landmarks are
hip-centred in both, so no alignment is applied.

Usage:
    python scripts/extract_world_landmarks.py   # once
    python scripts/measure_detector_agreement.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from extract_world_landmarks import OUTPUT_DIR, VIDEO_DIR, extract  # noqa: E402

ARM = list(range(11, 17))
VISIBILITY_THRESHOLD = 0.5
VARIANTS = {
    "smoothing off": dict(static_image_mode=False, model_complexity=1, smooth_landmarks=False),
    "heavy model": dict(static_image_mode=False, model_complexity=2),
    "static image mode": dict(static_image_mode=True, model_complexity=1),
}


def valid(landmarks, visibility):
    return np.isfinite(landmarks).all(axis=-1) & (visibility >= VISIBILITY_THRESHOLD)


def main():
    import mediapipe as mp

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No videos found under {VIDEO_DIR}")

    print(f"{'variant vs stored':<20}{'median':>9}{'p95':>9}   per clip (median)")
    for name, settings in VARIANTS.items():
        pooled, per_clip = [], []
        for video in videos:
            stored = np.load(OUTPUT_DIR / f"{video.stem}_world.npz")
            ref = stored["landmarks"].astype(np.float64)
            ref_ok = valid(ref, stored["visibility"])
            with mp.solutions.holistic.Holistic(**settings) as holistic:
                other, vis = extract(video, holistic)
            other = other.astype(np.float64)
            both = ref_ok & valid(other, vis)

            shoulders = ref_ok[:, 11] & ref_ok[:, 12]
            width = np.median(np.linalg.norm(ref[shoulders, 12] - ref[shoulders, 11], axis=-1))
            distance = np.linalg.norm(other[:, ARM] - ref[:, ARM], axis=-1)[both[:, ARM]]
            pooled.append(distance / width)
            per_clip.append(f"{video.stem} {100 * np.median(distance / width):.1f}")
        pooled = np.concatenate(pooled)
        print(f"{name:<20}{100 * np.median(pooled):>8.2f}%{100 * np.percentile(pooled, 95):>8.2f}%"
              f"   {', '.join(per_clip)}")


if __name__ == "__main__":
    main()
