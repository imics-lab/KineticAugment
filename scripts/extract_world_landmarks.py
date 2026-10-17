#!/usr/bin/env python3
"""
Extract MediaPipe world landmarks for metric body-model fitting.

The cached 543-landmark arrays hold `pose_landmarks`: image-normalised
coordinates where x is divided by image width and y by image height, so any
non-square video is anisotropically stretched. Those cannot be fitted to a
metric body model -- see Finding 9 in docs/V1_Problem_Scope.md.

`pose_world_landmarks` are metric, in real units, with the origin at the midpoint
between the hips. This script extracts them alongside per-landmark visibility.

Output is one .npz per clip containing:
    landmarks  (T, 33, 3) metres, hip-centred; NaN where no pose was detected
    visibility (T, 33)    MediaPipe's confidence, 0 where no pose was detected

Missing data is marked with NaN rather than zeros: the origin is a legitimate
position in hip-centred coordinates, so zeros cannot serve as a sentinel.

The existing image-space arrays are left untouched -- face and hand landmarks
have no world-space equivalent, so the 543-landmark format remains the source
for anything beyond the body.

Usage:
    python scripts/extract_world_landmarks.py
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

NUM_POSE_LANDMARKS = 33
VIDEO_DIR = Path("data/sign_videos")
OUTPUT_DIR = Path("data/test_output")


def extract(video_path: Path, holistic) -> tuple:
    capture = cv2.VideoCapture(str(video_path))
    landmarks, visibility = [], []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        world = getattr(result, "pose_world_landmarks", None)

        if world is None:
            landmarks.append(np.full((NUM_POSE_LANDMARKS, 3), np.nan))
            visibility.append(np.zeros(NUM_POSE_LANDMARKS))
        else:
            landmarks.append([[p.x, p.y, p.z] for p in world.landmark])
            visibility.append([p.visibility for p in world.landmark])

    capture.release()
    return np.array(landmarks, dtype=np.float32), np.array(visibility, dtype=np.float32)


def extract_all(videos, output_dir: Path) -> None:
    """
    Extract each clip with its own Holistic instance.

    Video mode is stateful -- a tracked region of interest and landmark smoothing.
    A shared instance carries that state across clips, making a clip's landmarks
    depend on which clip preceded it (up to 114 mm on the sample clips).
    """
    import mediapipe as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'clip':<10}{'frames':>8}{'detected':>10}{'median vis':>12}")
    print("-" * 40)

    for video in videos:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1
        ) as holistic:
            landmarks, visibility = extract(video, holistic)
        detected = np.isfinite(landmarks).all(axis=(1, 2))

        out = output_dir / f"{video.stem}_world.npz"
        np.savez_compressed(out, landmarks=landmarks, visibility=visibility)

        vis = visibility[detected, :NUM_POSE_LANDMARKS]
        print(f"{video.stem:<10}{len(landmarks):>8}{detected.sum():>9}/{len(landmarks)}"
              f"{np.median(vis) if vis.size else float('nan'):>11.2f}")


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No videos found under {VIDEO_DIR}")

    extract_all(videos, OUTPUT_DIR)
    print(f"\nWrote {len(videos)} files to {OUTPUT_DIR}/*_world.npz")


if __name__ == "__main__":
    main()
