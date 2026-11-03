#!/usr/bin/env python3
"""
Extract MediaPipe pose landmarks alone (legacy `solutions.pose.Pose`) for one video.

Spike for the 08925 head-placement experiment: a fresh `Pose` instance (video mode,
`model_complexity=1`, no segmentation) gives, per decoded frame, the 33 image landmarks (normalised
x, y, z), their visibility and presence, and the 33 world landmarks in metres. Frames where no pose
is detected hold NaN and `valid` is False.

Usage (from the repo root):
    PYTHONPATH=src:scripts .venv/bin/python scripts/extract_pose_only.py VIDEO.mp4

Output: <output-dir>/<clip>_pose_only.npz with image (T, 33, 3), visibility (T, 33), presence
(T, 33) when the landmarks carry it, world (T, 33, 3), valid (T,), width, height, fps.
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

OUTPUT_DIR = Path("data/test_output/exp_08925")
NUM_POSE_LANDMARKS = 33
POSE_SETTINGS = dict(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _points(landmark_list) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], dtype=np.float32)


def _field(landmark_list, name: str) -> np.ndarray:
    return np.array([getattr(p, name) for p in landmark_list.landmark], dtype=np.float32)


def extract_pose(video_path: Path, pose) -> dict:
    """Run `pose` over every frame of one clip; NaN where no pose is detected."""
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    image, world, visibility, presence, valid = [], [], [], [], []
    nan3 = np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    nan1 = np.full(NUM_POSE_LANDMARKS, np.nan, dtype=np.float32)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        found = result.pose_landmarks is not None and result.pose_world_landmarks is not None
        valid.append(found)
        image.append(_points(result.pose_landmarks) if found else nan3)
        world.append(_points(result.pose_world_landmarks) if found else nan3)
        visibility.append(_field(result.pose_landmarks, "visibility") if found else nan1)
        presence.append(_field(result.pose_landmarks, "presence") if found else nan1)
    capture.release()
    if not valid:
        raise ValueError(f"{video_path}: no frames decoded")

    out = {"image": np.stack(image), "visibility": np.stack(visibility),
           "world": np.stack(world), "valid": np.array(valid, dtype=bool),
           "width": width, "height": height, "fps": fps}
    presence = np.stack(presence)
    if np.isfinite(presence).any() and np.nanmax(presence) > 0:  # legacy Pose reports 0 presence
        out["presence"] = presence
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="path to the clip, e.g. 08925.mp4")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    import mediapipe as mp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with mp.solutions.pose.Pose(**POSE_SETTINGS) as pose:
        data = extract_pose(args.video, pose)
    out_path = args.output_dir / f"{args.video.stem}_pose_only.npz"
    np.savez_compressed(out_path, **data)
    print(f"{args.video.stem}: {len(data['valid'])} frames, {int(data['valid'].sum())} with a pose"
          f" -> {out_path}")


if __name__ == "__main__":
    main()
