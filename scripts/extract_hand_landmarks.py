#!/usr/bin/env python3
"""
Extract MediaPipe hand landmarks for the V1.2 hand round trip.

Per clip, a fresh legacy `Hands` instance (video mode, up to two hands) gives each
detected hand's 21 world landmarks -- the only installed MediaPipe API returning
metric hand coordinates -- and its 21 image landmarks. A fresh `Holistic`
instance on the same frame gives the pose, whose image wrists decide each hand's
anatomical side (`hand_observations.assign_sides`). Fresh instances per clip keep
a clip's landmarks independent of the clip processed before it.

Output: data/test_output/<clip>_hands.npz (see `HandClip`). The V1 body caches are
not touched.

Usage:
    python scripts/extract_hand_landmarks.py
"""

import hashlib
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

from kinetic_augment.body_model.hand_model import NUM_HAND_LANDMARKS  # noqa: E402
from kinetic_augment.body_model.hand_observations import (  # noqa: E402
    LABEL_SIDE_UNMIRRORED,
    MAX_WRIST_GAP,
    NUM_POSE_LANDMARKS,
    POSE_WRISTS,
    SIDE_MARGIN,
    SIDES,
    HandClip,
    assign_sides,
    save_hand_clip,
)

VIDEO_DIR = Path("data/sign_videos")
OUTPUT_DIR = Path("data/test_output")
HANDS_SETTINGS = dict(static_image_mode=False, max_num_hands=2, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
HOLISTIC_SETTINGS = dict(static_image_mode=False, model_complexity=1)


def _points(landmark_list) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], dtype=np.float32)


def observe_frame(hand_result, holistic_result, aspect: float) -> dict:
    """One frame's hand and pose observations, each hand on its anatomical side."""
    obs = {
        "world": np.full((2, NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float32),
        "image": np.full((2, NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float32),
        "present": np.zeros(2, dtype=bool),
        "handedness_score": np.full(2, np.nan, dtype=np.float32),
        "label_agrees": np.zeros(2, dtype=bool),
        "unassigned": 0,
        "pose_world": np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
        "pose_image": np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
        "pose_visibility": np.zeros(NUM_POSE_LANDMARKS, dtype=np.float32),
    }
    if holistic_result.pose_landmarks is not None:
        obs["pose_image"] = _points(holistic_result.pose_landmarks)
        obs["pose_visibility"] = np.array(
            [p.visibility for p in holistic_result.pose_landmarks.landmark], dtype=np.float32)
    if holistic_result.pose_world_landmarks is not None:
        obs["pose_world"] = _points(holistic_result.pose_world_landmarks)

    images = hand_result.multi_hand_landmarks or []
    worlds = hand_result.multi_hand_world_landmarks or []
    labels = hand_result.multi_handedness or []
    square = np.array([aspect, 1.0])  # normalised x, y -> fractions of image height
    wrists = np.array([[h.landmark[0].x, h.landmark[0].y] for h in images]).reshape(-1, 2)
    pose_wrists = obs["pose_image"][list(POSE_WRISTS), :2].astype(np.float64)
    sides = assign_sides(wrists * square, pose_wrists * square, MAX_WRIST_GAP, SIDE_MARGIN)

    for image, world, label, side in zip(images, worlds, labels, sides):
        if side < 0:
            obs["unassigned"] += 1
            continue
        best = label.classification[0]
        obs["world"][side] = _points(world)
        obs["image"][side] = _points(image)
        obs["present"][side] = True
        obs["handedness_score"][side] = best.score
        obs["label_agrees"][side] = LABEL_SIDE_UNMIRRORED.get(best.label) == side
    return obs


def extract_hands(video_path: Path, hands, holistic) -> HandClip:
    """Run both detectors over every frame of one clip."""
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows, stamps = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rows.append(observe_frame(hands.process(rgb), holistic.process(rgb), width / height))
    capture.release()
    if not rows:
        raise ValueError(f"{video_path}: no frames decoded")

    stacked = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    return HandClip(
        world=stacked["world"], image=stacked["image"], present=stacked["present"],
        handedness_score=stacked["handedness_score"], label_agrees=stacked["label_agrees"],
        unassigned=stacked["unassigned"].astype(np.int8), pose_world=stacked["pose_world"],
        pose_image=stacked["pose_image"], pose_visibility=stacked["pose_visibility"],
        timestamp_ms=np.array(stamps, dtype=np.float64), width=width, height=height, fps=fps,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_all(videos, output_dir) -> None:
    """Extract each clip with its own Hands and Holistic instances."""
    import mediapipe as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':<8}{'frames':>8}{'left':>7}{'right':>7}{'unassigned':>12}{'label agrees':>14}")
    for video in videos:
        video = Path(video)
        with mp.solutions.hands.Hands(**HANDS_SETTINGS) as hands, \
                mp.solutions.holistic.Holistic(**HOLISTIC_SETTINGS) as holistic:
            clip = extract_hands(video, hands, holistic)
        clip.meta = {
            "source": video.name,
            "source_sha256": _sha256(video),
            "mediapipe": mp.__version__,
            "hands_settings": HANDS_SETTINGS,
            "holistic_settings": HOLISTIC_SETTINGS,
            "side_rule": {"max_wrist_gap": MAX_WRIST_GAP, "side_margin": SIDE_MARGIN,
                          "units": "fraction of image height"},
            "detector_state": "fresh Hands and Holistic instances per clip",
        }
        save_hand_clip(output_dir / f"{video.stem}_hands.npz", clip)
        agrees = clip.label_agrees[clip.present].mean() if clip.present.any() else float("nan")
        print(f"{video.stem:<8}{len(clip.present):>8}{clip.present[:, 0].sum():>7}"
              f"{clip.present[:, 1].sum():>7}{clip.unassigned.sum():>12}{100 * agrees:>13.1f}%")


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No videos found under {VIDEO_DIR}")
    extract_all(videos, OUTPUT_DIR)
    print(f"\nWrote {len(videos)} files to {OUTPUT_DIR}/*_hands.npz (sides: {', '.join(SIDES)})")


if __name__ == "__main__":
    main()
