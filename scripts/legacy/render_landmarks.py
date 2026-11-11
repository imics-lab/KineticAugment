#!/usr/bin/env python3
"""
render_landmarks.py

Renders a two-colour overlay video:
  BLUE  = original MediaPipe landmarks
  RED   = SMPL-X round-trip result
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    TOTAL_LANDMARKS,
    LH_START, LH_END,
    RH_START, RH_END,
)


class MediaPipeExtractor:

    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic()

    def extract_from_video(self, path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)
            frames.append(self._extract(res))
        cap.release()
        return np.asarray(frames, np.float32), fps

    def _extract(self, res):
        lmk = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                lmk[i] = [lm.x, lm.y, lm.z]
        if res.face_landmarks:
            for i, lm in enumerate(res.face_landmarks.landmark):
                lmk[NUM_POSE_LANDMARKS + i] = [lm.x, lm.y, lm.z]
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                lmk[LH_START + i] = [lm.x, lm.y, lm.z]
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                lmk[RH_START + i] = [lm.x, lm.y, lm.z]
        return lmk


def _to_lmk(points):
    out = landmark_pb2.NormalizedLandmarkList()
    for x, y, z in points:
        out.landmark.add(x=float(x), y=float(y), z=float(z))
    return out


def render_overlay(original, roundtrip, out_path, fps):
    draw = mp.solutions.drawing_utils
    hol  = mp.solutions.holistic

    H, W = 720, 1280
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    T    = min(len(original), len(roundtrip))
    BLUE = (200, 80, 0)
    RED  = (0, 80, 200)

    for i in range(T):
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

        pose_orig = original[i, :NUM_POSE_LANDMARKS]
        if np.any(pose_orig != 0):
            draw.draw_landmarks(
                canvas, _to_lmk(pose_orig), hol.POSE_CONNECTIONS,
                draw.DrawingSpec(color=BLUE, thickness=2, circle_radius=3),
                draw.DrawingSpec(color=BLUE, thickness=2),
            )

        pose_rt = roundtrip[i, :NUM_POSE_LANDMARKS]
        if np.any(pose_rt != 0):
            draw.draw_landmarks(
                canvas, _to_lmk(pose_rt), hol.POSE_CONNECTIONS,
                draw.DrawingSpec(color=RED, thickness=2, circle_radius=3),
                draw.DrawingSpec(color=RED, thickness=2),
            )

        cv2.putText(
            canvas, "Original (blue)  |  SMPL-X round-trip (red)",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
        )
        writer.write(canvas)

    writer.release()
    print("Saved:", out_path)


def reconstruct_sequence(landmarks, wrapper):
    from kinetic_augment.legacy.mp_to_smplx import MediaPipeToSMPLX
    from kinetic_augment.legacy.smplx_to_mp import SMPLXToMediaPipe

    fitter    = MediaPipeToSMPLX()
    projector = SMPLXToMediaPipe(wrapper)

    roundtrip  = []
    zero_frame = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)

    for i, frame in enumerate(landmarks):
        if i % 20 == 0:
            print(f"  Frame {i}/{len(landmarks)}")

        if not np.any(frame[:NUM_POSE_LANDMARKS] != 0):
            roundtrip.append(zero_frame)
            continue

        params = fitter.fit(frame)
        batch  = {
            k: (v if v.ndim > 1 else v[np.newaxis])
            for k, v in params.items()
            if not k.startswith("_")
        }

        out = projector.project(
            batch,
            target_mp=frame[np.newaxis],
            return_full_frame=True,
        )
        roundtrip.append(np.asarray(out)[0])

    return np.stack(roundtrip)


def process_video(path, out_dir):
    print(f"\nProcessing: {path.name}")

    extractor = MediaPipeExtractor()
    landmarks, fps = extractor.extract_from_video(str(path))

    if len(landmarks) == 0:
        print("  No frames extracted, skipping.")
        return

    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
    wrapper = SMPLXWrapper(
        model_path=REPO_ROOT / "models",
        gender="neutral",
        use_pca=False,
    )

    roundtrip = reconstruct_sequence(landmarks, wrapper)
    out_path  = out_dir / f"{path.stem}_overlay.mp4"
    render_overlay(landmarks, roundtrip, out_path, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir",  type=Path, default=REPO_ROOT / "data" / "sign_videos")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "test_output" / "videos")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        print("No .mp4 files found in", args.video_dir)
        return

    for v in videos:
        process_video(v, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()