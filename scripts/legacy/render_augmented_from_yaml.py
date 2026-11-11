#!/usr/bin/env python3
"""
render_augmented_from_yaml.py

Standalone renderer that:
- extracts MediaPipe landmarks from videos
- loads augmentation pipelines from YAML only
- applies each augmentation pipeline separately
- renders one output video per pipeline

Each output video overlays:
  BLUE = original MediaPipe landmarks (pre-augmentation)
  RED  = augmented MediaPipe landmarks (post-augmentation)

This file is intentionally self-contained for YAML loading and
augmentation/render orchestration, so it does not depend on
kinetic_augment.legacy.augmentations.
"""

import argparse
import os
import re
import sys
from numbers import Number
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.framework.formats import landmark_pb2
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# Make project src importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kinetic_augment.utils.data_formats import (  # noqa: E402
    FACE_START,
    LH_END,
    LH_START,
    NUM_POSE_LANDMARKS,
    RH_END,
    RH_START,
    TOTAL_LANDMARKS,
    POSE_START,
)
from kinetic_augment.utils.landmark_maps import FACE_FULL_MAP, POSE_FULL_MAP  # noqa: E402


# =========================
# Augmentations
# =========================

def global_rotation(data: np.ndarray, max_angle_deg: dict, **kwargs) -> np.ndarray:
    """Apply a random global rotation to the entire sequence."""
    rx = np.random.uniform(-max_angle_deg.get("x", 0), max_angle_deg.get("x", 0))
    ry = np.random.uniform(-max_angle_deg.get("y", 0), max_angle_deg.get("y", 0))
    rz = np.random.uniform(-max_angle_deg.get("z", 0), max_angle_deg.get("z", 0))

    random_rotation = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)

    num_frames, num_landmarks, _ = data.shape
    reshaped_data = data.reshape(-1, 3)
    rotated_data = random_rotation.apply(reshaped_data)
    return rotated_data.reshape(num_frames, num_landmarks, 3).astype(data.dtype, copy=False)



def global_scaling(data: np.ndarray, scale_range: list, **kwargs) -> np.ndarray:
    """Apply a random uniform scaling to the entire sequence."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return (data * scale_factor).astype(data.dtype, copy=False)



def pose_flipping(data: np.ndarray, **kwargs) -> np.ndarray:
    """Mirror landmarks and swap left/right labels."""
    flipped_data = data.copy()

    # Geometric flip across the sagittal plane.
    flipped_data[:, :, 0] *= -1

    # Semantic swap using the unflipped source values.
    original_copy = data.copy()

    for left_idx, right_idx in POSE_FULL_MAP.items():
        flipped_data[:, POSE_START + left_idx, :] = original_copy[:, POSE_START + right_idx, :]

    for left_idx, right_idx in FACE_FULL_MAP.items():
        flipped_data[:, FACE_START + left_idx, :] = original_copy[:, FACE_START + right_idx, :]

    left_hand_data = original_copy[:, LH_START:LH_END, :]
    right_hand_data = original_copy[:, RH_START:RH_END, :]
    flipped_data[:, RH_START:RH_END, :] = left_hand_data
    flipped_data[:, LH_START:LH_END, :] = right_hand_data

    return flipped_data



def time_warping(
    data: np.ndarray,
    max_warp_factor: float,
    num_knots: int = 5,
    **kwargs,
) -> np.ndarray:
    """Non-linearly warp the time axis while preserving sequence length."""
    num_frames, num_landmarks, num_dims = data.shape
    if num_frames < 2:
        return data.copy()

    t_orig = np.arange(num_frames, dtype=np.float32)
    num_knots = max(3, min(num_knots, num_frames))

    knot_x = np.linspace(0, num_frames - 1, num_knots, dtype=np.float32)
    knot_y_offsets = (
        np.random.uniform(-max_warp_factor, max_warp_factor, size=num_knots).astype(np.float32)
        * num_frames
    )
    knot_y_offsets[0] = 0.0
    knot_y_offsets[-1] = 0.0

    spline = CubicSpline(knot_x, knot_x + knot_y_offsets)
    t_warped = spline(t_orig).astype(np.float32)
    t_warped = np.clip(t_warped, 0, num_frames - 1)
    t_warped = np.maximum.accumulate(t_warped)

    if t_warped[-1] == t_warped[0]:
        return data.copy()

    t_warped = t_warped - t_warped[0]
    t_warped *= (num_frames - 1) / t_warped[-1]

    warped_data = np.zeros_like(data)
    for i in range(num_landmarks):
        for j in range(num_dims):
            warped_data[:, i, j] = np.interp(t_warped, t_orig, data[:, i, j])

    return warped_data



def dynamic_trajectory_jittering(
    data: np.ndarray,
    amplitude: float,
    frequency: float,
    num_waves: int = 3,
    **kwargs,
) -> np.ndarray:
    """Add smooth, low-frequency noise to landmark trajectories."""
    num_frames, _, _ = data.shape
    if num_frames < 2:
        return data.copy()

    noise = np.zeros((num_frames, 3), dtype=np.float32)
    for dim in range(3):
        for _ in range(num_waves):
            freq = np.random.uniform(0.1, frequency)
            phase = np.random.uniform(0.0, np.pi * 2.0)
            noise[:, dim] += np.sin(np.linspace(0, freq * np.pi * 2.0, num_frames) + phase)

    max_abs = np.max(np.abs(noise))
    if max_abs > 0:
        noise /= max_abs
    noise *= amplitude

    jittered_data = data + noise[:, np.newaxis, :]
    return jittered_data.astype(data.dtype, copy=False)



def joint_angle_perturbation(data: np.ndarray, **kwargs) -> np.ndarray:
    print("  >> Warning: 'joint_angle_perturbation' is not implemented. Returning data as-is.")
    return data



def joint_coupled_noise(data: np.ndarray, **kwargs) -> np.ndarray:
    print("  >> Warning: 'joint_coupled_noise' is not implemented. Returning data as-is.")
    return data


AUGMENTATION_REGISTRY = {
    "global_rotation": global_rotation,
    "global_scaling": global_scaling,
    "pose_flipping": pose_flipping,
    "time_warping": time_warping,
    "trajectory_jitter": dynamic_trajectory_jittering,
    "dynamic_trajectory_jittering": dynamic_trajectory_jittering,
    "joint_angle_perturbation": joint_angle_perturbation,
    "joint_coupled_noise": joint_coupled_noise,
}


# =========================
# YAML loading / pipeline application
# =========================

def load_yaml_config(path: Path) -> dict:
    """Load augmentation pipelines from YAML only."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary with a top-level 'pipelines' key.")

    pipelines = config.get("pipelines")
    if pipelines is None:
        raise ValueError("Config is missing required top-level key: 'pipelines'.")
    if not isinstance(pipelines, list):
        raise ValueError("Config key 'pipelines' must be a list.")

    return config



def apply_pipeline(data: np.ndarray, pipeline: dict) -> np.ndarray:
    """Apply one configured augmentation pipeline to a landmark sequence."""
    augmented = data.copy()

    seed = pipeline.get("seed")
    if seed is not None:
        np.random.seed(seed)

    for aug in pipeline.get("augmentations", []):
        aug_name = aug.get("name")
        if aug_name is None:
            raise ValueError(f"Augmentation entry is missing 'name': {aug}")
        if aug_name not in AUGMENTATION_REGISTRY:
            supported = ", ".join(sorted(AUGMENTATION_REGISTRY))
            raise ValueError(f"Unknown augmentation '{aug_name}'. Supported: {supported}")

        params = aug.get("params", {})
        augmented = AUGMENTATION_REGISTRY[aug_name](augmented, **params)

    return augmented.astype(data.dtype, copy=False)


# =========================
# MediaPipe extraction / rendering
# =========================

class MediaPipeExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic()

    def extract_from_video(self, path: str):
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



def _to_lmk(points: np.ndarray):
    out = landmark_pb2.NormalizedLandmarkList()
    for x, y, z in points:
        out.landmark.add(x=float(x), y=float(y), z=float(z))
    return out



def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return safe.strip("_") or "pipeline"



def render_overlay(original_landmarks, augmented_landmarks, out_path: Path, fps: float, label: str = ""):
    """Render original landmarks in blue and augmented landmarks in red."""
    draw = mp.solutions.drawing_utils
    hol = mp.solutions.holistic

    height, width = 720, 1280
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    total_frames = min(len(original_landmarks), len(augmented_landmarks))
    blue = (200, 80, 0)
    red = (0, 80, 200)

    for i in range(total_frames):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        pose_original = original_landmarks[i, :NUM_POSE_LANDMARKS]
        if np.any(pose_original != 0):
            draw.draw_landmarks(
                canvas,
                _to_lmk(pose_original),
                hol.POSE_CONNECTIONS,
                draw.DrawingSpec(color=blue, thickness=2, circle_radius=3),
                draw.DrawingSpec(color=blue, thickness=2),
            )

        pose_augmented = augmented_landmarks[i, :NUM_POSE_LANDMARKS]
        if np.any(pose_augmented != 0):
            draw.draw_landmarks(
                canvas,
                _to_lmk(pose_augmented),
                hol.POSE_CONNECTIONS,
                draw.DrawingSpec(color=red, thickness=2, circle_radius=3),
                draw.DrawingSpec(color=red, thickness=2),
            )

        cv2.putText(
            canvas,
            f"Pipeline: {label} | Blue=pre-augment MP | Red=post-augment MP",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        writer.write(canvas)

    writer.release()
    print("Saved:", out_path)


# =========================
# Video processing
# =========================

def process_video(path: Path, out_dir: Path, config_path: Path):
    print(f"\nProcessing: {path.name}")

    extractor = MediaPipeExtractor()
    original_landmarks, fps = extractor.extract_from_video(str(path))

    if len(original_landmarks) == 0:
        print("  No frames extracted, skipping.")
        return

    config = load_yaml_config(config_path)
    pipelines = config["pipelines"]

    for pipeline in pipelines:
        pipeline_name = pipeline.get("name", "pipeline")
        safe_name = _safe_name(pipeline_name)
        print(f"  >> Pipeline: {pipeline_name}")

        augmented_landmarks = apply_pipeline(original_landmarks, pipeline)

        out_path = out_dir / f"{path.stem}_{safe_name}.mp4"
        render_overlay(
            original_landmarks,
            augmented_landmarks,
            out_path,
            fps,
            label=pipeline_name,
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "data" / "sign_videos")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "test_output" / "videos")
    parser.add_argument("--aug-config", type=Path, default=REPO_ROOT / "configs" / "legacy" / "augments.yaml", help="Path to YAML config.")
    parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to process")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        print("No .mp4 files found in", args.video_dir)
        return


    for video_path in videos[:args.num_videos]:
        process_video(video_path, args.output_dir, args.aug_config)

    print("\nDone.")


if __name__ == "__main__":
    main()
