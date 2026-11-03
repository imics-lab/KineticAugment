#!/usr/bin/env python3
"""
Extract MediaPipe FaceLandmarker output for the V1.3 face work.

Per clip, a fresh FaceLandmarker (Tasks API, VIDEO mode, one face) gives 478 landmarks, 52
blendshapes and the facial transformation matrix per frame. Frames are decoded with OpenCV
exactly as `extract_hand_landmarks.py` decodes them, so frame t here is frame t of
`<clip>_hands.npz`.

Output: <output-dir>/<clip>_face.npz (see `FaceClip`).

Usage:
    python scripts/extract_face_landmarks.py
    python scripts/extract_face_landmarks.py --clips-from data/round_trip_eval \
        --video-dir ~/Downloads/raw_videos --output-dir data/round_trip_eval
"""

import argparse
import hashlib
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

from kinetic_augment.body_model.face_observations import (  # noqa: E402
    NUM_BLENDSHAPES, NUM_FACE_POINTS, FaceClip, save_face_clip,
)

VIDEO_DIR = Path("data/sign_videos")
OUTPUT_DIR = Path("data/test_output")
MODEL = Path("models/mediapipe/face_landmarker.task")
SETTINGS = dict(num_faces=1, min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True)


def observe_frame(result) -> dict:
    """One frame's arrays; NaN when no face was found."""
    if not result.face_landmarks:
        return dict(landmarks=np.full((NUM_FACE_POINTS, 3), np.nan, np.float32),
                    blendshapes=np.full(NUM_BLENDSHAPES, np.nan, np.float32),
                    head_matrix=np.full((4, 4), np.nan, np.float32), valid=False, names=None)
    points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
    categories = result.face_blendshapes[0]
    return dict(
        landmarks=points,
        blendshapes=np.array([c.score for c in categories], dtype=np.float32),
        head_matrix=np.asarray(result.facial_transformation_matrixes[0],
                               dtype=np.float32).reshape(4, 4),
        valid=True,
        names=tuple(c.category_name for c in categories),
    )


def extract_face(video_path: Path, model_path: Path = MODEL) -> FaceClip:
    """Run a fresh FaceLandmarker over every frame of one clip."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                    running_mode=RunningMode.VIDEO, **SETTINGS)
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows, stamps, last = [], [], -1
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            stamps.append(stamp)
            # VIDEO mode needs strictly increasing integer timestamps.
            last = max(int(round(stamp)), last + 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rows.append(observe_frame(landmarker.detect_for_video(image, last)))
    capture.release()
    if not rows:
        raise ValueError(f"{video_path}: no frames decoded")

    names = next((row["names"] for row in rows if row["names"]), ())
    return FaceClip(
        landmarks=np.stack([row["landmarks"] for row in rows]),
        blendshapes=np.stack([row["blendshapes"] for row in rows]),
        head_matrix=np.stack([row["head_matrix"] for row in rows]),
        valid=np.array([row["valid"] for row in rows], dtype=bool),
        timestamp_ms=np.array(stamps, dtype=np.float64),
        width=width, height=height, fps=fps, blendshape_names=tuple(names),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_all(videos, output_dir, model_path: Path = MODEL) -> None:
    import mediapipe as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':<8}{'frames':>8}{'face %':>8}")
    for video in videos:
        video = Path(video)
        clip = extract_face(video, model_path)
        clip.meta = {"source": video.name, "source_sha256": _sha256(video),
                     "mediapipe": mp.__version__, "settings": SETTINGS,
                     "model": Path(model_path).name,
                     "detector_state": "fresh FaceLandmarker per clip"}
        save_face_clip(output_dir / f"{video.stem}_face.npz", clip)
        print(f"{video.stem:<8}{len(clip.valid):>8}{100 * clip.valid.mean():>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("videos", nargs="*", type=Path, help="videos (default: data/sign_videos)")
    parser.add_argument("--clips-from", type=Path,
                        help="take clip ids from the *_hands.npz files in this directory")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    if args.videos:
        videos = args.videos
    elif args.clips_from:
        stems = sorted(p.name[: -len("_hands.npz")] for p in args.clips_from.glob("*_hands.npz"))
        videos = [args.video_dir.expanduser() / f"{stem}.mp4" for stem in stems]
    else:
        videos = sorted(args.video_dir.glob("*.mp4"))
    missing = [v for v in videos if not Path(v).exists()]
    if missing or not videos:
        raise SystemExit(f"No videos, or missing: {missing[:5]}")
    extract_all(videos, args.output_dir, args.model)


if __name__ == "__main__":
    main()
