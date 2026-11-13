#!/usr/bin/env python3
"""Re-extract MediaPipe Holistic world landmarks under several configs/perturbations.

Writes ONLY into this work dir (never touches the snapshot's data/test_output).
Mirrors scripts/extract_world_landmarks.py:43-63 exactly, except for the Holistic
kwargs and an optional per-frame image transform.

Run with cwd = snapshot root (so data/sign_videos resolves).
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2
import numpy as np
import mediapipe as mp

OUT = Path(__file__).resolve().parent / "extract"
VIDEO_DIR = Path("data/sign_videos")
N = 33


def ident(f):
    return f


def resize75(f):
    h, w = f.shape[:2]
    return cv2.resize(f, (int(round(w * 0.75)), int(round(h * 0.75))), interpolation=cv2.INTER_AREA)


def pad_shift(f):
    # put the frame on a canvas 15% larger, offset toward bottom-right (person moves in image)
    h, w = f.shape[:2]
    ph, pw = int(0.15 * h), int(0.15 * w)
    canvas = np.zeros((h + ph, w + pw, 3), dtype=f.dtype)
    canvas[ph:, pw:] = f
    return canvas


def brighten(f):
    return cv2.convertScaleAbs(f, alpha=1.0, beta=25)


VARIANTS = {
    # name: (holistic kwargs, frame transform)
    "default":        (dict(static_image_mode=False, model_complexity=1), ident),
    "default_rerun":  (dict(static_image_mode=False, model_complexity=1), ident),
    "nosmooth":       (dict(static_image_mode=False, model_complexity=1, smooth_landmarks=False), ident),
    "static":         (dict(static_image_mode=True, model_complexity=1), ident),
    "resize75":       (dict(static_image_mode=False, model_complexity=1), resize75),
    "padshift":       (dict(static_image_mode=False, model_complexity=1), pad_shift),
    "bright":         (dict(static_image_mode=False, model_complexity=1), brighten),
    "nosmooth_resize75": (dict(static_image_mode=False, model_complexity=1, smooth_landmarks=False), resize75),
    "complexity2":    (dict(static_image_mode=False, model_complexity=2), ident),
}


def extract(video_path, holistic, tf):
    cap = cv2.VideoCapture(str(video_path))
    L, V = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = tf(frame)
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        world = getattr(res, "pose_world_landmarks", None)
        if world is None:
            L.append(np.full((N, 3), np.nan)); V.append(np.zeros(N))
        else:
            L.append([[p.x, p.y, p.z] for p in world.landmark])
            V.append([p.visibility for p in world.landmark])
    cap.release()
    return np.array(L, dtype=np.float32), np.array(V, dtype=np.float32)


def main():
    names = sys.argv[1:] or list(VARIANTS)
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    for name in names:
        kw, tf = VARIANTS[name]
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for v in videos:
            # fresh graph per video (the repo script reuses one graph across videos;
            # 'default' below is checked against the repo npz to see if that matters)
            h = mp.solutions.holistic.Holistic(**kw)
            try:
                L, V = extract(v, h, tf)
            finally:
                h.close()
            np.savez_compressed(d / f"{v.stem}_world.npz", landmarks=L, visibility=V)
            det = np.isfinite(L).all(axis=(1, 2)).sum()
            print(f"{name:<18}{v.stem}  frames={len(L)} detected={det}", flush=True)

    # also a single-graph-across-videos run, exactly like the repo script
    if not sys.argv[1:]:
        d = OUT / "default_sharedgraph"
        d.mkdir(parents=True, exist_ok=True)
        h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
        try:
            for v in videos:
                L, V = extract(v, h, ident)
                np.savez_compressed(d / f"{v.stem}_world.npz", landmarks=L, visibility=V)
                print(f"default_sharedgraph {v.stem} frames={len(L)}", flush=True)
        finally:
            h.close()


if __name__ == "__main__":
    main()
