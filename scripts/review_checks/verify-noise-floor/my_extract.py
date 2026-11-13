#!/usr/bin/env python3
"""Independent re-extraction (verifier). Writes only into this dir's ext/ folder.

Configs:
  shared_default : ONE Holistic(static_image_mode=False, model_complexity=1) across all clips,
                   clips in sorted order  (should equal repo npz if repo script is faithful)
  fresh_default  : new Holistic per clip, same kwargs
  fresh_raw      : new Holistic per clip, smooth_landmarks=False
  shared_raw     : ONE graph across clips, smooth_landmarks=False
Also records per-frame pose_landmarks (image space) for fresh_default to check ROI hypotheses.
Run with cwd = snapshot root.
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2
import numpy as np
import mediapipe as mp

OUT = Path(__file__).resolve().parent / "ext"
VIDEOS = sorted(Path("data/sign_videos").glob("*.mp4"))


def run_clip(h, v):
    cap = cv2.VideoCapture(str(v))
    L, V = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = h.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        w = r.pose_world_landmarks
        if w is None:
            L.append(np.full((33, 3), np.nan)); V.append(np.zeros(33))
        else:
            L.append([[p.x, p.y, p.z] for p in w.landmark]); V.append([p.visibility for p in w.landmark])
    cap.release()
    return np.array(L, np.float32), np.array(V, np.float32)


def save(name, stem, L, V):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{stem}_world.npz", landmarks=L, visibility=V)


def main():
    for name, kw in [("shared_default", dict(static_image_mode=False, model_complexity=1)),
                     ("shared_raw", dict(static_image_mode=False, model_complexity=1, smooth_landmarks=False))]:
        h = mp.solutions.holistic.Holistic(**kw)
        for v in VIDEOS:
            L, V = run_clip(h, v); save(name, v.stem, L, V); print(name, v.stem, len(L), flush=True)
        h.close()
    for name, kw in [("fresh_default", dict(static_image_mode=False, model_complexity=1)),
                     ("fresh_raw", dict(static_image_mode=False, model_complexity=1, smooth_landmarks=False))]:
        for v in VIDEOS:
            h = mp.solutions.holistic.Holistic(**kw)
            L, V = run_clip(h, v); h.close(); save(name, v.stem, L, V); print(name, v.stem, len(L), flush=True)


if __name__ == "__main__":
    main()
