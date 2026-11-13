"""Re-extract MediaPipe Holistic world landmarks under several configurations.

Reads videos from the snapshot (read-only) and writes .npz into this work dir only.
Configs:
  S  one shared Holistic across all clips, smooth_landmarks default (True)  -> replicates extract_world_landmarks.py
  F  fresh Holistic per clip, smooth_landmarks=True
  U  fresh Holistic per clip, smooth_landmarks=False (tracking still on)
  I  static_image_mode=True (no tracking, no smoothing)
"""
import os, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2, numpy as np
import mediapipe as mp

SNAP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
VIDEOS = sorted((SNAP / "data/sign_videos").glob("*.mp4"))

def extract(video, holistic):
    cap = cv2.VideoCapture(str(video))
    L, V, P = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w = r.pose_world_landmarks
        if w is None:
            L.append(np.full((33, 3), np.nan)); V.append(np.zeros(33)); P.append(np.zeros(33))
        else:
            L.append([[p.x, p.y, p.z] for p in w.landmark])
            V.append([p.visibility for p in w.landmark])
            P.append([p.presence for p in w.landmark])
    cap.release()
    return np.array(L, np.float32), np.array(V, np.float32), np.array(P, np.float32)

def run(cfg):
    d = OUT / cfg; d.mkdir(parents=True, exist_ok=True)
    H = mp.solutions.holistic.Holistic
    if cfg == "S":
        h = H(static_image_mode=False, model_complexity=1)
        for v in VIDEOS:
            l, vis, pres = extract(v, h)
            np.savez_compressed(d / f"{v.stem}_world.npz", landmarks=l, visibility=vis, presence=pres)
        h.close()
        return
    for v in VIDEOS:
        if cfg == "F":
            h = H(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        elif cfg == "U":
            h = H(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
        elif cfg == "I":
            h = H(static_image_mode=True, model_complexity=1)
        l, vis, pres = extract(v, h)
        h.close()
        np.savez_compressed(d / f"{v.stem}_world.npz", landmarks=l, visibility=vis, presence=pres)

for cfg in sys.argv[3:]:
    run(cfg)
    print("done", cfg, flush=True)
