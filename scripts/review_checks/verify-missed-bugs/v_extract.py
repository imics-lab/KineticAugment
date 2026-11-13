"""Independent check of N2: does extract_world_landmarks.py output depend on clip order?
Runs MediaPipe Holistic (same settings as the script) in three regimes, writes ONLY to this dir:
  shared  : one instance across clips in sorted order (what the script does)
  fresh   : a new instance per clip
  fresh2  : a second fresh run of 01987 (determinism control)
"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("GLOG_minloglevel", "3")
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp
SNAP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
vids = sorted((SNAP / "data/sign_videos").glob("*.mp4"))

def run(video, holistic):
    cap = cv2.VideoCapture(str(video)); L = []; V = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        r = holistic.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        w = r.pose_world_landmarks
        if w is None:
            L.append(np.full((33, 3), np.nan)); V.append(np.zeros(33))
        else:
            L.append([[p.x, p.y, p.z] for p in w.landmark]); V.append([p.visibility for p in w.landmark])
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return np.array(L, np.float32), np.array(V, np.float32), fps

mk = lambda: mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
h = mk()
for v in vids:
    L, V, fps = run(v, h); np.savez(OUT / f"shared_{v.stem}.npz", landmarks=L, visibility=V); print("shared", v.stem, L.shape, "fps", fps, flush=True)
h.close()
for v in vids:
    h = mk(); L, V, _ = run(v, h); h.close(); np.savez(OUT / f"fresh_{v.stem}.npz", landmarks=L, visibility=V); print("fresh", v.stem, flush=True)
h = mk(); L, V, _ = run(vids[1], h); h.close(); np.savez(OUT / f"fresh2_{vids[1].stem}.npz", landmarks=L, visibility=V); print("fresh2", vids[1].stem)
