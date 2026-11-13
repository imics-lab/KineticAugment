"""E3a: one-pass extraction of image-space pose + hands and world-space pose from the SAME
Holistic call, so image/world correspondence is exact. Mirrors scripts/extract_world_landmarks.py
(static_image_mode=False, model_complexity=1). Writes only into this work dir."""
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); os.environ.setdefault("GLOG_minloglevel", "3")
import cv2, numpy as np, mediapipe as mp
OUT = Path(__file__).resolve().parent / "extract"; OUT.mkdir(exist_ok=True)
for video in sorted(Path("data/sign_videos").glob("*.mp4")):
    cap = cv2.VideoCapture(str(video)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hol = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    img, wld, vis, lh, rh = [], [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok: break
        r = hol.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        nan33 = np.full((33, 3), np.nan); nan21 = np.full((21, 3), np.nan)
        img.append([[p.x, p.y, p.z] for p in r.pose_landmarks.landmark] if r.pose_landmarks else nan33)
        wld.append([[p.x, p.y, p.z] for p in r.pose_world_landmarks.landmark] if r.pose_world_landmarks else nan33)
        vis.append([p.visibility for p in r.pose_world_landmarks.landmark] if r.pose_world_landmarks else np.zeros(33))
        lh.append([[p.x, p.y, p.z] for p in r.left_hand_landmarks.landmark] if r.left_hand_landmarks else nan21)
        rh.append([[p.x, p.y, p.z] for p in r.right_hand_landmarks.landmark] if r.right_hand_landmarks else nan21)
    cap.release(); hol.close()
    np.savez_compressed(OUT / f"{video.stem}.npz", image=np.array(img), world=np.array(wld), visibility=np.array(vis),
                        left_hand=np.array(lh), right_hand=np.array(rh), width=W, height=H)
    print(video.stem, len(img), f"{W}x{H}", "pose frames", int(np.isfinite(np.array(wld)).all(axis=(1,2)).sum()),
          "LH", int(np.isfinite(np.array(lh)).all(axis=(1,2)).sum()), "RH", int(np.isfinite(np.array(rh)).all(axis=(1,2)).sum()))
