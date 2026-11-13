"""
extract_world_landmarks.py reuses ONE Holistic(static_image_mode=False) instance across all videos and
never calls holistic.reset(). Tracking/smoothing state from the end of clip N leaks into clip N+1.
Compare clip 01987 extracted (a) after 01986 with a shared instance, as the script does, vs (b) fresh.
Writes nothing; reads videos only.
"""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
spec = importlib.util.spec_from_file_location("ewl", "scripts/extract_world_landmarks.py")
ewl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ewl)
vids = sorted(Path("data/sign_videos").glob("*.mp4"))
for v in vids:
    cap = cv2.VideoCapture(str(v)); print(f"  {v.stem}: fps {cap.get(cv2.CAP_PROP_FPS):.2f}, frames {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}"); cap.release()

def run(seq):
    h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    try:
        return [ewl.extract(v, h) for v in seq]
    finally:
        h.close()

shared = run(vids)                                  # exactly what main() does
stored = {v.stem: np.load(f"data/test_output/{v.stem}_world.npz") for v in vids}
print("\n  shared-instance re-run vs stored npz (max |diff| over arm landmarks 11-16):")
for v, (L, _) in zip(vids, shared):
    S = stored[v.stem]["landmarks"]
    print(f"    {v.stem}: {np.nanmax(np.abs(L[:, 11:17] - S[:, 11:17]))*1000:.3f} mm")
SW = 0.33
print("\n  fresh instance per clip vs shared instance (as the script runs), arm landmarks 11-16:")
for i, v in enumerate(vids):
    fresh, _ = run([v])[0]
    L = shared[i][0]
    d = np.linalg.norm(fresh[:, 11:17] - L[:, 11:17], axis=-1)   # (T, 6)
    per_frame = np.nanmax(d, axis=1) * 1000
    first = per_frame[:5]
    print(f"    {v.stem}: first 5 frames max diff {np.round(first,2)} mm; frames with diff > 1 mm: "
          f"{int(np.sum(per_frame > 1))}/{len(per_frame)}; overall max {np.nanmax(per_frame):.2f} mm "
          f"({np.nanmax(per_frame)/1000/SW*100:.2f}% of ~0.33 m shoulder width)")
