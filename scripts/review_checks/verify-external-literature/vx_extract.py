"""Independent re-extraction (verifier). Writes only into this dir.
Configs:
  shared_s   one Holistic for all clips, defaults (replicates scripts/extract_world_landmarks.py)
  fresh_s    new Holistic per clip, defaults
  fresh_s2   same as fresh_s, second run (determinism control)
  shared_u   one Holistic for all clips, smooth_landmarks=False
  fresh_u    new Holistic per clip, smooth_landmarks=False
  static     new Holistic per clip, static_image_mode=True
  rev_s      one Holistic, clips in REVERSE order, defaults (order-dependence control)
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2, numpy as np
import mediapipe as mp

SNAP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
VIDEOS = sorted((SNAP / "data/sign_videos").glob("*.mp4"))
H = mp.solutions.holistic.Holistic

def run_clip(v, h):
    cap = cv2.VideoCapture(str(v)); L=[]; V=[]; pres=[]
    while True:
        ok, fr = cap.read()
        if not ok: break
        r = h.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        w = r.pose_world_landmarks
        if w is None:
            L.append(np.full((33,3), np.nan)); V.append(np.zeros(33)); pres.append(False)
        else:
            L.append([[p.x,p.y,p.z] for p in w.landmark]); V.append([p.visibility for p in w.landmark])
            pres.append(any(p.HasField('presence') for p in w.landmark))
    cap.release()
    return np.array(L,np.float32), np.array(V,np.float32), any(pres)

def save(cfg, v, L, V):
    d = OUT/cfg; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d/f"{v.stem}_world.npz", landmarks=L, visibility=V)

meta = {}
for v in VIDEOS:
    cap = cv2.VideoCapture(str(v))
    meta[v.stem] = dict(fps=cap.get(cv2.CAP_PROP_FPS), w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
print("video meta", json.dumps(meta))

anypres = False
for cfg in sys.argv[3:]:
    if cfg in ("shared_s","shared_u","rev_s"):
        kw = dict(static_image_mode=False, model_complexity=1)
        if cfg=="shared_u": kw["smooth_landmarks"]=False
        h = H(**kw)
        order = VIDEOS[::-1] if cfg=="rev_s" else VIDEOS
        for v in order:
            L,V,p = run_clip(v,h); anypres |= p; save(cfg,v,L,V)
        h.close()
    else:
        for v in VIDEOS:
            if cfg in ("fresh_s","fresh_s2"): h = H(static_image_mode=False, model_complexity=1)
            elif cfg=="fresh_u": h = H(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
            elif cfg=="static": h = H(static_image_mode=True, model_complexity=1)
            L,V,p = run_clip(v,h); anypres |= p; h.close(); save(cfg,v,L,V)
    print("done", cfg, flush=True)
print("presence HasField on any landmark:", anypres)
