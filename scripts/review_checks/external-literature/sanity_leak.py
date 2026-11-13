"""Independent sanity check of the cross-clip leakage claim (re-run 2026-10-05).
(1) Shared smoothed instance over 01986 -> 01987 must reproduce stored 01987 exactly.
(2) Fresh smoothed instance on 01987 alone: deviation from stored.
(3) Presence field + visibility smoothing evidence."""
import os, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2, numpy as np, mediapipe as mp
SNAP = Path(sys.argv[1]); ARM=[11,12,13,14,15,16]
def run(video, h):
    cap=cv2.VideoCapture(str(SNAP/f"data/sign_videos/{video}.mp4")); L=[]; V=[]; P=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        w=h.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)).pose_world_landmarks
        if w is None: L.append(np.full((33,3),np.nan)); V.append(np.zeros(33)); P.append(np.zeros(33))
        else:
            L.append([[p.x,p.y,p.z] for p in w.landmark]); V.append([p.visibility for p in w.landmark]); P.append([p.HasField('presence') for p in w.landmark])
    cap.release(); return np.array(L,np.float32), np.array(V,np.float32), np.array(P)
H=mp.solutions.holistic.Holistic
stored={c: np.load(SNAP/f"data/test_output/{c}_world.npz") for c in ["01986","01987"]}
h=H(static_image_mode=False, model_complexity=1)
a,_,pa=run("01986",h); b,vb,_=run("01987",h); h.close()
print("shared 01986 vs stored max|d| m:", float(np.nanmax(np.abs(a-stored['01986']['landmarks']))))
print("shared 01987 (after 01986) vs stored max|d| m:", float(np.nanmax(np.abs(b-stored['01987']['landmarks']))))
print("presence field set on any world landmark:", bool(pa.any()))
h=H(static_image_mode=False, model_complexity=1)
f,vf,_=run("01987",h); h.close()
d=np.nanmax(np.linalg.norm(f[:,ARM]-stored['01987']['landmarks'][:,ARM],axis=-1),axis=1)*1000
print("fresh 01987 vs stored, per-frame max arm dev mm: frame0 %.1f, median %.1f, max %.1f" % (d[0], np.nanmedian(d), np.nanmax(d)))
# arm bone lengths: stored vs fresh
def bl(x):
    pairs=[(11,13),(13,15),(12,14),(14,16)]
    return [float(np.nanmedian(np.linalg.norm(x[:,i]-x[:,j],axis=-1))) for i,j in pairs]
print("01987 median bone lengths stored (LU,LF,RU,RF):", np.round(bl(stored['01987']['landmarks']),4).tolist())
print("01987 median bone lengths fresh  (LU,LF,RU,RF):", np.round(bl(f),4).tolist())
print("01987 visibility L/R wrist mean stored vs fresh:", np.round([stored['01987']['visibility'][:,15].mean(), stored['01987']['visibility'][:,16].mean(), vf[:,15].mean(), vf[:,16].mean()],3).tolist())
