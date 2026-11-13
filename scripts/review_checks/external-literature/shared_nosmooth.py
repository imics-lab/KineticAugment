import os, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
import numpy as np, mediapipe as mp
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("rx", str(Path(__file__).parent/"reextract.py"))
SNAP=Path(sys.argv[1]); OUT=Path(sys.argv[2])
# minimal inline copy of extract() to avoid executing reextract's CLI
import cv2
def extract(video, h):
    cap=cv2.VideoCapture(str(video)); L=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        r=h.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)); w=r.pose_world_landmarks
        L.append(np.full((33,3),np.nan) if w is None else [[p.x,p.y,p.z] for p in w.landmark])
    cap.release(); return np.array(L,float)
vids=sorted((SNAP/"data/sign_videos").glob("*.mp4"))
h=mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
ARM=[11,12,13,14,15,16]
for v in vids:
    s=extract(v,h)
    u=np.load(OUT/f"U/{v.stem}_world.npz")["landmarks"].astype(float)
    d=np.nanmax(np.linalg.norm(s[:,ARM]-u[:,ARM],axis=-1),axis=1)*1000
    print(v.stem, "shared-vs-fresh (both unsmoothed): frame0", round(float(d[0]),1), "mm; median", round(float(np.nanmedian(d)),2), "mm; max", round(float(np.nanmax(d)),1))
h.close()
