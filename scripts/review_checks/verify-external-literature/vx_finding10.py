"""Recompute Finding 10 per-clip median lengths (vis>=0.5 gated, as the doc's table) per extraction config.
spread = std/mean (population) which is what reproduces the doc's 2.1%."""
import numpy as np, sys
from pathlib import Path
SNAP=Path(sys.argv[1]); EX=Path(sys.argv[2]); clips=["01986","01987","01988","01991","01992"]
B=[(11,12),(11,13),(13,15),(12,14),(14,16)]
def ld(cfg,c):
    p=SNAP/"data/test_output"/f"{c}_world.npz" if cfg=="stored" else EX/cfg/f"{c}_world.npz"
    z=np.load(p); L=z["landmarks"].astype(float); ok=np.isfinite(L).all(-1)&(z["visibility"]>=0.5); return L,ok
for cfg in ["stored","fresh_s","rev_s","fresh_u","static"]:
    T=[]
    for c in clips:
        L,ok=ld(cfg,c); row=[]
        for a,b in B:
            m=ok[:,a]&ok[:,b]; row.append(np.median(np.linalg.norm(L[m,a]-L[m,b],axis=1)))
        T.append(row)
    T=np.array(T); cv=T.std(0)/T.mean(0)*100; ratio=T[:,2]/T[:,1]
    print(f"[{cfg}] " + " | ".join(f"{c}:"+" ".join(f"{v:.4f}" for v in r) for c,r in zip(clips,T)))
    print(f"      CV% SW,LU,LF,RU,RF = {np.round(cv,1).tolist()}  fore/upper(L) per clip = {np.round(ratio,3).tolist()} CV {ratio.std()/ratio.mean()*100:.1f}%")
