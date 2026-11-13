"""Visibility is low-pass filtered (alpha=0.1) when smoothing is on. How often does the
project's observed mask (vis>=0.5) disagree between smoothed-fresh (F) and unsmoothed-fresh (U)?"""
import sys; from pathlib import Path; import numpy as np
W=Path(sys.argv[1]); idx={"L wrist":15,"R wrist":16,"L elbow":13,"R elbow":14,"L hip":23}
for name,i in idx.items():
    dis=tot=0; fs=us=0
    for p in sorted((W/"extract"/"F").glob("*_world.npz")):
        f=np.load(p)["visibility"][:,i]; u=np.load(W/"extract"/"U"/p.name)["visibility"][:,i]
        a=f>=0.5; b=u>=0.5; dis+=(a!=b).sum(); tot+=len(a); fs+=a.sum(); us+=b.sum()
    print(f"{name:8s} observed frac smoothed={fs/tot:.3f} unsmoothed={us/tot:.3f} mask disagreement={dis}/{tot} ({dis/tot*100:.1f}%)")
