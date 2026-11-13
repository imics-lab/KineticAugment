import numpy as np, sys
from pathlib import Path
OUT = Path(sys.argv[1]); ARM=[11,12,13,14,15,16]
for c in ["01987","01988","01991","01992"]:
    s=np.load(OUT/f"S/{c}_world.npz")["landmarks"].astype(float); f=np.load(OUT/f"F/{c}_world.npz")["landmarks"].astype(float)
    d=np.nanmax(np.linalg.norm(s[:,ARM]-f[:,ARM],axis=-1),axis=1)*1000
    T=len(d); q=[0,5,10,20,40,T-1]
    first_below = next((i for i in range(T) if np.all(d[i:]<5)), None)
    print(c, "max-arm-dev mm at frames", {i: round(float(d[i]),1) for i in q if i<T}, "| median over 2nd half:", round(float(np.nanmedian(d[T//2:])),2), "| frame after which always <5mm:", first_below)
