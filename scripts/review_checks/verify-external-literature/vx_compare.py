import numpy as np, sys
from pathlib import Path
SNAP=Path(sys.argv[1]); EX=Path(sys.argv[2])
clips=["01986","01987","01988","01991","01992"]
ARM=[11,12,13,14,15,16]
def ld(cfg,c):
    p = SNAP/"data/test_output"/f"{c}_world.npz" if cfg=="stored" else EX/cfg/f"{c}_world.npz"
    z=np.load(p); return z["landmarks"].astype(np.float64), z["visibility"].astype(np.float64)
def perframe(a,b):
    d=np.linalg.norm(a[:,ARM]-b[:,ARM],axis=2)*1000
    return np.nanmax(d,axis=1)
print("== stored vs shared_s (bitwise) and determinism fresh_s vs fresh_s2 ==")
for c in clips:
    s,_=ld("stored",c); a,_=ld("shared_s",c); f1,_=ld("fresh_s",c); f2,_=ld("fresh_s2",c)
    print(f" {c}: T={len(s)} max|stored-shared_s|={np.nanmax(np.abs(s-a)):.3g}  max|fresh_s-fresh_s2|={np.nanmax(np.abs(f1-f2)):.3g}")
for A,B,label in [("shared_s","fresh_s","LEAK smoothed"),("shared_u","fresh_u","LEAK unsmoothed"),("rev_s","fresh_s","reverse-order shared vs fresh")]:
    print(f"== {label}: {A} vs {B}, per-frame max arm dev (mm) ==")
    for c in clips:
        a,_=ld(A,c); b,_=ld(B,c); pf=perframe(a,b); T=len(pf); h=pf[T//2:]
        below = np.where(pf>=5)[0]; settle = None if len(below)==0 else (below[-1]+1 if below[-1]+1<T else None)
        print(f" {c}: f0={pf[0]:.1f} median={np.nanmedian(pf):.1f} 2nd-half median={np.nanmedian(h):.1f} max={np.nanmax(pf):.1f} last_frame>=5mm={below[-1] if len(below) else None}/{T-1}")
