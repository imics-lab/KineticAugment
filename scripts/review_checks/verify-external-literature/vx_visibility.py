"""Does MediaPipe visibility track positional uncertainty?  Two error proxies:
 (a) |second difference| (jitter + acceleration), per config
 (b) |static-detection - tracked-unsmoothed| per-frame disagreement between two per-frame estimators (fresh_u vs static)
Also: mask disagreement smoothed vs unsmoothed at vis>=0.5."""
import numpy as np, sys
from pathlib import Path
from scipy.stats import spearmanr
EX = Path(sys.argv[1]); clips=["01986","01987","01988","01991","01992"]
G = {"shoulders":[11,12],"elbows":[13,14],"wrists":[15,16],"hips":[23,24]}
def ld(cfg,c): z=np.load(EX/cfg/f"{c}_world.npz"); return z["landmarks"].astype(float), z["visibility"].astype(float)
for cfg in ["shared_s","fresh_u","static"]:
    out=[]
    for g,idx in G.items():
        vs=[];d2=[]
        for c in clips:
            L,V=ld(cfg,c)
            for j in idx:
                D=np.linalg.norm(L[:-2,j]-2*L[1:-1,j]+L[2:,j],axis=1)*1000
                ok=np.isfinite(D); vs.append(V[1:-1,j][ok]); d2.append(D[ok])
        vs=np.concatenate(vs); d2=np.concatenate(d2)
        rho=spearmanr(vs,d2).statistic if np.ptp(vs)>0 else np.nan
        lo=d2[vs<0.5]; hi=d2[vs>=0.5]
        out.append(f"{g}: rho={rho:+.2f} med|D2| vis<.5={np.median(lo) if len(lo) else np.nan:.1f}mm(n={len(lo)}) vis>=.5={np.median(hi):.1f}mm(n={len(hi)})")
    print(f"[{cfg}] |D2|:  " + " | ".join(out))
# (b) disagreement between per-frame estimators, centred (both hip-centred already)
out=[]
for g,idx in G.items():
    vs=[];dd=[]
    for c in clips:
        U,VU=ld("fresh_u",c); I,VI=ld("static",c)
        for j in idx:
            d=np.linalg.norm(U[:,j]-I[:,j],axis=1)*1000; ok=np.isfinite(d)
            vs.append(np.minimum(VU[:,j],VI[:,j])[ok]); dd.append(d[ok])
    vs=np.concatenate(vs); dd=np.concatenate(dd)
    rho=spearmanr(vs,dd).statistic if np.ptp(vs)>0 else np.nan
    q=np.quantile(vs,[0.25,0.75]); 
    out.append(f"{g}: rho={rho:+.2f} med|U-I| lowest-quartile-vis={np.median(dd[vs<=q[0]]):.1f}mm top-quartile={np.median(dd[vs>=q[1]]):.1f}mm")
print("[fresh_u vs static] " + " | ".join(out))
for j,name in [(15,"L wrist"),(16,"R wrist"),(13,"L elbow")]:
    dis=0;n=0
    for c in clips:
        _,VS=ld("fresh_s",c); _,VU=ld("fresh_u",c)
        a=VS[:,j]>=0.5; b=VU[:,j]>=0.5; dis+=(a!=b).sum(); n+=len(a)
    print(f"mask flip fresh_s vs fresh_u {name}: {dis}/{n} = {100*dis/n:.1f}%")
