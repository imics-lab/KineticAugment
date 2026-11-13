"""Finding 10 per-clip proportions recomputed for each extraction config (median over frames where
both endpoints have vis>=0.5), to see how much extraction state/smoothing moves them."""
import sys; from pathlib import Path; import numpy as np
S=Path(sys.argv[1]); W=Path(sys.argv[2])
pairs={"SW":(11,12),"LU":(11,13),"LF":(13,15),"RU":(12,14),"RF":(14,16)}
clips=["01986","01987","01988","01991","01992"]
for cfg,base in [("stored",S/"data/test_output"),("F",W/"extract/F"),("U",W/"extract/U"),("I",W/"extract/I")]:
    rows=[]
    for c in clips:
        d=np.load(base/f"{c}_world.npz"); x=d["landmarks"].astype(float); v=d["visibility"]
        r=[]
        for k,(a,b) in pairs.items():
            ok=(v[:,a]>=.5)&(v[:,b]>=.5)&np.isfinite(x[:,a]).all(1)&np.isfinite(x[:,b]).all(1)
            r.append(np.median(np.linalg.norm(x[ok,a]-x[ok,b],axis=-1)))
        rows.append(r)
    R=np.array(rows); spread=(R.max(0)-R.min(0))/R.mean(0)*100
    print(f"{cfg:6s} "+" | ".join(f"{c}: "+" ".join(f"{v:.4f}" for v in R[i]) for i,c in enumerate(clips)))
    print(f"       spread% SW,LU,LF,RU,RF = {np.round(spread,1).tolist()}   01987 LF/LU={R[1,2]/R[1,1]:.3f}")
