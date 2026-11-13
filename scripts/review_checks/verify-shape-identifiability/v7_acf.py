import numpy as np
from vcommon import load_clips, MP_SEG
acfs = []
for c, (P, V) in load_clips().items():
    ok = np.isfinite(P).all(-1) & (V >= 0.5)
    for a, b, name in MP_SEG:
        idx = np.where(ok[:, a] & ok[:, b])[0]
        L = np.linalg.norm(P[idx, b] - P[idx, a], axis=1); x = L - L.mean()
        r1 = (x[:-1] @ x[1:]) / (x @ x)
        neff = len(L) * (1 - r1) / (1 + r1)
        acfs.append(r1); print(f"{c} {name:<10} n={len(L):3d} lag1 {r1:.2f}  AR(1) n_eff {neff:5.1f}  gaps {np.sum(np.diff(idx)>1)}")
print("lag-1 range", min(acfs), max(acfs))
