"""Independent real-data checks: per-clip medians, L-R asymmetry, half-split
stability, and pose dependence with a deliberately MINIMAL regressor set
(1 regressor: |z component of bone direction|) and a circular-shift null.
Also an alternative split: thirds."""
import numpy as np
from vcommon import load_clips, MP_SEG

rng = np.random.default_rng(0)
clips = load_clips()
print("clip  seg         n   median(mm)  half1-half2(mm)   r(len,|dz|)  shift-null p   r(len,|dz|) 1-reg")
allp = []; asym = {}
for c, (P, V) in clips.items():
    ok = np.isfinite(P).all(-1) & (V >= 0.5)
    meds = {}
    for a, b, name in MP_SEG:
        m = ok[:, a] & ok[:, b]
        idx = np.where(m)[0]
        v = P[idx, b] - P[idx, a]; L = np.linalg.norm(v, axis=1)
        meds[name] = np.median(L)
        n = len(P)
        h1 = idx < n // 2
        dh = (np.median(L[h1]) - np.median(L[~h1])) * 1e3 if h1.sum() > 3 and (~h1).sum() > 3 else np.nan
        dz = np.abs(v[:, 2]) / L
        r = np.corrcoef(L, dz)[0, 1]
        null = np.array([abs(np.corrcoef(np.roll(L, k), dz)[0, 1]) for k in range(5, len(L) - 5)])
        p = (1 + (null >= abs(r)).sum()) / (1 + len(null))
        if name != "shoulder W": allp.append(p)
        print(f"{c} {name:<10} {len(L):4d} {meds[name]*1e3:9.1f} {dh:14.1f} {r:14.2f} {p:12.3f}")
    asym[c] = ((meds['L upper'] - meds['R upper']) * 1e3, (meds['L fore'] - meds['R fore']) * 1e3)
    fu = (meds['L fore'] + meds['R fore']) / (meds['L upper'] + meds['R upper'])
    print(f"   {c}: L-R upper {asym[c][0]:+.1f} mm, L-R fore {asym[c][1]:+.1f} mm; pooled fore/upper {fu:.3f}; "
          f"L fore/upper {meds['L fore']/meds['L upper']:.3f}  R {meds['R fore']/meds['R upper']:.3f}")
allp = np.array(allp)
print(f"\n1-regressor |dz| test: {np.sum(allp<0.05)}/{len(allp)} arm segments p<0.05 under circular-shift null")
