"""Autocorrelation-respecting null for s8: circularly shift the length series
against the pose regressors (preserves each series' autocorrelation)."""
import glob, os
import numpy as np
from common import SNAP
MP = {"L upper": (11, 13, 11, 13, 15), "L fore": (13, 15, 11, 13, 15),
      "R upper": (12, 14, 12, 14, 16), "R fore": (14, 16, 12, 14, 16)}
def r2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
out = []
for path in sorted(glob.glob(f"{SNAP}/data/test_output/*_world.npz")):
    clip = os.path.basename(path)[:5]; d = np.load(path); P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for name, (a, b, sh, el, wr) in MP.items():
        m = ok[:, sh] & ok[:, el] & ok[:, wr]
        v = P[m, b] - P[m, a]; L = np.linalg.norm(v, axis=1); n = len(L)
        u = P[m, el] - P[m, sh]; f = P[m, wr] - P[m, el]
        ang = np.degrees(np.arccos(np.clip((u*f).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(f,axis=1), -1, 1)))
        X = np.column_stack([np.ones(n), np.abs(v[:, 2]) / L, ang, v / L[:, None]])
        obs = r2(X, L)
        null = np.array([r2(X, np.roll(L, k)) for k in range(5, n - 5)])
        p = (1 + (null >= obs).sum()) / (1 + len(null))
        out.append((clip, name, obs, np.median(null), np.quantile(null, .95), p))
        print(f"{clip} {name:<8} n={n:3d} R2={obs:.2f}  shift-null median={np.median(null):.2f} 95%={np.quantile(null,.95):.2f}  p={p:.3f}")
ps = np.array([o[5] for o in out])
print(f"\n{(ps < 0.05).sum()}/{len(ps)} arm segments significant at p<0.05 under the circular-shift null; "
      f"median observed R2 {np.median([o[2] for o in out]):.2f} vs median null-95% {np.median([o[4] for o in out]):.2f}")
