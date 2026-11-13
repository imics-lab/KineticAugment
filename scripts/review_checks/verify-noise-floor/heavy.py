"""Full vs heavy model: floors, per-clip disagreement, per-axis, and L/R-swap check."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import pooled, load, sw, SCRIPT_SET, ARM, CLIPS
for src in ("T:default", "T:complexity2", "repo"):
    a, b, na, nb = pooled(src, SCRIPT_SET, [0, 1, 2]); print(f"floor {src:<14} A_xyz={a:.3f}% B_xyz={b:.3f}% budget={max(a,b):.3f}%")
SWAP = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
allax = []
for c in CLIPS:
    f, okf = load("T:default", c); h, okh = load("T:complexity2", c); s = sw(f, okf)
    n = min(len(f), len(h)); d, dsw, ax = [], [], []
    for t in range(n):
        for m in ARM:
            if okf[t, m] and okh[t, m]:
                d.append(np.linalg.norm(f[t, m] - h[t, m]) / s); ax.append(np.abs(f[t, m] - h[t, m]) / s)
                if okh[t, SWAP[m]]:
                    dsw.append(np.linalg.norm(f[t, m] - h[t, SWAP[m]]) / s)
    allax += ax
    print(f"  {c}: full-vs-heavy median={100*np.median(d):6.2f}%  swapped-L/R median={100*np.median(dsw):6.2f}%  n={len(d)}")
ax = np.array(allax) * 100
print("  per-axis medians x/y/z: " + " / ".join(f"{np.median(ax[:, k]):.2f}%" for k in range(3)))
