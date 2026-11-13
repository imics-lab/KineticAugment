"""Finding 10's per-clip bone-length table: stored (shared Holistic instance) vs fresh instance per clip."""
import sys, glob
import numpy as np
from pathlib import Path
B = [(11, 13), (13, 15), (12, 14), (14, 16)]
def table(root):
    rows = {}
    for p in sorted(glob.glob(f"{root}/*_world.npz")):
        d = np.load(p); L = d["landmarks"].astype(float); ok = np.isfinite(L).all(-1) & (d["visibility"] >= 0.5)
        both = ok[:, 11] & ok[:, 12]
        sw = np.median(np.linalg.norm(L[both, 11] - L[both, 12], axis=-1))
        r = [sw]
        for a, b in B:
            m = ok[:, a] & ok[:, b]
            r.append(np.median(np.linalg.norm(L[m, a] - L[m, b], axis=-1)))
        r.append(r[2] / r[1])  # L fore / L upper
        rows[Path(p).stem.split("_")[0]] = r
    return rows
a, b = table(sys.argv[1]), table(sys.argv[2])
print(f"  {'clip':<6}{'':>8}{'ShW':>8}{'Lup':>8}{'Lfore':>8}{'Rup':>8}{'Rfore':>8}{'Lf/Lu':>8}")
for k in a:
    for tag, r in (("stored", a[k]), ("fresh", b[k])):
        print(f"  {k:<6}{tag:>8}" + "".join(f"{x:8.4f}" for x in r))
for tag, t in (("stored", a), ("fresh", b)):
    lf = np.array([r[2] for r in t.values()])
    print(f"  {tag}: L forearm spread across clips (max-min)/mean = {100*(lf.max()-lf.min())/lf.mean():.1f}%")
