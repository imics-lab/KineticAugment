"""Is MediaPipe `visibility` usable as a 3-D positional uncertainty? Test on the unsmoothed (U) and
static (I) re-extractions: relationship between visibility and |second difference| (a jitter proxy),
per landmark group. Spearman rank correlation + median jitter by visibility bin."""
import sys, glob
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
W = Path(sys.argv[1])
groups = {"shoulders": [11, 12], "elbows": [13, 14], "wrists": [15, 16], "hips": [23, 24], "knees": [25, 26]}
for cfg in ["U", "I", "S"]:
    print(f"== config {cfg} ==")
    for g, idx in groups.items():
        V, J = [], []
        for p in sorted((W / "extract" / cfg).glob("*_world.npz")):
            d = np.load(p); x = d["landmarks"].astype(float); v = d["visibility"].astype(float)
            sd = np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=-1)   # (T-2, 33)
            for i in idx:
                ok = np.isfinite(sd[:, i])
                V.append(v[1:-1, i][ok]); J.append(sd[:, i][ok])
        V = np.concatenate(V); J = np.concatenate(J) * 1000
        rho = spearmanr(V, J).correlation if np.ptp(V) > 0 else float("nan")
        bins = [(0, .5), (.5, .9), (.9, .99), (.99, 1.01)]
        s = " ".join(f"[{a:.2f},{b:.2f}):n={((V>=a)&(V<b)).sum()},med={np.median(J[(V>=a)&(V<b)]) if ((V>=a)&(V<b)).any() else float('nan'):.1f}mm" for a, b in bins)
        print(f"  {g:9s} vis median={np.median(V):.3f} spearman(vis,|D2|)={rho:+.2f}  {s}")
