"""ACF of second difference under iid noise + real arm motion (no temporal correlation in the noise).
Tests whether raw lag-1 = -0.40 is evidence of correlated noise or just motion curvature in D."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as vf
from sim import truths, oe_track
rng = np.random.default_rng(2)
for s_mm in (0.5, 1.0, 1.5, 2.0, 3.0):
    for filt in (False, True):
        ser = []
        for _ in range(3):
            for c, j, tr, o in truths():
                p = tr + rng.normal(0, s_mm / 1000, tr.shape)
                if filt: p = oe_track(p, o)
                for a, b in vf.runs(o, 12):
                    for k in range(3):
                        x = p[a:b, k]; ser.append(x[:-2] - 2 * x[1:-1] + x[2:])
        r = vf.acf(ser)
        print(f"iid+motion s={s_mm} mm filt={filt!s:<5} D-ACF lag1={r[1]:+.3f} lag2={r[2]:+.3f}")
