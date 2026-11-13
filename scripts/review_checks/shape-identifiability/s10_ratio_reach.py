"""Doc line ~724 claims the 13% fore/upper discrepancy (obs 1.066 vs SMPL-X 0.946) is
'precisely what betas exists to absorb'. What fore/upper ratios can 10 betas reach
inside the prior? And what do the per-clip observed ratios need?"""
import numpy as np
from scipy.optimize import minimize
from common import NB, lengths
def ratio(b, side):
    L = lengths(b)
    return {"L": L[1] / L[0], "R": L[3] / L[2], "mean": (L[1] + L[3]) / (L[0] + L[2])}[side]
print(f"template ratio: L {ratio(np.zeros(NB),'L'):.3f}  R {ratio(np.zeros(NB),'R'):.3f}  mean {ratio(np.zeros(NB),'mean'):.3f}")
for r in (1.0, 3.06, 4.28, 10.0):
    out = []
    for side in ("L", "R", "mean"):
        ext = []
        for sgn in (+1, -1):
            res = minimize(lambda b: -sgn * ratio(b, side), np.zeros(NB) + 1e-3, method="SLSQP",
                           constraints=[{"type": "ineq", "fun": lambda b: r**2 - b @ b}])
            ext.append(ratio(res.x, side))
        out.append(f"{side}: {min(ext):.3f}-{max(ext):.3f}")
    print(f"|beta|<={r:5.2f}: " + "   ".join(out))
print("observed per-clip fore/upper (doc Finding 10, sides pooled): 1.068, 0.837, 1.072, 1.046, 1.103; pooled 1.066")
