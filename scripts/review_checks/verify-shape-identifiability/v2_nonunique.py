import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from vcommon import *

np.set_printoptions(precision=3, suppress=True, linewidth=160)
R = np.sqrt(chi2.ppf(0.5, 10))
print("chi2(10) median radius", R)
Lt = lengths(np.zeros(10))

def vol(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6) * 1000
F = model(10).faces.astype(np.int64)

def stats(b):
    V = verts(b)
    with torch.no_grad():
        J = forward(b).joints[0, :55].numpy()
    leg = np.linalg.norm(J[4]-J[1]) + np.linalg.norm(J[7]-J[4])
    return dict(height_cm=(V[:,1].max()-V[:,1].min())*100, Lleg_cm=leg*100, vol_L=vol(V, F), hipW_cm=np.linalg.norm(J[1]-J[2])*100)

def solve(sign, key):
    obj = lambda b: -sign * stats(b)[key]
    cons = [dict(type="eq", fun=lambda b: (lengths(b) - Lt) * 1e3),
            dict(type="ineq", fun=lambda b: R**2 - b @ b)]
    best = None
    for seed in range(3):
        x0 = np.random.default_rng(seed).normal(0, 0.3, 10)
        r = minimize(obj, x0, constraints=cons, method="SLSQP", options=dict(maxiter=300, ftol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return best.x

out = {}
for key in ("height_cm", "Lleg_cm", "vol_L"):
    hi, lo = solve(+1, key), solve(-1, key)
    sh, sl = stats(hi), stats(lo)
    print(f"[{key}] hi |b|={np.linalg.norm(hi):.2f} lo |b|={np.linalg.norm(lo):.2f}  max|dL|={max(np.abs(lengths(hi)-Lt).max(), np.abs(lengths(lo)-Lt).max())*1e3:.2e} mm")
    print("   hi", {k: round(v, 2) for k, v in sh.items()})
    print("   lo", {k: round(v, 2) for k, v in sl.items()})
    out[key] = (hi, lo)
np.save("pairs.npy", np.array([out[k] for k in ("height_cm", "Lleg_cm", "vol_L")]))
