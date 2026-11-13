import sys; sys.path.insert(0, "scripts/review_checks/verify-real-data-numbers")
import numpy as np
from scipy.optimize import minimize
import v_cf as C
# is 2.29 (sim) / 2.52 (rigid) a floor for per-clip constant shape + per-frame root?
for metric in ["sim", "rigid"]:
    allE = []
    for c in C.clips:
        for fr, ok in c["frames"]:
            R = C.opt_root(fr, ok, c["clen"], c["W"], metric, c["W"])
            allE += list(C.frame_err(fr, ok, R, c["clen"], c["W"], metric, c["W"]))
    print(f"{metric}: 3-dof per-frame root + per-clip median arm lengths + per-clip width: median {100*np.median(allE):.3f}%")
# optimise per-clip constant lengths+width (analytic root, rigid) to minimise clip SSE: does median-length beat it?
for metric in ["rigid"]:
    allE = []
    for c in C.clips:
        keys = C.CHAIN
        def unpack(x): return {k: x[i] for i, k in enumerate(keys)}, x[4]
        def fun(x):
            lens, w = unpack(x)
            return sum((C.frame_err(fr, ok, C.analytic_root(fr, ok), lens, w, metric, c["W"]) ** 2).sum() for fr, ok in c["frames"])
        x0 = np.array([c["clen"][k] for k in keys] + [c["W"]])
        r = minimize(fun, x0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-6, fatol=1e-10))
        lens, w = unpack(r.x)
        allE += [e for fr, ok in c["frames"] for e in C.frame_err(fr, ok, C.analytic_root(fr, ok), lens, w, metric, c["W"])]
    print(f"{metric}: analytic root + per-clip SSE-optimal constant lengths/width: median {100*np.median(allE):.3f}%")
