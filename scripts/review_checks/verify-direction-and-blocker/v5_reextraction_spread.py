"""Side observation: cached world landmarks vs the tester's re-extraction (same videos, stated-identical
Holistic settings). Per-clip arm-landmark spread in %SW, checked against frame shifts of -2..2."""
import numpy as np
X = "scripts/review_checks/direction-and-blocker/extract/"
allv = []
for c in ["01986", "01987", "01988", "01991", "01992"]:
    a = np.load("data/test_output/%s_world.npz" % c); A = a["landmarks"].astype(float); VA = a["visibility"]
    B = np.load(X + c + ".npz")["world"]
    best = {}
    for s in range(-2, 3):
        lo, hi = max(0, s), min(len(A), len(B) + s)
        dA, dB = A[lo:hi], B[lo - s:hi - s]
        ok = VA[lo:hi, 11:17] >= 0.5
        sw = np.linalg.norm(dA[:, 11] - dA[:, 12], axis=-1)[:, None]
        e = np.linalg.norm(dA[:, 11:17] - dB[:, 11:17], axis=-1) / sw * 100
        best[s] = np.nanmedian(np.where(ok, e, np.nan))
        if s == 0:
            v = np.where(ok, e, np.nan).ravel(); v = v[np.isfinite(v)]; allv.append(v)
            frac_exact = np.mean(np.nanmax(np.where(ok, e, np.nan), axis=1) < 1e-3)
    print(c, "median arm diff %%SW by shift:", {k: round(float(v), 2) for k, v in best.items()}, "| frames bit-identical (shift 0): %.2f" % frac_exact)
allv = np.concatenate(allv)
print("pooled shift-0: median %.2f p95 %.2f %%SW (n=%d)" % (np.median(allv), np.percentile(allv, 95), len(allv)))
