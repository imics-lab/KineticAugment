"""Compare stored world landmarks vs re-extracted configs; quantify smoothing + cross-clip state leakage."""
import sys, glob
from pathlib import Path
import numpy as np
SNAP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
ARM = [11, 12, 13, 14, 15, 16]
clips = ["01986", "01987", "01988", "01991", "01992"]

def load(p):
    d = np.load(p); return d["landmarks"].astype(np.float64), d["visibility"].astype(np.float64), (d["presence"] if "presence" in d else None)

print("== 1. Stored npz vs re-extracted S (shared instance, default smoothing) ==")
for c in clips:
    a, va, _ = load(SNAP / f"data/test_output/{c}_world.npz")
    b, vb, pb = load(OUT / f"S/{c}_world.npz")
    if a.shape != b.shape:
        print(c, "shape mismatch", a.shape, b.shape); continue
    print(f"  {c}: T={len(a)}  max|dL|={np.nanmax(np.abs(a-b)):.3e} m  max|dVis|={np.nanmax(np.abs(va-vb)):.3e}  presence field range S: [{np.nanmin(pb):.3f},{np.nanmax(pb):.3f}]")

print("\n== 2. Cross-clip state leakage: S (shared) vs F (fresh per clip), arm landmarks, mm ==")
for c in clips:
    s, _, _ = load(OUT / f"S/{c}_world.npz"); f, _, _ = load(OUT / f"F/{c}_world.npz")
    d = np.linalg.norm(s[:, ARM] - f[:, ARM], axis=-1) * 1000  # (T,6)
    per_frame = np.nanmax(d, axis=1)
    nz = np.where(per_frame > 1e-3)[0]
    print(f"  {c}: frames differing(>1um)={len(nz)}/{len(s)}  first5 max-dev mm={np.round(per_frame[:5],2).tolist()}  overall max={np.nanmax(per_frame):.2f} mm  last differing frame={nz.max() if len(nz) else None}")

print("\n== 3. Smoothing effect: F (smoothed) vs U (unsmoothed), arm landmarks ==")
for c in clips:
    f, vf, _ = load(OUT / f"F/{c}_world.npz"); u, vu, _ = load(OUT / f"U/{c}_world.npz")
    d = np.linalg.norm(f[:, ARM] - u[:, ARM], axis=-1) * 1000
    print(f"  {c}: median |F-U| = {np.nanmedian(d):.2f} mm, p95 = {np.nanpercentile(d,95):.2f} mm, max = {np.nanmax(d):.2f} mm")

print("\n== 4. Raw frame-to-frame step on arms (median, mm) per config ==")
for cfg in ["F", "U", "I"]:
    vals = []
    for c in clips:
        x, _, _ = load(OUT / f"{cfg}/{c}_world.npz")
        st = np.linalg.norm(np.diff(x[:, ARM], axis=0), axis=-1) * 1000
        vals.append(np.nanmedian(st))
    print(f"  {cfg}: per-clip median step mm = {np.round(vals,2).tolist()}")

print("\n== 5. Static (I) vs tracked/unsmoothed (U): per-frame diff in arms (mm) ==")
for c in clips:
    i, _, _ = load(OUT / f"I/{c}_world.npz"); u, _, _ = load(OUT / f"U/{c}_world.npz")
    d = np.linalg.norm(i[:, ARM] - u[:, ARM], axis=-1) * 1000
    print(f"  {c}: detected I={np.isfinite(i).all(axis=(1,2)).sum()}/{len(i)} median |I-U| = {np.nanmedian(d):.2f} mm, p95={np.nanpercentile(d,95):.2f}")
