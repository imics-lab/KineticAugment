"""E3f: separate 'transfer method error' from 'MediaPipe world/image heads disagree'.
(a) Oracle linear transfer: per clip, least-squares 2x3 map from world segment motion to image
    segment motion fitted ON the motion pairs themselves (best any linear camera could do).
(b) Bone-length stability and proportions in isotropic image space (X=xW, Y=yH, Z=zW) vs world,
    i.e. could an augmentation be applied directly to image-space landmarks?"""
import numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent / "extract"
SEGS = [(13, 15), (14, 16), (11, 13), (12, 14)]
print("(a) oracle per-clip linear transfer, k=3, relative error")
allrel = []
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I, Pw, V = d["image"] * np.array([W, H, W]), d["world"].astype(np.float64), d["visibility"]
    X, Y = [], []
    k = 3
    for t in range(len(I) - k):
        for a, b in SEGS:
            if not ((V[t, [a, b]] > 0.5).all() and (V[t+k, [a, b]] > 0.5).all()): continue
            dw = (Pw[t+k, b] - Pw[t+k, a]) - (Pw[t, b] - Pw[t, a])
            di = (I[t+k, b, :2] - I[t+k, a, :2]) - (I[t, b, :2] - I[t, a, :2])
            sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
            if np.linalg.norm(di) / sw * 100 < 2: continue
            X.append(dw); Y.append(di)
    X, Y = np.array(X), np.array(Y)
    M, *_ = np.linalg.lstsq(X, Y, rcond=None)
    rel = np.linalg.norm(X @ M - Y, axis=1) / np.linalg.norm(Y, axis=1)
    allrel.append(rel)
    print(f"  {f.stem}: n={len(X)} relative error median {np.median(rel):.2f} p95 {np.percentile(rel,95):.2f}")
allrel = np.concatenate(allrel)
print(f"  pooled: median {np.median(allrel):.2f} p95 {np.percentile(allrel,95):.2f}")

print("\n(b) bone length CV over time and forearm/upper-arm ratio (frames with vis>0.5 on the bone)")
print(f"  {'clip':6s} {'space':10s} {'CV upperL':>9s} {'CV foreL':>9s} {'CV upperR':>9s} {'CV foreR':>9s} {'fore/upper L':>12s} {'R':>6s}")
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    V = d["visibility"]
    for name, P in (("world", d["world"]), ("image-iso", d["image"] * np.array([W, H, W])),
                    ("image-xy", d["image"][..., :2] * np.array([W, H]))):
        out, lens = [], {}
        for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]:
            m = (V[:, a] > 0.5) & (V[:, b] > 0.5)
            l = np.linalg.norm(P[m, b] - P[m, a], axis=1)
            lens[(a, b)] = np.median(l); out.append(np.std(l) / np.mean(l) * 100)
        print(f"  {f.stem:6s} {name:10s} " + " ".join(f"{x:9.1f}" for x in out) +
              f" {lens[(13,15)]/lens[(11,13)]:12.2f} {lens[(14,16)]/lens[(12,14)]:6.2f}")
