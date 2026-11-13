"""E3c: why does weak perspective fit worse than affine? Inspect affine singular values and
per-landmark residuals; test whether world and image differ by a reflection or anisotropy."""
import numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent / "extract"
UPPER = list(range(0, 23))
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I = d["image"] * np.array([W, H, W]); Pw = d["world"].astype(np.float64); V = d["visibility"]
    svs, dets, aniso = [], [], []
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in UPPER if ok[i]]
        if len(idx) < 6: continue
        A = np.c_[Pw[t, idx], np.ones(len(idx))]
        M, *_ = np.linalg.lstsq(A, I[t, idx, :2], rcond=None)
        L = M[:3].T
        S = np.linalg.svd(L, compute_uv=False)
        svs.append(S); 
        # 3D affine to X,Y,Z too
        M3, *_ = np.linalg.lstsq(A, I[t, idx], rcond=None)
        dets.append(np.sign(np.linalg.det(M3[:3])))
        aniso.append(np.linalg.norm(L[0]) / np.linalg.norm(L[1]))
    svs = np.array(svs)
    print(f"{f.stem} {W}x{H}: affine 2x3 singular values median {np.median(svs,0).round(1)} ratio {np.median(svs[:,0]/svs[:,1]):.2f}; "
          f"row-norm ratio x/y {np.median(aniso):.2f}; 3D affine det sign +:{(np.array(dets)>0).mean():.2f}")
# One frame, show world vs image of shoulders/elbows/wrists
d = np.load(D / "01986.npz"); W, H = int(d["width"]), int(d["height"])
t = 10
print("01986 f10 world (m):"); print(np.round(d["world"][t, 11:17], 3))
print("01986 f10 image iso px:"); print(np.round((d["image"][t] * np.array([W, H, W]))[11:17], 1))
