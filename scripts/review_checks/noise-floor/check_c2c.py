"""Per-axis split of the full-vs-heavy disagreement, and of benign-perturbation disagreement."""
from pathlib import Path
import numpy as np
EX = Path(__file__).resolve().parent / "extract"
ARM = [11, 12, 13, 14, 15, 16]
for vb in ["complexity2", "resize75", "padshift", "nosmooth"]:
    ax = [[], [], []]
    for c in ["01986", "01987", "01988", "01991", "01992"]:
        a = np.load(EX / "default" / f"{c}_world.npz"); b = np.load(EX / vb / f"{c}_world.npz")
        A, B = a["landmarks"].astype(float), b["landmarks"].astype(float)
        n = min(len(A), len(B)); A, B = A[:n], B[:n]
        ok = (a["visibility"][:n] >= .5) & (b["visibility"][:n] >= .5)
        sw = np.nanmedian(np.linalg.norm(A[:, 11] - A[:, 12], axis=-1))
        for m in ARM:
            for k in range(3):
                ax[k].extend(np.abs(A[ok[:, m], m, k] - B[ok[:, m], m, k]) / sw * 100)
    print(f"default vs {vb:<12} median |diff| % sw   x={np.median(ax[0]):.2f}  y={np.median(ax[1]):.2f}  z={np.median(ax[2]):.2f}")
