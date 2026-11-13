"""Is the full-vs-heavy disagreement a left/right swap? Compare swapped correspondences,
restricted to frames where both models report visibility >= 0.5. Also image-space check."""
from pathlib import Path
import numpy as np
EX = Path(__file__).resolve().parent / "extract"
SW = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
for c in ["01986", "01987", "01988", "01991", "01992"]:
    a = np.load(EX / "default" / f"{c}_world.npz"); b = np.load(EX / "complexity2" / f"{c}_world.npz")
    A, B = a["landmarks"].astype(float), b["landmarks"].astype(float)
    ok = (a["visibility"] >= .5) & (b["visibility"] >= .5)
    sw = np.nanmedian(np.linalg.norm(A[:, 11] - A[:, 12], axis=-1))
    same, swp = [], []
    for m, n in SW.items():
        k = ok[:, m] & ok[:, n]
        same.append(np.median(np.linalg.norm(A[k, m] - B[k, m], axis=-1)) / sw * 100)
        swp.append(np.median(np.linalg.norm(A[k, m] - B[k, n], axis=-1)) / sw * 100)
    # hand-to-hand distance sanity: wrist separation in each model
    k = ok[:, 15] & ok[:, 16]
    wa = np.median(np.linalg.norm(A[k, 15] - A[k, 16], axis=-1)); wb = np.median(np.linalg.norm(B[k, 15] - B[k, 16], axis=-1))
    print(f"{c}: same-side median % sw " + " ".join(f"{v:5.1f}" for v in same) +
          " | swapped " + " ".join(f"{v:5.1f}" for v in swp) + f" | wrist sep c1={wa:.3f} c2={wb:.3f}")
