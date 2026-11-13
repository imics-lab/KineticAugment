"""Sanity check the model_complexity 1 vs 2 disagreement and per-clip structure."""
from pathlib import Path
import numpy as np
EX = Path(__file__).resolve().parent / "extract"
ARM = [11, 12, 13, 14, 15, 16]
for c in ["01986", "01987", "01988", "01991", "01992"]:
    a = np.load(EX / "default" / f"{c}_world.npz"); b = np.load(EX / "complexity2" / f"{c}_world.npz")
    A, B = a["landmarks"].astype(float), b["landmarks"].astype(float)
    swa = np.nanmedian(np.linalg.norm(A[:, 11] - A[:, 12], axis=-1))
    swb = np.nanmedian(np.linalg.norm(B[:, 11] - B[:, 12], axis=-1))
    ua = np.nanmedian(np.linalg.norm(A[:, 13] - A[:, 11], axis=-1)); ub = np.nanmedian(np.linalg.norm(B[:, 13] - B[:, 11], axis=-1))
    d = np.linalg.norm(A[:, ARM] - B[:, ARM], axis=-1) / swa * 100
    va, vb = a["visibility"][:, ARM], b["visibility"][:, ARM]
    print(f"{c}: sw c1={swa:.3f} c2={swb:.3f}  Lupperarm c1={ua:.3f} c2={ub:.3f}  "
          f"per-landmark median diff % sw: " + " ".join(f"{np.nanmedian(d[:, k]):.1f}" for k in range(6))
          + f"  vis med c1={np.median(va):.2f} c2={np.median(vb):.2f}")
