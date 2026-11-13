"""Check that our default re-extraction reproduces the repo npz, and how variants differ."""
import glob
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent / "extract"
REPO = Path("data/test_output")
ARM = [11, 12, 13, 14, 15, 16]

def load(p):
    d = np.load(p); return d["landmarks"].astype(np.float64), d["visibility"].astype(np.float64)

for var in ["default", "default_rerun", "default_sharedgraph", "nosmooth", "static", "complexity2",
            "resize75", "padshift", "bright", "nosmooth_resize75"]:
    out = []
    for rp in sorted(REPO.glob("*_world.npz")):
        a, _ = load(rp); b, _ = load(W / var / rp.name)
        n = min(len(a), len(b))
        ok = np.isfinite(a[:n]).all(axis=(1, 2)) & np.isfinite(b[:n]).all(axis=(1, 2))
        sw = np.median(np.linalg.norm(a[ok, 11] - a[ok, 12], axis=-1))
        d = np.linalg.norm(a[:n][ok][:, ARM] - b[:n][ok][:, ARM], axis=-1) / sw
        out.append(d.ravel())
    d = np.concatenate(out)
    print(f"{var:<20} vs repo npz (6 arm lmks): max={d.max()*100:.4f}%  median={np.median(d)*100:.4f}%  "
          f"p95={np.percentile(d,95)*100:.4f}%")
