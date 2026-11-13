import sys, numpy as np
from pathlib import Path
SNAP = Path(sys.argv[1]); E = Path(sys.argv[2])
ARM = [11, 12, 13, 14, 15, 16]
def mm(a, b):
    d = np.linalg.norm(a[:, ARM] - b[:, ARM], axis=-1) * 1000  # (T, 6)
    return d
for stem in ["01986", "01987", "01988", "01991", "01992"]:
    stored = np.load(SNAP / f"data/test_output/{stem}_world.npz")["landmarks"].astype(float)
    sh = np.load(E / f"shared_{stem}.npz")["landmarks"].astype(float)
    fr = np.load(E / f"fresh_{stem}.npz")["landmarks"].astype(float)
    d_sh = mm(stored, sh); d_fr = mm(stored, fr)
    per_frame_max = np.nanmax(d_fr, axis=1)
    print(f"{stem}: T={len(stored)} shared-vs-stored max {np.nanmax(d_sh):.4f} mm | fresh-vs-stored: frame0 max {per_frame_max[0]:.2f} mm, "
          f"median per-frame max {np.nanmedian(per_frame_max):.2f} mm, median all {np.nanmedian(d_fr):.2f} mm, last10 median per-frame max {np.nanmedian(per_frame_max[-10:]):.2f} mm, overall max {np.nanmax(d_fr):.2f}")
f2 = np.load(E / "fresh2_01987.npz")["landmarks"].astype(float); f1 = np.load(E / "fresh_01987.npz")["landmarks"].astype(float)
print("determinism control fresh vs fresh2 01987 max diff mm:", np.nanmax(mm(f1, f2)))
