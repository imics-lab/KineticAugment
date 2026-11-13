"""A/B ratios per band: observed (repo, raw) vs iid+real-motion simulation (pooled medians, same as script)."""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import est_A, est_B, pooled, SCRIPT_SET
from sim import truths, oe_track
rng = np.random.default_rng(1)
bands = {"z": [2], "xy": [0, 1], "xyz": [0, 1, 2]}
for src in ("repo", "fresh_raw"):
    out = []
    for b, d in bands.items():
        a, bb, _, _ = pooled(src, SCRIPT_SET, d); out.append(f"{b}: A={a:.4f} B={bb:.4f} A/B={a/bb:.2f}")
    print(f"observed {src:<10}", " | ".join(out))
print("iid stationary theory A/B: z 0.18, xy 0.46, xyz 0.62")
for filt in (False, True):
    for s_mm in (0.5, 1.0, 2.0):
        s = s_mm / 1000; res = {b: ([], []) for b in bands}
        for _ in range(3):
            for c, j, tr, o in truths():
                p = tr + rng.normal(0, s, tr.shape)
                if filt: p = oe_track(p, o)
                for b, d in bands.items():
                    res[b][0].append(est_A(p, o, d)); res[b][1].append(est_B(p, o, d))
        print(f"sim real-motion iid filt={filt!s:<5} s={s_mm} mm:",
              " | ".join(f"{b}: A/B={np.nanmedian(res[b][0])/np.nanmedian(res[b][1]):.2f}" for b in bands))
