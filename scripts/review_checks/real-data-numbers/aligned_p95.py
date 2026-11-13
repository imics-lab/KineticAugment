"""Aligned per-joint p95 (doc's older Step 5 table) + IK-frame direction error at full precision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import *  # noqa
sk = SMPLXSkeleton(model_path="models")
al = {m: [] for m in MP}; ik_dir = []
for name, pose, oks, width in load_clips():
    for frame, ok in zip(pose, oks):
        present = [m for m in MP if ok[m]]
        if len(present) < 3: continue
        pts, _ = ik_points(frame, ok, sk)
        src = np.array([pts[m] for m in present]); tgt = np.array([frame[m] for m in present])
        a, _ = align(src, tgt, "sim")
        for i, m in enumerate(present): al[m].append(np.linalg.norm(a[i] - tgt[i]) / width)
        for p, c, _ in BONES:
            if ok[p] and ok[c]: ik_dir.append(angle_deg(pts[c] - pts[p], frame[c] - frame[p]))
for m in MP: print(f"{NAMES[m]:<11} aligned median {pct(al[m]):.2f}  p95 {pct(al[m],95):.2f}")
print(f"IK-frame direction error: median {np.median(ik_dir):.2e} deg, max {np.max(ik_dir):.2e} deg (n={len(ik_dir)})")
