"""Direct (converter) error stratified by how many arm landmarks were valid in the frame:
fewer correspondences = fewer residual degrees of freedom after the in-sample 6-dof rigid fit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import *  # noqa
sk = SMPLXSkeleton(model_path="models")
by = {}
sh_by = {}
for name, pose, oks, width in load_clips():
    for frame, ok in zip(pose, oks):
        present = [m for m in MP if ok[m]]
        if len(present) < 3: continue
        out = reconstruct_body(fit_body(frame, ok, sk), sk)
        e = [np.linalg.norm(out[m] - frame[m]) / width for m in present]
        by.setdefault(len(present), []).extend(e)
        sh_by.setdefault(len(present), []).extend(e[:2])  # shoulders only (always present first)
for k in sorted(by):
    print(f"{k} valid arm landmarks: frames {len(by[k])//k:4d}  direct median {pct(by[k]):.2f}%  shoulders-only median {pct(sh_by[k]):.2f}%")
