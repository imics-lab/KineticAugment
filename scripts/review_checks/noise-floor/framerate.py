"""Frame-rate sensitivity of the script's estimators: decimate existing tracks by 2 and 3
(post-hoc; equivalent to sampling the same detections at 15/10 fps) and re-pool A/B (xyz)."""
import importlib.util
from pathlib import Path
import numpy as np
SNAP = Path.cwd(); EX = Path(__file__).resolve().parent / "extract"
spec = importlib.util.spec_from_file_location("mnf", SNAP / "scripts/measure_noise_floor.py")
mnf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mnf)
LMS = list(mnf.IN_SCOPE.values())
for variant in ["repo", "nosmooth"]:
    for step in (1, 2, 3):
        A, B = [], []
        for c in ["01986", "01987", "01988", "01991", "01992"]:
            p = (SNAP / "data/test_output" if variant == "repo" else EX / variant) / f"{c}_world.npz"
            d = np.load(p); pose = d["landmarks"].astype(float)[::step]
            ok_all = (np.isfinite(pose).all(-1) & (d["visibility"][::step] >= 0.5))
            sw = mnf.shoulder_width(pose, ok_all)
            for j in LMS:
                tr, ok = pose[:, j], ok_all[:, j]
                if ok.mean() < 0.5: continue
                a = mnf.method_a(tr, ok, [0, 1, 2]) / sw; b = mnf.method_b(tr, ok, [0, 1, 2]) / sw
                if np.isfinite(a): A.append(a)
                if np.isfinite(b): B.append(b)
        print(f"{variant:<9} every {step} frame(s) (~{30//step} fps): A_xyz={np.median(A)*100:.3f}%  "
              f"B_xyz={np.median(B)*100:.3f}%  budget=max={max(np.median(A),np.median(B))*100:.3f}%")
