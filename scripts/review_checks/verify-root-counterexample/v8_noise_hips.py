"""Does pooling the hip landmarks (which sit at MediaPipe's world origin by construction) bias the noise floor?"""
import glob, sys, numpy as np
sys.path.insert(0, "scripts")
import measure_noise_floor as nf
for drop in (False, True):
    vals = {"A": [], "B": []}
    for path in sorted(glob.glob("data/test_output/*_world.npz")):
        pose, ok_all = nf.load_clip(path); sw = nf.shoulder_width(pose, ok_all)
        for name, idx in nf.IN_SCOPE.items():
            if drop and "hip" in name:
                continue
            track, ok = pose[:, idx], ok_all[:, idx]
            if ok.mean() < nf.MIN_OBSERVED_FRACTION:
                continue
            for k, fn in (("A", nf.method_a), ("B", nf.method_b)):
                v = fn(track, ok, [0, 1, 2]) / sw
                if np.isfinite(v):
                    vals[k].append(v)
    print(("without hips" if drop else "as shipped  "), "xyz pooled median A %.3f%%  B %.3f%%  (n=%d)" % (
        np.median(vals["A"]) * 100, np.median(vals["B"]) * 100, len(vals["A"])))
