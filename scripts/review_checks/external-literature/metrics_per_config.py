"""Run the project's own noise-floor estimators and round-trip harness on each re-extracted config.
Imports the snapshot's scripts unchanged; only swaps the input file list."""
import sys, types
from pathlib import Path
import numpy as np
SNAP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(SNAP / "scripts"))
import measure_noise_floor as mnf
import measure_round_trip as mrt
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

def noise(files):
    bands = {"xy": [0, 1], "z": [2], "xyz": [0, 1, 2]}
    rows = {b: {"A": [], "B": []} for b in bands}
    for p in files:
        pose, ok_all = mnf.load_clip(str(p))
        sw = mnf.shoulder_width(pose, ok_all)
        for name, idx in mnf.IN_SCOPE.items():
            track, ok = pose[:, idx], ok_all[:, idx]
            if ok.mean() < mnf.MIN_OBSERVED_FRACTION:
                continue
            for band, dims in bands.items():
                a = mnf.method_a(track, ok, dims) / sw
                b = mnf.method_b(track, ok, dims) / sw
                if np.isfinite(a): rows[band]["A"].append(a)
                if np.isfinite(b): rows[band]["B"].append(b)
    return {b: (np.median(rows[b]["A"]) * 100, np.median(rows[b]["B"]) * 100) for b in bands}

skel = SMPLXSkeleton(model_path=str(SNAP / "models"))
sets = {"stored": sorted((SNAP / "data/test_output").glob("*_world.npz"))}
for cfg in ["S", "F", "U", "I"]:
    sets[cfg] = sorted((OUT / cfg).glob("*_world.npz"))

print(f"{'set':<8}{'A_xyz':>8}{'B_xyz':>8}{'A_xy':>8}{'B_xy':>8}{'A_z':>8}{'B_z':>8}{'direct':>9}{'aligned':>9}  per-clip direct")
for name, files in sets.items():
    nf = noise(files)
    mrt.glob = types.SimpleNamespace(glob=lambda pat, _f=files: [str(x) for x in _f])
    direct, aligned, coverage, length, per_clip, skipped, frames = mrt.run_real(skel)
    d = np.median(np.concatenate([direct[m] for m in mrt.NAMES])) * 100
    a = np.median(np.concatenate([aligned[m] for m in mrt.NAMES])) * 100
    pcs = " ".join(f"{v*100:.2f}" for v in per_clip.values())
    print(f"{name:<8}{nf['xyz'][0]:>7.2f}%{nf['xyz'][1]:>7.2f}%{nf['xy'][0]:>7.2f}%{nf['xy'][1]:>7.2f}%{nf['z'][0]:>7.2f}%{nf['z'][1]:>7.2f}%{d:>8.2f}%{a:>8.2f}%  {pcs}")
