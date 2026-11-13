"""
measure_noise_floor.py compares a 2-axis band ('xy': norm of two per-axis SDs) to a 1-axis band ('z').
For isotropic noise that ratio is 1/sqrt(2), not 1. Re-run the script's OWN estimators per single
axis on the real clips, and check method B's motion gate against known noise.
"""
import glob, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("mnf", "scripts/measure_noise_floor.py")
mnf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mnf)

# --- 1. real data, the script's own estimators, one axis at a time --------------------------
bands = {"x": [0], "y": [1], "z": [2], "xy": [0, 1]}
rows = {b: {"A": [], "B": []} for b in bands}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    pose, ok_all = mnf.load_clip(path)
    sw = mnf.shoulder_width(pose, ok_all)
    for name, idx in mnf.IN_SCOPE.items():
        track, ok = pose[:, idx], ok_all[:, idx]
        if ok.mean() < mnf.MIN_OBSERVED_FRACTION:
            continue
        for b, dims in bands.items():
            for k, fn in (("A", mnf.method_a), ("B", mnf.method_b)):
                v = fn(track, ok, dims) / sw
                if np.isfinite(v):
                    rows[b][k].append(v)
med = {b: {k: 100 * np.median(rows[b][k]) for k in "AB"} for b in bands}
print("Real clips, script's own estimators, pooled median %% shoulder width:")
for b in bands:
    print(f"  {b:>3}: A {med[b]['A']:.4f}%   B {med[b]['B']:.4f}%")
print(f"  script's headline ratio  B_z / B_xy            = {med['z']['B']/med['xy']['B']:.2f}  ('z is 1.2x noisier')")
print(f"  per-axis ratio           B_z / mean(B_x, B_y)  = {med['z']['B']/np.mean([med['x']['B'], med['y']['B']]):.2f}")
print(f"  per-axis ratio           A_z / mean(A_x, A_y)  = {med['z']['A']/np.mean([med['x']['A'], med['y']['A']]):.2f}")

# --- 2. method B gate bias on KNOWN noise ---------------------------------------------------
rng = np.random.default_rng(0)
T = 4000
def b_nogate(track, dims):
    d = track[:-2] - 2 * track[1:-1] + track[2:]
    return float(np.linalg.norm(np.std(d[:, dims], axis=0)) / np.sqrt(6))
print("\nMethod B on synthetic tracks with KNOWN per-axis sigma = 1 mm (truth for xyz band: sqrt(3) = 1.732 mm)")
for label, signal in (("stationary + iid noise", np.zeros((T, 3))),
                      ("slow sinusoid (0.2 Hz @30fps, 5 cm) + iid noise",
                       0.05e3 * np.sin(2 * np.pi * 0.2 * np.arange(T) / 30)[:, None] * np.array([1, 0.5, 0.2]))):
    tr = signal + rng.normal(0, 1.0, (T, 3))
    ok = np.ones(T, bool)
    print(f"  {label:<48}: script (gated) {mnf.method_b(tr, ok, [0,1,2]):.3f}   ungated {b_nogate(tr, [0,1,2]):.3f}")
