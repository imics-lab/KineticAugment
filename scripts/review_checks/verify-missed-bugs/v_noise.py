"""A004-gate, A004-dim, N4: own re-implementation of Methods A/B; validate vs script output; per-axis; synthetic calibration."""
import glob, numpy as np
IN = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
def A(tr, ok, dims):
    st = [np.linalg.norm(tr[t + 1, dims] - tr[t, dims]) for t in range(len(tr) - 1) if ok[t] and ok[t + 1]]
    return np.percentile(st, 10) if len(st) >= 5 else np.nan
def B(tr, ok, dims, gated=True):
    sp = np.full(len(tr), np.nan)
    for t in range(len(tr) - 1):
        if ok[t] and ok[t + 1]: sp[t] = np.linalg.norm(tr[t + 1] - tr[t])
    if np.all(np.isnan(sp)): return np.nan
    g = np.nanpercentile(sp, 25)
    D = [tr[t, dims] - 2 * tr[t + 1, dims] + tr[t + 2, dims] for t in range(len(tr) - 2)
         if ok[t] and ok[t + 1] and ok[t + 2] and (not gated or np.nanmax(sp[t:t + 2]) <= g)]
    if len(D) < 5: return np.nan
    return np.linalg.norm(np.std(np.array(D), axis=0)) / np.sqrt(6)
bands = {"xy": [0, 1], "z": [2], "xyz": [0, 1, 2], "x": [0], "y": [1]}
rows = {b: {"A": [], "B": []} for b in bands}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    both = ok[:, 11] & ok[:, 12]; sw = np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1))
    for i in IN:
        if ok[:, i].mean() < 0.5: continue
        for b, dims in bands.items():
            for k, fn in (("A", A), ("B", B)):
                v = fn(P[:, i], ok[:, i], dims) / sw
                if np.isfinite(v): rows[b][k].append(v)
med = {b: {k: 100 * np.median(v) for k, v in m.items()} for b, m in rows.items()}
for b in bands: print(f"real {b:<4} A {med[b]['A']:.4f}%  B {med[b]['B']:.4f}%")
print(f"script-style ratio B_z/B_xy = {med['z']['B']/med['xy']['B']:.2f}  (script prints this as 'z is Nx noisier')")
rms_xy_B = np.sqrt((med['x']['B']**2 + med['y']['B']**2) / 2); rms_xy_A = np.sqrt((med['x']['A']**2 + med['y']['A']**2) / 2)
print(f"per-axis ratio B: z/rms(x,y) = {med['z']['B']/rms_xy_B:.2f}; z/x {med['z']['B']/med['x']['B']:.2f}; z/y {med['z']['B']/med['y']['B']:.2f}")
print(f"per-axis ratio A: z/rms(x,y) = {med['z']['A']/rms_xy_A:.2f}")
# ---- synthetic calibration
rng = np.random.default_rng(7); T = 300; ok = np.ones(T, bool)
def sim(signal):
    out = {"A_z/xy": [], "B_z/xy": [], "B_gated_xyz": [], "B_ungated_xyz": []}
    for _ in range(200):
        tr = signal + rng.normal(0, 1, (T, 3))
        out["A_z/xy"].append(A(tr, ok, [2]) / A(tr, ok, [0, 1])); out["B_z/xy"].append(B(tr, ok, [2]) / B(tr, ok, [0, 1]))
        out["B_gated_xyz"].append(B(tr, ok, [0, 1, 2])); out["B_ungated_xyz"].append(B(tr, ok, [0, 1, 2], gated=False))
    return {k: round(float(np.median(v)), 3) for k, v in out.items()}
print("isotropic iid sigma=1/axis, stationary (true xyz norm = 1.732):", sim(np.zeros((T, 3))))
t = np.arange(T)[:, None]
print("isotropic iid sigma=1/axis + slow sinusoid amp 20, period 150 frames:", sim(20 * np.sin(2 * np.pi * t / 150 + np.array([0, 1, 2]))))
