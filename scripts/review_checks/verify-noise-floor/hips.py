import sys; sys.path.insert(0, "scripts")
import numpy as np, glob
import measure_noise_floor as m
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    pose, ok = m.load_clip(f); sw = m.shoulder_width(pose, ok)
    for j in (23, 24):
        fr = ok[:, j].mean()
        a = m.method_a(pose[:, j], ok[:, j], [0,1,2]) / sw
        b = m.method_b(pose[:, j], ok[:, j], [0,1,2]) / sw
        sp = m._speed(pose[:, j], ok[:, j])
        print(f, j, f"frac={fr:.2f} A={a*100:.4f}% B={b*100:.4f}% median|p|={np.nanmedian(np.linalg.norm(pose[ok[:,j],j],axis=-1)):.4f} m  speed p25={np.nanpercentile(sp,25) if np.isfinite(sp).any() else np.nan:.2e}")
