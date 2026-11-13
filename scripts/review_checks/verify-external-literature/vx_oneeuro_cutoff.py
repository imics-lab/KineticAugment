"""Implied One-Euro cutoff (MediaPipe world-landmark filter: min_cutoff 0.1, beta 40, value scaling disabled)
at the frames the noise-floor estimators sample. Uses the smoothed output speed (per axis) as a proxy for the
filter's internal derivative estimate (which is itself low-passed at 1 Hz, so this is approximate)."""
import glob, numpy as np
FPS = 30.0
rows = []
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    z = np.load(p); L = z["landmarks"].astype(float); V = z["visibility"]
    for j in [11,12,13,14,15,16]:
        ok = np.isfinite(L[:, j]).all(1) & (V[:, j] >= 0.5)
        v = np.diff(L[:, j], axis=0) * FPS          # m/s per axis
        okv = ok[1:] & ok[:-1]
        sp3 = np.linalg.norm(v, axis=1)[okv]; ax = np.abs(v[okv])
        gate = np.percentile(sp3, 25); slow = sp3 <= gate
        p10 = sp3 <= np.percentile(sp3, 10)
        rows.append((np.median(0.1 + 40*ax[slow], axis=0).mean(), np.median(0.1+40*ax[p10],axis=0).mean(), np.median(0.1+40*ax,axis=0).mean(), np.median(sp3[slow])*1000/FPS))
r = np.array(rows)
print(f"median implied cutoff Hz: slowest-25% frames {np.median(r[:,0]):.2f} (range {r[:,0].min():.2f}-{r[:,0].max():.2f}); p10 frames {np.median(r[:,1]):.2f}; all frames {np.median(r[:,2]):.2f}")
print(f"median 3D step in slowest-25% frames: {np.median(r[:,3]):.2f} mm/frame")
print(f"time constant at cutoff c: tau=1/(2 pi c): c=0.1 -> {1/(2*np.pi*0.1):.2f}s ; c={np.median(r[:,0]):.2f} -> {1/(2*np.pi*np.median(r[:,0])):.2f}s")
