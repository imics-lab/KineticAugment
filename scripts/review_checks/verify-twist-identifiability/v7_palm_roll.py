# Is palm roll about the forearm observable from MediaPipe pose landmarks 17-22?
import glob, numpy as np
SIDES = {"L": (11, 13, 15, 17, 19), "R": (12, 14, 16, 18, 20)}   # shoulder, elbow, wrist, pinky, index
vis_all = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L = d["landmarks"].astype(float); V = d["visibility"]
    vis_all.append(V[:, 17:23])
    line = f.split("/")[-1][:5]
    for s, (S, E, W, P, I) in SIDES.items():
        ok = (V[:, [S, E, W, P, I]] >= 0.5).all(1) & np.isfinite(L[:, [S, E, W, P, I]]).all((1, 2))
        rolls = []
        for fr in np.where(ok)[0]:
            a = L[fr, W]-L[fr, E]; a /= np.linalg.norm(a)
            up = L[fr, S]-L[fr, E]                          # reference: elbow bend plane
            n = np.cross(L[fr, I]-L[fr, W], L[fr, P]-L[fr, W])
            ref = up-(up@a)*a; nn = n-(n@a)*a
            if np.linalg.norm(ref) < 0.02 or np.linalg.norm(nn) < 1e-6: continue   # near-straight elbow: skip
            ref /= np.linalg.norm(ref); nn /= np.linalg.norm(nn)
            rolls.append(np.degrees(np.arctan2(np.cross(ref, nn)@a, ref@nn)))
        r = np.unwrap(np.radians(rolls)); r = np.degrees(r)
        if len(r) > 2:
            line += f" | {s}: n={len(r):3d} roll sd {r.std():5.1f} range {np.ptp(r):5.1f} deg"
    print(line)
v = np.concatenate(vis_all)
print("visibility of pose landmarks 17-22: median %.2f, frac>=0.5 %.2f" % (np.median(v), np.mean(v >= 0.5)))
print("  per landmark medians:", np.round(np.median(v, 0), 2))
