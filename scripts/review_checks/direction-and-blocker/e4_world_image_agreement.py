"""E4: sanity check on E3f -- do MediaPipe's world and image heads agree about 2-D segment geometry?
Per frame, compare the image-plane direction of each arm segment (isotropic pixels, x=xW, y=yH) with
the xy components of the same world segment (world axes are camera-aligned, origin at hip centre).
Also compare frame-to-frame CHANGES of the 2-D angle (what an augmentation delta would need)."""
import numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent / "extract"
SEGS = {"L upper": (11, 13), "L fore": (13, 15), "R upper": (12, 14), "R fore": (14, 16)}
def ang(v): return np.degrees(np.arctan2(v[..., 1], v[..., 0]))
def wrap(a): return (a + 180) % 360 - 180
absdiff, dchg_i, dchg_w, inplane = [], [], [], []
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I = d["image"][..., :2] * np.array([W, H]); P = d["world"].astype(float); V = d["visibility"]
    for nm, (a, b) in SEGS.items():
        ok = (V[:, a] > 0.5) & (V[:, b] > 0.5) & np.isfinite(P[:, [a, b]]).all((1, 2))
        vi, vw = I[:, b] - I[:, a], P[:, b] - P[:, a]
        # in-plane fraction of world segment (|xy| / |xyz|); low => foreshortened, 2-D angle ill-defined
        frac = np.linalg.norm(vw[:, :2], axis=1) / np.linalg.norm(vw, axis=1)
        m = ok & (frac > 0.5)
        absdiff.append(np.abs(wrap(ang(vi[m]) - ang(vw[m, :2]))))
        inplane.append(frac[ok])
        for k in (3,):
            idx = np.where(m[:-k] & m[k:])[0]
            dchg_i.append(wrap(ang(vi[idx + k]) - ang(vi[idx])))
            dchg_w.append(wrap(ang(vw[idx + k, :2]) - ang(vw[idx, :2])))
ad = np.concatenate(absdiff); ci, cw = np.concatenate(dchg_i), np.concatenate(dchg_w)
print(f"segments with world in-plane fraction >0.5: n={len(ad)}; in-plane fraction median {np.median(np.concatenate(inplane)):.2f}")
print(f"|2-D angle(image) - 2-D angle(world xy)|: median {np.median(ad):.1f} deg, p95 {np.percentile(ad,95):.1f} deg")
big = np.abs(ci) > 5
print(f"3-frame change in 2-D angle, frames where image angle changes >5 deg: n={big.sum()}")
print(f"   corr(image change, world change) = {np.corrcoef(ci[big], cw[big])[0,1]:.3f}")
print(f"   |world change - image change| / |image change|: median {np.median(np.abs(cw[big]-ci[big])/np.abs(ci[big])):.2f}, p95 {np.percentile(np.abs(cw[big]-ci[big])/np.abs(ci[big]),95):.2f}")
print(f"   sign agreement: {np.mean(np.sign(ci[big]) == np.sign(cw[big]))*100:.1f}%")
