"""E11: what happens to the image-space HANDS under each integration design, at ZERO augmentation?
The classifier consumes (T,543,3) image landmarks; hands (2x21) are separate MediaPipe outputs whose
landmark 0 is the wrist. Full replacement moves the pose wrist to its SMPL-X reconstruction (projected
with a per-frame weak-perspective camera) while hands stay put; delta transfer leaves both untouched.
Reports |pose wrist - hand wrist| in the image, % image shoulder width."""
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
D = Path(__file__).resolve().parent / "extract"
sk = SMPLXSkeleton(model_path="models")
def wp_fit(P, Q):
    def res(x):
        return ((np.exp(x[0]) * P @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]) - Q).ravel()
    best = None
    for rv0 in ([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]):
        r = least_squares(res, np.r_[np.log(400.0), rv0, Q.mean(0)])
        if best is None or r.cost < best.cost: best = r
    x = best.x
    return np.exp(x[0]) * R.from_rotvec(x[1:4]).as_matrix()[:2], x[4:6]
orig, repl = [], []
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I, Pw, V = d["image"] * np.array([W, H, W]), d["world"].astype(float), d["visibility"]
    hands = {15: d["left_hand"] * np.array([W, H, W]), 16: d["right_hand"] * np.array([W, H, W])}
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in range(23) if ok[i]]
        if len(idx) < 6: continue
        try:
            out = reconstruct_body(fit_body(Pw[t], ok, sk), sk)
        except ValueError:
            continue
        Lin, tr = wp_fit(Pw[t, idx], I[t, idx, :2])
        sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
        for w_mp, hand in hands.items():
            # MediaPipe Holistic 'left_hand' is the signer's left hand; pose 15 = left wrist
            if not (ok[w_mp] and np.isfinite(hand[t, 0]).all()): continue
            orig.append(np.linalg.norm(I[t, w_mp, :2] - hand[t, 0, :2]) / sw * 100)
            repl.append(np.linalg.norm(Lin @ out[w_mp] + tr - hand[t, 0, :2]) / sw * 100)
o, r = np.array(orig), np.array(repl)
print(f"wrist/hand pairs: {len(o)}")
print(f"original data      |pose wrist - hand wrist|: median {np.median(o):.1f} p95 {np.percentile(o,95):.1f} %SW")
print(f"full replacement   |pose wrist - hand wrist|: median {np.median(r):.1f} p95 {np.percentile(r,95):.1f} %SW (zero augmentation)")
print(f"delta transfer     identical to original at zero augmentation (by construction)")
