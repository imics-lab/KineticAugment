"""A001, N5 (betas), N6 (origin), N8 (collinear), N7 (elbow jumps/fabrication), N9 (antipode)."""
import glob, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, _swing_rotation
sk = SMPLXSkeleton(model_path="models")
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
def smplx_joints(model, body_pose, go, transl, betas):
    with torch.no_grad():
        return model(body_pose=t(body_pose), global_orient=t(go), transl=t(transl), betas=t(betas), return_verts=False).joints[0].numpy().astype(float)
rng = np.random.default_rng(11)
# ---- N7 elbow jumps on wrist-validity toggles; fabricated wrists
tog, both = [], []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); OK = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    def ff(fr, ok):
        try: return fit_body(fr, ok, sk)
        except ValueError: return None
    fits = [ff(fr, ok) for fr, ok in zip(P, OK)]
    for i in range(1, len(fits)):
        if fits[i] is None or fits[i-1] is None: continue
        for elbow_bp, wrist in ((17, 15), (18, 16)):  # body_pose joint index = joint-1 (18->17, 19->18)
            a = R.from_rotvec(fits[i-1].body_pose.reshape(21, 3)[elbow_bp]); c = R.from_rotvec(fits[i].body_pose.reshape(21, 3)[elbow_bp])
            jump = (a.inv() * c).magnitude()
            if OK[i-1, wrist] != OK[i, wrist]: tog.append(jump)
            elif OK[i-1, wrist] and OK[i, wrist]: both.append(jump)
tog, both = np.array(tog), np.array(both)
print(f"N7: elbow rotation jump on wrist-validity toggle: n={len(tog)}, median {np.median(tog):.2f} rad, max {tog.max():.2f}; both-observed: n={len(both)}, median {np.median(both):.3f}, p99 {np.percentile(both,99):.2f}, max {both.max():.2f}; jumps>1rad: toggles {np.sum(tog>1)}, non-toggle {np.sum(both>1)}")
# ---- N9 antipode
a = np.array([1.0, 0, 0]); prev = None; maxjump = 0
for th in np.linspace(np.pi - 1e-3, np.pi + 1e-3, 11):
    b = np.array([np.cos(th), np.sin(th) * 0.0 + np.sin(th), 1e-4]); b /= np.linalg.norm(b)
    rv = R.from_matrix(_swing_rotation(a, b)).as_rotvec()
    if prev is not None: maxjump = max(maxjump, np.linalg.norm(rv - prev))
    prev = rv
print(f"N9: swing rotvec max step across antipode sweep (0.2 mrad steps): {maxjump:.2f} rad")
