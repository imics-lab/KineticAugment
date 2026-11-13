import glob, warnings, traceback
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
warnings.filterwarnings("ignore")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE, BodyFit
from kinetic_augment.body_model.skeleton import SMPLXSkeleton, SyntheticSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics

sk = SMPLXSkeleton(model_path="models")
d = np.load("data/test_output/01988_world.npz"); L = d["landmarks"].astype(float); V = d["visibility"]

# (1) 180deg about shoulder axis invariance of arm landmarks
worst = 0; n = 0
for t in range(0, L.shape[0], 7):
    lm = L[t]; ok = np.isfinite(lm).all(1) & (V[t] >= 0.5)
    try: fc = fit_body(lm, ok, sk)
    except ValueError: continue
    rc = reconstruct_body(fc, sk)
    axis = rc[11] - rc[12]; axis /= np.linalg.norm(axis)
    Q = R.from_rotvec(axis * np.pi).as_matrix()
    G2 = Q @ R.from_rotvec(fc.global_orient).as_matrix()
    rots = solve_swing_ik(np.where(ok[:, None], lm, np.nan), sk, SMPLX_ARM_BONES, global_orient=R.from_matrix(G2).as_rotvec())
    posed = forward_kinematics(rots, sk)
    tr = rc[11] - posed[16]
    f2 = BodyFit(rots[1:22].reshape(63), rots[0], tr)
    r2 = reconstruct_body(f2, sk)
    worst = max(worst, max(np.linalg.norm(r2[m] - rc[m]) for _, m in ARM_CORRESPONDENCE if ok[m])); n += 1
    up_old = R.from_rotvec(fc.global_orient).as_matrix() @ [0, 1, 0]; up_new = G2 @ [0, 1, 0]
print("(1) 180deg about fitted shoulder axis: frames %d, worst arm-landmark change %.2e m; last frame up y old %.2f new %.2f" % (n, worst, up_old[1], up_new[1]))

# (2) betas ignored by reconstruct_body
lm = L[40]; ok = np.isfinite(lm).all(1) & (V[40] >= 0.5)
fc = fit_body(lm, ok, sk)
b = np.zeros(10); b[:3] = [2, -1.5, 1]
fb = BodyFit(fc.body_pose, fc.global_orient, fc.transl, b)
r0, r1 = reconstruct_body(fc, sk), reconstruct_body(fb, sk)
t = lambda a: torch.tensor(a, dtype=torch.float32)[None]
with torch.no_grad():
    Jb = sk.model(body_pose=t(fb.body_pose), global_orient=t(fb.global_orient), transl=t(fb.transl), betas=t(b), return_verts=False).joints[0].numpy()
print("(2) reconstruct_body change when betas set: %.2e m; real forward with betas vs reconstruct_body: max %.1f mm" % (
    np.nanmax(np.abs(r1 - r0)), 1000 * max(np.linalg.norm(Jb[s] - r1[m]) for s, m in ARM_CORRESPONDENCE)))
import inspect, kinetic_augment.body_model.converter as C
print("    'betas' referenced in reconstruct_body source:", "betas" in inspect.getsource(C.reconstruct_body))
# with skeleton carrying the same betas, does it match?
skb = SMPLXSkeleton(model_path="models", betas=b)
r3 = reconstruct_body(fb, skb)
print("    reconstruct_body with SMPLXSkeleton(betas=b) vs real forward: max %.2e m" % max(np.linalg.norm(Jb[s] - r3[m]) for s, m in ARM_CORRESPONDENCE))

# (3) SyntheticSkeleton
try:
    fit_body(lm, ok, SyntheticSkeleton()); print("(3) synthetic: OK")
except Exception as e:
    print("(3) fit_body with SyntheticSkeleton:", type(e).__name__, e)

# (4) pipeline reachability
from kinetic_augment.pipeline.pipeline import Pipeline
from kinetic_augment.pipeline import presets
for y in sorted(glob.glob("configs/*.yaml")):
    try:
        p = Pipeline.from_yaml(y); print("(4) yaml", y, "OK mode", p.mode)
    except Exception as e:
        print("(4) yaml", y, "FAILS:", type(e).__name__, str(e)[:110])
for name in presets.list_presets():
    c = presets.get_preset(name)
    print("(4) preset", name, "mode", c.mode)
from kinetic_augment.pipeline.chain import AugmentationChain
from kinetic_augment.augmentations.intrinsic import JointAnglePerturbation, LimbLengthScaling
try:
    ch = AugmentationChain(mode="landmark"); ch.add(JointAnglePerturbation({"stddev": 0.1}))
    print("(4) landmark chain accepted JAP")
except Exception as e:
    print("(4) landmark chain + JAP:", type(e).__name__, str(e)[:100])
# LimbLengthScaling semantics with har config's params
llsc = LimbLengthScaling({"scale_range": [0.9, 1.1], "joint_group": "legs"})
out = llsc.apply({"betas": np.zeros(10), "body_pose": np.zeros(63)})
print("(4) LimbLengthScaling(scale_range=[0.9,1.1]) on zero betas ->", np.round(out["betas"], 3))
