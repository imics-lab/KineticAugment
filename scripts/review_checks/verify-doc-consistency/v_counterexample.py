"""Sweep the frozen-root counterexample (doc table L525-530; review 0.7 rad numbers)."""
from types import SimpleNamespace
import numpy as np
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity

full = SMPLXSkeleton(model_path="models")
sk = SimpleNamespace(rest_joints=full.rest_joints[:22], parents=full.parents[:22], num_joints=22)
CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]


def errs(truth):
    J = forward_kinematics(truth, sk)
    lm = np.full((33, 3), np.nan)
    for s, m in CORR:
        lm[m] = J[s]
    Jf = forward_kinematics(solve_swing_ik(lm, sk, SMPLX_ARM_BONES), sk)
    src = np.array([Jf[s] for s, _ in CORR]); tgt = np.array([lm[m] for _, m in CORR])
    raw = np.linalg.norm(src - tgt, axis=1)
    sim = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1)
    rig = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False)) - tgt, axis=1)
    return raw, sim, rig

# review's exact case
t = np.zeros((22, 3)); t[0] = [0, 0, 0.7]
raw, sim, rig = errs(t)
print(f"review case 0.7 rad about z, rest arms: raw max {raw.max()*1e3:.2f} mean {raw.mean()*1e3:.2f}; sim max {sim.max()*1e3:.2f} mean {sim.mean()*1e3:.2f} mm")

for label, arm_sd, axes in [("rest arms, random axis", 0.0, "rand"), ("random arms N(0,0.4), random axis", 0.4, "rand"),
                            ("random arms N(0,0.3), y axis", 0.3, "y"), ("rest arms, x/y/z axes", 0.0, "xyz")]:
    print(f"-- {label}")
    for deg in (10, 20, 45):
        rng = np.random.default_rng(0)
        R_, S_, G_ = [], [], []
        trials = 3 if axes == "xyz" else 300
        for k in range(trials):
            t = np.zeros((22, 3))
            if axes == "rand":
                a = rng.normal(size=3); a /= np.linalg.norm(a)
            elif axes == "y":
                a = np.array([0, 1.0, 0])
            else:
                a = np.eye(3)[k]
            t[0] = a * np.deg2rad(deg)
            for j, *_ in SMPLX_ARM_BONES:
                t[j] = rng.normal(0, arm_sd, 3) if arm_sd else 0
            r, s, g = errs(t)
            R_.append(r); S_.append(s); G_.append(g)
        R_, S_, G_ = map(np.concatenate, (R_, S_, G_))
        print(f"  {deg:2d} deg: raw mean {R_.mean()*1e3:6.2f} max {R_.max()*1e3:6.1f} | sim mean {S_.mean()*1e3:6.2f} max {S_.max()*1e3:6.1f} | rigid mean {G_.mean()*1e3:6.2f} max {G_.max()*1e3:6.1f} mm")
