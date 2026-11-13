"""E2: Does the new converter make zero-noise intrinsic augmentation a no-op?
The doc's motivation (Sec.3) is that zero-noise JointAnglePerturbation moved landmarks by 0.56
under the legacy encoding. Here: real world landmarks -> fit_body -> JointAnglePerturbation /
JointCoupledNoise / LimbLengthScaling with zero magnitude -> reconstruct_body -> displacement."""
import warnings
import numpy as np
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, BodyFit, ARM_CORRESPONDENCE
from kinetic_augment.augmentations.intrinsic import (JointAnglePerturbation, JointCoupledNoise,
                                                     LimbLengthScaling)

sk = SMPLXSkeleton(model_path="models")
CLIPS = ["01986", "01987", "01988", "01991", "01992"]
MP = [m for _, m in ARM_CORRESPONDENCE]
VIS = 0.5

def fits():
    for c in CLIPS:
        d = np.load(f"data/test_output/{c}_world.npz")
        L, V = d["landmarks"].astype(np.float64), d["visibility"]
        for t in range(len(L)):
            ok = (V[t] > VIS) & np.isfinite(L[t]).all(1)
            try:
                f = fit_body(L[t], ok, sk)
            except ValueError:
                continue
            sw = np.linalg.norm(L[t, 11] - L[t, 12])
            yield c, t, f, ok, sw

def as_params(f):
    return {"body_pose": f.body_pose[None].copy(), "global_orient": f.global_orient[None].copy(),
            "betas": f.betas[None].copy()}

def back(p, f):
    return BodyFit(body_pose=p["body_pose"][0], global_orient=p["global_orient"][0],
                   transl=f.transl, betas=p["betas"][0])

augs = {
    "JointAnglePerturbation(stddev=0, arms, respect_limits=True)":
        JointAnglePerturbation({"stddev": 0.0, "joint_groups": ["left_arm", "right_arm"]}),
    "JointAnglePerturbation(stddev=0, arms, respect_limits=False)":
        JointAnglePerturbation({"stddev": 0.0, "joint_groups": ["left_arm", "right_arm"]}, respect_limits=False),
    "JointCoupledNoise(stddev=0)": JointCoupledNoise({"stddev": 0.0, "affected_limbs": ["arms"]}),
    "LimbLengthScaling(range=(1.0,1.0)) [betas += 1 -> shape SHOULD change]":
        LimbLengthScaling({"scale_range": (1.0, 1.0), "components": [0, 1]}),
}
res = {k: [] for k in augs}
changed_pose = {k: 0 for k in augs}
n = 0
for c, t, f, ok, sw in fits():
    base = reconstruct_body(f, sk)
    n += 1
    for k, a in augs.items():
        p = a.apply(as_params(f))
        if not np.allclose(p["body_pose"], f.body_pose[None]):
            changed_pose[k] += 1
        out = reconstruct_body(back(p, f), sk)
        disp = np.linalg.norm(out[MP] - base[MP], axis=1)
        res[k].append(disp / sw * 100)
print(f"frames fitted: {n}")
for k in augs:
    r = np.concatenate(res[k])
    print(f"{k}\n   frames with body_pose altered: {changed_pose[k]}/{n}; landmark displacement "
          f"%SW median {np.median(r):.2f}  p95 {np.percentile(r,95):.2f}  max {r.max():.2f}; "
          f"frames with any joint moved >1%SW: {sum((x>1).any() for x in res[k])}/{n}")
