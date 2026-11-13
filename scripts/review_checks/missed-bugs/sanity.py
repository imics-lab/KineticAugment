"""Checks that came back CLEAN (reported so the report can say what is NOT a bug)."""
import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

sk = SMPLXSkeleton(model_path="models")
R = sk.rest_joints
print("1. Left/right: SMPL-X rest x of joints 16 (L shoulder) %.3f, 17 (R shoulder) %.3f, 20 (L wrist) %.3f, 21 (R wrist) %.3f"
      % (R[16, 0], R[17, 0], R[20, 0], R[21, 0]))
xs = []
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    L = np.load(p)["landmarks"].astype(float)
    good = np.isfinite(L).all((1, 2))
    xs.append(L[good, 11, 0] - L[good, 12, 0])
xs = np.concatenate(xs)
print("   MP world: x(left_shoulder 11) - x(right_shoulder 12) > 0 in %.1f%% of frames (median %+.3f m)"
      % (100 * np.mean(xs > 0), np.median(xs)))

print("\n2. Umeyama / Kabsch edge cases")
rng = np.random.default_rng(0)
worst = {"refl_det": 0, "planar_err": 0, "scale_err": 0}
for _ in range(2000):
    x = rng.normal(size=(6, 3))
    Rt = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
    s_true = rng.uniform(0.5, 2)
    y = s_true * x @ Rt.T + rng.normal(size=3)
    s, Rf, t = similarity_transform(x, y)
    worst["scale_err"] = max(worst["scale_err"], abs(s - s_true))
    xp = x.copy(); xp[:, 2] = 0                      # planar source
    yp = xp @ Rt.T
    _, Rp, tp = similarity_transform(xp, yp, allow_scale=False)
    worst["planar_err"] = max(worst["planar_err"], np.abs(apply_similarity(xp, 1, Rp, tp) - yp).max())
    _, Rr, _ = similarity_transform(x, x * [1, 1, -1])
    worst["refl_det"] = max(worst["refl_det"], abs(np.linalg.det(Rr) - 1))
print("   2000 random cases: max |scale err| %.1e, planar rigid residual %.1e, |det-1| on mirrored target %.1e"
      % (worst["scale_err"], worst["planar_err"], worst["refl_det"]))
# Umeyama optimality on a mirrored target: compare against brute-force best proper rotation
x = rng.normal(size=(8, 3)); y = x * [1, 1, -1]
_, Rr, tr = similarity_transform(x, y, allow_scale=False)
res = np.sum((apply_similarity(x, 1, Rr, tr) - y) ** 2)
best = min(np.sum(((x - x.mean(0)) @ Rotation.random(random_state=i).as_matrix().T - (y - y.mean(0))) ** 2) for i in range(20000))
print("   mirrored target: Umeyama SSE %.4f vs best of 20000 random proper rotations %.4f (Umeyama must be <=)" % (res, best))

print("\n3. Converter output vs REAL SMPL-X forward on real frames (suite currently cannot check this: fixture error)")
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
worst = 0; n = 0
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(p)
    for lm, vis in list(zip(d["landmarks"].astype(float), d["visibility"]))[::7]:
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        try:
            f = fit_body(lm, ok, sk)
        except ValueError:
            continue
        ours = reconstruct_body(f, sk)
        with torch.no_grad():
            J = sk.model(body_pose=t(f.body_pose), global_orient=t(f.global_orient), transl=t(f.transl),
                         betas=t(f.betas), return_verts=False).joints[0].numpy()
        worst = max(worst, max(np.linalg.norm(ours[m] - J[s]) for s, m in ARM_CORRESPONDENCE)); n += 1
print("   %d frames: max |reconstruct_body - SMPL-X forward| = %.2e m" % (n, worst))

print("\n4. 'free scale hides 0.61 pts': is the direct output already the rigid optimum?")
gap = []
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(p)
    for lm, vis in zip(d["landmarks"].astype(float), d["visibility"]):
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        try:
            out = reconstruct_body(fit_body(lm, ok, sk), sk)
        except ValueError:
            continue
        pres = [m for _, m in ARM_CORRESPONDENCE if ok[m]]
        src, tgt = out[pres], lm[pres]
        rig = apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False))
        gap.append(np.abs(np.sum((rig - tgt) ** 2) - np.sum((src - tgt) ** 2)))
print("   max |SSE(rigid re-aligned) - SSE(direct)| = %.2e m^2  -> the whole direct/aligned gap is the free scale" % max(gap))
