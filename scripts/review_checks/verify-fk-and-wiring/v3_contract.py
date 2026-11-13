"""Independent checks of the Step 7 contract claims (converter.py)."""
import glob, warnings, numpy as np, torch
from scipy.spatial.transform import Rotation as Rot
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, BodyFit, ARM_CORRESPONDENCE
warnings.filterwarnings("ignore")
ARM = [m for _, m in ARM_CORRESPONDENCE]
neutral = SMPLXSkeleton(model_path="models")

def smplx_joints(skel_model, fit, betas=None):
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None]
    b = fit.betas if betas is None else betas
    with torch.no_grad():
        return skel_model(body_pose=T(fit.body_pose), global_orient=T(fit.global_orient),
                          transl=T(fit.transl), betas=T(b), return_verts=False).joints[0, :55].double().numpy()

def lm_from(skel, rots, R=np.eye(3), t=np.zeros(3)):
    j = forward_kinematics(rots, skel) @ R.T + t
    L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: L[m] = j[s]
    return L, np.isfinite(L).all(1)

def arm_pose(rng, root_sigma=0.0):
    r = np.zeros((55, 3)); r[0] = rng.normal(0, root_sigma, 3) if root_sigma else 0
    for j, *_ in SMPLX_ARM_BONES: r[j] = rng.normal(0, 0.4, 3)
    return r

def kabsch(src, dst):   # my own rigid fit, independent of alignment.py
    sc, dc = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, dc - R @ sc

rng = np.random.default_rng(777)
print("== S7-forward: SMPL-X(returned params) vs reconstruct_body ==")
for label, betas in [("betas=0", None), ("|b|~2", rng.normal(0, 0.65, 10)), ("test betas", np.array([1.5, -1, .5, .8, -.3, 0, 0, 0, 0, 0])), ("|b|~6", rng.normal(0, 2, 10))]:
    skel = neutral if betas is None else SMPLXSkeleton(model_path="models", betas=betas)
    e_ret, e_hand = [], []
    for k in range(20):
        R = Rot.random(random_state=k).as_matrix(); t = rng.normal(0, 1, 3)
        L, v = lm_from(skel, arm_pose(rng, 0.5), R, t)
        fit = fit_body(L, v, skel)
        ours = reconstruct_body(fit, skel)[ARM]
        sj = smplx_joints(skel.model, fit)
        e_ret.append(np.linalg.norm(ours - np.array([sj[s] for s, _ in ARM_CORRESPONDENCE]), axis=1).max())
        if betas is not None:
            sj2 = smplx_joints(skel.model, fit, betas=betas)
            e_hand.append(np.linalg.norm(ours - np.array([sj2[s] for s, _ in ARM_CORRESPONDENCE]), axis=1).max())
    msg = f"  {label:10s} |b|={0 if betas is None else np.linalg.norm(betas):.2f}: fit.betas returned all-zero={not np.any(fit.betas)}; max err with returned betas {max(e_ret)*1e3:.4g} mm"
    if e_hand: msg += f"; with skeleton's betas passed by hand {max(e_hand):.2e} m"
    print(msg)

# real frames, neutral skeleton
errs, nframes = [], 0
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    z = np.load(f); Ls, Vs = z["landmarks"].astype(np.float64), z["visibility"]
    for L, vis in zip(Ls, Vs):
        ok = vis > 0.5
        try: fit = fit_body(L, ok, neutral)
        except ValueError: continue
        nframes += 1
        ours = reconstruct_body(fit, neutral); sj = smplx_joints(neutral.model, fit)
        errs.append(max(np.linalg.norm(ours[m] - sj[s]) for s, m in ARM_CORRESPONDENCE))
print(f"  real frames (neutral): n={nframes} max={max(errs):.2e} m")

print("== S7-fold: reconstruct_body(fit) vs R@FK(ik)+t with my own Kabsch ==")
fold, nopivot = [], []
for k in range(100):
    skel = neutral if k % 2 == 0 else SMPLXSkeleton(model_path="models", betas=rng.normal(0, 1, 10)) if k % 10 == 1 else neutral
    R0 = Rot.random(random_state=1000 + k).as_matrix(); t0 = rng.normal(0, 1, 3)
    L, v = lm_from(skel, arm_pose(rng, 0.8), R0, t0)
    fit = fit_body(L, v, skel)
    ik = solve_swing_ik(np.where(v[:, None], L, np.nan), skel, SMPLX_ARM_BONES)
    posed = forward_kinematics(ik, skel)
    src = np.array([posed[s] for s, _ in ARM_CORRESPONDENCE]); dst = L[ARM]
    R, t = kabsch(src, dst)
    expect = posed @ R.T + t
    rec = reconstruct_body(fit, skel)
    fold.append(max(np.linalg.norm(rec[m] - expect[s]) for s, m in ARM_CORRESPONDENCE))
    # control: drop pivot compensation
    bad = BodyFit(fit.body_pose, fit.global_orient, t.copy())
    rb = reconstruct_body(bad, skel)
    nopivot.append(max(np.linalg.norm(rb[m] - expect[s]) for s, m in ARM_CORRESPONDENCE))
print(f"  fold max err {max(fold):.2e} m over 100; no-pivot control median {np.median(nopivot)*1e3:.0f} mm, |root rest|={np.linalg.norm(neutral.rest_joints[0])*1e3:.0f} mm")

print("== whole-body rotation: error vs truth (root rotation estimate, Step 8 territory) ==")
for deg in (0, 10, 30, 90, 180):
    w = []
    for k in range(200):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        R0 = Rot.from_rotvec(ax * np.deg2rad(deg)).as_matrix()
        L, v = lm_from(neutral, arm_pose(rng, 0), R0, rng.normal(0, .5, 3))
        rec = reconstruct_body(fit_body(L, v, neutral), neutral)
        w.append(np.linalg.norm(rec[ARM] - L[ARM], axis=1).max())
    print(f"  {deg:3d} deg: worst-joint median {np.median(w)*1e3:.1f} mm, max {max(w)*1e3:.1f} mm")

print("== S7-only-params ==")
L, v = lm_from(neutral, arm_pose(rng, .5))
fit = fit_body(L, v, neutral); a = reconstruct_body(fit, neutral)
fit2 = BodyFit(fit.body_pose, fit.global_orient, fit.transl, betas=rng.normal(0, 3, 10))
print("  random fit.betas changes reconstruct_body by", np.nanmax(np.abs(reconstruct_body(fit2, neutral) - a)))
shaped = SMPLXSkeleton(model_path="models", betas=np.array([1.5, -1, .5, .8, -.3, 0, 0, 0, 0, 0]))
print("  same fit, shaped vs neutral skeleton: max diff %.1f mm" % (np.nanmax(np.linalg.norm(reconstruct_body(fit, shaped) - a, axis=1)) * 1e3))

print("== S7-nan ==")
rots = arm_pose(rng, .3); L, v = lm_from(neutral, rots)
for inv in ([15], [13, 15], [13]):
    vv = v.copy(); vv[inv] = False
    out = reconstruct_body(fit_body(L, vv, neutral), neutral)
    nan_rows = [i for i in range(33) if np.isnan(out[i]).all()]
    print(f"  invalid {inv}: NaN rows={len(nan_rows)}, invalid rows NaN? {[bool(np.isnan(out[i]).all()) for i in inv]}, "
          f"err at invalid rows mm={[round(float(np.linalg.norm(out[i]-L[i]))*1e3,1) for i in inv]}")
# invalid values never influence the fit
diffs = 0; comps = 0
for k in range(5):
    L, v = lm_from(neutral, arm_pose(rng, .5))
    for mask_bits in range(1, 64):
        inv = [m for i, m in enumerate(ARM) if mask_bits >> i & 1]
        vv = v.copy(); vv[inv] = False
        if vv[ARM].sum() < 3: continue
        f0 = fit_body(L, vv, neutral)
        for junk in (np.nan, 0.0, 1e3, None):
            LL = L.copy(); LL[inv] = rng.normal(0, 5, (len(inv), 3)) if junk is None else junk
            f1 = fit_body(LL, vv, neutral); comps += 1
            diffs += not (np.array_equal(f0.body_pose, f1.body_pose) and np.array_equal(f0.global_orient, f1.global_orient) and np.array_equal(f0.transl, f1.transl))
print(f"  junk in invalid rows: {diffs} bitwise differences in {comps} comparisons")

print("== S7-min3 / degeneracy / orphans ==")
L, v = lm_from(neutral, arm_pose(rng, .3))
for nvalid in range(0, 7):
    vv = np.zeros(33, bool); vv[ARM[:nvalid]] = True
    try: fit_body(L, vv, neutral); r = "fit"
    except ValueError: r = "ValueError"
    print(f"  {nvalid} valid -> {r}", end=";")
print()
# collinear: straight left arm (zero rotations at 16,18) only left arm valid; rotate world about arm axis
rots = np.zeros((55, 3)); rots[16] = [0, 0, 0.3]
base_j = forward_kinematics(rots, neutral)
axis = base_j[20] - base_j[16]; axis /= np.linalg.norm(axis)
for ang in (0.5, 1.0, 2.0):
    R0 = Rot.from_rotvec(axis * ang).as_matrix()
    # rotate about the left shoulder so the left arm stays fixed
    p = base_j[16]; Lw, _ = lm_from(neutral, rots, R0, p - R0 @ p)
    vv = np.zeros(33, bool); vv[[11, 13, 15]] = True
    fit = fit_body(Lw, vv, neutral)
    out = reconstruct_body(fit, neutral)
    print(f"  collinear L-arm only, truth rotated {ang} rad about arm: fit OK, R shoulder err {np.linalg.norm(out[12]-Lw[12])*1e3:.0f} mm, L wrist err {np.linalg.norm(out[15]-Lw[15])*1e3:.2g} mm")

def orphan(inv, valid_only=None, trials=200):
    e = []
    for k in range(trials):
        L, v = lm_from(neutral, arm_pose(rng, 0), np.eye(3), rng.normal(0, .3, 3))
        vv = v.copy()
        if valid_only is not None: vv[:] = False; vv[valid_only] = True
        vv[inv] = False
        out = reconstruct_body(fit_body(L, vv, neutral), neutral)
        obs = [m for m in ARM if vv[m]]
        # "fully observed": joints whose bone chain is solvable
        e.append(np.linalg.norm(out[obs] - L[obs], axis=1).max())
    return np.median(e) * 1e3, max(e) * 1e3
print("  L elbow invalid (L wrist valid -> orphan): worst observed joint median %.1f / max %.1f mm" % orphan([13]))
print("  L shoulder invalid: median %.1f / max %.1f mm" % orphan([11]))
print("  L wrist invalid (no orphan): median %.2g / max %.2g mm" % orphan([15]))
print("  only L sh, R sh, L wrist valid: median %.1f / max %.1f mm" % orphan([], valid_only=[11, 12, 15]))

# real-data prevalence of orphan configurations (vis>0.5)
cnt = {"elbow_inv_wrist_ok": 0, "shoulder_inv_elbow_ok": 0, "total": 0}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    z = np.load(f)
    for vis in z["visibility"]:
        ok = vis > 0.5; cnt["total"] += 1
        cnt["elbow_inv_wrist_ok"] += int((not ok[13] and ok[15]) or (not ok[14] and ok[16]))
        cnt["shoulder_inv_elbow_ok"] += int((not ok[11] and ok[13]) or (not ok[12] and ok[14]))
print("  real data:", cnt)
