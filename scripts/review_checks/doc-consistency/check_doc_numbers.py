"""Check the numeric claims in docs/V1_Problem_Scope.md and presentation_one.md.

Run from the snapshot root with PYTHONPATH=<snap>/src.
"""
import glob
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

VIS = 0.5
NAMES = {11: "L sho", 12: "R sho", 13: "L elb", 14: "R elb", 15: "L wri", 16: "R wri"}
BONES = [(11, 13, "L upper"), (13, 15, "L fore"), (12, 14, "R upper"), (14, 16, "R fore")]

sk = SMPLXSkeleton(model_path="models")
R = sk.rest_joints


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# --------------------------------------------------------------------------
section("1. Rest skeleton lengths (doc lines 283-289, 604-609)")
L = lambda a, b: np.linalg.norm(R[a] - R[b])
print(f"L upper arm (16->18) {L(16,18):.4f}   R upper arm (17->19) {L(17,19):.4f}")
print(f"L forearm  (18->20) {L(18,20):.4f}   R forearm  (19->21) {L(19,21):.4f}")
print(f"L thigh (1->4) {L(1,4):.4f}  L shin (4->7) {L(4,7):.4f}  shoulder width (16-17) {L(16,17):.4f}")
print(f"fore/upper L {L(18,20)/L(16,18):.3f}  R {L(19,21)/L(17,19):.3f}")

# --------------------------------------------------------------------------
section("2. Visibility / coverage (Finding 5 table, line 333-337, Step 8 nose claim)")
files = sorted(glob.glob("data/test_output/*_world.npz"))
idx = {"nose": 0, "L sho": 11, "R sho": 12, "L elb": 13, "R elb": 14, "L wri": 15, "R wri": 16,
       "L hip": 23, "R hip": 24, "L knee": 25, "R knee": 26, "L ank": 27, "R ank": 28}
allvis = []
per_clip_cov = {k: [] for k in idx}
per_clip_minvis = {k: [] for k in idx}
for f in files:
    d = np.load(f)
    v = d["visibility"].astype(float)
    allvis.append(v)
    for k, i in idx.items():
        per_clip_cov[k].append(np.nanmean(v[:, i] >= VIS))
        per_clip_minvis[k].append(np.nanmedian(v[:, i]))
allv = np.concatenate(allvis)
print(f"{'lm':<7}{'pooled cov':>11}{'clip-avg cov':>13}{'mean vis':>10}{'median vis':>11}  per-clip median vis")
for k, i in idx.items():
    col = allv[:, i]
    print(f"{k:<7}{np.nanmean(col >= VIS)*100:>10.1f}%{np.mean(per_clip_cov[k])*100:>12.1f}%"
          f"{np.nanmean(col):>10.3f}{np.nanmedian(col):>11.3f}  "
          + " ".join(f"{x:.2f}" for x in per_clip_minvis[k]))

# --------------------------------------------------------------------------
section("3. Pooled / per-clip observed bone lengths (lines 604-609, Finding 10)")
pooled = {n: [] for *_, n in BONES}
for f in files:
    d = np.load(f)
    p = d["landmarks"].astype(float)
    ok = np.isfinite(p).all(-1) & (d["visibility"] >= VIS)
    row = []
    sw = np.linalg.norm(p[ok[:, 11] & ok[:, 12], 11] - p[ok[:, 11] & ok[:, 12], 12], axis=-1)
    row.append(np.median(sw))
    for a, b, n in BONES:
        m = ok[:, a] & ok[:, b]
        l = np.linalg.norm(p[m, a] - p[m, b], axis=-1)
        pooled[n].extend(l)
        row.append(np.median(l))
    print(Path(f).stem, " ".join(f"{x:.4f}" for x in row), f"fore/upper(L) {row[2]/row[1]:.3f}")
rest_len = {"L upper": L(16, 18), "L fore": L(18, 20), "R upper": L(17, 19), "R fore": L(19, 21)}
ratios = []
for n, v in pooled.items():
    r = np.median(v) / rest_len[n]
    ratios.append(r)
    print(f"pooled {n:<8} observed {np.median(v):.4f}  rest {rest_len[n]:.4f}  ratio {r:.3f}")
print(f"ratio std {np.std(ratios):.3f}  range {max(ratios)-min(ratios):.3f}")

# --------------------------------------------------------------------------
section("4. Forearm length drift: world landmarks vs presentation's image-space table")
for f in files:
    d = np.load(f)
    p = d["landmarks"].astype(float)
    ok = np.isfinite(p).all(-1) & (d["visibility"] >= VIS)
    out = []
    for a, b, n in BONES:
        m = ok[:, a] & ok[:, b]
        l = np.linalg.norm(p[m, a] - p[m, b], axis=-1)
        # "drift" = (p95 - p5)/median, and CV
        out.append(f"{n}: cv {np.std(l)/np.mean(l)*100:4.1f}% range {(l.max()-l.min())/np.median(l)*100:5.1f}%")
    print(Path(f).stem, " | ".join(out))

cached = sorted(glob.glob("data/test_output/*_landmarks.npy"))
print("\nCached image-space pose_landmarks: median fore/upper (left arm), xy vs xyz")
for f in cached:
    x = np.load(f).astype(float)
    pose = x[:, :33] if x.ndim == 3 else x
    sh, el, wr = pose[:, 11], pose[:, 13], pose[:, 15]
    good = (np.abs(sh).sum(-1) > 0) & (np.abs(el).sum(-1) > 0) & (np.abs(wr).sum(-1) > 0)
    up3 = np.linalg.norm(el - sh, axis=-1)[good]; fo3 = np.linalg.norm(wr - el, axis=-1)[good]
    up2 = np.linalg.norm((el - sh)[:, :2], axis=-1)[good]; fo2 = np.linalg.norm((wr - el)[:, :2], axis=-1)[good]
    print(f"{Path(f).stem} shape {x.shape}  xyz ratio-of-medians {np.median(fo3)/np.median(up3):.3f}"
          f"  median-of-ratios {np.median(fo3/up3):.3f}   xy ratio-of-medians {np.median(fo2)/np.median(up2):.3f}"
          f"  median-of-ratios {np.median(fo2/up2):.3f}   forearm xy range {(fo2.max()-fo2.min())/np.median(fo2)*100:.1f}%")


# --------------------------------------------------------------------------
section("5. Root-rotation counterexample table (doc lines 525-530)")


def rot_trial(skel, deg, axis):
    truth = np.zeros((skel.num_joints, 3))
    truth[0] = axis / np.linalg.norm(axis) * np.deg2rad(deg)
    rng = np.random.default_rng()
    tj = forward_kinematics(truth, skel)
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = tj[s]
    fitted = solve_swing_ik(lm, skel, SMPLX_ARM_BONES)
    fj = forward_kinematics(fitted, skel)
    src = np.array([fj[s] for s, _ in ARM_CORRESPONDENCE])
    tgt = np.array([lm[m] for _, m in ARM_CORRESPONDENCE])
    raw = np.linalg.norm(src - tgt, axis=1)
    al = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1)
    rig = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False)) - tgt, axis=1)
    return raw, al, rig


def rot_trial_pose(skel, deg, axis, rng):
    truth = np.zeros((skel.num_joints, 3))
    truth[0] = axis / np.linalg.norm(axis) * np.deg2rad(deg)
    for joint, *_ in SMPLX_ARM_BONES:
        truth[joint] = rng.normal(0, 0.4, 3)
    tj = forward_kinematics(truth, skel)
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = tj[s]
    fitted = solve_swing_ik(lm, skel, SMPLX_ARM_BONES)
    fj = forward_kinematics(fitted, skel)
    src = np.array([fj[s] for s, _ in ARM_CORRESPONDENCE])
    tgt = np.array([lm[m] for _, m in ARM_CORRESPONDENCE])
    raw = np.linalg.norm(src - tgt, axis=1)
    al = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1)
    rig = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False)) - tgt, axis=1)
    return raw, al, rig


print("Single axes, rest arms (mm): raw mean/max | similarity-aligned mean/max | rigid-aligned mean/max")
for deg in (10, 20, 30, 40.107, 45):
    for name, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        raw, al, rig = rot_trial(sk, deg, np.array(ax, float))
        print(f"  {deg:>6} deg axis {name}: raw {raw.mean()*1e3:7.2f}/{raw.max()*1e3:7.2f} | "
              f"sim {al.mean()*1e3:6.2f}/{al.max()*1e3:6.2f} | rigid {rig.mean()*1e3:6.2f}/{rig.max()*1e3:6.2f}")
print("Random axes + random arm poses (200 trials, seed 42), pooled over joints*trials (mm)")
for deg in (10, 20, 30, 45):
    rng = np.random.default_rng(42)
    R_, A_, G_ = [], [], []
    for _ in range(200):
        ax = rng.normal(size=3)
        raw, al, rig = rot_trial_pose(sk, deg, ax, rng)
        R_.append(raw); A_.append(al); G_.append(rig)
    R_, A_, G_ = map(np.concatenate, (R_, A_, G_))
    print(f"  {deg:>3} deg: raw mean {R_.mean()*1e3:7.2f} max {R_.max()*1e3:7.2f} | sim mean {A_.mean()*1e3:6.2f}"
          f" max {A_.max()*1e3:6.2f} | rigid mean {G_.mean()*1e3:6.2f} max {G_.max()*1e3:6.2f}")


# --------------------------------------------------------------------------
section("6. Real data: aligned residual, Procrustes scale, post-alignment direction error, length error")


class LenSkeleton:
    """SMPL-X skeleton with the four arm bones rescaled to given lengths."""

    def __init__(self, base, lengths):
        self.parents = base.parents
        self.rest_joints = base.rest_joints.copy()
        self._num = base.num_joints
        children = {j: [c for c in range(base.num_joints) if base.parents[c] == j] for j in range(base.num_joints)}

        def descendants(j):
            out = []
            for c in children[j]:
                out.append(c)
                out.extend(descendants(c))
            return out

        for (j, c, *_), Lnew in zip(SMPLX_ARM_BONES, lengths):
            if Lnew is None or not np.isfinite(Lnew):
                continue
            v = self.rest_joints[c] - self.rest_joints[j]
            delta = v / np.linalg.norm(v) * Lnew - v
            for k in [c] + descendants(c):
                self.rest_joints[k] = self.rest_joints[k] + delta

    @property
    def num_joints(self):
        return self._num


def frame_errors(frame, ok, skel, go=None):
    masked = np.where(ok[:, None], frame, np.nan)
    rots = solve_swing_ik(masked, skel, SMPLX_ARM_BONES, global_orient=go)
    fj = forward_kinematics(rots, skel)
    present = [m for s, m in ARM_CORRESPONDENCE if ok[m]]
    src = np.array([fj[s] for s, m in ARM_CORRESPONDENCE if ok[m]])
    tgt = np.array([frame[m] for m in present])
    sim = similarity_transform(src, tgt)
    al = apply_similarity(src, *sim)
    return present, src, tgt, al, sim, fj


def load(f):
    d = np.load(f)
    p = d["landmarks"].astype(float)
    ok = np.isfinite(p).all(-1) & (d["visibility"] >= VIS)
    both = ok[:, 11] & ok[:, 12]
    w = float(np.median(np.linalg.norm(p[both, 11] - p[both, 12], axis=-1)))
    return p, ok, w


base_err, scales, dir_err, len_al, len_direct = [], [], {n: [] for *_, n in BONES}, [], []
smap = dict((m, s) for s, m in ARM_CORRESPONDENCE)
for f in files:
    p, okall, w = load(f)
    for frame, ok in zip(p, okall):
        if sum(ok[m] for m in NAMES) < 3:
            continue
        present, src, tgt, al, sim, fj = frame_errors(frame, ok, sk)
        base_err.extend(np.linalg.norm(al - tgt, axis=1) / w)
        scales.append(sim[0])
        alpos = {m: a for m, a in zip(present, al)}
        for a, b, n in BONES:
            if ok[a] and ok[b]:
                vo = frame[b] - frame[a]
                vr = alpos[b] - alpos[a]
                c = np.dot(vo, vr) / np.linalg.norm(vo) / np.linalg.norm(vr)
                dir_err[n].append(np.degrees(np.arccos(np.clip(c, -1, 1))))
                len_al.append(abs(np.linalg.norm(vr) - np.linalg.norm(vo)) / w)
                vd = fj[smap[b]] - fj[smap[a]]
                len_direct.append(abs(np.linalg.norm(vd) - np.linalg.norm(vo)) / w)
print(f"baseline aligned pooled median: {np.median(base_err)*100:.2f}%  (doc 5.94%)")
print(f"Procrustes scale median {np.median(scales):.4f}  p5 {np.percentile(scales,5):.3f}  p95 {np.percentile(scales,95):.3f}  (doc 0.9726, 0.91-1.02)")
for n, v in dir_err.items():
    print(f"post-alignment direction error {n}: median {np.median(v):.2f} deg")
print(f"aligned length error pooled median {np.median(len_al)*100:.2f}%  (doc 6.66%)")
print(f"unaligned (rest-skeleton vs observed) length error pooled median {np.median(len_direct)*100:.2f}%")


# --------------------------------------------------------------------------
section("7. Oracle decomposition attempt (doc lines 651-656: 5.94 / 5.13 / 3.86 / 2.64)")


def oracle_root(frame, ok, skel, w):
    def obj(rv):
        present, src, tgt, al, sim, fj = frame_errors(frame, ok, skel, go=rv)
        return np.sum(np.linalg.norm(al - tgt, axis=1) ** 2)
    best = None
    for init in (np.zeros(3), np.array([0, 0.3, 0]), np.array([0, -0.3, 0])):
        r = minimize(obj, init, method="Powell", options={"xtol": 1e-6, "ftol": 1e-12, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    present, src, tgt, al, sim, fj = frame_errors(frame, ok, skel, go=best.x)
    return np.linalg.norm(al - tgt, axis=1) / w


def frame_lengths(frame, ok):
    out = []
    for (j, c, ma, mb) in SMPLX_ARM_BONES:
        out.append(np.linalg.norm(frame[mb] - frame[ma]) if ok[ma] and ok[mb] else None)
    return out


rng = np.random.default_rng(0)
res = {"baseline": [], "root": [], "len_frame": [], "len_clip": [], "both_frame": [], "both_clip": [],
       "len+span_frame": []}
for f in files:
    p, okall, w = load(f)
    clip_len = []
    for (j, c, ma, mb) in SMPLX_ARM_BONES:
        m = okall[:, ma] & okall[:, mb]
        clip_len.append(float(np.median(np.linalg.norm(p[m, mb] - p[m, ma], axis=-1))))
    sk_clip = LenSkeleton(sk, clip_len)
    for frame, ok in zip(p, okall):
        if sum(ok[m] for m in NAMES) < 3:
            continue
        present, src, tgt, al, sim, fj = frame_errors(frame, ok, sk)
        res["baseline"].extend(np.linalg.norm(al - tgt, axis=1) / w)
        res["root"].extend(oracle_root(frame, ok, sk, w))
        skf = LenSkeleton(sk, frame_lengths(frame, ok))
        present, src, tgt, al, sim, fj = frame_errors(frame, ok, skf)
        res["len_frame"].extend(np.linalg.norm(al - tgt, axis=1) / w)
        present, src, tgt, al, sim, fj = frame_errors(frame, ok, sk_clip)
        res["len_clip"].extend(np.linalg.norm(al - tgt, axis=1) / w)
        res["both_frame"].extend(oracle_root(frame, ok, skf, w))
        res["both_clip"].extend(oracle_root(frame, ok, sk_clip, w))
for k, v in res.items():
    if v:
        b = np.median(res["baseline"])
        print(f"{k:<14} pooled median {np.median(v)*100:5.2f}%   reduction {(1-np.median(v)/b)*100:5.1f}%")
