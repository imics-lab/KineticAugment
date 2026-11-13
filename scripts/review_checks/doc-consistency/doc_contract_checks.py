"""Empirical checks behind doc-consistency claims in V1_Problem_Scope.md / presentation_one.md.

Run from the snapshot root with PYTHONPATH=<snap>/src.
P1  Can a 7-number similarity map world landmarks onto image-normalised landmarks? (doc 470-479)
P2  Position recovery vs rotation recovery on synthetic data (doc 176, 686, 749 vs 579-583)
P3  Bone-direction error in the converter's DIRECT output (doc 600 'exact by construction')
P4  Does fit_body reject image-space input? (contract enforcement)
P5  fore/upper ratio: doc Finding 9 (1.228/0.614) vs presentation (1.42/0.53)
P6  Presentation 'drift' table for 01986 (thigh 3.5%, forearm 30%) and leg observability
P7  Presentation hand-block depth claim (pose wrist z=-1.196, 'a metre from the arm')
P8  Meaningful numbers per frame actually fitted (presentation: 24)
"""
import glob
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

VIS = 0.5
sk = SMPLXSkeleton(model_path="models")
DIMS = {"01986": (320, 240), "01987": (1280, 720), "01988": (640, 480),
        "01991": (288, 192), "01992": (288, 192)}
clips = sorted(glob.glob("data/test_output/*_world.npz"))


def section(t):
    print("\n" + "=" * 76 + "\n" + t + "\n" + "=" * 76)


def load_pair(f):
    stem = Path(f).stem.split("_")[0]
    d = np.load(f)
    W = d["landmarks"].astype(float)
    V = d["visibility"].astype(float)
    I = np.load(f"data/test_output/{stem}_landmarks.npy").astype(float)[:, :33]
    return stem, W, V, I


# ---------------------------------------------------------------------------
section("P1. World -> image-normalised: is a 7-number similarity an adequate 'camera'?")
UPPER = list(range(0, 17)) + [23, 24]
print("residual = per-landmark RMS after fit, as % of that frame's target shoulder width")
print(f"{'clip':<6}{'aspect':>7} | {'sim/frame norm':>14} {'sim/frame px':>13} {'sim/clip px':>12} "
      f"{'affine/frame':>13} | {'|Ay|/|Ax| (W/H)':>16}")
allres = {k: [] for k in ["sn", "sp", "sc", "af"]}
for f in clips:
    stem, W, V, I = load_pair(f)
    w, h = DIMS[stem]
    Ipx = I * np.array([w, h, w])
    res = {k: [] for k in allres}
    ratios = []
    clipW, clipI = [], []
    for t in range(len(W)):
        if not np.isfinite(W[t]).all() or np.abs(I[t]).sum() == 0:
            continue
        m = [i for i in UPPER if V[t, i] >= VIS]
        if len(m) < 6:
            continue
        src, tn, tp = W[t, m], I[t, m], Ipx[t, m]
        swn = np.linalg.norm(I[t, 11] - I[t, 12])
        swp = np.linalg.norm(Ipx[t, 11] - Ipx[t, 12])
        for key, tgt, sw in (("sn", tn, swn), ("sp", tp, swp)):
            s, R, tr = similarity_transform(src, tgt)
            r = np.sqrt(np.mean(np.sum((apply_similarity(src, s, R, tr) - tgt) ** 2, axis=1))) / sw
            res[key].append(r)
        X = np.hstack([src, np.ones((len(m), 1))])
        A, *_ = np.linalg.lstsq(X, tn, rcond=None)
        r = np.sqrt(np.mean(np.sum((X @ A - tn) ** 2, axis=1))) / swn
        res["af"].append(r)
        lin = A[:3].T  # rows map to x, y, z outputs
        ratios.append(np.linalg.norm(lin[1]) / np.linalg.norm(lin[0]))
        clipW.append(src); clipI.append(tp)
    # one similarity for the whole clip ("where the camera stood")
    cs, cR, ct = similarity_transform(np.vstack(clipW), np.vstack(clipI))
    for src, tgt in zip(clipW, clipI):
        pass
    k = 0
    for t in range(len(W)):
        if not np.isfinite(W[t]).all() or np.abs(I[t]).sum() == 0:
            continue
        m = [i for i in UPPER if V[t, i] >= VIS]
        if len(m) < 6:
            continue
        src, tgt = W[t, m], Ipx[t, m]
        swp = np.linalg.norm(Ipx[t, 11] - Ipx[t, 12])
        res["sc"].append(np.sqrt(np.mean(np.sum((apply_similarity(src, cs, cR, ct) - tgt) ** 2, axis=1))) / swp)
    for kk in allres:
        allres[kk].extend(res[kk])
    print(f"{stem:<6}{w/h:>7.3f} | {np.median(res['sn'])*100:>13.1f}% {np.median(res['sp'])*100:>12.1f}% "
          f"{np.median(res['sc'])*100:>11.1f}% {np.median(res['af'])*100:>12.1f}% | {np.median(ratios):>8.3f} ({w/h:.3f})")
print(f"{'pooled':<13} | {np.median(allres['sn'])*100:>13.1f}% {np.median(allres['sp'])*100:>12.1f}% "
      f"{np.median(allres['sc'])*100:>11.1f}% {np.median(allres['af'])*100:>12.1f}%")
print("compare: converter direct round trip 6.55%, aligned 5.94% (world->SMPL-X->world)")

# ---------------------------------------------------------------------------
section("P2. Synthetic: positions recovered, rotations NOT (twist), vs doc lines 176/686/749")
rng = np.random.default_rng(42)
pos_err, rot_err = [], {j: [] for j, *_ in SMPLX_ARM_BONES}
idem_err = []
for _ in range(200):
    truth = np.zeros((sk.num_joints, 3))
    for j, *_ in SMPLX_ARM_BONES:
        truth[j] = rng.normal(0, 0.4, 3)
    J = forward_kinematics(truth, sk)
    L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        L[m] = J[s]
    est = solve_swing_ik(L, sk, SMPLX_ARM_BONES)
    Je = forward_kinematics(est, sk)
    pos_err.append(max(np.linalg.norm(Je[s] - J[s]) for s, _ in ARM_CORRESPONDENCE))
    for j, *_ in SMPLX_ARM_BONES:
        d = Rotation.from_rotvec(est[j]).inv() * Rotation.from_rotvec(truth[j])
        rot_err[j].append(np.degrees(d.magnitude()))
    # idempotence: IK output is in the zero-twist family; re-solving recovers it exactly
    L2 = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        L2[m] = Je[s]
    est2 = solve_swing_ik(L2, sk, SMPLX_ARM_BONES)
    idem_err.append(max(np.degrees((Rotation.from_rotvec(est2[j]).inv() * Rotation.from_rotvec(est[j])).magnitude())
                        for j, *_ in SMPLX_ARM_BONES))
print(f"worst joint POSITION error over 200 poses: {max(pos_err):.3e} m")
names = {16: "L shoulder", 18: "L elbow", 17: "R shoulder", 19: "R elbow"}
for j, v in rot_err.items():
    print(f"  {names[j]:<11} local ROTATION error: median {np.median(v):6.2f} deg  p95 {np.percentile(v,95):6.2f}  max {np.max(v):6.2f}")
print(f"re-solve of the IK's own (zero-twist) output: worst rotation diff {max(idem_err):.2e} deg")

# ---------------------------------------------------------------------------
section("P3. Bone-direction error in the DIRECT converter output (doc line 600)")
BONES = [(11, 13, "L upper"), (13, 15, "L fore"), (12, 14, "R upper"), (14, 16, "R fore")]
dir_direct = {n: [] for *_, n in BONES}
go_mag = []
for f in clips:
    d = np.load(f)
    P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= VIS)
    for frame, o in zip(P, ok):
        if sum(o[m] for _, m in ARM_CORRESPONDENCE) < 3:
            continue
        fit = fit_body(frame, o, sk)
        out = reconstruct_body(fit, sk)
        go_mag.append(np.degrees(np.linalg.norm(fit.global_orient)))
        for a, b, n in BONES:
            if o[a] and o[b]:
                u, v = frame[b] - frame[a], out[b] - out[a]
                c = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
                dir_direct[n].append(np.degrees(np.arccos(np.clip(c, -1, 1))))
for n, v in dir_direct.items():
    print(f"  {n:<8} direct-output direction error: median {np.median(v):5.2f} deg  p95 {np.percentile(v,95):5.2f}")
print(f"  fitted |global_orient| (rigid frame fold-in): median {np.median(go_mag):.1f} deg")

# ---------------------------------------------------------------------------
section("P4. Does fit_body refuse image-space input? (contract: world landmarks in metres)")
stem, W, V, I = load_pair(clips[0])
frame = I[10].copy()
ok = np.ones(33, bool)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        fit = fit_body(frame, ok, sk)
        out = reconstruct_body(fit, sk)
        e = np.nanmax(np.linalg.norm(out[[11, 12, 13, 14, 15, 16]] - frame[[11, 12, 13, 14, 15, 16]], axis=1))
        print(f"  accepted image-space frame silently: warnings={len(caught)}, "
              f"max landmark error {e:.3f} (image units), |transl|={np.linalg.norm(fit.transl):.3f}")
    except Exception as ex:  # noqa
        print("  raised:", type(ex).__name__, ex)

# ---------------------------------------------------------------------------
section("P5. fore/upper (left arm) ratio: doc Finding 9 (xyz 1.228, xy 0.614) vs presentation (1.42 / 0.53)")
allx, ally, per = [], [], []
for f in clips:
    stem, W, V, I = load_pair(f)
    good = (np.abs(I[:, [11, 13, 15]]).sum(-1) > 0).all(-1)
    sh, el, wr = I[good, 11], I[good, 13], I[good, 15]
    r3 = np.linalg.norm(wr - el, axis=-1) / np.linalg.norm(el - sh, axis=-1)
    r2 = np.linalg.norm((wr - el)[:, :2], axis=-1) / np.linalg.norm((el - sh)[:, :2], axis=-1)
    allx.extend(r3); ally.extend(r2)
    per.append((stem, np.median(r3), np.median(r2)))
    okw = np.isfinite(W).all(-1) & (V >= VIS)
    mw = okw[:, 11] & okw[:, 13] & okw[:, 15]
    rw = np.linalg.norm(W[mw, 15] - W[mw, 13], axis=-1) / np.linalg.norm(W[mw, 13] - W[mw, 11], axis=-1)
    print(f"  {stem}: image xyz {np.median(r3):.3f}  image xy {np.median(r2):.3f}  world(vis>=.5) {np.median(rw):.3f}")
print(f"  pooled median-of-ratios: image xyz {np.median(allx):.3f}  xy {np.median(ally):.3f}")
print(f"  mean of clip medians:    image xyz {np.mean([p[1] for p in per]):.3f}  xy {np.mean([p[2] for p in per]):.3f}")
print(f"  median of clip medians:  image xyz {np.median([p[1] for p in per]):.3f}  xy {np.median([p[2] for p in per]):.3f}")
print("  reference: SMPL-X rest L 0.946 / R 0.909; Winter/Drillis-Contini segment table 0.146H/0.186H = 0.785")

# ---------------------------------------------------------------------------
section("P6. Presentation 'drift' table, clip 01986 image-space (thigh 3.5/3.7, shoulders 2.0/3.2, upper 7.7/11.9, forearm 30.0/28.1)")
stem, W, V, I = load_pair(clips[0])
segs = [("Thigh L", 23, 25), ("Shoulders", 11, 12), ("Upper arm L", 11, 13), ("Forearm L", 13, 15),
        ("Thigh R", 24, 26), ("Upper arm R", 12, 14), ("Forearm R", 14, 16)]
print(f"  {'bone':<12}{'cv xy':>8}{'cv xyz':>8}{'rng xy':>8}{'rng xyz':>9}{'p95-p5 xy':>10}{'p95-p5 xyz':>11}")
for n, a, b in segs:
    l2 = np.linalg.norm((I[:, b] - I[:, a])[:, :2], axis=-1)
    l3 = np.linalg.norm(I[:, b] - I[:, a], axis=-1)
    cv = lambda l: np.std(l) / np.mean(l) * 100
    rg = lambda l: (l.max() - l.min()) / np.mean(l) * 100
    pp = lambda l: (np.percentile(l, 95) - np.percentile(l, 5)) / np.median(l) * 100
    print(f"  {n:<12}{cv(l2):>7.1f}%{cv(l3):>7.1f}%{rg(l2):>7.1f}%{rg(l3):>8.1f}%{pp(l2):>9.1f}%{pp(l3):>10.1f}%")
print(f"  01986 image-space knee y>1 (below frame): L {np.mean(I[:,25,1]>1)*100:.0f}%  R {np.mean(I[:,26,1]>1)*100:.0f}%; "
      f"hip y>1: L {np.mean(I[:,23,1]>1)*100:.0f}%")
print(f"  01986 world visibility mean: knee L {V[:,25].mean():.3f} R {V[:,26].mean():.3f}; hip L {V[:,23].mean():.3f}")

# ---------------------------------------------------------------------------
section("P7. Presentation hand-block depth claim (frame 0: pose wrist z=-1.196, hand wrist z~0)")
LH, RH = 33 + 468, 33 + 468 + 21
for f in clips:
    stem = Path(f).stem.split("_")[0]
    X = np.load(f"data/test_output/{stem}_landmarks.npy").astype(float)
    w, h = DIMS[stem]
    print(f"  {stem} frame0: pose Lwrist z {X[0,15,2]:+.3f}  Rwrist z {X[0,16,2]:+.3f} | "
          f"Lhand[0] z {X[0,LH,2]:+.4f} (present {np.abs(X[0,LH]).sum()>0})  Rhand[0] z {X[0,RH,2]:+.4f} "
          f"(present {np.abs(X[0,RH]).sum()>0}) | 1.196 image-widths = {1.196*w:.0f} px")

# ---------------------------------------------------------------------------
section("P8. How many body_pose numbers does V1 actually set on the real clips? (presentation: 24)")
counts = []
for f in clips:
    d = np.load(f)
    P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= VIS)
    for frame, o in zip(P, ok):
        if sum(o[m] for _, m in ARM_CORRESPONDENCE) < 3:
            continue
        fit = fit_body(frame, o, sk)
        bp = fit.body_pose.reshape(21, 3)
        counts.append(int((np.linalg.norm(bp, axis=1) > 1e-12).sum()))
vals, cnt = np.unique(counts, return_counts=True)
print("  nonzero body_pose joints per frame:", dict(zip(vals.tolist(), cnt.tolist())),
      "-> numbers =", {int(v) * 3: int(c) for v, c in zip(vals, cnt)}, "+ 3 global_orient + 3 transl")
