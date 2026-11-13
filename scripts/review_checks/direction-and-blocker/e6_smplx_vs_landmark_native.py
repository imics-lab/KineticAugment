"""E6: Does an SMPL-X round trip buy anything for an elbow-angle augmentation, versus doing it
directly on the observed (world) landmarks?
Augmentation: 'flex the elbow by +10 deg'. Three routes, on real world landmarks:
  R1 full replacement : fit_body -> rotvec_y += s*10deg -> reconstruct_body -> output
  R2 delta transfer   : P + [recon(theta') - recon(theta)]
  R3 landmark-native  : rotate the observed wrist about the observed elbow by 10 deg about the
                        arm-plane normal n = upper x fore (no model, no fit)
Measures, % shoulder width (SW): zero-augmentation error, wrist motion, R2-vs-R3 agreement,
achieved change in elbow bend, and forearm-length preservation."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, BodyFit

sk = SMPLXSkeleton(model_path="models")
DEG = 10.0
def bend(S, E, Wr):
    u, v = E - S, Wr - E
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))

# which sign of local-y increases bend for each elbow? (determined on the rest skeleton)
from kinetic_augment.body_model.kinematics import forward_kinematics
sign = {}
for j, (s, w) in {18: (16, 20), 19: (17, 21)}.items():
    rots = np.zeros((55, 3)); rots[j, 1] = np.radians(DEG)
    P = forward_kinematics(rots, sk)
    sign[j] = 1.0 if bend(P[s], P[j], P[w]) > 1e-6 and \
        np.cross(P[j] - P[s], P[w] - P[j])[2] * 0 == 0 and True else 1.0
# bend() is unsigned; at rest bend is 0 either way, so pick sign that bends toward +z (front)
for j, w in ((18, 20), (19, 21)):
    for sgn in (1.0, -1.0):
        rots = np.zeros((55, 3)); rots[j, 1] = sgn * np.radians(DEG)
        if forward_kinematics(rots, sk)[w][2] > sk.rest_joints[w][2]:
            sign[j] = sgn
print("flexion sign on local y:", sign)

z1, r1w, r2w, r3w, agree_ang, mag_ratio, dbend2, dbend3, flen1, flen2 = ([] for _ in range(10))
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        try:
            f = fit_body(L[t], ok, sk)
        except ValueError:
            continue
        sw = np.linalg.norm(L[t, 11] - L[t, 12])
        base = reconstruct_body(f, sk)
        for j, (s_mp, e_mp, w_mp) in ((18, (11, 13, 15)), (19, (12, 14, 16))):
            if not ok[w_mp]:
                continue
            S, E, Wr = L[t, s_mp], L[t, e_mp], L[t, w_mp]
            b0 = bend(S, E, Wr)
            if b0 < 15 or b0 > 160:   # need a defined, non-degenerate arm plane
                continue
            bp = f.body_pose.reshape(21, 3).copy(); bp[j - 1, 1] += sign[j] * np.radians(DEG)
            aug = reconstruct_body(BodyFit(bp.reshape(63), f.global_orient, f.transl, f.betas), sk)
            # R1 full replacement
            z1.append(np.linalg.norm(base[w_mp] - Wr) / sw * 100)
            r1 = aug[w_mp]
            # R2 delta transfer
            r2 = Wr + (aug[w_mp] - base[w_mp])
            # R3 landmark-native: rotate wrist about elbow, axis = arm-plane normal, increasing bend
            n = np.cross(E - S, Wr - E); n /= np.linalg.norm(n)
            r3 = E + R.from_rotvec(n * np.radians(DEG)).apply(Wr - E)
            if bend(S, E, r3) < b0:
                r3 = E + R.from_rotvec(-n * np.radians(DEG)).apply(Wr - E)
            d2, d3 = r2 - Wr, r3 - Wr
            r2w.append(np.linalg.norm(d2) / sw * 100); r3w.append(np.linalg.norm(d3) / sw * 100)
            agree_ang.append(np.degrees(np.arccos(np.clip(d2 @ d3 / np.linalg.norm(d2) / np.linalg.norm(d3), -1, 1))))
            mag_ratio.append(np.linalg.norm(d2) / np.linalg.norm(d3))
            dbend2.append(bend(S, E, r2) - b0); dbend3.append(bend(S, E, r3) - b0)
            flen1.append(np.linalg.norm(r1 - aug[e_mp]) / np.linalg.norm(Wr - E))
            flen2.append(np.linalg.norm(r2 - E) / np.linalg.norm(Wr - E))
A = {k: np.array(v) for k, v in dict(z1=z1, r2w=r2w, r3w=r3w, agree=agree_ang, ratio=mag_ratio,
                                     db2=dbend2, db3=dbend3, fl1=flen1, fl2=flen2).items()}
q = lambda x: f"median {np.median(x):.2f}, p5 {np.percentile(x,5):.2f}, p95 {np.percentile(x,95):.2f}"
print(f"arms analysed: {len(A['z1'])}")
print(f"R1 full replacement, wrist error at ZERO augmentation (%SW): {q(A['z1'])}")
print(f"R2/R3 at zero augmentation: 0 by construction")
print(f"wrist motion for a +{DEG:.0f} deg flex (%SW): R2 {q(A['r2w'])} | R3 {q(A['r3w'])}")
print(f"R2 vs R3 wrist-motion direction disagreement (deg): {q(A['agree'])}")
print(f"R2/R3 wrist-motion magnitude ratio: {q(A['ratio'])}")
print(f"achieved change in elbow bend (deg), target +{DEG:.0f}: R2 {q(A['db2'])} | R3 {q(A['db3'])}")
print(f"observed-forearm length ratio after aug: R1 (SMPL-X length / observed) {q(A['fl1'])} | R2 {q(A['fl2'])} | R3 1.0 exactly")
