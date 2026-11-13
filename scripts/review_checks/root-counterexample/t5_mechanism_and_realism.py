"""A. mechanism: which joint rotations can move a shoulder; B. counterexample shoulder-line error;
C. real-data shoulder-line angle (the gate-relevant root component); D. hip-centre availability;
E. head-nod sensitivity of a shoulders+nose root on the real SMPL-X mesh nose;
F. nose motion relative to the torso on the real clips; torso lean implied by each root estimator."""
import glob
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

sk = SMPLXSkeleton(model_path="models"); rest = sk.rest_joints
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- A
print("== A. d(shoulder position)/d(rotation of joint j), finite difference, max over 3 axes (m/rad) ==")
for label, base in (("rest pose", np.zeros((55, 3))),
                    ("random pose", np.random.default_rng(0).normal(0, 0.3, (55, 3)))):
    J0 = forward_kinematics(base, sk)
    sens = {}
    for j in range(55):
        g = 0.0
        for a in range(3):
            p = base.copy(); p[j, a] += 1e-6
            g = max(g, np.abs(forward_kinematics(p, sk)[[16, 17]] - J0[[16, 17]]).max() / 1e-6)
        if g > 1e-6: sens[j] = round(g, 4)
    print(f"  {label}: joints that move a shoulder -> {sens}")
anc = []; j = 16
while j >= 0: anc.append(int(j)); j = sk.parents[j]
print(f"  ancestor chain of L shoulder (16): {anc};  arm-IK driven joints: {[b[0] for b in SMPLX_ARM_BONES]}")

# ---------------------------------------------------------------- B
truth = np.zeros((55, 3)); truth[0] = [0, 0, 0.7]
T = forward_kinematics(truth, sk)
L = np.full((33, 3), np.nan)
for s, m in [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]: L[m] = T[s]
F = forward_kinematics(solve_swing_ik(L, sk, SMPLX_ARM_BONES), sk)
def ang(a, b): return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
print("\n== B. Counterexample: the one segment the review's direction check omits ==")
print(f"  shoulder-line (L-R) direction error: {ang(F[16]-F[17], T[16]-T[17]):.3f} deg  (root angle {np.degrees(0.7):.3f} deg)")
print(f"  shoulder width fitted {np.linalg.norm(F[16]-F[17]):.6f} vs truth {np.linalg.norm(T[16]-T[17]):.6f} m")
print(f"  upper arm/forearm lengths equal: {np.allclose([np.linalg.norm(F[a]-F[b]) for a,b in ((16,18),(18,20),(17,19),(19,21))], [np.linalg.norm(T[a]-T[b]) for a,b in ((16,18),(18,20),(17,19),(19,21))], atol=1e-12)}")
print(f"  rest L/R shoulders at z-rest: arm span in T-pose (wrist-to-wrist) {np.linalg.norm(rest[20]-rest[21]):.3f} m -> rotation lever is maximal in T-pose")

# ---------------------------------------------------------------- C, D, F
VIS = 0.5
rest_sl = rest[16] - rest[17]
def yaw_roll(v):
    # MP world: x to subject's left (image right), y down, z away from camera
    return np.degrees(np.arctan2(-v[2], v[0])), np.degrees(np.arctan2(-v[1], v[0]))
print(f"\n== C. Real clips: observed shoulder line vs zero-root rest shoulder line ==")
print(f"  rest shoulder line yaw/roll in MP axes after Rx(180) gauge: {yaw_roll(np.diag([1,-1,-1]) @ rest_sl)}  (identity: {yaw_roll(rest_sl)})")
allang, allyaw, allroll = [], [], []
hip_vis, hip_mid_norm, nose_rel = [], [], {}
with torch.no_grad():
    rest127 = sk.model(return_verts=False).joints[0].numpy().astype(np.float64)
REST_NOSE = rest127[55]
print(f"{'clip':>8s}{'n':>5s}{'angle med':>10s}{'p95':>7s}{'max':>7s}{'|yaw| med':>10s}{'p95':>7s}{'|roll| med':>11s}{'p95':>7s}")
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); P = d["landmarks"].astype(np.float64); V = d["visibility"]
    ok = np.isfinite(P).all(-1) & (V >= VIS)
    clip = Path(path).stem.replace("_world", "")
    sel = ok[:, 11] & ok[:, 12]
    v = P[sel, 11] - P[sel, 12]
    a = np.array([ang(x, rest_sl) for x in v]); yr = np.array([yaw_roll(x) for x in v])
    allang += list(a); allyaw += list(np.abs(yr[:, 0])); allroll += list(np.abs(yr[:, 1]))
    print(f"{clip:>8s}{sel.sum():5d}{np.median(a):10.2f}{np.percentile(a,95):7.2f}{a.max():7.2f}"
          f"{np.median(np.abs(yr[:,0])):10.2f}{np.percentile(np.abs(yr[:,0]),95):7.2f}{np.median(np.abs(yr[:,1])):11.2f}{np.percentile(np.abs(yr[:,1]),95):7.2f}")
    fin = np.isfinite(P[:, 23]).all(-1) & np.isfinite(P[:, 24]).all(-1)
    hip_vis += list((V[fin, 23] >= VIS) & (V[fin, 24] >= VIS))
    hip_mid_norm += list(np.linalg.norm((P[fin, 23] + P[fin, 24]) / 2, axis=-1))
    # F: nose in a torso frame built from shoulders + MP hip centre (origin)
    s2 = sel & ok[:, 0]
    rel = []
    for Lf in P[s2]:
        x = Lf[11] - Lf[12]; x /= np.linalg.norm(x)
        mid = (Lf[11] + Lf[12]) / 2
        up = -mid; up -= (up @ x) * x; up /= np.linalg.norm(up)   # hip -> ... note: up points from shoulders to hips
        up = -up                                                   # now hip -> shoulder ("up" the torso)
        fwd = np.cross(x, up)
        n = Lf[0] - mid
        rel.append([n @ x, n @ up, n @ fwd])
    nose_rel[clip] = np.array(rel)
pa = np.array(allang)
print(f"  pooled: angle median {np.median(pa):.2f} p95 {np.percentile(pa,95):.2f} max {pa.max():.2f} deg; "
      f"|yaw| median {np.median(allyaw):.2f} p95 {np.percentile(allyaw,95):.2f}; |roll| median {np.median(allroll):.2f} p95 {np.percentile(allroll,95):.2f}")
print(f"  frames with shoulder-line angle >= 40.1 deg (the counterexample): {np.mean(pa >= 40.1)*100:.2f}%;  >= 30 deg: {np.mean(pa >= 30)*100:.2f}%;  >= 20 deg: {np.mean(pa>=20)*100:.2f}%")

print("\n== D. Is the MediaPipe hip centre available when hips are 'not visible'? ==")
hv = np.array(hip_vis); hn = np.array(hip_mid_norm)
print(f"  frames with both hips vis>=0.5: {hv.mean()*100:.1f}%;  |hip midpoint| (should be ~0 by MP world convention): median {np.median(hn)*1e3:.2f} mm, max {hn.max()*1e3:.2f} mm, n={len(hn)}")

print("\n== F. Nose position relative to shoulders, in a shoulder + hip-centre torso frame (mm) ==")
print("  axes: x = along shoulder line, up = toward shoulders from hips (orthogonalised), fwd = x cross up")
print(f"{'clip':>8s}{'n':>5s}{'x mean':>8s}{'x std':>7s}{'up mean':>8s}{'up std':>7s}{'fwd mean':>9s}{'fwd std':>8s}{'pitch std':>10s}{'pitch rng':>10s}")
for c, r in nose_rel.items():
    pitch = np.degrees(np.arctan2(r[:, 2], r[:, 1]))
    print(f"{c:>8s}{len(r):5d}{r[:,0].mean()*1e3:8.1f}{r[:,0].std()*1e3:7.1f}{r[:,1].mean()*1e3:8.1f}{r[:,1].std()*1e3:7.1f}"
          f"{r[:,2].mean()*1e3:9.1f}{r[:,2].std()*1e3:8.1f}{pitch.std():10.2f}{np.ptp(pitch):10.2f}")
# same quantity in SMPL-X rest (pelvis as hip proxy, and true hip-joint midpoint)
for lab, hipc in (("pelvis(0)", rest[0]), ("hip-joint midpoint (1,2)", (rest[1] + rest[2]) / 2)):
    x = rest[16] - rest[17]; x /= np.linalg.norm(x); mid = (rest[16] + rest[17]) / 2
    up = mid - hipc; up -= (up @ x) * x; up /= np.linalg.norm(up); fwd = np.cross(x, up)
    n = REST_NOSE - mid
    # SMPL-X is right-handed with +z forward; MP: z away from camera, so the subject's forward is -z.
    # Both frames use the same construction, so compare the nose angle about the shoulder axis with sign by fwd.
    print(f"  SMPL-X rest, hip proxy {lab}: nose (x, up, fwd) = {np.array([n@x, n@up, n@fwd])*1e3} mm, "
          f"pitch {np.degrees(np.arctan2(n@fwd, n@up)):.2f} deg")
allr = np.vstack(list(nose_rel.values()))
print(f"  observed pooled pitch (nose forward of torso axis): median {np.median(np.degrees(np.arctan2(allr[:,2], allr[:,1]))):.2f} deg")
print("  NOTE sign of fwd: x cross up. In SMPL-X rest (x=left, up=+y) fwd=+z = facing direction.")
print("        In MP world (x=left, up=-y) fwd = x cross (-y) = -z = toward camera = facing direction. Same meaning.")

# ---------------------------------------------------------------- E
print("\n== E. Head motion leaks into a shoulders+nose root (real SMPL-X mesh nose, joints[55]) ==")
def frame_from(lsh, rsh, third):
    x = lsh - rsh; x = x / np.linalg.norm(x)
    y = third - (lsh + rsh) / 2; y = y - (y @ x) * x; y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)
F_REST = frame_from(rest[16], rest[17], REST_NOSE)
def t(a): return torch.tensor(a, dtype=torch.float32)[None]
for label, joint_rots in (
        ("nod down 15 (neck 7.5 + head 7.5, about x)", {12: [7.5, 0, 0], 15: [7.5, 0, 0]}),
        ("nod down 30 (neck 15 + head 15)", {12: [15, 0, 0], 15: [15, 0, 0]}),
        ("nod up 20 (neck -10 + head -10)", {12: [-10, 0, 0], 15: [-10, 0, 0]}),
        ("head turn 40 (head yaw, about y)", {15: [0, 40, 0]}),
        ("head tilt 20 (head roll, about z)", {15: [0, 0, 20]}),
        ("neck forward 20 only (about x)", {12: [20, 0, 0]})):
    bp = np.zeros((21, 3))
    for jj, e in joint_rots.items(): bp[jj - 1] = Rotation.from_euler("xyz", e, degrees=True).as_rotvec()
    with torch.no_grad():
        J = sk.model(body_pose=t(bp.reshape(63)), return_verts=False).joints[0].numpy().astype(np.float64)
    R = frame_from(J[16], J[17], J[55]) @ F_REST.T   # true root is identity
    rv = Rotation.from_matrix(R).as_rotvec()
    total = np.degrees(np.linalg.norm(rv))
    print(f"  {label:44s}: spurious torso root {total:6.2f} deg, axis {rv/ (np.linalg.norm(rv)+1e-15)}; "
          f"arm landmarks change? shoulders moved {np.abs(J[[16,17]]-rest127[[16,17]]).max()*1e3:.2e} mm")
