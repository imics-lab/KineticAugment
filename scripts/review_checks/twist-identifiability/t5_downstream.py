"""A003 step 4b: does twist (non-)identifiability matter downstream?

(a) Which local axis is the SMPL-X elbow hinge?  (geometry with the real model)
(b) The repo's joint-limit table (joint_mapping.SMPLX_JOINT_LIMITS) applied with zero noise
    (clamp_joint_angles, which JointAnglePerturbation calls by default) to anatomically
    reasonable real-data parameters: how far does it move the wrist?
(c) Two parameterisations with IDENTICAL joints (zero-twist vs hinge twist) receive the SAME
    additive rotvec noise (JointAnglePerturbation's model): do the augmented landmarks agree?
(d) Is forearm/palm twist observable from MediaPipe at all?  roll of the (wrist,index,pinky)
    plane about the forearm across frames, pose landmarks 15/17/19 (L) and 16/18/20 (R).
(e) Orientation of the DELIVERED mesh in the landmark frame (nose vertex vs shoulders).
"""
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from armlib import REST, ARMS, MODEL, unit, rest_dirs, forward, twist_deg, swing, hinge_ik, frame_rot
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.joint_mapping import clamp_joint_angles, SMPLX_JOINT_LIMITS
import torch

sk = SMPLXSkeleton(model_path="models")
with torch.no_grad():
    REST_NOSE = MODEL.v_template[9120].numpy()

print("(a) SMPL-X left elbow local axes (rest T-pose, palms down; body forward = +z, up = +y)")
J0, V0 = forward(np.zeros((55, 3)))
for name, rv in (("rotvec x +90", [np.pi/2, 0, 0]), ("rotvec y -90", [0, -np.pi/2, 0]),
                 ("rotvec z +90", [0, 0, np.pi/2])):
    p = np.zeros((55, 3)); p[18] = rv
    J, _ = forward(p)
    d = J[20] - J0[20]
    print(f"  {name}: wrist moves {np.linalg.norm(d)*1000:6.1f} mm, direction {np.round(unit(d) if np.linalg.norm(d)>1e-6 else d, 2)}")
print("  -> local x ~ forearm axis (twist), local -y swings forearm forward (+z) = flexion, local z swings it"
      " up/down = varus/valgus. SMPLify's angle prior uses the same convention (penalises +y on left elbow).")
print(f"  repo table left_elbow: {SMPLX_JOINT_LIMITS['left_elbow']}  (labels x 'flexion only', y 'very limited')")

p = np.zeros((55, 3)); p[18] = [0, -np.radians(90), 0]
Jf, _ = forward(p)
q = p.copy(); q[18] = clamp_joint_angles("left_elbow", p[18])
Jc, _ = forward(q)
print(f"  anatomical 90 deg left-elbow flexion {p[18].round(3)} -> clamped {q[18].round(3)}; "
      f"wrist moved {np.linalg.norm(Jc[20]-Jf[20])*1000:.0f} mm by a ZERO-noise clamp")


def torso_rotation(lsh, rsh, nose):
    return frame_rot(REST[16] - REST[17], REST_NOSE - 0.5 * (REST[16] + REST[17]),
                     lsh - rsh, nose - 0.5 * (lsh + rsh))


MP = {"L": (11, 13, 15, 17, 19), "R": (12, 14, 16, 18, 20)}
clamp_move = {"TORSO": [], "HINGE": []}
aug_div = []
roll = {"L": [], "R": []}
vis_hand = []
nose_z_rel = []
head_y_rel = []
rng = np.random.default_rng(1)
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    dat = np.load(path); Ls = dat["landmarks"].astype(np.float64); Vs = dat["visibility"]
    clip_roll = {"L": [], "R": []}
    for fi, (fr, vis) in enumerate(zip(Ls, Vs)):
        ok = np.isfinite(fr).all(1) & (vis >= 0.5)
        vis_hand.append(vis[[17, 18, 19, 20]])
        try:
            fit = fit_body(fr, ok, sk)
        except ValueError:
            continue
        # (e) delivered mesh orientation
        deliv = np.zeros((55, 3)); deliv[0] = fit.global_orient; deliv[1:22] = fit.body_pose.reshape(21, 3)
        if fi % 10 == 0:
            Jd, Vd = forward(deliv, fit.transl)
            sh_mid = 0.5 * (Jd[16] + Jd[17])
            nose_z_rel.append(Vd[9120, 2] - sh_mid[2]); head_y_rel.append(Vd[9120, 1] - sh_mid[1])
        masked = np.where(ok[:, None], fr, np.nan)
        Rt = torso_rotation(fr[11], fr[12], fr[0])
        tor = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=Rt.as_rotvec())
        hin = tor.copy(); have = []
        for side in ("L", "R"):
            a, b, c, idx, pk = MP[side]
            if not (ok[a] and ok[b] and ok[c]):
                continue
            u = unit(fr[b] - fr[a]); v = unit(fr[c] - fr[b])
            ul, vl = Rt.inv().apply(u), Rt.inv().apply(v)
            s, e, w, ax = ARMS[side]
            sols = [sl for sl in hinge_ik(side, ul, vl) if 0 <= sl[1] <= 160]
            if sols:
                hin[s] = sols[0][0].as_rotvec(); hin[e] = ax * np.radians(sols[0][1]); have.append(side)
            # (d) palm roll about the forearm: hand-plane normal projected perpendicular to forearm
            if ok[idx] and ok[pk]:
                n = np.cross(fr[idx] - fr[c], fr[pk] - fr[c])
                n_perp = unit(n - (n @ v) * v)
                ref = unit(np.cross(v, u)) if np.linalg.norm(np.cross(v, u)) > 0.2 else None
                if ref is not None:  # roll relative to the arm plane normal (frame-free)
                    clip_roll[side].append(np.degrees(np.arctan2(np.cross(ref, n_perp) @ v, ref @ n_perp)))
        if not have:
            continue
        Jt, _ = forward(tor); Jh, _ = forward(hin)
        # (b) zero-noise clamp, as JointAnglePerturbation(respect_limits=True) applies
        names = {16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow", 19: "right_elbow"}
        for tag, P, Jbase in (("TORSO", tor, Jt), ("HINGE", hin, Jh)):
            Q = P.copy()
            for j, nm in names.items():
                Q[j] = clamp_joint_angles(nm, P[j])
            Jq, _ = forward(Q)
            clamp_move[tag].append(max(np.linalg.norm(Jq[k] - Jbase[k]) for k in (20, 21)))
        # (c) same additive noise (stddev 0.1 rad, the JointAnglePerturbation default), no clamp
        for _ in range(3):
            noise = {j: rng.normal(0, 0.1, 3) for j in (16, 17, 18, 19)}
            A = tor.copy(); B = hin.copy()
            for j, nz in noise.items():
                A[j] = A[j] + nz; B[j] = B[j] + nz
            Ja, _ = forward(A); Jb, _ = forward(B)
            for side in have:
                w = ARMS[side][2]
                aug_div.append((np.linalg.norm(Ja[w] - Jb[w]), np.linalg.norm(Ja[w] - Jt[w])))
    for side in ("L", "R"):
        if clip_roll[side]:
            r = np.unwrap(np.radians(clip_roll[side]))
            roll[side].append((path[-15:-10], np.degrees(np.std(r)), np.degrees(np.ptp(r)), len(r)))

print("\n(b) zero-noise joint-limit clamp (repo table) on real-data parameters, max wrist displacement")
for tag, m in clamp_move.items():
    m = np.array(m) * 1000
    print(f"  {tag:<6} n={len(m)}: median {np.median(m):.0f} mm, p95 {np.percentile(m,95):.0f} mm, "
          f"frac > 50 mm {np.mean(m>50):.2f}")
print("\n(c) same rotvec noise (sd 0.1 rad) applied to two parameterisations with identical joints")
D = np.array(aug_div) * 1000
print(f"  n={len(D)}: wrist disagreement between the two augmented results: median {np.median(D[:,0]):.1f} mm, "
      f"p95 {np.percentile(D[:,0],95):.1f} mm; for scale, the augmentation's own wrist displacement median "
      f"{np.median(D[:,1]):.1f} mm; ratio median {np.median(D[:,0]/D[:,1]):.2f}")
print("\n(d) palm-plane roll about the forearm (MediaPipe pose 15/17/19, 16/18/20), per clip")
for side in ("L", "R"):
    for clip, sd, ptp, n in roll[side]:
        print(f"  {side} {clip}: n={n:3d} roll std {sd:5.1f} deg, range {ptp:6.1f} deg")
vh = np.array(vis_hand)
print(f"  visibility of pose landmarks 17-20 (pinky/index): median {np.median(vh):.2f}, frac >= 0.5 {np.mean(vh>=0.5):.2f}")
print("\n(e) DELIVERED mesh in the landmark frame (MediaPipe world: y down, -z toward camera)")
nz = np.array(nose_z_rel); hy = np.array(head_y_rel)
print(f"  nose vertex minus shoulder midpoint: y median {np.median(hy):+.3f} m (negative = above), "
      f"z median {np.median(nz):+.3f} m (negative = toward camera); frac nose BELOW shoulders {np.mean(hy>0):.2f}, "
      f"frac nose facing AWAY from camera {np.mean(nz>0):.2f}")
