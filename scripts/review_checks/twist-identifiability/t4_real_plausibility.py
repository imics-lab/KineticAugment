"""A003 step 4: anatomical plausibility of the analytic-IK parameters on the five real clips.

Three parameterisations of the SAME observed arm directions:
  DELIVERED  fit_body() exactly as shipped (root from the rigid fit; swing IK solved with root = I).
  TORSO      swing IK solved relative to a torso frame estimated from L/R shoulder + nose, matched
             to SMPL-X's rest shoulders + nose vertex 9120 (a proxy for the Step 8 design).
  HINGE      same torso frame, but humerus twist chosen so the elbow is a pure flexion hinge
             (the anatomical member of the equivalence class), flexion branch.
All three reproduce the same elbow/wrist bone directions; they differ only in twist.
"""
import glob
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as R
from armlib import REST, ARMS, MODEL, unit, rest_dirs, forward, geo_deg, twist_deg, swing, hinge_ik, frame_rot
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.constraints.joint_limits import JointLimitConstraint
import torch

sk = SMPLXSkeleton(model_path="models")
with torch.no_grad():
    REST_NOSE = MODEL.v_template[9120].numpy()
MP = {"L": (11, 13, 15), "R": (12, 14, 16)}
NOSE = 0
jl = JointLimitConstraint()


def torso_rotation(lsh, rsh, nose, rl, rr, rn):
    """rotation taking rest (shoulders, nose) frame to observed: x = R->L shoulder, 2nd = mid->nose."""
    return frame_rot(rl - rr, rn - 0.5 * (rl + rr), lsh - rsh, nose - 0.5 * (lsh + rsh))


def arm_metrics(side, rot55, R_parent, u, v):
    """metrics for one arm given local rotations rot55 and the shoulder's parent world rotation."""
    s, e, w, ax = ARMS[side]
    h, f = rest_dirs(side)
    Rsh = R.from_rotvec(rot55[s]); Rel = R.from_rotvec(rot55[e])
    # humerus direction in the TORSO frame (parent frame of the shoulder)
    d = Rsh.apply(h)
    elev = np.degrees(np.arccos(np.clip(np.dot(d, [0, -1, 0]), -1, 1)))  # 0 = hanging down
    g = Rel.apply(f)  # forearm in humerus-parent frame
    offhinge = abs(np.degrees(np.arcsin(np.clip(g @ ax, -1, 1)) - np.arcsin(np.clip(f @ ax, -1, 1))))
    rv = rot55[e]; flex_side = np.dot(rv, ax)  # >0 flexion side, <0 hyperextension side
    return dict(sh_angle=np.degrees(np.linalg.norm(rot55[s])), elev=elev, fwd=d[2],
                el_angle=np.degrees(np.linalg.norm(rv)), offhinge=offhinge, flex_side=flex_side,
                bend=np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


rows = {k: {"L": [], "R": []} for k in ("DELIVERED", "TORSO", "HINGE")}
viol = {k: Counter() for k in ("DELIVERED", "TORSO", "HINGE")}
nframes = {k: 0 for k in viol}
hinge_info = {"L": [], "R": []}
temporal = {"L": [], "R": []}
mesh_samples = []
go_deg = []

for path in sorted(glob.glob("data/test_output/*_world.npz")):
    dat = np.load(path); Ls = dat["landmarks"].astype(np.float64); Vs = dat["visibility"]
    prev = None
    for fi, (fr, vis) in enumerate(zip(Ls, Vs)):
        ok = np.isfinite(fr).all(1) & (vis >= 0.5)
        try:
            fit = fit_body(fr, ok, sk)
        except ValueError:
            prev = None
            continue
        masked = np.where(ok[:, None], fr, np.nan)
        deliv = np.zeros((55, 3)); deliv[0] = fit.global_orient; deliv[1:22] = fit.body_pose.reshape(21, 3)
        go_deg.append(np.degrees(np.linalg.norm(fit.global_orient)))
        # TORSO frame
        Rt = torso_rotation(fr[11], fr[12], fr[NOSE], REST[16], REST[17], REST_NOSE)
        tor = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=Rt.as_rotvec())
        hin = tor.copy()
        for name, rot55, Rpar in (("DELIVERED", deliv, R.identity()), ("TORSO", tor, Rt)):
            nframes[name] += 1
            for vv in jl.check({"body_pose": rot55[1:22].reshape(1, 63)}):
                viol[name][(vv.joint_name, vv.axis)] += 1
        for side in ("L", "R"):
            a, b, c = MP[side]
            if not (ok[a] and ok[b] and ok[c]):
                continue
            u = unit(fr[b] - fr[a]); v = unit(fr[c] - fr[b])
            # delivered: shoulder parent frame is identity (solve ran with root = I)
            rows["DELIVERED"][side].append(arm_metrics(side, deliv, R.identity(), u, v))
            ul, vl = Rt.inv().apply(u), Rt.inv().apply(v)
            rows["TORSO"][side].append(arm_metrics(side, tor, Rt, ul, vl))
            sols = hinge_ik(side, ul, vl)
            s, e, w, ax = ARMS[side]
            h, f = rest_dirs(side)
            flex = [sl for sl in sols if 0 <= sl[1] <= 160]
            if flex:
                Rs, ph = flex[0]
                psi = twist_deg(swing(h, ul).inv() * Rs, h)
                hin[s] = Rs.as_rotvec(); hin[e] = ax * np.radians(ph)
                hinge_info[side].append((psi, ph, True))
            else:
                hinge_info[side].append((np.nan, np.nan, False))
            # temporal: change in local rotations vs change in bone directions (TORSO param.)
            if prev is not None and prev[0] == fi - 1 and side in prev[1]:
                pu, pv, prot = prev[1][side]
                dir_change = max(np.degrees(np.arccos(np.clip(u @ pu, -1, 1))),
                                 np.degrees(np.arccos(np.clip(v @ pv, -1, 1))))
                rot_change = max(geo_deg(tor[s], prot[s]), geo_deg(tor[e], prot[e]))
                temporal[side].append((dir_change, rot_change))
        nframes["HINGE"] += 1
        for vv in jl.check({"body_pose": hin[1:22].reshape(1, 63)}):
            viol["HINGE"][(vv.joint_name, vv.axis)] += 1
        for side in ("L", "R"):
            if hinge_info[side] and hinge_info[side][-1][2]:
                pass
        prev = (fi, {sd: (unit(fr[MP[sd][1]] - fr[MP[sd][0]]), unit(fr[MP[sd][2]] - fr[MP[sd][1]]), tor.copy())
                     for sd in ("L", "R") if all(ok[i] for i in MP[sd])})
        if fi % 20 == 0:
            mesh_samples.append((path[-15:-10], fi, tor.copy(), hin.copy()))


def q(a, p):
    return np.nanpercentile(a, p)


print(f"frames fitted {nframes['DELIVERED']}; delivered |global_orient| median {np.median(go_deg):.1f} deg "
      f"(a y-down, camera-facing input needs ~180 deg)")
for name in ("DELIVERED", "TORSO"):
    print(f"\n=== {name} parameterisation ===")
    for side in ("L", "R"):
        r = rows[name][side]
        A = {k: np.array([x[k] for x in r]) for k in r[0]}
        print(f" {side} arm (n={len(r)}): elbow bend (bone angle, 0=straight) med {np.median(A['bend']):.0f} "
              f"[p5 {q(A['bend'],5):.0f}, p95 {q(A['bend'],95):.0f}]; frac bend<25deg {np.mean(A['bend']<25):.2f}")
        print(f"   shoulder |rotvec| med {np.median(A['sh_angle']):.0f} p95 {q(A['sh_angle'],95):.0f} max {A['sh_angle'].max():.0f} deg;"
              f" humerus elevation from body-down med {np.median(A['elev']):.0f} [p5 {q(A['elev'],5):.0f}, p95 {q(A['elev'],95):.0f}];"
              f" frac above horizontal {np.mean(A['elev']>90):.2f}; frac pointing behind body (fwd<-0.2) {np.mean(A['fwd']<-0.2):.2f}")
        print(f"   elbow |rotvec| med {np.median(A['el_angle']):.0f} max {A['el_angle'].max():.0f} deg; off-hinge (varus/valgus) "
              f"med {np.median(A['offhinge']):.0f} p95 {q(A['offhinge'],95):.0f} max {A['offhinge'].max():.0f} deg; "
              f"frac >20deg {np.mean(A['offhinge']>20):.2f}; frac on hyperextension side {np.mean(A['flex_side']<0):.2f}")
print("\n=== HINGE (anatomical twist) member of the same equivalence class, TORSO frame ===")
for side in ("L", "R"):
    H = np.array(hinge_info[side], dtype=float)
    okh = H[:, 2] > 0
    psi = H[okh, 0]; ph = H[okh, 1]
    print(f" {side}: hinge-solvable with flexion in [0,160] deg: {okh.mean():.2f} of {len(H)} frames")
    print(f"   humerus twist the zero-twist policy discards: |psi| med {np.median(np.abs(psi)):.0f} "
          f"p95 {q(np.abs(psi),95):.0f} max {np.abs(psi).max():.0f} deg; frac |psi|>90 {np.mean(np.abs(psi)>90):.2f}, "
          f">120 {np.mean(np.abs(psi)>120):.2f}")
    print(f"   hinge flexion med {np.median(ph):.0f} [p5 {q(ph,5):.0f}, p95 {q(ph,95):.0f}] deg")
print("\n=== repo JointLimitConstraint.check violations (frames with >=1 violation on that joint/axis) ===")
for name in viol:
    top = ", ".join(f"{j}.{a}:{n}" for (j, a), n in sorted(viol[name].items(), key=lambda t: -t[1]))
    print(f" {name:<9} n={nframes[name]}: {top}")
print("\n=== temporal stability, TORSO param. (consecutive frames) ===")
for side in ("L", "R"):
    T = np.array(temporal[side])
    ratio = T[:, 1] / np.maximum(T[:, 0], 1e-3)
    print(f" {side}: n={len(T)} bone-direction change med {np.median(T[:,0]):.1f} max {T[:,0].max():.1f} deg; "
          f"local-rotation change med {np.median(T[:,1]):.1f} max {T[:,1].max():.1f} deg; "
          f"frames with rot change > 3x dir change and >10deg: {int(np.sum((ratio>3)&(T[:,1]>10)))}")
print("\n=== mesh: TORSO (zero twist) vs HINGE (anatomical twist), same arm joints ===")
for clip, fi, a, b in mesh_samples[:12]:
    Ja, Va = forward(a); Jb, Vb = forward(b)
    dj = max(np.linalg.norm(Ja[k] - Jb[k]) for k in (16, 17, 18, 19, 20, 21))
    dv = np.linalg.norm(Va - Vb, axis=1)
    print(f"  {clip} f{fi:03d}: max arm-joint diff {dj*1000:.4f} mm | vertex diff max {dv.max()*1000:6.1f} mm, "
          f"verts moved >10 mm {int((dv>0.01).sum())}")
