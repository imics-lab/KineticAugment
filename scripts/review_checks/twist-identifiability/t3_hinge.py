"""A003 step 2b: which twists are identifiable?  Real SMPL-X forward (float64).

(1) Under SMPL-X's native parameterisation (3-DOF ball elbow) -> see t1/t1b: humerus twist AND
    forearm twist are both invisible to all 55 joints.
(2) Under an ANATOMICAL hinge elbow (elbow = rotation about its flexion axis only), humerus twist
    becomes identifiable from elbow+wrist positions when the elbow is bent: show the solution set
    is exactly two discrete branches (flexion vs mirror), not a continuum.
(3) Straight-elbow degeneracy: sensitivity of the recovered humerus twist to landmark noise as a
    function of elbow bend; and whether the SMPL-X rest geometry ever lets the bones be collinear
    under the hinge.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from armlib import REST, ARMS, LS, LE, LW, unit, rest_dirs, forward, geo_deg, twist_deg, swing, hinge_ik

h, f = rest_dirs("L")
ax = ARMS["L"][3]
print(f"rest L humerus dir {np.round(h,3)}  forearm dir {np.round(f,3)}  "
      f"rest angle between bones {np.degrees(np.arccos(h@f)):.1f} deg")
print(f"  components along hinge axis (-y): humerus {h@ax:+.3f}, forearm {f@ax:+.3f}")
# under the hinge, angle(h, R(phi) f) ranges over [min,max]; collinear only if min == 0
phis = np.radians(np.linspace(-180, 180, 7201))
angs = np.array([np.degrees(np.arccos(np.clip(h @ R.from_rotvec(ax * p).apply(f), -1, 1))) for p in phis])
print(f"  under a pure hinge the humerus-forearm angle spans [{angs.min():.2f}, {angs.max():.2f}] deg; "
      f"minimum (straightest arm) at phi = {np.degrees(phis[angs.argmin()]):+.1f} deg "
      f"({'hyperextension' if phis[angs.argmin()] < 0 else 'flexion'})")

# ---- (2) uniqueness under hinge, verified with the real model -------------------------------------
print("\n(2) hinge-elbow arm: solve (shoulder 3-DOF, elbow flexion 1-DOF) from elbow+wrist positions")
rng = np.random.default_rng(0)
for trial in range(5):
    p = np.zeros((55, 3))
    p[LS] = R.from_euler("zyx", rng.uniform([-80, -40, -60], [10, 40, 60]), degrees=True).as_rotvec()
    phi_true = rng.uniform(20, 130)
    p[LE] = ax * np.radians(phi_true)
    J, _ = forward(p)
    u = unit(J[LE] - J[LS]); v = unit(J[LW] - J[LE])
    sols = hinge_ik("L", u, v)
    line = f"  truth flex {phi_true:6.1f}:"
    for Rs, phi in sols:
        q = np.zeros((55, 3)); q[LS] = Rs.as_rotvec(); q[LE] = ax * np.radians(phi)
        Jq, _ = forward(q)
        perr = max(np.linalg.norm(Jq[k] - J[k]) for k in (LS, LE, LW))
        line += (f" | branch flex {phi:+7.1f} deg, shoulder dev from truth {geo_deg(p[LS], Rs.as_rotvec()):6.1f} deg,"
                 f" pos err {perr:.1e} m")
    print(line)
print("  -> exactly two hinge solutions reproduce the positions: the true one (0 deg deviation) and a mirror"
      " branch that bends the elbow the other way; joint limits (no hyperextension) select one.")

# continuum check: twist humerus by psi with the elbow held on its hinge -> wrist moves
p = np.zeros((55, 3))
p[LS] = R.from_euler("z", -60, degrees=True).as_rotvec(); p[LE] = ax * np.radians(90)
J0, _ = forward(p)
print("  hinge held, humerus twisted (no compensation): wrist displacement")
for psi in (1, 5, 10, 30):
    q = p.copy(); q[LS] = (R.from_rotvec(p[LS]) * R.from_rotvec(np.radians(psi) * h)).as_rotvec()
    J, _ = forward(q)
    print(f"    psi {psi:3d} deg -> wrist moves {np.linalg.norm(J[LW]-J0[LW])*1000:6.1f} mm, elbow {np.linalg.norm(J[LE]-J0[LE])*1000:.3f} mm")

# ---- (3) straight-elbow degeneracy: twist sensitivity to noise vs bend ---------------------------
print("\n(3) humerus-twist recoverability vs elbow bend (hinge model, isotropic landmark noise)")
print("    bend = angle between humerus and forearm bone vectors (0 = straight)")
L1 = np.linalg.norm(REST[LE] - REST[LS]); L2 = np.linalg.norm(REST[LW] - REST[LE])
for sigma_mm in (2.0, 5.0):
    print(f"  sigma = {sigma_mm} mm per landmark, 400 draws per row")
    for target_bend in (5, 10, 15, 25, 45, 90):
        # pick phi giving this bend (on the flexion side)
        k = np.argmin(np.abs(angs - target_bend) + 1e3 * (phis < 0))
        phi = phis[k]
        p = np.zeros((55, 3)); p[LS] = R.from_euler("z", -60, degrees=True).as_rotvec(); p[LE] = ax * phi
        J, _ = forward(p)
        u0 = unit(J[LE] - J[LS]); v0 = unit(J[LW] - J[LE])
        sw0 = swing(h, u0)
        tw_true = twist_deg(sw0.inv() * R.from_rotvec(p[LS]), h)
        errs = []; nsol = 0
        for _ in range(400):
            Jn = {k2: J[k2] + rng.normal(0, sigma_mm / 1000, 3) for k2 in (LS, LE, LW)}
            u = unit(Jn[LE] - Jn[LS]); v = unit(Jn[LW] - Jn[LE])
            sols = hinge_ik("L", u, v)
            if not sols:
                continue
            nsol += 1
            # choose the flexion branch closest to the truth's flexion sign
            Rs, ph = min(sols, key=lambda s: abs(s[1] - np.degrees(phi)))
            tw = twist_deg(swing(h, u).inv() * Rs, h)
            errs.append((tw - tw_true + 180) % 360 - 180)
        errs = np.array(errs)
        print(f"    bend {angs[k]:5.1f} deg (flex {np.degrees(phi):6.1f}): humerus-twist error "
              f"std {errs.std():6.1f} deg, |err| p95 {np.percentile(np.abs(errs),95):6.1f} deg, "
              f"solvable {nsol}/400")
