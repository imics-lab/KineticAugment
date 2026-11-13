# Same additive rotvec noise on two parameter sets with identical joints (differ only by a compensated humerus twist).
import glob, numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body
sk = SMPLXSkeleton(model_path="models"); rest = sk.rest_joints
FLIP = R.from_euler('x', 180, degrees=True).as_matrix()
rng = np.random.default_rng(0)
fits = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f)
    for lm, vis in zip(d["landmarks"].astype(float), d["visibility"]):
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        if not ok[[11, 12, 13, 14, 15, 16]].all(): continue
        try: fit = fit_body(lm @ FLIP.T, ok, sk)
        except ValueError: continue
        r = np.zeros((55, 3)); r[1:22] = fit.body_pose.reshape(21, 3); fits.append(r)
print("frames with all 6 arm landmarks:", len(fits))
for th in (10, 30):
    diff, disp, jchk = [], [], []
    for r in fits:
        b = r.copy()
        for S, E in ((16, 18), (17, 19)):
            u = rest[E]-rest[S]; u /= np.linalg.norm(u)
            T = R.from_rotvec(np.radians(th)*u)
            b[S] = (R.from_rotvec(r[S])*T).as_rotvec(); b[E] = (T.inv()*R.from_rotvec(r[E])).as_rotvec()
        Ja, Jb = forward_kinematics(r, sk), forward_kinematics(b, sk)
        jchk.append(np.abs(Ja-Jb).max())
        for _ in range(5):
            n = np.zeros((55, 3)); n[[16, 17, 18, 19]] = rng.normal(0, 0.1, (4, 3))
            Na, Nb = forward_kinematics(r+n, sk), forward_kinematics(b+n, sk)
            for w in (20, 21):
                diff.append(np.linalg.norm(Na[w]-Nb[w])); disp.append(np.linalg.norm(Na[w]-Ja[w]))
    diff, disp = np.array(diff)*1000, np.array(disp)*1000
    print(f"twist {th:2d} deg: joints identical to {max(jchk):.1e} m; augmented wrists differ median {np.median(diff):.1f} mm "
          f"p95 {np.percentile(diff,95):.1f} mm; augmentation displacement median {np.median(disp):.1f} mm; ratio {np.median(diff)/np.median(disp):.2f}")
