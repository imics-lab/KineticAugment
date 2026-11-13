"""E5: Are SMPL-X *local* elbow angles from the V1 converter anatomically meaningful?
Intrinsic augmentation (JointAnglePerturbation + joint limits) acts on local axis-angle values.
Under the zero-twist policy the elbow's local frame is fixed by the shoulder's swing only, so the
elbow rotation need not be a hinge rotation about SMPL-X's flexion axis (local y; see E2b).
Measures, on real world landmarks via converter.fit_body:
  - elbow local rotation decomposed into swing-twist about the forearm rest axis (x) and the
    flexion-axis share: |component about y| / |rotvec|
  - how much shoulder twist would make the elbow a pure hinge (shows twist IS identifiable from
    a bent elbow, so 'twist = 0' is a choice that breaks local-angle semantics, not a free default)."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body

sk = SMPLXSkeleton(model_path="models")
share_y, off_hinge_deg, bend_deg = [], [], []
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        try:
            f = fit_body(L[t], ok, sk)
        except ValueError:
            continue
        bp = f.body_pose.reshape(21, 3)
        for elbow_j, wrist_mp in ((18, 15), (19, 16)):
            if not ok[wrist_mp]:
                continue
            rv = bp[elbow_j - 1]
            ang = np.linalg.norm(rv)
            if ang < np.radians(15):      # nearly straight arm: hinge axis ill-defined
                continue
            axis = rv / ang
            share_y.append(abs(axis[1]))
            off_hinge_deg.append(np.degrees(np.arccos(min(1.0, abs(axis[1])))))
            bend_deg.append(np.degrees(ang))
sy, oh, bd = map(np.array, (share_y, off_hinge_deg, bend_deg))
print(f"bent elbows (>15 deg) analysed: {len(sy)}; local rotation magnitude median {np.median(bd):.1f} deg")
print(f"angle between elbow rotation axis and SMPL-X flexion axis (local y): median {np.median(oh):.1f} deg, "
      f"p5 {np.percentile(oh,5):.1f}, p95 {np.percentile(oh,95):.1f}")
print(f"fraction of bent elbows whose axis is >30 deg off the flexion axis: {np.mean(oh>30)*100:.1f}%")
