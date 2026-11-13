"""(a) Size of the global_orient jump if the nose drops out and Step 8 falls back to the shoulder-only
minimal rotation; (b) are body_pose arm rotations torso-relative? wrist height relative to the shoulder in
the fitted body's canonical frame (global_orient removed) vs the observed wrist height relative to the shoulder."""
import glob, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, sys.argv[1])
import conftest  # noqa
import step8_proto as P8
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.converter import fit_body
sk = SMPLXSkeleton(model_path="models"); REF = sys.argv[2] if len(sys.argv) > 2 else "hip"; print("reference up-axis:", REF)
jump, rows = [], {"shipped": [], "nose": [], "hip": []}
obs = []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); Pm = d["landmarks"].astype(np.float64); V = d["visibility"]
    okall = np.isfinite(Pm).all(-1) & (V >= 0.5)
    for L, ok in zip(Pm, okall):
        try:
            fs = fit_body(L, ok, sk); fn = P8.fit_body(L, ok, sk, apex="nose")
            fh = P8.fit_body(L, ok, sk, apex="hip"); f0 = P8.fit_body(L, ok, sk, apex=None)
        except ValueError:
            continue
        jump.append(np.degrees((Rotation.from_rotvec(fn.global_orient).inv() * Rotation.from_rotvec(f0.global_orient)).magnitude()))
        if not ok[15]: continue
        # observed: left wrist below left shoulder along the observed torso 'up' (hip-centre -> shoulder mid)
        up = np.array([0.0, -1.0, 0.0]) if REF == "camera" else (L[11] + L[12]) / 2 / np.linalg.norm((L[11] + L[12]) / 2)
        obs.append((L[15] - L[11]) @ up)
        for k, f in (("shipped", fs), ("nose", fn), ("hip", fh)):
            rots = np.zeros((55, 3)); rots[1:22] = f.body_pose.reshape(21, 3)   # canonical: global_orient removed
            J = forward_kinematics(rots, sk)
            rows[k].append(J[20, 1] - J[16, 1])   # SMPL-X canonical +y = up the torso
j = np.array(jump)
print(f"(a) nose-apex root vs shoulder-only fallback root: median {np.median(j):.1f} deg, min {j.min():.1f}, max {j.max():.1f} (n={len(j)})")
o = np.array(obs)
print(f"(b) left wrist height above left shoulder along the torso axis (m): observed median {np.median(o):+.3f}")
for k, v in rows.items():
    v = np.array(v)
    print(f"    {k:8s} body_pose in canonical frame: median {np.median(v):+.3f}; sign agrees with observation in {np.mean(np.sign(v)==np.sign(o))*100:.1f}% of frames; corr {np.corrcoef(v, o)[0,1]:+.3f}")
