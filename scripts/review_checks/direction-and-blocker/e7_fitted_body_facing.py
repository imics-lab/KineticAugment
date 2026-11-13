"""E7: follow-up to E6. Which way does the CURRENT converter's fitted SMPL-X body face, and with
which sign does it bend the elbows?  MediaPipe world: x image-right, y down, z away from camera
(signer faces camera => body front ~ -z, body up ~ -y). SMPL-X canonical: y up, z = body front.
If the fitted body faces away (front ~ +z), arms that bend toward the camera are encoded as elbow
HYPER-extension, so every local-angle augmentation and joint limit acts with the wrong sign."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body

sk = SMPLXSkeleton(model_path="models")
fwd_z, up_y, ly, ry, per_clip = [], [], [], [], {}
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    fz = []
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        try:
            f = fit_body(L[t], ok, sk)
        except ValueError:
            continue
        Rg = R.from_rotvec(f.global_orient)
        fz.append(Rg.apply([0, 0, 1])[2]); up_y.append(Rg.apply([0, 1, 0])[1])
        bp = f.body_pose.reshape(21, 3)
        if ok[15]: ly.append(bp[17, 1])
        if ok[16]: ry.append(bp[18, 1])
    per_clip[c] = np.array(fz); fwd_z += fz
fz, uy, ly, ry = map(np.array, (fwd_z, up_y, ly, ry))
print(f"frames: {len(fz)}")
print(f"fitted body FRONT, z-component in MediaPipe world (camera-facing => ~ -1): median {np.median(fz):+.2f}; "
      f"fraction facing away (z>0): {np.mean(fz>0)*100:.1f}%")
for c, v in per_clip.items():
    print(f"   {c}: median front_z {np.median(v):+.2f}, facing away {np.mean(v>0)*100:.0f}%")
print(f"fitted body UP, y-component (upright => ~ -1): median {np.median(uy):+.2f}")
print(f"left elbow local-y (flexion is NEGATIVE for left, per E2b/E6): median {np.degrees(np.median(ly)):+.1f} deg, "
      f"fraction positive {np.mean(ly>0)*100:.1f}%")
print(f"right elbow local-y (flexion is POSITIVE for right): median {np.degrees(np.median(ry)):+.1f} deg, "
      f"fraction negative {np.mean(ry<0)*100:.1f}%")
