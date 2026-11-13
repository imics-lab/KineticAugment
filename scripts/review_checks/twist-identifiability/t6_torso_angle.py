"""Size of the torso-frame rotation (shoulders+nose vs SMPL-X rest) on real clips, and its
distance from a pure 180-deg flip about x (the MediaPipe-world -> SMPL-X axis change)."""
import glob, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from armlib import REST, MODEL, frame_rot
with torch.no_grad():
    RN = MODEL.v_template[9120].numpy()
ang, dflip = [], []
flip = R.from_euler("x", 180, degrees=True)
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(p)
    for fr, v in zip(d["landmarks"].astype(float), d["visibility"]):
        if min(v[0], v[11], v[12]) < 0.5: continue
        Rt = frame_rot(REST[16]-REST[17], RN-0.5*(REST[16]+REST[17]), fr[11]-fr[12], fr[0]-0.5*(fr[11]+fr[12]))
        ang.append(np.degrees(Rt.magnitude())); dflip.append(np.degrees((flip.inv()*Rt).magnitude()))
print(f"n={len(ang)} torso-frame rotation angle median {np.median(ang):.1f} deg [p5 {np.percentile(ang,5):.1f}, p95 {np.percentile(ang,95):.1f}]; "
      f"distance from pure 180-deg x-flip median {np.median(dflip):.1f} deg, p95 {np.percentile(dflip,95):.1f}")
