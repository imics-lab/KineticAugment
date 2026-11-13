"""Extra checks for the fk-and-wiring cluster (run against the refreshed snapshot).

E1  "invalid landmarks never influence the fit": exhaustive over validity masks, bitwise.
E2  orphan correspondences: a valid landmark whose bone cannot be solved still enters the
    rigid alignment at its REST position. Synthetic magnitude + real-data frequency/effect.
E3  doc Step 7 table: synthetic root unrotated exact, root rotated 30 deg "70 mm worst".
E4  hidden FK assumption: expression = 0 (SMPL-X expression blendshapes shift joints).
E5  gender coverage / betas length handling of SMPLXSkeleton.
E6  rotated landmark frame: fold is exact, but is the frame estimate right? (truth error)
E7  DeprecationWarning visibility under default filters when called from a library module.
"""
import glob
import importlib.util
import itertools
import os
import subprocess
import sys
import textwrap

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

torch.set_grad_enabled(False)
rng = np.random.default_rng(2026)
ARM_MP = [m for _, m in ARM_CORRESPONDENCE]
skel = SMPLXSkeleton(model_path="models")


def synth(sk, root=np.zeros(3), R=np.eye(3), t=np.zeros(3), sig=0.4):
    rots = np.zeros((55, 3)); rots[0] = root
    for j, *_ in SMPLX_ARM_BONES:
        rots[j] = rng.normal(0, sig, 3)
    J = forward_kinematics(rots, sk) @ R.T + t
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = J[s]
    return lm, np.isfinite(lm).all(1)


def fit_arrays(f):
    return np.concatenate([f.body_pose, f.global_orient, f.transl, f.betas])


# ---------------------------------------------------------------- E1
print("=== E1: invalid landmark VALUES never influence the fit (bitwise) ===")
n_masks = n_cmp = n_diff = n_raise_mismatch = 0
for bits in itertools.product([0, 1], repeat=6):
    if sum(bits) < 3:
        continue
    n_masks += 1
    for trial in range(5):
        lm, _ = synth(skel, t=rng.normal(0, 1, 3))
        v = np.zeros(33, bool)
        for b, m in zip(bits, ARM_MP):
            v[m] = bool(b)
        base = fit_arrays(fit_body(lm, v, skel))
        for junk in (np.nan, 0.0, 99.0, None):
            lm2 = lm.copy()
            for b, m in zip(bits, ARM_MP):
                if not b:
                    lm2[m] = rng.normal(0, 5, 3) if junk is None else junk
            # also corrupt every non-arm landmark
            lm2[[i for i in range(33) if i not in ARM_MP]] = rng.normal(0, 5, (27, 3))
            other = fit_arrays(fit_body(lm2, v, skel))
            n_cmp += 1
            n_diff += int(not np.array_equal(base, other))
print(f"  masks with >=3 valid: {n_masks}; comparisons: {n_cmp}; fits that differed bitwise: {n_diff}")

# ---------------------------------------------------------------- E2
print("\n=== E2: orphan correspondences (valid landmark whose bone is unsolved) ===")
cases = {
    "L sh, R sh, L wrist only (no bone solvable)": [11, 12, 15],
    "all but L elbow (L wrist orphaned)": [11, 12, 14, 15, 16],
    "all but L shoulder (L elbow, L wrist orphaned)": [12, 13, 14, 15, 16],
    "all but R shoulder": [11, 13, 14, 15, 16],
}
for name, valid in cases.items():
    errs_obs, errs_obs_drop = [], []
    for _ in range(200):
        lm, _ = synth(skel, t=rng.normal(0, 1, 3))
        v = np.zeros(33, bool); v[valid] = True
        out = reconstruct_body(fit_body(lm, v, skel), skel)
        errs_obs.append(max(np.linalg.norm(out[m] - lm[m]) for m in valid))
    print(f"  {name:48s}: max error over the VALID landmarks: median {np.median(errs_obs)*1000:6.1f} mm, "
          f"max {np.max(errs_obs)*1000:6.1f} mm (raises: no)")

# real-data frequency and effect
files = sorted(glob.glob("data/test_output/*_world.npz"))
n_frames = n_orphan = 0
eff = []
solvable = {13: (11, 13), 15: (13, 15), 14: (12, 14), 16: (14, 16)}  # landmark -> bone it needs
for path in files:
    d = np.load(path)
    pose = d["landmarks"].astype(np.float64)
    ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    for fr, ok in zip(pose, ok_all):
        if sum(ok[m] for m in ARM_MP) < 3:
            continue
        n_frames += 1
        # an elbow/wrist is "orphaned" when valid but its bone (and, transitively, the parent bone) unsolved
        orphans = []
        for m, (a, b) in solvable.items():
            if ok[m] and not (ok[a] and ok[b]):
                orphans.append(m)
        # a wrist whose elbow bone is solved but upper arm is not is still positioned by an
        # unsolved upper arm -> also orphaned in effect
        for w, e, s in ((15, 13, 11), (16, 14, 12)):
            if ok[w] and ok[e] and not ok[s] and w not in orphans:
                orphans.append(w)
        if not orphans:
            continue
        n_orphan += 1
        try:
            out_a = reconstruct_body(fit_body(fr, ok, skel), skel)
        except ValueError:
            continue
        ok_b = ok.copy(); ok_b[orphans] = False
        try:
            out_b = reconstruct_body(fit_body(fr, ok_b, skel), skel)
        except ValueError:
            continue
        keep = [m for m in ARM_MP if ok_b[m]]
        sw = np.linalg.norm(fr[11] - fr[12]) if ok[11] and ok[12] else 0.33
        ea = np.median([np.linalg.norm(out_a[m] - fr[m]) for m in keep]) / sw * 100
        eb = np.median([np.linalg.norm(out_b[m] - fr[m]) for m in keep]) / sw * 100
        eo = np.median([np.linalg.norm(out_a[m] - fr[m]) for m in orphans]) / sw * 100
        eff.append((ea, eb, eo))
eff = np.array(eff) if eff else np.zeros((0, 3))
print(f"  real clips: {n_orphan} of {n_frames} fittable frames contain an orphaned correspondence")
if len(eff):
    print(f"    on those frames, median error of the NON-orphan landmarks: with orphan in alignment "
          f"{np.median(eff[:,0]):.2f}% sw vs orphan excluded {np.median(eff[:,1]):.2f}% sw "
          f"(frames where excluding helps: {np.mean(eff[:,1] < eff[:,0])*100:.0f}%)")
    print(f"    orphan landmarks' own median error: {np.median(eff[:,2]):.2f}% sw")

# ---------------------------------------------------------------- E3
print("\n=== E3: doc Step 7 synthetic table, recomputed with the harness's own function ===")
spec = importlib.util.spec_from_file_location("mrt", "scripts/measure_round_trip.py")
mrt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mrt)
print(f"  root unrotated (200 trials): worst {mrt.run_synthetic(skel, 0):.2e} m   (doc: 5.3e-15 m)")
print(f"  root rotated 30 deg (200 trials): worst {mrt.run_synthetic(skel, 30)*1000:.1f} mm   (doc: 70 mm worst)")

# ---------------------------------------------------------------- E4
print("\n=== E4: hidden assumption -- expression = 0 ===")
m = skel.model
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
rots = np.zeros((55, 3)); rots[0] = rng.normal(0, 0.5, 3); rots[1:22] = rng.normal(0, 0.3, (21, 3))
ref0 = m(global_orient=t(rots[0]), body_pose=t(rots[1:22].reshape(-1)), expression=torch.zeros(1, 10),
         return_verts=False).joints[0, :55].numpy()
refx = m(global_orient=t(rots[0]), body_pose=t(rots[1:22].reshape(-1)), expression=t(rng.normal(0, 2, 10)),
         return_verts=False).joints[0, :55].numpy()
dj = np.linalg.norm(refx - ref0, axis=1)
print(f"  random expression (sigma 2) moves SMPL-X joints: body 0..21 max {dj[:22].max()*1000:.3f} mm; "
      f"jaw/eyes 22..24 max {dj[22:25].max()*1000:.3f} mm; hands max {dj[25:].max()*1000:.3f} mm")

# ---------------------------------------------------------------- E5
print("\n=== E5: SMPLXSkeleton gender / betas handling ===")
print("  model files present:", sorted(os.listdir("models/smplx")))
for g in ("male", "female"):
    try:
        SMPLXSkeleton(model_path="models", gender=g); print(f"  gender={g}: constructed")
    except Exception as ex:
        print(f"  gender={g}: {type(ex).__name__}: {str(ex)[:90]}")
for nb in (5, 16):
    try:
        s = SMPLXSkeleton(model_path="models", betas=rng.normal(0, 1, nb))
        print(f"  betas of length {nb}: constructed (rest joints shape {s.rest_joints.shape})")
    except Exception as ex:
        print(f"  betas of length {nb}: {type(ex).__name__}: {str(ex)[:100]}")

# ---------------------------------------------------------------- E6
print("\n=== E6: rotated landmark frame -- fold is exact, frame ESTIMATE vs truth ===")
for deg in (0, 10, 30, 90, 180):
    errs = []
    for _ in range(100):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        R = Rotation.from_rotvec(ax * np.deg2rad(deg)).as_matrix()
        lm, v = synth(skel, R=R, t=rng.normal(0, 1, 3))
        out = reconstruct_body(fit_body(lm, v, skel), skel)
        errs.append(max(np.linalg.norm(out[m] - lm[m]) for m in ARM_MP))
    print(f"  whole landmark set rigidly rotated {deg:3d} deg: worst-joint error median "
          f"{np.median(errs)*1000:6.1f} mm, max {np.max(errs)*1000:6.1f} mm")

# ---------------------------------------------------------------- E7
print("\n=== E7: DeprecationWarning visibility (default warning filters) ===")
tmpd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(tmpd, "_lib_caller.py"), "w") as fh:
    fh.write("def make():\n    from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX\n"
             "    return MediaPipeToSMPLX()\n")
env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "") + os.pathsep + tmpd)
for label, code in (("called from __main__", "from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX; MediaPipeToSMPLX()"),
                    ("called from an imported library module", "import _lib_caller; _lib_caller.make()")):
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print(f"  {label:40s}: warning printed = {'DeprecationWarning' in r.stderr}")
