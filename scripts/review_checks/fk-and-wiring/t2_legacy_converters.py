"""A007: do the public converter classes (mp_to_smplx / smplx_to_mp) still
encode/decode raw displacements, and do they use SMPL-X at all?"""
import glob
import inspect

import numpy as np
import torch

from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX
from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe
from kinetic_augment.body_model import converter
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.utils.data_formats import TOTAL_LANDMARKS, LH_START, RH_START
import kinetic_augment.body_model as bm

torch.set_grad_enabled(False)

# ---- real image-space frames (the format existing callers use)
img = np.load(sorted(glob.glob("data/test_output/*_landmarks.npy"))[0]).astype(np.float32)
img = img.reshape(img.shape[0], TOTAL_LANDMARKS, 3)
frame = next(f for f in img if np.any(f[LH_START:LH_START + 21] != 0) and np.any(f[:33] != 0))

fitter = MediaPipeToSMPLX()
params = fitter.fit(frame)
bp = params["body_pose"][0]
print("=== 1. Encoding (MediaPipeToSMPLX.fit) on a real image-space frame ===")
print("  global_orient all zero:", bool(np.all(params["global_orient"] == 0)),
      "| betas all zero:", bool(np.all(params["betas"] == 0)))
checks = {  # body_pose slot -> expected raw displacement
    17: frame[13] - frame[11],  # slot of SMPL-X joint 18 (left elbow) <- L shoulder->elbow vector
    19: frame[15] - frame[13],  # joint 20 (left wrist) <- L elbow->wrist
    18: frame[14] - frame[12],
    20: frame[16] - frame[14],
    3: frame[25] - frame[23],   # joint 4 (left knee) <- L hip->knee
}
for slot, vec in checks.items():
    print(f"  body_pose slot {slot:2d} (SMPL-X joint {slot+1:2d}) == raw child-parent displacement: "
          f"{np.array_equal(bp[slot*3:slot*3+3], vec.astype(np.float32))}  value={np.round(bp[slot*3:slot*3+3],4)}")
lh = params["left_hand_pose"][0]
H = frame[LH_START:LH_START + 21]
print("  left_hand_pose slot 0 == H[2]-H[1] (raw finger displacement):",
      np.array_equal(lh[0:3], (H[2] - H[1]).astype(np.float32)))
# Semantic check: SMPL-X joint 18's rotation drives the forearm (18->20), but the
# encoder stores the UPPER-arm vector (shoulder->elbow) there.
print("  NOTE: SMPL-X joint 18's rotation drives bone 18->20 (forearm); slot 17 holds the upper-arm vector.")

# ---- metric world frame embedded in a 543 array, so magnitudes are metres
world = np.load(sorted(glob.glob("data/test_output/*_world.npz"))[0])
W = world["landmarks"].astype(np.float64)
vis = world["visibility"]
fi = int(np.argmax((vis[:, 11:17] > 0.5).all(axis=1)))
wf = W[fi]
full = np.zeros((TOTAL_LANDMARKS, 3), np.float32)
full[:33] = wf
p2 = fitter.fit(full)
print("\n=== 2. Legacy params through the REAL SMPL-X model (world frame %d) ===" % fi)
skel = SMPLXSkeleton(model_path="models")
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).reshape(1, -1)
j = skel.model(body_pose=t(p2["body_pose"][0]), global_orient=t(p2["global_orient"][0]),
               betas=t(p2["betas"][0]), return_verts=False).joints[0].numpy().astype(np.float64)
src = np.array([j[s] for s, _ in ARM_CORRESPONDENCE])
tgt = np.array([wf[m] for _, m in ARM_CORRESPONDENCE])
sw = np.linalg.norm(wf[11] - wf[12])
err_sim = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1)
err_rig = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False)) - tgt, axis=1)
print(f"  legacy params -> SMPL-X -> best SIMILARITY alignment: mean {err_sim.mean()*1000:.1f} mm "
      f"({err_sim.mean()/sw*100:.1f}% shoulder width), max {err_sim.max()*1000:.1f} mm")
print(f"  legacy params -> SMPL-X -> best RIGID alignment     : mean {err_rig.mean()*1000:.1f} mm "
      f"({err_rig.mean()/sw*100:.1f}% sw), max {err_rig.max()*1000:.1f} mm")
ok = np.isfinite(wf).all(1) & (vis[fi] >= 0.5)
nb = reconstruct_body(fit_body(wf, ok, skel), skel)
e_new = np.array([np.linalg.norm(nb[m] - wf[m]) for _, m in ARM_CORRESPONDENCE])
print(f"  new converter fit_body->reconstruct_body (direct, no extra alignment): mean {e_new.mean()*1000:.1f} mm "
      f"({e_new.mean()/sw*100:.1f}% sw), max {e_new.max()*1000:.1f} mm")
angles = np.linalg.norm(p2["body_pose"][0].reshape(21, 3), axis=1)
print("  legacy 'rotation angles' (rad) in driven slots = bone lengths in metres:",
      np.round(angles[[17, 18, 19, 20]], 4).tolist(),
      " vs measured bone lengths:", np.round([np.linalg.norm(wf[13]-wf[11]), np.linalg.norm(wf[14]-wf[12]),
                                              np.linalg.norm(wf[15]-wf[13]), np.linalg.norm(wf[16]-wf[14])], 4).tolist())

print("\n=== 3. Decoding (SMPLXToMediaPipe.project) ===")
print("  signature:", inspect.signature(SMPLXToMediaPipe.project))


class Tripwire:
    def __getattr__(self, name):
        raise RuntimeError(f"wrapper.{name} touched")

    def __call__(self, *a, **k):
        raise RuntimeError("wrapper called")


proj = SMPLXToMediaPipe(Tripwire())
B = frame[None]
out_same = proj.project(params, B)[0]
print("  project() with a tripwire wrapper raised nothing -> SMPL-X never consulted")
print("  round trip with target_mp=original: max |out-in| over all 543 =",
      float(np.abs(out_same - frame).max()))
out_zero = proj.project(params, np.zeros_like(B))[0]
print("  target_mp = zeros: shoulders out =", out_zero[11].tolist(), out_zero[12].tolist(),
      "| L elbow out == stored displacement:", np.allclose(out_zero[13], frame[13] - frame[11]))
other = img[len(img) // 2]
out_other = proj.project(params, other[None])[0]
print("  target_mp = a different frame: L shoulder taken from target:", np.allclose(out_other[11], other[11]),
      "| face copied from target verbatim:", np.array_equal(out_other[33:501], other[33:501]))
z = {k: np.zeros_like(v) for k, v in params.items()}
out_z = proj.project(z, B)[0]
print("  all-zero params + target_mp=original -> max |output-original| =", float(np.abs(out_z - frame).max()),
      "(params carry nothing; target is passed through)")

print("\n=== 4. What is 'public'? ===")
print("  body_model.__all__:", bm.__all__)
print("  get_mp_to_smplx() is legacy class:", bm.get_mp_to_smplx() is MediaPipeToSMPLX,
      "| converter.fit_body exported from package:", hasattr(bm, "fit_body"))
