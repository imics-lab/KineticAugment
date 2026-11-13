"""Independent check of A007: do MediaPipeToSMPLX / SMPLXToMediaPipe still use raw
displacements, and is the analytic path wired into them?"""
import inspect, warnings, numpy as np, torch
import kinetic_augment.body_model.mp_to_smplx as m2s
import kinetic_augment.body_model.smplx_to_mp as s2m
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
from kinetic_augment.body_model.joint_mapping import MEDIAPIPE_POSE_LANDMARKS as MP, SMPLX_BODY_JOINTS as SJ

# 1. static wiring: does either legacy module reference the analytic modules?
for mod in (m2s, s2m):
    src = inspect.getsource(mod)
    refs = [n for n in ("swing_ik", "solve_swing_ik", "forward_kinematics", "fit_body(", "SMPLXSkeleton", "torch", "smplx.create") if n in src]
    code_refs = [n for n in refs if any(n in line and not line.strip().startswith(("#",)) and '"' not in line for line in src.splitlines())]
    print(mod.__name__, "mentions:", refs, "| in non-string code lines:", code_refs)

img = np.load("data/test_output/01986_landmarks.npy")       # (T,543,3) image-space
world = np.load("data/test_output/01986_world.npz")
W, V = world["landmarks"].astype(np.float64), world["visibility"]

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    enc = m2s.MediaPipeToSMPLX(); dec = s2m.SMPLXToMediaPipe(smplx_wrapper=object())
print("warnings on construct:", [(w.category.__name__, "converter" in str(w.message)) for w in caught])

# 2. slots == raw child-parent displacement, on image-space real frames
bones = [(p, c, cs) for p, c, cs in m2s.BODY_BONES]
max_dev, n_slots = 0.0, 0
for t in range(0, img.shape[0], 7):
    prm = enc.fit(img[t])
    bp = prm["body_pose"][0]
    for p, c, cs in bones:
        slot = SJ[cs] - 1
        disp = (img[t, MP[c]] - img[t, MP[p]]).astype(np.float32)
        max_dev = max(max_dev, np.abs(bp[slot*3:slot*3+3] - disp).max()); n_slots += 1
    assert np.all(prm["global_orient"] == 0) and np.all(prm["betas"] == 0)
print(f"image-space: {n_slots} driven slots, max |slot - (child-parent)| = {max_dev:.1e}; global_orient, betas all zero")
# which slots are filled, and what it stores at each
print("slot -> (stored at SMPL-X joint, bone it stores):", [(SJ[cs]-1, cs, f"{p}->{c}") for p, c, cs in bones])

# 3. decode: zero params + target reproduces target exactly; params carry nothing
t = 10
zero = {k: np.zeros_like(v) for k, v in enc.fit(img[t]).items()}
out0 = dec.project(zero, img[t][None])
print("zero params + target_mp -> max |out - target| =", float(np.abs(out0[0] - img[t]).max()))
# real params, but a DIFFERENT target frame: output follows target, not params
prm = enc.fit(img[t])
out_other = dec.project(prm, img[t + 20][None])
print("params(frame t) + target(frame t+20): face rows equal target(t+20)?",
      np.array_equal(out_other[0, 33:501], img[t + 20, 33:501]),
      "| L shoulder equals target(t+20) shoulder?", np.allclose(out_other[0, 11], img[t + 20, 11]))

# 4. world-space frame through legacy encoding vs analytic, both evaluated in real SMPL-X
skel = SMPLXSkeleton(model_path="models")
model = skel.model
def smplx_arm(bp, go, tr=np.zeros(3)):
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None]
    with torch.no_grad():
        j = model(body_pose=T(bp), global_orient=T(go), transl=T(tr), return_verts=False).joints[0, :55].double().numpy()
    return np.array([j[s] for s, _ in ARM_CORRESPONDENCE])
arm_mp = [m for _, m in ARM_CORRESPONDENCE]
rows = []
for t in range(W.shape[0]):
    ok = V[t] > 0.5
    if not ok[arm_mp].all(): continue
    target = W[t][arm_mp]
    sw = np.linalg.norm(W[t, 11] - W[t, 12])
    full = np.zeros((543, 3), np.float32); full[:33] = W[t]
    lp = enc.fit(full)
    leg = smplx_arm(lp["body_pose"][0], lp["global_orient"][0])
    s, R, tt = similarity_transform(leg, target, allow_scale=True)
    e_leg = np.linalg.norm(apply_similarity(leg, s, R, tt) - target, axis=1).mean()
    fit = fit_body(W[t], ok, skel)
    ana = smplx_arm(fit.body_pose, fit.global_orient, fit.transl)
    e_ana = np.linalg.norm(ana - target, axis=1).mean()
    rows.append((e_leg, e_ana, sw, np.linalg.norm(lp["body_pose"][0][17*3:18*3])))
rows = np.array(rows)
print(f"world frames used: {len(rows)}")
print(f"legacy params -> SMPL-X, best similarity: mean arm error median {np.median(rows[:,0])*1e3:.1f} mm ({np.median(rows[:,0]/rows[:,2])*100:.0f}% sw)")
print(f"fit_body params -> SMPL-X, direct (no extra align): median {np.median(rows[:,1])*1e3:.1f} mm ({np.median(rows[:,1]/rows[:,2])*100:.1f}% sw)")
print(f"legacy 'angle' at slot 17 (rad) == upper-arm length (m): median {np.median(rows[:,3]):.4f}")
