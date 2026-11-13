# Independent check: is the fitted SMPL-X body upside down relative to the MediaPipe data?
import glob, numpy as np, torch, smplx
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body
sk = SMPLXSkeleton(model_path="models")
model = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False)
n = up_ang = 0; head_above = 0; angs = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L, V = d["landmarks"], d["visibility"]
    for t in range(len(L)):
        ok = np.isfinite(L[t]).all(-1) & (V[t] >= 0.5)
        if not (ok[11] and ok[12] and ok[23] and ok[24]): continue
        try: fit = fit_body(L[t], ok, sk)
        except Exception: continue
        out = model(global_orient=torch.tensor(fit.global_orient[None], dtype=torch.float32),
                    body_pose=torch.tensor(np.asarray(fit.body_pose).reshape(1, -1)[:, :63], dtype=torch.float32),
                    transl=torch.tensor(np.asarray(fit.transl)[None], dtype=torch.float32))
        J = out.joints[0].detach().numpy()
        smplx_up = J[12] - J[0]                      # neck - pelvis in output space
        mp_up = (L[t, 11] + L[t, 12]) / 2 - (L[t, 23] + L[t, 24]) / 2   # shoulder mid - hip mid (data)
        a = np.degrees(np.arccos(np.dot(smplx_up, mp_up) / np.linalg.norm(smplx_up) / np.linalg.norm(mp_up)))
        angs.append(a); n += 1
        sh_mid = (L[t, 11] + L[t, 12]) / 2
        head_above += np.dot(J[15] - sh_mid, mp_up) > 0
print(f"frames {n}  median angle(SMPL-X torso up, data torso up) = {np.median(angs):.1f} deg  "
      f"[min {np.min(angs):.1f}, max {np.max(angs):.1f}]  head-joint above shoulders: {head_above}/{n}")
