import glob, numpy as np, torch
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
sk = SMPLXSkeleton(model_path="models"); model = sk.model
res = {k: [] for k in ["f64_arccos", "f32joints_arccos64", "f32joints_f32norm_arccos", "f32_all"]}
for f in sorted(glob.glob("data/test_output/*_landmarks.npy")):
    pose = np.load(f).astype(np.float64)[:, :33]
    for fr in pose:
        if not np.all(np.isfinite(fr[11:17])) or np.any(np.all(fr[11:17] == 0, axis=1)):
            continue
        rots = solve_swing_ik(fr, sk, SMPLX_ARM_BONES)
        P = forward_kinematics(rots, sk)
        with torch.no_grad():
            J = model(body_pose=torch.tensor(rots[1:22].reshape(1, 63), dtype=torch.float32),
                      global_orient=torch.tensor(rots[0][None], dtype=torch.float32)).joints[0, :55].numpy()
        for a, b, ma, mb in [(16, 18, 11, 13), (18, 20, 13, 15), (17, 19, 12, 14), (19, 21, 14, 16)]:
            o = fr[mb] - fr[ma]
            v = P[b] - P[a]
            res["f64_arccos"].append(np.degrees(np.arccos(np.clip(v @ o / np.linalg.norm(v) / np.linalg.norm(o), -1, 1))))
            v32 = (J[b] - J[a])
            res["f32joints_arccos64"].append(np.degrees(np.arccos(np.clip(v32.astype(np.float64) @ o / np.linalg.norm(v32.astype(np.float64)) / np.linalg.norm(o), -1, 1))))
            u1 = v32 / np.linalg.norm(v32); u2 = (o / np.linalg.norm(o)).astype(np.float32)
            res["f32joints_f32norm_arccos"].append(np.degrees(np.arccos(np.clip(np.float64(u1 @ u2), -1, 1))))
            res["f32_all"].append(float(np.degrees(np.arccos(np.clip(u1 @ u2, np.float32(-1), np.float32(1))))))
for k, v in res.items():
    print(f"{k:28s} median {np.median(v):.4f} max {np.max(v):.4f} deg  n={len(v)}")
