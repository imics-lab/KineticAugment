import numpy as np, torch, smplx
m = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False)
def fore(axis, ang):
    bp = torch.zeros(1, 63); bp[0, 17*3 + axis] = ang   # body_pose idx 17 = joint 18 = left_elbow
    J = m(body_pose=bp).joints[0].detach().numpy(); v = J[20] - J[18]; return v / np.linalg.norm(v)
base = fore(0, 0.0)
for ax in range(3):
    print("axis", "xyz"[ax], "60deg -> forearm moves %.1f deg" % np.degrees(np.arccos(np.clip(base @ fore(ax, np.radians(60)), -1, 1))))
