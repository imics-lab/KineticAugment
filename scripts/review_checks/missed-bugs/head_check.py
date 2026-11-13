"""Real SMPL-X forward on fit_body output: where does the model's head land vs MediaPipe's nose?"""
import glob
import numpy as np
import torch
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

C = np.diag([1.0, -1.0, -1.0])
sk = SMPLXSkeleton(model_path="models")


def smplx_joints(fit):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
    with torch.no_grad():
        return sk.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient),
                        transl=t(fit.transl), betas=t(fit.betas), return_verts=False).joints[0].numpy()


raw, conv, sh = [], [], []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path)
    for lm, vis in zip(d["landmarks"].astype(np.float64), d["visibility"]):
        if not np.isfinite(lm).all():
            continue
        ok = vis >= 0.5
        try:
            f_raw = fit_body(lm, ok, sk)
            f_c = fit_body(lm @ C.T, ok, sk)
        except ValueError:
            continue
        head_raw = smplx_joints(f_raw)[15]            # SMPL-X head joint, MP frame
        head_c = smplx_joints(f_c)[15] @ C            # back to MP frame
        raw.append(np.linalg.norm(head_raw - lm[0]))
        conv.append(np.linalg.norm(head_c - lm[0]))
        sh.append(np.linalg.norm(lm[11] - lm[12]))
raw, conv, sh = map(np.asarray, (raw, conv, sh))
print(f"frames {len(raw)}")
print(f"SMPL-X head joint -> MediaPipe nose distance, CURRENT fit : median {1000*np.median(raw):.0f} mm "
      f"({100*np.median(raw/sh):.0f}% shoulder width), min {1000*raw.min():.0f} mm")
print(f"SMPL-X head joint -> MediaPipe nose distance, with C first: median {1000*np.median(conv):.0f} mm "
      f"({100*np.median(conv/sh):.0f}% shoulder width), max {1000*conv.max():.0f} mm")
