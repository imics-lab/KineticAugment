"""Independent check: is the current fit_body output upside-down / backward in MediaPipe world frame?
Uses only data-derived axes (no assumption about MediaPipe's camera convention) plus the real smplx forward."""
import glob, warnings
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
warnings.filterwarnings("ignore")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

sk = SMPLXSkeleton(model_path="models")
rows = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L = d["landmarks"].astype(float); V = d["visibility"]
    for t in range(L.shape[0]):
        lm = L[t]; ok = np.isfinite(lm).all(1) & (V[t] >= 0.5)
        try:
            fit = fit_body(lm, ok, sk)
        except ValueError:
            continue
        with torch.no_grad():
            J = sk.model(body_pose=torch.tensor(fit.body_pose, dtype=torch.float32)[None],
                         global_orient=torch.tensor(fit.global_orient, dtype=torch.float32)[None],
                         transl=torch.tensor(fit.transl, dtype=torch.float32)[None],
                         betas=torch.zeros(1, 10), return_verts=False).joints[0].numpy().astype(float)
        rec = reconstruct_body(fit, sk)
        armdiff = max(np.linalg.norm(rec[m] - J[s]) for s, m in ARM_CORRESPONDENCE)
        sw = np.linalg.norm(lm[11] - lm[12])
        smid = 0.5 * (lm[11] + lm[12])
        # data-derived MP body axes: left = L-R shoulder; up = nose - shoulder mid, orthogonalised
        left = (lm[11] - lm[12]); left /= np.linalg.norm(left)
        upn = lm[0] - smid; upn -= upn.dot(left) * left; upn /= np.linalg.norm(upn)
        front_mp = np.cross(left, upn)  # SMPL-X convention x=left,y=up,z=front -> z = x cross y
        # fitted body axes from real forward pass joints
        fl = J[16] - J[17]; fl /= np.linalg.norm(fl)
        fu = J[12] - J[0]   # neck - pelvis
        fu -= fu.dot(fl) * fl; fu /= np.linalg.norm(fu)
        ff = np.cross(fl, fu)
        Gm = R.from_rotvec(fit.global_orient).as_matrix()
        fshmid = 0.5 * (J[16] + J[17])
        head_above = (J[15] - fshmid).dot(upn) > 0    # SMPL-X head joint relative to MP's own 'up'
        mp_nose_above_y = lm[0][1] < smid[1]           # y-down frame
        smplx_head_above_y = J[15][1] < fshmid[1]
        rows.append(dict(armdiff=armdiff, up_dot=fu.dot(upn), front_dot=ff.dot(front_mp),
                         up_y=fu[1], front_z=ff[1*0+2], mp_front_z=front_mp[2], G_y=(Gm @ [0,1,0])[1],
                         head_above=head_above, mp_nose_above_y=mp_nose_above_y, smplx_head_above_y=smplx_head_above_y,
                         head_nose=np.linalg.norm(J[15] - lm[0]) / sw,
                         hip=np.linalg.norm(0.5*(J[1]+J[2]) - 0.5*(lm[23]+lm[24])) / sw,
                         rot_angle=np.degrees(np.linalg.norm(fit.global_orient)),
                         elbowL_y=np.degrees(fit.body_pose.reshape(21,3)[17,1]),
                         elbowR_y=np.degrees(fit.body_pose.reshape(21,3)[18,1]),
                         clip=f.split('/')[-1][:5]))
import collections
a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
n = len(rows)
print("frames", n, "max |real forward - reconstruct_body| on arm joints (m):", a["armdiff"].max())
print("cos(fitted up, MP data up from nose): median %.3f  min %.3f max %.3f; frac<0: %.3f" % (np.median(a["up_dot"]), a["up_dot"].min(), a["up_dot"].max(), (a["up_dot"]<0).mean()))
print("cos(fitted front, MP data front): median %.3f min %.3f max %.3f; frac<0: %.3f" % (np.median(a["front_dot"]), a["front_dot"].min(), a["front_dot"].max(), (a["front_dot"]<0).mean()))
print("fitted up y-comp (MP y-down; upright=>neg): median %.3f; fitted front z: median %.3f; MP-data front z: median %.3f" % (np.median(a["up_y"]), np.median(a["front_z"]), np.median(a["mp_front_z"])))
print("global_orient magnitude deg: median %.1f p5 %.1f p95 %.1f" % tuple(np.percentile(a["rot_angle"], [50,5,95])))
print("SMPL-X head above its shoulders along MP's own up: %.1f%%; MP nose above shoulders (y-down): %.1f%%; SMPL-X head above shoulders (y-down): %.1f%%" % (100*a["head_above"].mean(), 100*a["mp_nose_above_y"].mean(), 100*a["smplx_head_above_y"].mean()))
print("|SMPL-X head - MP nose| %%SW median %.0f; |hip mid diff| %%SW median %.0f" % (100*np.median(a["head_nose"]), 100*np.nanmedian(a["hip"])))
print("left elbow local-y deg: median %.1f frac>0 %.3f | right elbow local-y: median %.1f frac<0 %.3f" % (np.median(a["elbowL_y"]), (a["elbowL_y"]>0).mean(), np.median(a["elbowR_y"]), (a["elbowR_y"]<0).mean()))
for c in sorted(set(a["clip"])):
    m = a["clip"] == c
    print(" ", c, "n", m.sum(), "up_dot med %.2f front_dot med %.2f" % (np.median(a["up_dot"][m]), np.median(a["front_dot"][m])))
