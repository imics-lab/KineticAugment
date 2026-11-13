"""
Does fit_body produce anatomically meaningful SMPL-X parameters on REAL
MediaPipe world landmarks, given that MediaPipe world is y-down / z-away-from-
camera and SMPL-X is y-up / z-forward?

Run from snapshot root with PYTHONPATH=snap/src.
"""
import glob
import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

VIS = 0.5
NOSE, LS, RS, LE, RE, LW, RW, LH, RH = 0, 11, 12, 13, 14, 15, 16, 23, 24
# MediaPipe world (x: subject-left on unmirrored video, y: DOWN, z: away from camera)
# -> SMPL-X canonical (x: left, y: UP, z: forward/toward camera) is a 180 deg turn about x.
C = np.diag([1.0, -1.0, -1.0])

sk = SMPLXSkeleton(model_path="models")


def unit(v):
    return v / np.linalg.norm(v)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1)))


stats = {k: [] for k in ["dat_up_y", "dat_fwd_z", "lr_x", "chir",
                         "fit_up_vs_data_up", "fit_fwd_vs_data_fwd", "go_deg",
                         "fitC_up_vs_data_up", "fitC_fwd_vs_data_fwd", "goC_deg",
                         "phys_elbow_above", "fit_elbow_above", "fitC_elbow_above",
                         "phys_wrist_front", "fit_wrist_front", "fitC_wrist_front",
                         "err_raw", "err_C"]}

for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path)
    P = d["landmarks"].astype(np.float64)
    V = d["visibility"]
    for lm, vis in zip(P, V):
        if not np.isfinite(lm).all():
            continue
        ok = np.isfinite(lm).all(1) & (vis >= VIS)
        sh_mid = (lm[LS] + lm[RS]) / 2
        hip_mid = (lm[LH] + lm[RH]) / 2
        data_up = sh_mid - hip_mid                      # physical up, in MP frame
        lr = lm[LS] - lm[RS]                             # subject's left, in MP frame
        nose_rel = lm[NOSE] - sh_mid
        # forward = nose direction orthogonalised against left and up
        e_l = unit(lr)
        e_u = unit(data_up - np.dot(data_up, e_l) * e_l)
        fwd = nose_rel - np.dot(nose_rel, e_l) * e_l - np.dot(nose_rel, e_u) * e_u
        e_f = unit(fwd)
        stats["dat_up_y"].append(e_u[1])
        stats["dat_fwd_z"].append(e_f[2])
        stats["lr_x"].append(e_l[0])
        stats["chir"].append(np.dot(np.cross(e_l, e_u), e_f))  # +1 = same handedness as SMPL-X (x-left,y-up,z-fwd)

        try:
            fit = fit_body(lm, ok, sk)
            fitC = fit_body(lm @ C.T, ok, sk)
        except ValueError:
            continue
        for tag, f, Cm in (("fit", fit, np.eye(3)), ("fitC", fitC, C)):
            R = Rotation.from_rotvec(f.global_orient).as_matrix()
            # fitted body's up / forward axes, expressed back in the MP frame
            up_mp = Cm.T @ (R @ [0, 1, 0])
            fwd_mp = Cm.T @ (R @ [0, 0, 1])
            key = "" if tag == "fit" else "C"
            stats[f"fit{key}_up_vs_data_up"].append(ang(up_mp, e_u))
            stats[f"fit{key}_fwd_vs_data_fwd"].append(ang(fwd_mp, e_f))
            stats[f"go{key}_deg"].append(np.degrees(np.linalg.norm(f.global_orient)))
            # body-local (canonical) arm semantics: FK with global_orient = 0
            rots = np.zeros((55, 3))
            rots[1:22] = f.body_pose.reshape(21, 3)
            J = forward_kinematics(rots, sk)
            above = np.mean([J[18][1] > J[16][1], J[19][1] > J[17][1]])
            front = np.mean([J[20][2] > J[16][2], J[21][2] > J[17][2]])
            stats[f"{tag}_elbow_above"].append(above)
            stats[f"{tag}_wrist_front"].append(front)
        # physical truth from the data itself (in the data's own torso frame)
        stats["phys_elbow_above"].append(np.mean([np.dot(lm[LE] - lm[LS], e_u) > 0,
                                                  np.dot(lm[RE] - lm[RS], e_u) > 0]))
        stats["phys_wrist_front"].append(np.mean([np.dot(lm[LW] - lm[LS], e_f) > 0,
                                                  np.dot(lm[RW] - lm[RS], e_f) > 0]))
        # direct error is identical either way (rigid transform C is invisible to it)
        out = reconstruct_body(fit, sk)
        outC = reconstruct_body(fitC, sk) @ C  # back to MP frame (C is its own inverse)
        m = [mp for _, mp in ARM_CORRESPONDENCE if ok[mp]]
        stats["err_raw"].append(np.mean([np.linalg.norm(out[i] - lm[i]) for i in m]))
        stats["err_C"].append(np.mean([np.linalg.norm(outC[i] - lm[i]) for i in m]))

s = {k: np.asarray(v) for k, v in stats.items()}
print(f"frames analysed: {len(s['go_deg'])}")
print("--- MediaPipe world-frame convention, measured from the data ---")
print(f"  data 'up' (hip->shoulder) y-component: median {np.median(s['dat_up_y']):+.3f} (−1 => y is DOWN)")
print(f"  data 'forward' (toward nose) z-comp  : median {np.median(s['dat_fwd_z']):+.3f} (−1 => facing −z)")
print(f"  subject-left (R->L shoulder) x-comp  : median {np.median(s['lr_x']):+.3f}")
print(f"  chirality (left x up . forward)      : min {s['chir'].min():+.3f} max {s['chir'].max():+.3f} (+1 = proper rotation from SMPL-X)")
print("--- CURRENT fit_body on raw MP world landmarks ---")
print(f"  |global_orient| median {np.median(s['go_deg']):.1f} deg  (range {s['go_deg'].min():.1f}..{s['go_deg'].max():.1f})")
print(f"  angle(fitted body up, true up)      : median {np.median(s['fit_up_vs_data_up']):.1f} deg")
print(f"  angle(fitted body fwd, true fwd)    : median {np.median(s['fit_fwd_vs_data_fwd']):.1f} deg")
print(f"  body-local: elbow ABOVE shoulder    : {100*s['fit_elbow_above'].mean():.1f}% of arms (truth {100*s['phys_elbow_above'].mean():.1f}%)")
print(f"  body-local: wrist IN FRONT of shldr : {100*s['fit_wrist_front'].mean():.1f}% of arms (truth {100*s['phys_wrist_front'].mean():.1f}%)")
print("--- fit_body after applying the 180-deg x convention C=diag(1,-1,-1) first ---")
print(f"  |global_orient| median {np.median(s['goC_deg']):.1f} deg")
print(f"  angle(fitted body up, true up)      : median {np.median(s['fitC_up_vs_data_up']):.1f} deg")
print(f"  angle(fitted body fwd, true fwd)    : median {np.median(s['fitC_fwd_vs_data_fwd']):.1f} deg")
print(f"  body-local: elbow ABOVE shoulder    : {100*s['fitC_elbow_above'].mean():.1f}% of arms")
print(f"  body-local: wrist IN FRONT of shldr : {100*s['fitC_wrist_front'].mean():.1f}% of arms")
print("--- direct round-trip error cannot see the difference ---")
print(f"  mean per-frame direct error raw {1000*s['err_raw'].mean():.4f} mm, with C {1000*s['err_C'].mean():.4f} mm, "
      f"max |diff| {1000*np.abs(s['err_raw']-s['err_C']).max():.2e} mm")
