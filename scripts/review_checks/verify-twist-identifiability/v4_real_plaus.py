# Plausibility of fit_body output on the real world-landmark clips.
import glob, numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
FLIP = R.from_euler('x', 180, degrees=True).as_matrix()  # MP (y down, faces -z) -> SMPL-X (y up, faces +z)
ARMS = {"L": (16, 18, 20), "R": (17, 19, 21)}

def analyse(flip):
    st = {k: [] for k in ["go", "head_below", "face_away", "nose_head_dist"]}
    arm = {s: {"elev": [], "behind": [], "hyper": [], "flexz": []} for s in ARMS}
    for f in sorted(glob.glob("data/test_output/*_world.npz")):
        d = np.load(f); L = d["landmarks"].astype(float); V = d["visibility"]
        for fr in range(len(L)):
            lm = L[fr].copy(); ok = np.isfinite(lm).all(1) & (V[fr] >= 0.5)
            if flip: lm = lm @ FLIP.T
            try: fit = fit_body(lm, ok, sk)
            except ValueError: continue
            rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
            J = forward_kinematics(rots, sk) + fit.transl
            Rg = R.from_rotvec(fit.global_orient).as_matrix()
            # undo flip so judgements are in MP's own frame (y down, camera toward -z)
            Jmp = J @ FLIP if flip else J            # FLIP is symmetric/involutive
            Rmp = FLIP @ Rg if flip else Rg
            st["go"].append(np.degrees(np.linalg.norm(fit.global_orient)))
            sh_y = 0.5 * (Jmp[16, 1] + Jmp[17, 1])
            st["head_below"].append(Jmp[15, 1] > sh_y)          # y down: larger y = lower
            fwd = Rmp @ np.array([0, 0, 1.0])                   # body forward in MP frame
            st["face_away"].append(fwd[2] > 0)                  # camera side is -z
            if ok[0]:
                st["nose_head_dist"].append(np.linalg.norm(Jmp[15] - L[fr, 0]))
            # arm angles in the BODY frame (spine/collars are zero, so body frame = global_orient)
            for s, (S, E, W) in ARMS.items():
                if not (ok[[11, 13, 15][0 if s == "L" else 0]] if False else True): pass
                mpS, mpE, mpW = (11, 13, 15) if s == "L" else (12, 14, 16)
                if not (ok[mpS] and ok[mpE] and ok[mpW]): continue
                ub = Rg.T @ (J[E] - J[S]); ub /= np.linalg.norm(ub)
                arm[s]["elev"].append(np.degrees(np.arccos(np.clip(ub @ [0, -1, 0], -1, 1))))
                arm[s]["behind"].append(ub[2] < 0)
                # forearm direction expressed in the humerus frame (world rot of shoulder joint)
                chain = [0, 3, 6, 9, 13 if s == "L" else 14, S]
                Wsh = np.eye(3)
                for j in chain: Wsh = Wsh @ R.from_rotvec(rots[j]).as_matrix()
                fh = Wsh.T @ (J[W] - J[E]); fh /= np.linalg.norm(fh)
                arm[s]["flexz"].append(fh[2]); arm[s]["hyper"].append(fh[2] < -0.05)
    n = len(st["go"])
    print(f"  frames fitted {n}; global_orient angle median {np.median(st['go']):.1f} deg")
    print(f"  SMPL-X head joint below shoulders (MP frame): {np.mean(st['head_below'])*100:.1f}%")
    print(f"  body forward points away from camera: {np.mean(st['face_away'])*100:.1f}%")
    print(f"  head joint to MP nose distance median {np.median(st['nose_head_dist']):.3f} m")
    for s in ARMS:
        a = arm[s]; e = np.array(a["elev"])
        print(f"  {s} arm n={len(e)}: upper-arm elevation from straight-down med {np.median(e):.0f} deg, "
              f"above horizontal {np.mean(e>90)*100:.0f}%, behind body {np.mean(a['behind'])*100:.0f}%, "
              f"forearm bends to humerus-frame -z (hyperext., z<-0.05) {np.mean(a['hyper'])*100:.0f}%, "
              f"median z {np.median(a['flexz']):+.2f}")

print("AS DELIVERED (fit_body on raw MP world landmarks)"); analyse(False)
print("WITH FIXED MP->SMPL-X AXIS FLIP (180 deg about x) applied before fit_body"); analyse(True)
