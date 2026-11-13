"""Throwaway prototype of the Step 8 design implied by the red-phase tests in
snap/tests/test_converter.py. NOT project code; lives only in the review scratch dir.

root  = rotation taking the rest (shoulder line, apex) frame onto the observed one;
        apex = nose when valid (fixes rotation about the shoulder axis), otherwise the
        minimal rotation aligning the shoulder lines.
arms  = swing IK solved relative to that root.
transl= mean residual over valid correspondences (translation only, so directions stay exact).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.converter import (  # reuse the shipped pieces
    ARM_CORRESPONDENCE, BodyFit, MIN_CORRESPONDENCES, reconstruct_body)
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, _swing_rotation, solve_swing_ik

NOSE_MP, L_SH, R_SH = 0, 11, 12
_EPS = 1e-9


def _frame(left, right, apex):
    x = left - right
    x = x / np.linalg.norm(x)
    y = apex - (left + right) / 2
    y = y - (y @ x) * x
    n = np.linalg.norm(y)
    if n < _EPS:
        return None
    y = y / n
    return np.stack([x, y, np.cross(x, y)], axis=1)


def torso_rotation(rest_left, rest_right, obs_left, obs_right, rest_apex=None, obs_apex=None):
    if rest_apex is not None and obs_apex is not None:
        fr, fo = _frame(rest_left, rest_right, rest_apex), _frame(obs_left, obs_right, obs_apex)
        if fr is not None and fo is not None:
            return fo @ fr.T
    a = rest_left - rest_right
    b = obs_left - obs_right
    return _swing_rotation(a / np.linalg.norm(a), b / np.linalg.norm(b))


def fit_body(landmarks, validity, skeleton, apex="nose", hip_centre=None, post="translation"):
    landmarks = np.asarray(landmarks, dtype=np.float64)
    usable = np.asarray(validity, dtype=bool) & np.isfinite(landmarks).all(axis=1)
    if not (usable[L_SH] and usable[R_SH]):
        raise ValueError("both shoulders are needed to fix the torso frame")
    pairs = [(s, m) for s, m in ARM_CORRESPONDENCE if usable[m]]
    if len(pairs) < MIN_CORRESPONDENCES:
        raise ValueError("too few correspondences")
    rest = skeleton.rest_joints
    if apex == "nose" and usable[NOSE_MP]:
        R = torso_rotation(rest[16], rest[17], landmarks[L_SH], landmarks[R_SH],
                           skeleton.rest_nose, landmarks[NOSE_MP])
    elif apex == "hip":   # MediaPipe world origin is the hip centre
        R = torso_rotation(rest[16], rest[17], landmarks[L_SH], landmarks[R_SH],
                           (rest[1] + rest[2]) / 2, np.zeros(3) if hip_centre is None else hip_centre)
    else:
        R = torso_rotation(rest[16], rest[17], landmarks[L_SH], landmarks[R_SH])
    masked = np.where(usable[:, None], landmarks, np.nan)
    rots = solve_swing_ik(masked, skeleton, SMPLX_ARM_BONES,
                          global_orient=Rotation.from_matrix(R).as_rotvec())
    J = forward_kinematics(rots, skeleton)
    src = np.array([J[s] for s, _ in pairs]); tgt = np.array([landmarks[m] for _, m in pairs])
    if post == "translation":
        return BodyFit(body_pose=rots[1:22].reshape(63).copy(), global_orient=rots[0].copy(),
                       transl=(tgt - src).mean(axis=0))
    # post == "rigid": an extra rigid fit folded into global_orient, like the shipped converter
    from kinetic_augment.body_model.alignment import similarity_transform
    _, Rr, tr = similarity_transform(src, tgt, allow_scale=False)
    root = rest[0]
    Rroot = Rr @ Rotation.from_rotvec(rots[0]).as_matrix()
    return BodyFit(body_pose=rots[1:22].reshape(63).copy(), global_orient=Rotation.from_matrix(Rroot).as_rotvec(),
                   transl=Rr @ root + tr - root)
