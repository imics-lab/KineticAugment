"""
V1 MediaPipe <-> SMPL-X body converter.

Contract:
    world-coordinate body landmarks + per-landmark validity
        -> SMPL-X parameters (body_pose, global_orient, transl, betas)
        -> reconstructed world-coordinate body landmarks

Reconstruction receives only the parameters. Everything needed to rebuild the
landmarks lives in them, so the round trip genuinely passes through SMPL-X
rather than subtracting and re-adding the input. That includes body shape: the
skeleton passed to `reconstruct_body` supplies the model, and `fit.betas` the
body it is put in.

Fitting runs in an order that keeps the solved quantities from fighting:

  1. Root rotation from the torso: the shoulder axis fixes two degrees of
     freedom, and the nose -- or, without it, the hip centre -- fixes rotation
     about that axis.
  2. Arm rotations by swing IK, solved *relative to* that root, so each bone
     points along its observed direction in world space.
  3. Translation by least squares over the valid correspondences.

Rotating the body after the arms are solved -- what a whole-body rigid alignment
does -- turns correct bone directions away from the observations, so no rotation
is fitted after step 2. Scale is never fitted: both spaces are metric, and a free
scale absorbs correspondence error into something that looks like a fit.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik, swing_rotation

NUM_POSE_LANDMARKS = 33
NUM_BETAS = 10

NOSE_MP = 0
LEFT_SHOULDER_MP, RIGHT_SHOULDER_MP = 11, 12
LEFT_SHOULDER_SMPLX, RIGHT_SHOULDER_SMPLX = 16, 17
LEFT_HIP_MP, RIGHT_HIP_MP = 23, 24
LEFT_HIP_SMPLX, RIGHT_HIP_SMPLX = 1, 2
# Outward tilt of an arm bone left hanging because it was not observed (`fit_body`).
HANG_ABDUCTION_DEG = 10.0

# (SMPL-X joint, MediaPipe pose landmark) for the joints V1 models. Legs are
# excluded because MediaPipe reports visibility near zero for them in
# upper-body-framed video; face and hands are out of V1's scope.
ARM_CORRESPONDENCE = [
    (16, 11),  # left shoulder
    (17, 12),  # right shoulder
    (18, 13),  # left elbow
    (19, 14),  # right elbow
    (20, 15),  # left wrist
    (21, 16),  # right wrist
]


@dataclass
class BodyFit:
    """
    Genuine SMPL-X parameters; valid input to the real model's forward pass.

    `betas` holds as many coefficients as the fitting skeleton's shape space --
    300 for a default `SMPLXSkeleton` -- so a consumer must build its SMPL-X model
    with `num_betas=len(betas)`. Truncating to the conventional ten describes a
    different body.

    `placed` marks, per MediaPipe landmark, which positions the parameters
    actually determine. A joint that was not observed leaves its bone hanging down
    the torso (so the mesh is not in a T-pose); reconstruction reports such
    landmarks as missing rather than inventing them. None means every modelled
    landmark was placed.
    """

    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    betas: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BETAS))
    placed: Optional[np.ndarray] = None


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(v)
    return None if norm < 1e-12 else v / norm


def _torso_frame(axis: np.ndarray, toward_apex: np.ndarray) -> Optional[np.ndarray]:
    """Right-handed frame: x along the shoulder axis, y toward the apex."""
    up = _unit(toward_apex)
    if up is None:
        return None
    z = np.cross(axis, up)
    norm = np.linalg.norm(z)
    if norm < 1e-9:
        return None
    z = z / norm
    return np.column_stack([axis, np.cross(z, axis), z])


def torso_rotation(
    rest_left: np.ndarray,
    rest_right: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    rest_apex: Optional[np.ndarray] = None,
    obs_apex: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rotation carrying the rest torso orientation onto the observed one.

    The shoulder axis fixes two of the three rotational degrees of freedom. The
    third, rotation about the shoulder axis, moves neither shoulder. It is taken
    from the apex -- a point off the shoulder line, such as the nose -- when one
    is supplied and not degenerate, and otherwise from the minimal rotation
    between the two shoulder axes. That last resort is only safe when both
    frames share an up direction, which MediaPipe's y-down world does not, so
    `fit_body` always supplies an apex.

    That third freedom cannot change any reconstructed arm landmark: the arms are
    solved in world directions and hang from shoulders it leaves in place. The
    apex makes the root anatomically plausible without touching round-trip error.
    """
    rest_left, rest_right = np.asarray(rest_left, float), np.asarray(rest_right, float)
    obs_left, obs_right = np.asarray(obs_left, float), np.asarray(obs_right, float)

    axis_rest = _unit(rest_right - rest_left)
    axis_obs = _unit(obs_right - obs_left)
    if axis_rest is None or axis_obs is None:
        raise ValueError("shoulders coincide, so the torso axis is undefined")

    if rest_apex is not None and obs_apex is not None:
        frame_rest = _torso_frame(axis_rest, np.asarray(rest_apex, float)
                                  - 0.5 * (rest_left + rest_right))
        frame_obs = _torso_frame(axis_obs, np.asarray(obs_apex, float)
                                 - 0.5 * (obs_left + obs_right))
        if frame_rest is not None and frame_obs is not None:
            return frame_obs @ frame_rest.T

    return swing_rotation(axis_rest, axis_obs)


def _torso_apex(landmarks: np.ndarray, usable: np.ndarray, skeleton):
    """
    The point that fixes rotation about the shoulder axis: (rest, observed).

    The nose when it is visible. Otherwise the hip centre -- observed when both
    hips are valid, and otherwise the origin, because MediaPipe's world landmarks
    are hip-centred (within ~1 cm on the sample clips). Without either, the root
    would fall back to the minimal rotation between shoulder axes, which in
    MediaPipe's y-down frame turns the body upside down.
    """
    if usable[NOSE_MP]:
        return skeleton.rest_nose, landmarks[NOSE_MP]

    rest = skeleton.rest_joints
    rest_hips = 0.5 * (rest[LEFT_HIP_SMPLX] + rest[RIGHT_HIP_SMPLX])
    if usable[LEFT_HIP_MP] and usable[RIGHT_HIP_MP]:
        return rest_hips, 0.5 * (landmarks[LEFT_HIP_MP] + landmarks[RIGHT_HIP_MP])
    return rest_hips, np.zeros(3)


def _placed_landmarks(usable: np.ndarray) -> np.ndarray:
    """
    Landmarks whose whole bone chain from the shoulder was observed. A valid wrist
    below an invalid elbow is not placed: no observed direction reaches it.
    """
    placed = np.zeros(NUM_POSE_LANDMARKS, dtype=bool)
    placed[[LEFT_SHOULDER_MP, RIGHT_SHOULDER_MP]] = True
    for _, _, mp_parent, mp_child in SMPLX_ARM_BONES:   # parents precede children
        placed[mp_child] = placed[mp_parent] and usable[mp_parent] and usable[mp_child]
    return placed


def _hang_unplaced_bones(rotations: np.ndarray, skeleton, placed: np.ndarray) -> np.ndarray:
    """
    Arm bones with no observed direction, turned to hang down the torso instead of resting.

    SMPL-X rests in a T-pose, so an unsolved bone would stick straight out. In sign-language
    video an arm MediaPipe cannot see is almost always lowered out of frame, so it hangs down
    the rest torso frame's -y, tilted `HANG_ABDUCTION_DEG` outward to clear the hip. Only the
    mesh shows this: the bones' landmarks stay unplaced, and reconstruction reports them missing.
    """
    rest = skeleton.rest_joints
    parents = np.asarray(skeleton.parents, dtype=int)
    local = Rotation.from_rotvec(rotations).as_matrix()
    root, a = local[0], np.radians(HANG_ABDUCTION_DEG)

    def world(j):
        out = np.eye(3)
        while j >= 0:
            out = local[j] @ out
            j = parents[j]
        return out

    for joint, child, _, mp_child in SMPLX_ARM_BONES:   # parents precede children
        if placed[mp_child]:
            continue
        outward = np.sign(rest[joint, 0])   # SMPL-X's left side is +x
        target = root @ np.array([outward * np.sin(a), -np.cos(a), 0.0])
        d_rest = _unit(rest[child] - rest[joint])
        local[joint] = swing_rotation(d_rest, world(parents[joint]).T @ target)
    return Rotation.from_matrix(local).as_rotvec()


def fit_body(landmarks: np.ndarray, validity: np.ndarray, skeleton) -> BodyFit:
    """
    Fit SMPL-X parameters to one frame of world-coordinate body landmarks.

    Args:
        landmarks: (33, 3) MediaPipe world landmarks, metres.
        validity: (33,) bool; False marks a landmark to ignore, e.g. low
            visibility. Invalid landmarks never influence the fit.
        skeleton: SMPL-X skeleton source exposing `rest_nose`, in the body shape
            to fit; its `betas` are recorded in the result. See `fit_shape`.

    Raises:
        ValueError: either shoulder is invalid. The shoulders anchor the torso
            frame; without both, the root rotation is undefined.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    usable = np.asarray(validity, dtype=bool) & np.isfinite(landmarks).all(axis=1)

    if not (usable[LEFT_SHOULDER_MP] and usable[RIGHT_SHOULDER_MP]):
        raise ValueError("both shoulders must be valid: they anchor the torso frame")

    rest = skeleton.rest_joints
    rest_apex, obs_apex = _torso_apex(landmarks, usable, skeleton)
    root = torso_rotation(
        rest[LEFT_SHOULDER_SMPLX], rest[RIGHT_SHOULDER_SMPLX],
        landmarks[LEFT_SHOULDER_MP], landmarks[RIGHT_SHOULDER_MP],
        rest_apex=rest_apex, obs_apex=obs_apex,
    )

    masked = np.where(usable[:, None], landmarks, np.nan)
    rotations = solve_swing_ik(masked, skeleton, SMPLX_ARM_BONES,
                               global_orient=Rotation.from_matrix(root).as_rotvec())
    placed = _placed_landmarks(usable)
    rotations = _hang_unplaced_bones(rotations, skeleton, placed)
    posed = forward_kinematics(rotations, skeleton)

    residuals = [landmarks[m] - posed[s] for s, m in ARM_CORRESPONDENCE if placed[m]]

    return BodyFit(
        body_pose=rotations[1:22].reshape(63).copy(),
        global_orient=rotations[0].copy(),
        transl=np.mean(residuals, axis=0),
        betas=np.array(getattr(skeleton, "betas", np.zeros(NUM_BETAS)), dtype=np.float64),
        placed=placed,
    )


def _in_shape(skeleton, betas: np.ndarray):
    """`skeleton` in the body shape `betas` describes."""
    betas = np.asarray(betas, dtype=np.float64)
    if hasattr(skeleton, "shaped"):
        return skeleton.shaped(betas)
    if np.any(betas != 0):
        raise ValueError("this skeleton has no shape space, so it cannot apply nonzero betas")
    return skeleton


def reconstruct_body(fit: BodyFit, skeleton) -> np.ndarray:
    """
    Rebuild world-coordinate landmarks from SMPL-X parameters alone.

    `skeleton` supplies the body model; the body shape comes from `fit.betas`.

    Returns (33, 3). Landmarks outside V1's model, and modelled landmarks the fit
    could not place (`fit.placed`), are NaN rather than invented.
    """
    body = _in_shape(skeleton, fit.betas)
    rotations = np.zeros((body.num_joints, 3))
    rotations[0] = fit.global_orient
    rotations[1:22] = np.asarray(fit.body_pose).reshape(21, 3)

    joints = forward_kinematics(rotations, body) + np.asarray(fit.transl)

    out = np.full((NUM_POSE_LANDMARKS, 3), np.nan)
    for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
        out[mp_idx] = joints[smplx_idx]
    if fit.placed is not None:
        out[~np.asarray(fit.placed, dtype=bool)] = np.nan
    return out
