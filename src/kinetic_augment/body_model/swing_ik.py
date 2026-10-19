"""
Analytic swing inverse kinematics: MediaPipe landmarks -> SMPL-X rotations.

For each bone, the rotation carrying its rest direction onto its observed
direction splits into a *swing* -- the minimal rotation between two directions --
and a *twist* about the bone's own axis. Joint positions constrain the swing
exactly and constrain the twist not at all, since twisting a bone moves no joint.
So the swing is solved in closed form and the twist is set to zero by policy,
rather than left for an optimiser to resolve arbitrarily.

Rotations are solved in parent-before-child order and expressed relative to the
parent, so each local rotation is the smallest one that explains its bone given
where the parent already placed it.

This produces genuine SMPL-X axis-angle parameters. It is deterministic, has no
local minima, and needs no optimisation loop.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# (driven_joint, child_joint, mp_parent_landmark, mp_child_landmark)
# The rotation stored at `driven_joint` orients the bone running to `child_joint`,
# which is observed as the displacement between the two MediaPipe landmarks.
BoneSpec = Tuple[int, int, int, int]

# The four joints Finding 5 of the V1 scope established are constrainable on
# upper-body-framed signing video. Legs are omitted: MediaPipe extrapolates them
# past the image boundary, so fitting them would fit the model's prior.
SMPLX_ARM_BONES: List[BoneSpec] = [
    (16, 18, 11, 13),  # left shoulder  -> upper arm
    (18, 20, 13, 15),  # left elbow     -> forearm
    (17, 19, 12, 14),  # right shoulder -> upper arm
    (19, 21, 14, 16),  # right elbow    -> forearm
]

_EPS = 1e-12


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    return None if n < _EPS else v / n


def swing_rotation(d_from: np.ndarray, d_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix carrying unit vector `d_from` onto `d_to`."""
    axis = np.cross(d_from, d_to)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(d_from, d_to))

    if sin_a < _EPS:
        if cos_a > 0.0:
            return np.eye(3)
        # Exactly antiparallel: the axis is undefined, so any perpendicular one
        # gives a valid 180-degree rotation.
        fallback = np.cross(d_from, [1.0, 0.0, 0.0])
        if np.linalg.norm(fallback) < 1e-9:
            fallback = np.cross(d_from, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(fallback / np.linalg.norm(fallback) * np.pi).as_matrix()

    return Rotation.from_rotvec(axis / sin_a * np.arctan2(sin_a, cos_a)).as_matrix()


def _observed(landmarks: np.ndarray, index: int) -> bool:
    """
    Whether a landmark carries an actual measurement.

    Two missing-data conventions are rejected. Image-space MediaPipe writes exact
    zeros for undetected landmarks. World landmarks are hip-centred, so the origin
    is a legitimate position there and cannot double as a sentinel; those use NaN
    instead. Treating either as real data produces a silently wrong fit, or an
    SVD that fails to converge.
    """
    point = landmarks[index]
    return bool(np.all(np.isfinite(point)) and np.any(point != 0.0))


def solve_swing_ik(
    landmarks: np.ndarray,
    skeleton,
    bones: Sequence[BoneSpec],
    global_orient: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit local joint rotations so each bone points along its observed direction.

    Args:
        landmarks: (L, 3) landmark positions. Only directions are read, so the
            result is invariant to the scale and translation of this input.
        skeleton: object exposing `rest_joints`, `parents` and `num_joints`.
        bones: which joint each bone drives and which landmarks observe it.
        global_orient: optional (3,) axis-angle root rotation. Bones are solved
            relative to it, so supplying it changes the local rotations rather
            than being overwritten by them.

    Returns:
        (num_joints, 3) local rotations in axis-angle form. Joints with no bone,
        and bones whose landmarks are undetected, are left at zero.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    needed = max(max(b[2], b[3]) for b in bones) + 1 if bones else 0
    if landmarks.ndim != 2 or landmarks.shape[1] != 3 or landmarks.shape[0] < needed:
        raise ValueError(
            f"landmarks must be (L, 3) with L >= {needed}, got {landmarks.shape}"
        )

    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=int)
    num_joints = skeleton.num_joints

    local = np.tile(np.eye(3), (num_joints, 1, 1))
    if global_orient is not None:
        local[0] = Rotation.from_rotvec(np.asarray(global_orient, dtype=np.float64)).as_matrix()

    driven = {spec[0]: spec for spec in bones}
    world = np.tile(np.eye(3), (num_joints, 1, 1))

    for j in range(num_joints):
        parent = parents[j]
        parent_world = world[parent] if parent >= 0 else np.eye(3)

        spec = driven.get(j)
        if spec is not None:
            _, child, mp_parent, mp_child = spec
            if _observed(landmarks, mp_parent) and _observed(landmarks, mp_child):
                d_rest = _unit(rest[child] - rest[j])
                d_obs = _unit(landmarks[mp_child] - landmarks[mp_parent])
                if d_rest is not None and d_obs is not None:
                    # Undo the parent's rotation so the solved value is the local
                    # one: the parent has already carried this bone part of the way.
                    local[j] = swing_rotation(d_rest, parent_world.T @ d_obs)

        world[j] = parent_world @ local[j]

    return Rotation.from_matrix(local).as_rotvec()
