"""
Head placement from several observed face points (08925 head-placement spike).

`head_converter.split_neck_head` swings the neck so one point, the SMPL-X ear midpoint, lands on
its observation; the head then takes the exact remainder of the head's world rotation. This module
generalises the target: with the head's world rotation R_head fixed, each rigid head point k sits
at J_head + R_head o_k (o_k its rest offset from the head joint), so every observed point votes for
a head-joint position observed_k - R_head o_k. The neck swings so the neck-to-head-joint direction
points at the weighted mean of those votes. Twist share and the exact head remainder are as in
`split_neck_head`; the neck bone has fixed length, so only the direction can be matched.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .assembly import HEAD, NECK, global_rotations
from .head_converter import NECK_YAW_SHARE, twist_angle
from .kinematics import forward_kinematics
from .swing_ik import swing_rotation


@dataclass
class PointNeck:
    neck: np.ndarray  # (3,) neck local axis-angle
    head: np.ndarray  # (3,) head local axis-angle
    head_joint_target: np.ndarray  # (3,) weighted mean of the per-point head-joint votes
    points_used: int  # points with positive weight and a finite observation


def _frame(rotations, transl, skeleton) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Torso world rotation above the neck, neck joint position, neck bone, its unit axis."""
    rot = np.array(rotations, dtype=np.float64)
    rot[[NECK, HEAD]] = 0.0
    parents = skeleton.parents
    torso = global_rotations(rot, parents)[parents[NECK]]
    p_neck = (forward_kinematics(rot, skeleton) + np.asarray(transl))[NECK]
    rest = skeleton.rest_joints
    bone = rest[HEAD] - rest[NECK]
    return torso, p_neck, bone, bone / np.linalg.norm(bone)


def rest_neck(rotations, transl, skeleton, head_world) -> PointNeck:
    """Neck at rest; the head takes all of `head_world` (the baseline placement)."""
    torso, p_neck, bone, _ = _frame(rotations, transl, skeleton)
    head = torso.T @ head_world
    return PointNeck(np.zeros(3), Rotation.from_matrix(head).as_rotvec(), p_neck + torso @ bone, 0)


def fit_neck_to_points(
    rotations, transl, skeleton, offsets, observed, weights, head_world
) -> Optional[PointNeck]:
    """
    Neck and head local rotations that put the rigid head's points on weighted observations.

    Args:
        rotations: (55, 3) body rotations; neck and head entries are ignored.
        transl: (3,) body translation.
        skeleton: SMPLXSkeleton in the clip's shape.
        offsets: (K, 3) each point's rest offset from the rest head joint, metres.
        observed: (K, 3) the observed points in the pose-world frame, metres.
        weights: (K,) point weights; 0 drops a point.
        head_world: (3, 3) the head's world rotation, reproduced exactly.

    Returns:
        None when no point has positive weight and a finite observation.
    """
    offsets, observed = np.asarray(offsets, np.float64), np.asarray(observed, np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    use = (weights > 0) & np.isfinite(observed).all(axis=1) & np.isfinite(offsets).all(axis=1)
    if not use.any() or not np.isfinite(head_world).all():
        return None
    votes = observed[use] - offsets[use] @ head_world.T
    target = (weights[use] @ votes) / weights[use].sum()

    torso, p_neck, bone, axis = _frame(rotations, transl, skeleton)
    direction = target - p_neck
    swing_world = swing_rotation(torso @ axis, direction / np.linalg.norm(direction))
    swing = torso.T @ swing_world @ torso
    relative = torso.T @ head_world
    twist = Rotation.from_rotvec(NECK_YAW_SHARE * twist_angle(relative, axis) * axis)
    neck = swing @ twist.as_matrix()
    head = (torso @ neck).T @ head_world
    return PointNeck(
        Rotation.from_matrix(neck).as_rotvec(),
        Rotation.from_matrix(head).as_rotvec(),
        target,
        int(use.sum()),
    )
