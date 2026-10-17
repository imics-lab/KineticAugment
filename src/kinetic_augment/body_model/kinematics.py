"""
Forward kinematics over a rest-pose skeleton.

Composes per-joint local rotations down the kinematic tree to produce world
joint positions. This mirrors the rigid transform chain SMPL-X applies to its
own joints, so the positions returned here are directly comparable to the first
55 entries of the model's `joints` output.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def forward_kinematics(rotations: np.ndarray, skeleton) -> np.ndarray:
    """
    Pose a skeleton by composing local rotations down its kinematic chain.

    Args:
        rotations: (J, 3) local rotations in axis-angle form, one per joint,
            ordered to match the skeleton. Index 0 is the root orientation.
        skeleton: object exposing `rest_joints` (J, 3), `parents` (J,) and
            `num_joints`.

    Returns:
        (J, 3) posed joint positions in world space.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    expected = (skeleton.num_joints, 3)
    if rotations.shape != expected:
        raise ValueError(f"rotations must have shape {expected}, got {rotations.shape}")

    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=int)
    local_rot = Rotation.from_rotvec(rotations).as_matrix()

    # Offset of each joint from its parent at rest; the root keeps its absolute
    # rest position so the posed chain stays anchored where the model puts it.
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]

    world_rot = np.zeros_like(local_rot)
    world_pos = np.zeros_like(rest)

    for j in range(skeleton.num_joints):
        p = parents[j]
        if p < 0:
            world_rot[j] = local_rot[j]
            world_pos[j] = offsets[j]
        else:
            world_rot[j] = world_rot[p] @ local_rot[j]
            world_pos[j] = world_rot[p] @ offsets[j] + world_pos[p]

    return world_pos
