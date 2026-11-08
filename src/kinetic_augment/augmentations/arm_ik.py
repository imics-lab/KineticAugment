"""
Analytic two-bone arm IK and the torso frame for the week-4 augmentations.

`two_bone_ik` moves one wrist to a world target by turning the shoulder and the elbow, keeping
the elbow in its current swivel plane and the hand at its world orientation (the fingers keep
their local rotations, so they follow the hand). Limb-length scaling, movement-size scaling and
location jitter use it to place a wrist. Everything works on one frame of SMPL-X local axis-angle
rotations (55, 3) and the rest joints (55, 3) the forward kinematics reads; callers skip frames
whose rotations are not finite.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .tier1 import SMPLX_ARM

NECK, SPINE3 = 12, 9
LEFT_SHOULDER, RIGHT_SHOULDER = SMPLX_ARM["left"]["shoulder"], SMPLX_ARM["right"]["shoulder"]
REACH_MARGIN = 1e-6  # metres kept away from full extension and full fold
DEGENERATE = 1e-6  # below this length a pole or direction is undefined


def world_pose(rotations, rest_joints, parents) -> Tuple[np.ndarray, np.ndarray]:
    """
    World rotation matrices (J, 3, 3) and joint positions (J, 3) of one frame.

    The same chain as `kinematics.forward_kinematics` and `demo_smplx_augmentation.world_pose`:
    each bone offset is `rest[j] - rest[parent[j]]`, the root sits at `rest[0]` and no
    translation is applied. Parents must precede their children.
    """
    local = Rotation.from_rotvec(np.asarray(rotations, dtype=np.float64)).as_matrix()
    rest = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    rot, pos = np.zeros_like(local), np.zeros_like(rest)
    for j in range(len(local)):
        p = parents[j]
        if p < 0:
            rot[j], pos[j] = local[j], rest[j]
        else:
            rot[j] = rot[p] @ local[j]
            pos[j] = rot[p] @ (rest[j] - rest[p]) + pos[p]
    return rot, pos


def torso_frame(pos) -> Tuple[np.ndarray, np.ndarray]:
    """
    The torso's origin (3,) and axes (3, 3) from posed joint positions (55, 3).

    The origin is the mid-shoulder point. The rows are x = from the left shoulder (16) to the
    right shoulder (17), y = up, from spine3 (9) to the neck (12), made orthogonal to x, and
    z = x × y, so the axes are a proper rotation. With the subject's left at +x and up at +y (the
    SMPL-X rest), x points to the subject's right and z points behind the body.
    """
    pos = np.asarray(pos, dtype=np.float64)
    origin = (pos[LEFT_SHOULDER] + pos[RIGHT_SHOULDER]) / 2.0
    x = _unit(pos[RIGHT_SHOULDER] - pos[LEFT_SHOULDER])
    up = pos[NECK] - pos[SPINE3]
    y = _unit(up - np.dot(up, x) * x)
    z = np.cross(x, y)
    return origin, np.stack([x, y, z])


def two_bone_ik(rotations, rest_joints, parents, side: str, target) -> Tuple[np.ndarray, float]:
    """
    One frame's rotations (55, 3) with the `side` wrist moved to `target`, and the residual (m).

    The shoulder–target distance is clipped to what the arm can reach (rest bone lengths a, b,
    kept 1e-6 m from full extension and full fold), so an unreachable target straightens the arm
    toward it and the residual is the distance left over. The elbow stays on the same side of the
    shoulder–target line as before (the torso's z axis is the pole when it lies on the line). The
    shoulder swings the upper arm onto the solved elbow, the elbow swings the forearm onto the
    solved wrist, and the wrist's local rotation is reset so its world rotation is unchanged.
    Only the three joints of that arm change.
    """
    arm = SMPLX_ARM[side]
    shoulder, elbow, wrist = arm["shoulder"], arm["elbow"], arm["wrist"]
    rest = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    target = np.asarray(target, dtype=np.float64)
    out = np.array(rotations, dtype=np.float64, copy=True)

    rot, pos = world_pose(out, rest, parents)
    wrist_world = rot[wrist].copy()
    s, e, w = pos[shoulder], pos[elbow], pos[wrist]
    a = np.linalg.norm(rest[elbow] - rest[shoulder])
    b = np.linalg.norm(rest[wrist] - rest[elbow])

    d = target - s
    distance = np.linalg.norm(d)
    u = _unit(d) if distance >= DEGENERATE else _unit(w - s)
    if u is None:  # target and wrist both on the shoulder: keep the upper-arm direction
        u = _unit(e - s)
    length = np.clip(distance, abs(a - b) + REACH_MARGIN, a + b - REACH_MARGIN)

    p = _pole(e - s, u, pos)
    cos_alpha = np.clip((a * a + length * length - b * b) / (2.0 * a * length), -1.0, 1.0)
    sin_alpha = np.sqrt(1.0 - cos_alpha * cos_alpha)
    e_new = s + a * (cos_alpha * u + sin_alpha * p)
    w_new = s + length * u

    delta = _swing(e - s, e_new - s)
    _rotate_in_world(out, rot, shoulder, parents[shoulder], delta)
    rot, pos = world_pose(out, rest, parents)
    delta = _swing(pos[wrist] - pos[elbow], w_new - pos[elbow])
    _rotate_in_world(out, rot, elbow, parents[elbow], delta)
    rot, _ = world_pose(out, rest, parents)
    out[wrist] = Rotation.from_matrix(rot[parents[wrist]].T @ wrist_world).as_rotvec()
    return out, float(np.linalg.norm(w_new - target))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def _pole(upper, u, pos):
    """The unit direction, perpendicular to `u`, from the shoulder–target line to the elbow."""
    pole = upper - np.dot(upper, u) * u
    if np.linalg.norm(pole) >= DEGENERATE:
        return pole / np.linalg.norm(pole)
    _, axes = torso_frame(pos)
    for axis in (axes[2], axes[1], axes[0]):  # behind the body first; any axis off the line
        pole = axis - np.dot(axis, u) * u
        if np.linalg.norm(pole) >= DEGENERATE:
            return pole / np.linalg.norm(pole)
    raise ValueError("torso frame is degenerate")  # pragma: no cover - axes are orthonormal


def _swing(d_from, d_to) -> np.ndarray:
    """The smallest rotation (3, 3) turning direction `d_from` onto `d_to`."""
    f, t = _unit(d_from), _unit(d_to)
    if f is None or t is None:
        return np.eye(3)
    axis = np.cross(f, t)
    sin = np.linalg.norm(axis)
    cos = float(np.dot(f, t))
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        # Opposite directions: half a turn about any axis perpendicular to f.
        helper = np.eye(3)[int(np.argmin(np.abs(f)))]
        axis = np.cross(f, helper)
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(axis / sin * np.arctan2(sin, cos)).as_matrix()


def _rotate_in_world(rotations, world_rot, joint, parent, delta):
    """Turn `joint` (and its child chain) by world rotation `delta` about the joint, in place."""
    local = Rotation.from_rotvec(rotations[joint]).as_matrix()
    new_local = world_rot[parent].T @ delta @ world_rot[parent] @ local
    rotations[joint] = Rotation.from_matrix(new_local).as_rotvec()
