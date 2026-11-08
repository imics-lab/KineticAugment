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

    The elbow is a hinge. Its local rotation only turns further about its own local axis (the
    axis of its current axis-angle rotation; for an unbent elbow, the normal of the arm's plane,
    or a bend toward the torso's z axis when the arm is straight). That turn sets the
    shoulder–wrist distance to |target − shoulder|, clipped to what the arm can reach (rest bone
    lengths a, b, kept 1e-6 m from full extension and full fold, and the hinge's own range). The
    shoulder then turns the now-rigid arm so the wrist lies on the shoulder–target line and the
    elbow stays on the same side of that line as before (the torso's z axis is the pole when the
    elbow lies on the line). Finally the wrist's local rotation is reset so its world rotation is
    unchanged; the fingers keep their local rotations. Only the three joints of that arm change.
    An unreachable target straightens the arm toward it, and the residual is the distance left.
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

    # Elbow: turn about its hinge until the shoulder–wrist distance is `length`.
    hinge = _hinge(out[elbow], rot[shoulder], e - s, w - e, p)
    theta = _hinge_angle(e - s, w - e, hinge, (length * length - a * a - b * b) / 2.0)
    turn = Rotation.from_rotvec(rot[shoulder].T @ hinge * theta)
    out[elbow] = (turn * Rotation.from_rotvec(out[elbow])).as_rotvec()

    # Shoulder: carry the rigid arm's frame (wrist direction, elbow side) onto (u, p).
    rot, pos = world_pose(out, rest, parents)
    reach = pos[wrist] - s
    along = _unit(reach)
    side_dir = _unit((pos[elbow] - s) - np.dot(pos[elbow] - s, along) * along)
    if side_dir is None:  # arm straight after the hinge turn: any perpendicular will do
        side_dir = _unit(np.cross(hinge, along))
    old = np.stack([along, side_dir, np.cross(along, side_dir)], axis=1)
    new = np.stack([u, p, np.cross(u, p)], axis=1)
    _rotate_in_world(out, rot, shoulder, parents[shoulder], new @ old.T)

    rot, _ = world_pose(out, rest, parents)
    out[wrist] = Rotation.from_matrix(rot[parents[wrist]].T @ wrist_world).as_rotvec()
    return out, float(np.linalg.norm(s + np.linalg.norm(reach) * u - target))


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


def _hinge(elbow_rotvec, shoulder_world, upper, fore, pole):
    """
    The elbow's hinge axis in world coordinates (unit, (3,)).

    A bent elbow keeps the axis of its local rotation (in the shoulder's frame), so further
    flexion stays about that one local axis. An unbent elbow uses the normal of the arm's plane,
    and a straight arm bends in the plane holding the pole (the torso's z axis on the line).
    """
    angle = np.linalg.norm(elbow_rotvec)
    if angle > DEGENERATE:
        return shoulder_world @ (np.asarray(elbow_rotvec) / angle)
    normal = np.cross(_unit(upper), _unit(fore))
    if np.linalg.norm(normal) > DEGENERATE:
        return normal / np.linalg.norm(normal)
    return _unit(np.cross(upper, pole))


def _hinge_angle(upper, fore, hinge, target_dot) -> float:
    """
    The smallest turn of `fore` about `hinge` that makes upper · fore equal `target_dot`.

    upper · R(θ)fore = C + A cos θ + B sin θ; when `target_dot` lies outside the reachable range
    the turn goes to the nearest end of it (the arm straightens or folds as far as it can).
    """
    parallel = np.dot(fore, hinge) * hinge
    c0 = np.dot(upper, parallel)
    a_coef = np.dot(upper, fore - parallel)
    b_coef = np.dot(upper, np.cross(hinge, fore))
    radius = np.hypot(a_coef, b_coef)
    if radius < 1e-12:  # the forearm lies on the hinge axis: turning changes nothing
        return 0.0
    phi = np.arctan2(b_coef, a_coef)
    spread = np.arccos(np.clip((target_dot - c0) / radius, -1.0, 1.0))
    options = [_wrap(phi + spread), _wrap(phi - spread)]
    return float(min(options, key=abs))


def _wrap(angle) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _rotate_in_world(rotations, world_rot, joint, parent, delta):
    """Turn `joint` (and its child chain) by world rotation `delta` about the joint, in place."""
    local = Rotation.from_rotvec(rotations[joint]).as_matrix()
    new_local = world_rot[parent].T @ delta @ world_rot[parent] @ local
    rotations[joint] = Rotation.from_matrix(new_local).as_rotvec()
