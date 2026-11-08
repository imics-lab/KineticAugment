"""
Tests for the analytic two-bone arm IK and the torso frame (`augmentations.arm_ik`).

The fixture is the real neutral SMPL-X skeleton in its rest pose with both elbows bent 60
degrees (`bent_pose`), so the arm has a well-defined swivel plane.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.augmentations.arm_ik import torso_frame, two_bone_ik, world_pose
from kinetic_augment.augmentations.tier1 import SMPLX_ARM
from tests.clip_fixtures import get_skeleton
from tests.hand_assets import requires_smplx

pytestmark = requires_smplx

SIDES = ("left", "right")


def bent_pose(degrees: float = 60.0) -> np.ndarray:
    """
    Rest-pose rotations (55, 3) with both elbows bent `degrees` about their local y axis.

    In the SMPL-X rest pose the left arm points along +x and the right along -x, so a turn of
    -degrees (left) and +degrees (right) about y swings each forearm forward, toward +z.
    """
    rotations = np.zeros((55, 3))
    angle = np.radians(degrees)
    rotations[SMPLX_ARM["left"]["elbow"]] = [0.0, -angle, 0.0]
    rotations[SMPLX_ARM["right"]["elbow"]] = [0.0, angle, 0.0]
    return rotations


def _setup():
    skeleton = get_skeleton()
    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=int)
    return bent_pose(), rest, parents


def _bone_lengths(rest, side):
    arm = SMPLX_ARM[side]
    a = np.linalg.norm(rest[arm["elbow"]] - rest[arm["shoulder"]])
    b = np.linalg.norm(rest[arm["wrist"]] - rest[arm["elbow"]])
    return a, b


@pytest.mark.parametrize("side", SIDES)
def test_ik_reaches_reachable_target(side):
    rotations, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    rot0, pos0 = world_pose(rotations, rest, parents)
    target = pos0[arm["wrist"]] + np.array([0.03, -0.02, 0.04])

    out, residual = two_bone_ik(rotations, rest, parents, side, target)
    rot1, pos1 = world_pose(out, rest, parents)

    assert residual < 1e-6
    assert np.linalg.norm(pos1[arm["wrist"]] - target) < 1e-6
    a, b = _bone_lengths(rest, side)
    assert abs(np.linalg.norm(pos1[arm["elbow"]] - pos1[arm["shoulder"]]) - a) < 1e-9
    assert abs(np.linalg.norm(pos1[arm["wrist"]] - pos1[arm["elbow"]]) - b) < 1e-9
    assert np.allclose(rot1[arm["wrist"]], rot0[arm["wrist"]], atol=1e-9)
    # The shoulder itself does not move, and every joint outside this arm (the fingers, the
    # other arm, the torso and head) keeps its local rotation bit for bit.
    assert np.allclose(pos1[arm["shoulder"]], pos0[arm["shoulder"]], atol=1e-12)
    others = np.setdiff1d(np.arange(55), list(arm.values()))
    assert np.array_equal(out[others], rotations[others])


@pytest.mark.parametrize("side", SIDES)
def test_ik_elbow_stays_a_hinge(side):
    """The elbow's local rotation only turns further about its own local flexion axis."""
    rotations, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    _, pos0 = world_pose(rotations, rest, parents)
    target = pos0[arm["wrist"]] + np.array([0.03, -0.02, 0.04])  # about 5.4 cm

    out, residual = two_bone_ik(rotations, rest, parents, side, target)

    assert residual < 1e-6
    old = Rotation.from_rotvec(rotations[arm["elbow"]])
    axis = rotations[arm["elbow"]] / np.linalg.norm(rotations[arm["elbow"]])
    change = (Rotation.from_rotvec(out[arm["elbow"]]) * old.inv()).as_rotvec()
    off_hinge = change - np.dot(change, axis) * axis
    assert np.linalg.norm(off_hinge) < 1e-6
    assert abs(np.dot(change, axis)) > np.radians(1.0)  # the elbow did bend


@pytest.mark.parametrize("side", SIDES)
def test_ik_keeps_swivel_plane(side):
    rotations, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    _, pos0 = world_pose(rotations, rest, parents)
    shoulder = pos0[arm["shoulder"]]
    target = pos0[arm["wrist"]] + np.array([0.03, -0.02, 0.04])
    u = (target - shoulder) / np.linalg.norm(target - shoulder)

    def off_line(point):
        v = point - shoulder
        return v - np.dot(v, u) * u

    pole = off_line(pos0[arm["elbow"]])
    out, _ = two_bone_ik(rotations, rest, parents, side, target)
    _, pos1 = world_pose(out, rest, parents)
    assert np.dot(off_line(pos1[arm["elbow"]]), pole) > 0


@pytest.mark.parametrize("side", SIDES)
def test_ik_unreachable_target_extends(side):
    rotations, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    _, pos0 = world_pose(rotations, rest, parents)
    shoulder = pos0[arm["shoulder"]]
    a, b = _bone_lengths(rest, side)
    direction = np.array([0.3, -0.5, 0.6])
    direction /= np.linalg.norm(direction)
    target = shoulder + direction * (a + b + 1.0)

    out, residual = two_bone_ik(rotations, rest, parents, side, target)
    _, pos1 = world_pose(out, rest, parents)

    # The elbow is a hinge about its own local axis, and the SMPL-X rest arm is not straight, so
    # its full extension falls a few millimetres short of a + b (8.6 mm left, under 1 mm right).
    # The arm straightens toward the target as far as the hinge allows, and the residual is the
    # distance left over.
    assert np.all(np.isfinite(out))
    wrist = pos1[arm["wrist"]]
    reach = np.linalg.norm(wrist - shoulder)
    assert a + b - 0.01 < reach <= a + b
    assert np.allclose((wrist - shoulder) / reach, direction, atol=1e-9)
    assert abs(residual - np.linalg.norm(wrist - target)) < 1e-9
    assert abs(residual - 1.0) < 0.01
    # Fully extended: a target further along the same line leaves the arm where it is.
    further, _ = two_bone_ik(rotations, rest, parents, side, shoulder + direction * (a + b + 2.0))
    assert np.allclose(further, out, atol=1e-9)


@pytest.mark.parametrize("side", SIDES)
def test_ik_zero_move_is_identity(side):
    rotations, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    _, pos0 = world_pose(rotations, rest, parents)

    out, residual = two_bone_ik(rotations, rest, parents, side, pos0[arm["wrist"]])

    assert residual < 1e-9
    assert np.allclose(out, rotations, atol=1e-9)


def test_world_pose_matches_forward_kinematics():
    from kinetic_augment.body_model.kinematics import forward_kinematics

    skeleton = get_skeleton()
    rotations = Rotation.random(55, random_state=3).as_rotvec() * 0.3
    _, pos = world_pose(rotations, skeleton.rest_joints, skeleton.parents)
    assert np.allclose(pos, forward_kinematics(rotations, skeleton), atol=1e-6)


def test_torso_frame_is_orthonormal():
    rotations, rest, parents = _setup()
    rotations[0] = [0.2, -0.4, 0.1]
    rotations[9] = [0.1, 0.05, -0.2]
    _, pos = world_pose(rotations, rest, parents)

    origin, axes = torso_frame(pos)

    assert np.allclose(origin, (pos[16] + pos[17]) / 2)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0)
    x = (pos[17] - pos[16]) / np.linalg.norm(pos[17] - pos[16])
    assert np.allclose(axes[0], x)
    assert np.dot(axes[1], pos[12] - pos[9]) > 0


@pytest.mark.parametrize("side", SIDES)
def test_ik_straight_arm_uses_torso_pole(side):
    """
    A straight arm has no swivel plane, so the torso's z axis (behind the body) is the pole.

    The SMPL-X rest arm is not straight (the elbow sits 1.6-5.1 cm off the shoulder-wrist line),
    so this copy of the rest joints moves the elbow onto that line, keeping the wrist where it is.
    """
    _, rest, parents = _setup()
    arm = SMPLX_ARM[side]
    rest = rest.copy()
    s_rest, w_rest = rest[arm["shoulder"]], rest[arm["wrist"]]
    a, b = _bone_lengths(rest, side)
    rest[arm["elbow"]] = s_rest + (w_rest - s_rest) * a / (a + b)
    rotations = np.zeros((55, 3))
    _, pos0 = world_pose(rotations, rest, parents)
    shoulder, elbow = pos0[arm["shoulder"]], pos0[arm["elbow"]]
    line = (pos0[arm["wrist"]] - shoulder) / np.linalg.norm(pos0[arm["wrist"]] - shoulder)
    off = (elbow - shoulder) - np.dot(elbow - shoulder, line) * line
    assert np.linalg.norm(off) < 1e-9  # truly collinear: the pole falls back to the torso
    target = shoulder + 0.7 * (pos0[arm["wrist"]] - shoulder)

    out, residual = two_bone_ik(rotations, rest, parents, side, target)
    _, pos1 = world_pose(out, rest, parents)

    assert np.all(np.isfinite(out))
    assert residual < 1e-6
    assert np.linalg.norm(pos1[arm["wrist"]] - target) < 1e-6
    _, axes = torso_frame(pos0)
    z = axes[2] - np.dot(axes[2], line) * line
    z /= np.linalg.norm(z)
    moved = pos1[arm["elbow"]] - shoulder
    moved_off = moved - np.dot(moved, line) * line
    assert np.linalg.norm(moved_off) > 0.05  # the elbow left the line
    assert np.allclose(moved_off / np.linalg.norm(moved_off), z, atol=1e-9)  # straight back
