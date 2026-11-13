"""Tests for the Tier 1 augmentations: sampling, transforms, re-timing and the frame check."""

import numpy as np
import pytest

from kinetic_augment.augmentations.tier1 import (
    SMPLX_ARM,
    ArmJointDraw,
    _world,
    perturb_arm_joints,
    POSE_MIRROR,
    TIER1,
    WeakCamera,
    about,
    duration_times,
    fit_weak_camera,
    frame_loss,
    in_frame,
    reflect_x,
    resample,
    resample_points,
    sample_tier1,
    uncentre_range,
    uncentre_shift,
    view_rotation,
)


def draws(count, seed=0):
    rng = np.random.default_rng(seed)
    return [sample_tier1(rng) for _ in range(count)]


class TestSampling:
    def test_every_value_is_inside_its_range(self):
        for d in draws(500):
            assert abs(d.yaw_deg) <= TIER1.yaw_deg
            assert abs(d.pitch_deg) <= TIER1.pitch_deg
            assert TIER1.duration[0] <= d.duration <= TIER1.duration[1]
            assert TIER1.scale[0] <= d.scale <= TIER1.scale[1]
            assert all(0.0 <= u < 1.0 for u in d.uncentre_u)
            for j in d.arm_joints:
                assert abs(j.angle_deg) <= TIER1.arm_joint_deg
                assert np.linalg.norm(j.axis) == pytest.approx(1.0)

    def test_the_same_seed_gives_the_same_draw(self):
        assert draws(3, seed=7) == draws(3, seed=7)

    def test_probabilities_match_the_ranges(self):
        sample = draws(4000)
        assert np.mean([d.mirror for d in sample]) == pytest.approx(TIER1.mirror_p, abs=0.03)
        joints = np.mean([len(d.arm_joints) for d in sample]) / 6
        assert joints == pytest.approx(TIER1.arm_joint_p, abs=0.02)

    def test_duration_is_symmetric_in_log(self):
        logs = np.log([d.duration for d in draws(4000)])
        assert np.median(logs) == pytest.approx(0.0, abs=0.02)


class TestSpatial:
    def test_zero_rotation_is_the_identity(self):
        assert np.array_equal(view_rotation(0.0, 0.0), np.eye(3))

    def test_yaw_turns_about_the_vertical_axis(self):
        rot = view_rotation(90.0, 0.0)
        assert rot @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
        assert rot @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0])

    def test_pitch_is_applied_after_yaw_in_the_fixed_frame(self):
        rot = view_rotation(90.0, 90.0)
        # x turns to -z under the yaw, then the pitch about the fixed x takes -z to +y.
        assert rot @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])

    def test_maps_act_about_the_pivot(self):
        pivot = np.array([1.0, 2.0, 3.0])
        points = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        scaled = about(points, pivot, 0.5 * np.eye(3))
        assert scaled == pytest.approx(np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0]]))

    def test_reflecting_twice_gives_the_points_back(self):
        points = np.random.default_rng(1).normal(size=(4, 33, 3))
        once = reflect_x(points, 0.3)
        assert once[..., 0] == pytest.approx(0.6 - points[..., 0])
        assert reflect_x(once, 0.3) == pytest.approx(points)

    def test_pose_mirror_swaps_pairs_and_is_its_own_inverse(self):
        assert POSE_MIRROR[11] == 12 and POSE_MIRROR[16] == 15 and POSE_MIRROR[0] == 0
        assert np.array_equal(POSE_MIRROR[POSE_MIRROR], np.arange(33))


class TestTemporal:
    def test_factor_one_keeps_every_frame(self):
        assert np.array_equal(duration_times(62, 1.0), np.arange(62.0))

    def test_ends_are_kept_and_length_scales(self):
        slow, fast = duration_times(61, 1.25), duration_times(61, 0.8)
        assert len(slow) == 76 and len(fast) == 49
        for times in (slow, fast):
            assert times[0] == 0.0 and times[-1] == 60.0
            assert np.all(np.diff(times) > 0)

    def test_resampling_interpolates_linearly(self):
        frames = [np.array([0.0]), np.array([10.0]), np.array([20.0])]
        out = resample(frames, np.array([0.0, 0.25, 1.5, 2.0]))
        assert np.concatenate(out) == pytest.approx([0.0, 2.5, 15.0, 20.0])

    def test_a_missing_neighbour_is_not_blended(self):
        frames = [np.array([0.0]), None, np.array([np.nan])]
        out = resample(frames, np.array([0.75, 1.0, 1.4]))
        assert out[0] == pytest.approx([0.0]) and out[1] is None
        assert out[2] is None

    def test_points_resample_independently(self):
        seq = np.zeros((3, 2, 3))
        seq[:, 0, 0] = [0.0, 1.0, 2.0]
        seq[1, 1] = np.nan
        out = resample_points(seq, np.array([0.5, 1.5]))
        assert out[:, 0, 0] == pytest.approx([0.5, 1.5])
        assert np.isfinite(out[:, 1]).all()


def synthetic_camera(count=5, points=6, seed=0):
    rng = np.random.default_rng(seed)
    world = rng.normal(scale=0.2, size=(count, points, 3))
    scale = np.array([-1.4, 1.9])
    offset = rng.uniform(0.3, 0.7, size=(count, 2))
    image = scale * world[..., :2] + offset[:, None]
    return world, image, scale, offset


class TestCamera:
    def test_a_weak_camera_is_recovered(self):
        world, image, scale, offset = synthetic_camera()
        camera = fit_weak_camera(world, image)
        assert camera.scale == pytest.approx(scale)
        assert camera.offset == pytest.approx(offset)
        assert camera.residual == pytest.approx(0.0, abs=1e-12)
        assert camera.project(world) == pytest.approx(image)

    def test_square_pixels_share_one_scale(self):
        world, _, _, offset = synthetic_camera()
        size = (640, 480)
        px_per_m = 700.0
        image = (px_per_m * world[..., :2]) / np.array(size) + offset[:, None]
        camera = fit_weak_camera(world, image, pixel_size=size)
        assert np.array(camera.scale) * size == pytest.approx([px_per_m, px_per_m])
        assert camera.project(world) == pytest.approx(image)

    def test_frames_without_points_borrow_the_nearest_offset(self):
        world, image, _, offset = synthetic_camera()
        world[2] = np.nan
        camera = fit_weak_camera(world, image)
        assert camera.offset[2] == pytest.approx(offset[1])

    def test_world_shift_moves_the_projection_by_the_image_shift(self):
        world, image, _, _ = synthetic_camera()
        camera = fit_weak_camera(world, image)
        moved = camera.project(world + camera.world_shift([0.05, -0.02]))
        assert moved - image == pytest.approx(np.broadcast_to([0.05, -0.02], image.shape))

    def test_resampling_the_camera_follows_the_clip(self):
        camera = WeakCamera((1.0, 1.0), np.array([[0.0, 0.0], [1.0, 2.0]]), 0.0)
        assert camera.resampled(np.array([0.5])).offset == pytest.approx(np.array([[0.5, 1.0]]))


class TestFrameCheck:
    def test_points_on_the_edge_are_inside_and_nan_is_outside(self):
        uv = np.array([[0.0, 1.0], [1.01, 0.5], [np.nan, 0.5]])
        assert in_frame(uv).tolist() == [True, False, False]

    def test_the_range_keeps_in_frame_points_in_and_ignores_the_rest(self):
        uv = np.array([[0.2, 0.95], [0.5, 0.4], [0.5, 1.2]])  # the last is already out
        bounds = uncentre_range(uv, 0.1)
        assert bounds == pytest.approx(np.array([[-0.1, 0.1], [-0.1, 0.05]]))
        for u in ([0.0, 0.0], [0.999, 0.999], [0.5, 0.3]):
            shift = uncentre_shift(uv, u, 0.1)
            assert in_frame(uv[:2] + shift).all()

    def test_the_range_always_contains_zero(self):
        uv = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert uncentre_range(uv, 0.1) == pytest.approx(np.zeros((2, 2)))

    def test_frame_loss_counts_points_that_left(self):
        before = np.array([[0.5, 0.5], [0.5, 0.95], [0.5, 1.5]])
        assert frame_loss(before, before + [0.0, 0.1]) == pytest.approx(0.5)
        assert frame_loss(before[2:], before[2:]) is None


# SMPL-X's first 22 parents, and a rest pose with arms out to the sides (y up).
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def rest_pose():
    rest = np.zeros((22, 3))
    for j in range(1, 22):
        rest[j] = rest[PARENTS[j]] + [0.0, 0.1, 0.0]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm = SMPLX_ARM[side]
        collar = PARENTS[arm["shoulder"]]
        rest[arm["shoulder"]] = rest[collar] + [sign * 0.15, 0.0, 0.0]
        rest[arm["elbow"]] = rest[arm["shoulder"]] + [sign * 0.28, 0.0, 0.0]
        rest[arm["wrist"]] = rest[arm["elbow"]] + [sign * 0.25, 0.0, 0.0]
    return rest


def bent(degrees=60.0):
    """55 rotations with both elbows bent forward by `degrees`."""
    rotations = np.zeros((55, 3))
    rotations[18] = [0.0, np.radians(degrees), 0.0]
    rotations[19] = [0.0, -np.radians(degrees), 0.0]
    return rotations


def arm_angle(rotations, side):
    _, pos = _world(rotations, rest_pose(), PARENTS)
    arm = SMPLX_ARM[side]
    upper = pos[arm["elbow"]] - pos[arm["shoulder"]]
    fore = pos[arm["wrist"]] - pos[arm["elbow"]]
    return np.degrees(np.arccos(np.dot(upper, fore) / np.linalg.norm(upper) / np.linalg.norm(fore)))


class TestArmJoints:
    def test_zero_angles_change_nothing(self):
        joints = [
            ArmJointDraw(s, j, 0.0, (1.0, 0.0, 0.0))
            for s in ("left", "right")
            for j in ("shoulder", "elbow", "wrist")
        ]
        out = perturb_arm_joints(bent(), rest_pose(), PARENTS, joints)
        assert out == pytest.approx(bent(), abs=1e-12)

    def test_zero_angle_leaves_a_straight_elbow_straight(self):
        # The 5° floor limits a drawn bend; it must not lift an elbow nothing was drawn for.
        out = perturb_arm_joints(
            np.zeros((55, 3)),
            rest_pose(),
            PARENTS,
            [ArmJointDraw("left", "elbow", 0.0, (1.0, 0.0, 0.0))],
        )
        assert out == pytest.approx(np.zeros((55, 3)), abs=1e-12)

    def test_the_elbow_bends_by_the_drawn_angle(self):
        out = perturb_arm_joints(
            bent(60.0), rest_pose(), PARENTS, [ArmJointDraw("left", "elbow", 3.5, (1.0, 0.0, 0.0))]
        )
        assert arm_angle(out, "left") == pytest.approx(63.5)
        assert arm_angle(out, "right") == pytest.approx(60.0)

    def test_the_elbow_bend_is_kept_anatomical(self):
        out = perturb_arm_joints(
            bent(148.0),
            rest_pose(),
            PARENTS,
            [ArmJointDraw("right", "elbow", 4.0, (1.0, 0.0, 0.0))],
        )
        assert arm_angle(out, "right") == pytest.approx(150.0)

    def test_a_straight_elbow_still_bends(self):
        out = perturb_arm_joints(
            np.zeros((55, 3)),
            rest_pose(),
            PARENTS,
            [ArmJointDraw("left", "elbow", 4.0, (1.0, 0.0, 0.0))],
        )
        assert arm_angle(out, "left") == pytest.approx(5.0)  # 0° + 4° lifted to the minimum

    def test_a_shoulder_turn_moves_the_arm_rigidly_by_the_angle(self):
        rotations = bent()
        out = perturb_arm_joints(
            rotations,
            rest_pose(),
            PARENTS,
            [ArmJointDraw("left", "shoulder", 4.0, (0.0, 0.0, 1.0))],
        )
        _, before = _world(rotations, rest_pose(), PARENTS)
        _, after = _world(out, rest_pose(), PARENTS)
        arm = SMPLX_ARM["left"]
        old = before[arm["elbow"]] - before[arm["shoulder"]]
        new = after[arm["elbow"]] - after[arm["shoulder"]]
        assert np.linalg.norm(new) == pytest.approx(np.linalg.norm(old))
        angle = np.degrees(np.arccos(np.dot(old, new) / np.linalg.norm(old) ** 2))
        assert angle == pytest.approx(4.0)
        assert arm_angle(out, "left") == pytest.approx(60.0)  # the elbow keeps its bend

    def test_joints_below_the_wrist_keep_their_rotations(self):
        rotations = bent()
        rotations[25:40] = 0.1
        out = perturb_arm_joints(
            rotations, rest_pose(), PARENTS, [ArmJointDraw("left", "wrist", -3.0, (0.0, 1.0, 0.0))]
        )
        assert np.array_equal(out[22:], rotations[22:])
        assert not np.allclose(out[20], rotations[20])
