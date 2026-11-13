"""
Tests for analytic swing inverse kinematics (V1 round-trip, Step 4).

The synthetic tests are closed-loop with known ground truth: generate landmarks
from rotations we chose, fit, and check we get those rotations back. Real
MediaPipe data can never support that, because nobody knows the true elbow
rotation in a video.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.skeleton import SyntheticSkeleton, SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES

SMPLX_MODEL_PATH = "models"

try:
    import torch
    import smplx
    _smplx_ok = True
except ImportError:
    _smplx_ok = False

requires_smplx = pytest.mark.skipif(not _smplx_ok, reason="torch/smplx not installed")

# Synthetic chain: joint index doubles as landmark index. The root bone is left
# out because the root's rotation is global orientation, solved separately -- and
# because the synthetic root sits at the origin, which would collide with
# MediaPipe's all-zeros "undetected" sentinel.
SYNTHETIC_BONES = [(1, 2, 1, 2), (2, 3, 2, 3)]


def rotvec_z(deg):
    return np.array([0.0, 0.0, np.deg2rad(deg)])


class TestSwingIKOnSyntheticChain:
    def test_rest_pose_landmarks_give_zero_rotations(self):
        s = SyntheticSkeleton()
        landmarks = forward_kinematics(np.zeros((s.num_joints, 3)), s)
        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)
        np.testing.assert_allclose(rots, np.zeros((s.num_joints, 3)), atol=1e-12)

    def test_recovers_a_known_single_rotation(self):
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = rotvec_z(90)
        landmarks = forward_kinematics(truth, s)

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        np.testing.assert_allclose(rots, truth, atol=1e-9)

    def test_recovers_chained_rotations(self):
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = rotvec_z(45)
        truth[2] = rotvec_z(-30)
        landmarks = forward_kinematics(truth, s)

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        np.testing.assert_allclose(rots, truth, atol=1e-9)

    def test_is_scale_invariant(self):
        # Swing IK reads directions only, so a uniform scale must change nothing.
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = rotvec_z(60)
        landmarks = forward_kinematics(truth, s)

        a = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)
        b = solve_swing_ik(landmarks * 7.5, s, SYNTHETIC_BONES)

        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_undetected_landmark_leaves_its_rotation_at_zero(self):
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = rotvec_z(90)
        landmarks = forward_kinematics(truth, s)
        landmarks[3] = 0.0  # MediaPipe writes zeros for undetected landmarks

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        np.testing.assert_allclose(rots[2], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(rots[1], truth[1], atol=1e-9)

    def test_antiparallel_bone_does_not_produce_nan(self):
        # A bone folded exactly 180 degrees has an undefined rotation axis.
        # Landmarks are offset off the origin so none collides with the
        # all-zeros "undetected" sentinel.
        s = SyntheticSkeleton()
        landmarks = forward_kinematics(np.zeros((s.num_joints, 3)), s) + [10.0, 5.0, 0.0]
        landmarks[2] = landmarks[1] - np.array([1.0, 0.0, 0.0])
        landmarks[3] = landmarks[2] - np.array([1.0, 0.0, 0.0])

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        assert np.all(np.isfinite(rots))
        # IK solves directions, not absolute positions, so compare bone vectors.
        posed = forward_kinematics(rots, s)
        np.testing.assert_allclose(posed[2] - posed[1], landmarks[2] - landmarks[1], atol=1e-9)

    def test_twist_about_the_bone_axis_is_not_recoverable(self):
        # Documents a real limitation: rotation about a bone's own axis moves no
        # joint, so positions carry no evidence of it. The fit returns zero twist
        # and the reconstructed positions still match.
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = np.array([np.deg2rad(50), 0.0, 0.0])  # about +x, along the bone
        landmarks = forward_kinematics(truth, s)

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        assert not np.allclose(rots[1], truth[1], atol=1e-3)
        np.testing.assert_allclose(forward_kinematics(rots, s), landmarks, atol=1e-9)

    def test_non_finite_landmark_leaves_its_rotation_at_zero(self):
        # World landmarks are hip-centred, so the origin is a legitimate position
        # and cannot double as the "missing" marker. NaN is used instead.
        s = SyntheticSkeleton()
        truth = np.zeros((s.num_joints, 3))
        truth[1] = rotvec_z(90)
        landmarks = forward_kinematics(truth, s)
        landmarks[3] = np.nan

        rots = solve_swing_ik(landmarks, s, SYNTHETIC_BONES)

        assert np.all(np.isfinite(rots))
        np.testing.assert_allclose(rots[2], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(rots[1], truth[1], atol=1e-9)

    def test_rejects_landmarks_of_wrong_shape(self):
        s = SyntheticSkeleton()
        with pytest.raises(ValueError):
            solve_swing_ik(np.zeros((2, 3)), s, SYNTHETIC_BONES)


@requires_smplx
class TestSwingIKOnSMPLX:
    def _landmarks_from_rotations(self, skeleton, rots):
        """Pose SMPL-X, then lay the corresponded joints out at MediaPipe indices."""
        joints = forward_kinematics(rots, skeleton)
        mp = np.zeros((33, 3))
        for parent_j, child_j, mp_p, mp_c in SMPLX_ARM_BONES:
            mp[mp_p] = joints[parent_j]
            mp[mp_c] = joints[child_j]
        return mp

    def test_recovers_arm_rotations_from_their_own_landmarks(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        rng = np.random.default_rng(7)
        truth = np.zeros((55, 3))
        for joint, _, _, _ in SMPLX_ARM_BONES:
            truth[joint] = rng.normal(0, 0.25, 3)

        landmarks = self._landmarks_from_rotations(s, truth)
        rots = solve_swing_ik(landmarks, s, SMPLX_ARM_BONES)

        # Twist is unobservable, so compare reconstructed positions, not rotations.
        reposed = forward_kinematics(rots, s)
        original = forward_kinematics(truth, s)
        for _, child, _, _ in SMPLX_ARM_BONES:
            np.testing.assert_allclose(reposed[child], original[child], atol=1e-9)

    def test_leaves_unconstrained_joints_at_zero(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        rng = np.random.default_rng(3)
        truth = np.zeros((55, 3))
        for joint, _, _, _ in SMPLX_ARM_BONES:
            truth[joint] = rng.normal(0, 0.2, 3)
        landmarks = self._landmarks_from_rotations(s, truth)

        rots = solve_swing_ik(landmarks, s, SMPLX_ARM_BONES)

        driven = {b[0] for b in SMPLX_ARM_BONES}
        for j in range(55):
            if j not in driven:
                np.testing.assert_allclose(rots[j], np.zeros(3), atol=1e-12)

    def test_arm_bones_target_the_documented_joints(self):
        # left/right shoulder (16, 17) and left/right elbow (18, 19) -- the four
        # joints Finding 5 established are constrainable on this data.
        assert {b[0] for b in SMPLX_ARM_BONES} == {16, 17, 18, 19}
