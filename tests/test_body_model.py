"""
Unit tests for the body_model module.

Tests cover:
- Joint mapping between SMPL-X and MediaPipe
- SMPL-X wrapper functionality (requires models)
- SMPLXParamsContainer (no models required)
"""

import numpy as np
import pytest
from pathlib import Path

# Import mapping module (always available)
from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    SMPLX_HAND_JOINTS,
    MEDIAPIPE_POSE_LANDMARKS,
    MEDIAPIPE_HAND_LANDMARKS,
    SMPLX_TO_MEDIAPIPE_POSE,
    SMPLX_JOINT_LIMITS,
    get_corresponding_joints,
    get_smplx_joint_index,
    get_mediapipe_landmark_index,
    clamp_joint_angles,
    get_kinematic_chain,
    LIMB_GROUPS,
)

# Check if SMPL-X is available
try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

# smplx.create() appends the model type to the path it is given, so it must
# receive the models/ root -- passing models/smplx resolves to models/smplx/smplx
# and fails. The asset check needs the subdirectory, so the two are separate.
MODEL_PATH = Path(__file__).parent.parent / 'models'
SMPLX_ASSET_DIR = MODEL_PATH / 'smplx'
MODELS_DOWNLOADED = SMPLX_ASSET_DIR.exists() and (SMPLX_ASSET_DIR / 'SMPLX_NEUTRAL.npz').exists()


class TestJointMapping:
    """Tests for joint_mapping.py - no external dependencies required."""

    def test_smplx_body_joints_count(self):
        """SMPL-X should have 22 body joints."""
        assert len(SMPLX_BODY_JOINTS) == 22

    def test_smplx_hand_joints_count(self):
        """SMPL-X should have 15 hand joints per hand."""
        assert len(SMPLX_HAND_JOINTS) == 15

    def test_mediapipe_pose_landmarks_count(self):
        """MediaPipe pose should have 33 landmarks."""
        assert len(MEDIAPIPE_POSE_LANDMARKS) == 33

    def test_mediapipe_hand_landmarks_count(self):
        """MediaPipe hand should have 21 landmarks."""
        assert len(MEDIAPIPE_HAND_LANDMARKS) == 21

    def test_smplx_to_mediapipe_mapping_valid(self):
        """All keys in mapping should exist in respective dictionaries."""
        for smplx_name, mp_name in SMPLX_TO_MEDIAPIPE_POSE.items():
            # SMPL-X name should be valid
            assert smplx_name in SMPLX_BODY_JOINTS or smplx_name in ['left_eye', 'right_eye', 'jaw']
            # MediaPipe name should be valid
            assert mp_name in MEDIAPIPE_POSE_LANDMARKS

    def test_get_corresponding_joints_mediapipe(self):
        """Test getting joint correspondence from MediaPipe perspective."""
        mapping = get_corresponding_joints('mediapipe')

        # Should return dict of int -> int
        assert isinstance(mapping, dict)
        assert all(isinstance(k, int) for k in mapping.keys())
        assert all(isinstance(v, int) for v in mapping.values())

        # Should have some mappings
        assert len(mapping) > 0

    def test_get_corresponding_joints_smplx(self):
        """Test getting joint correspondence from SMPL-X perspective."""
        mapping = get_corresponding_joints('smplx')

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_get_corresponding_joints_invalid(self):
        """Test invalid source format raises error."""
        with pytest.raises(ValueError):
            get_corresponding_joints('invalid_format')

    def test_get_smplx_joint_index_valid(self):
        """Test getting SMPL-X joint index by name."""
        assert get_smplx_joint_index('pelvis') == 0
        assert get_smplx_joint_index('left_shoulder') == 16
        assert get_smplx_joint_index('right_wrist') == 21

    def test_get_smplx_joint_index_invalid(self):
        """Test invalid joint name raises error."""
        with pytest.raises(ValueError):
            get_smplx_joint_index('invalid_joint')

    def test_get_mediapipe_landmark_index_valid(self):
        """Test getting MediaPipe landmark index by name."""
        assert get_mediapipe_landmark_index('nose') == 0
        assert get_mediapipe_landmark_index('left_shoulder') == 11
        assert get_mediapipe_landmark_index('right_ankle') == 28

    def test_get_mediapipe_landmark_index_invalid(self):
        """Test invalid landmark name raises error."""
        with pytest.raises(ValueError):
            get_mediapipe_landmark_index('invalid_landmark')

    def test_joint_limits_structure(self):
        """Test joint limits have correct structure."""
        for joint_name, limits in SMPLX_JOINT_LIMITS.items():
            assert isinstance(limits, dict)
            for axis, (min_val, max_val) in limits.items():
                assert axis in ['x', 'y', 'z']
                assert min_val <= max_val
                # Angles should be in reasonable range (radians)
                assert -2 * np.pi <= min_val <= 2 * np.pi
                assert -2 * np.pi <= max_val <= 2 * np.pi

    def test_clamp_joint_angles_within_limits(self):
        """Test that angles within limits are unchanged."""
        angles = np.array([0.0, 0.0, 0.0])
        clamped = clamp_joint_angles('left_elbow', angles)
        np.testing.assert_array_almost_equal(angles, clamped)

    def test_clamp_joint_angles_exceeds_limits(self):
        """Test that angles exceeding limits are clamped."""
        # Left elbow has x limit of (0, 2.62)
        angles = np.array([5.0, 0.0, 0.0])  # Exceeds max
        clamped = clamp_joint_angles('left_elbow', angles)

        assert clamped[0] <= 2.62
        assert clamped[0] >= 0

    def test_clamp_joint_angles_unknown_joint(self):
        """Test that unknown joints pass through unchanged."""
        angles = np.array([1.0, 2.0, 3.0])
        clamped = clamp_joint_angles('unknown_joint', angles)
        np.testing.assert_array_equal(angles, clamped)

    def test_get_kinematic_chain_arm(self):
        """Test kinematic chain for arm."""
        chain = get_kinematic_chain('left_wrist', 'pelvis')

        # Should include shoulder, elbow, wrist
        assert 'left_shoulder' in chain
        assert 'left_elbow' in chain
        assert 'left_wrist' in chain

        # Should be in correct order (root to end)
        shoulder_idx = chain.index('left_shoulder')
        elbow_idx = chain.index('left_elbow')
        wrist_idx = chain.index('left_wrist')

        assert shoulder_idx < elbow_idx < wrist_idx

    def test_get_kinematic_chain_leg(self):
        """Test kinematic chain for leg."""
        chain = get_kinematic_chain('right_ankle', 'pelvis')

        assert 'right_hip' in chain
        assert 'right_knee' in chain
        assert 'right_ankle' in chain

    def test_limb_groups_structure(self):
        """Test limb groups are properly defined."""
        assert 'left_arm' in LIMB_GROUPS
        assert 'right_arm' in LIMB_GROUPS
        assert 'left_leg' in LIMB_GROUPS
        assert 'right_leg' in LIMB_GROUPS
        assert 'spine' in LIMB_GROUPS

        # Each group should have valid joint names
        for group_name, joints in LIMB_GROUPS.items():
            for joint in joints:
                assert joint in SMPLX_BODY_JOINTS


@pytest.mark.skipif(not SMPLX_AVAILABLE, reason="PyTorch/SMPL-X not installed")
@pytest.mark.skipif(not MODELS_DOWNLOADED, reason="SMPL-X models not downloaded")
class TestSMPLXWrapper:
    """Tests for smplx_wrapper.py - requires SMPL-X models."""

    @pytest.fixture
    def smplx_wrapper(self):
        """Create SMPL-X wrapper for testing."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
        return SMPLXWrapper(model_path=MODEL_PATH, gender='neutral')

    def test_wrapper_initialization(self, smplx_wrapper):
        """Test that wrapper initializes correctly."""
        assert smplx_wrapper.model is not None
        assert smplx_wrapper.gender == 'neutral'

    def test_get_default_params(self, smplx_wrapper):
        """Test default parameter generation."""
        params = smplx_wrapper.get_default_params(batch_size=2)

        assert 'body_pose' in params
        assert 'betas' in params
        assert params['body_pose'].shape == (2, 63)
        assert params['betas'].shape == (2, 10)

    def test_forward_default_pose(self, smplx_wrapper):
        """Test forward pass with default pose."""
        params = smplx_wrapper.get_default_params(batch_size=1)
        output = smplx_wrapper.forward(**params)

        assert 'joints' in output
        assert output['joints'].shape[0] == 1
        assert output['joints'].shape[2] == 3  # 3D coordinates

    def test_forward_with_pose(self, smplx_wrapper):
        """Test forward pass with custom pose."""
        body_pose = np.random.randn(1, 63).astype(np.float32) * 0.1
        output = smplx_wrapper.forward(body_pose=body_pose)

        assert 'joints' in output
        assert output['joints'].shape[0] == 1

    def test_rest_pose_joints(self, smplx_wrapper):
        """Test getting rest pose joint positions."""
        joints = smplx_wrapper.get_rest_pose_joints()

        assert joints.shape == (1, smplx_wrapper.num_output_joints, 3)
        # In rest pose, joints should be centered around origin
        center = joints.mean(axis=1)
        assert np.abs(center).max() < 1.0  # Should be near origin

    def test_joint_counts_are_distinct(self, smplx_wrapper):
        """num_model_joints is the kinematic chain; num_output_joints is wider."""
        assert smplx_wrapper.num_model_joints == 55
        assert smplx_wrapper.num_output_joints == 127
        assert smplx_wrapper.num_output_joints > smplx_wrapper.num_model_joints

    def test_num_joints_is_deprecated(self, smplx_wrapper):
        """The ambiguous alias still works but warns and never sizes output."""
        with pytest.deprecated_call():
            assert smplx_wrapper.num_joints == smplx_wrapper.num_model_joints

    def test_model_joints_is_leading_slice_of_joints(self, smplx_wrapper):
        """model_joints must be exactly the kinematic prefix of joints."""
        params = smplx_wrapper.get_default_params(batch_size=1)
        output = smplx_wrapper.forward(**params, return_vertices=False)

        assert output['joints'].shape == (1, smplx_wrapper.num_output_joints, 3)
        assert output['model_joints'].shape == (1, smplx_wrapper.num_model_joints, 3)
        np.testing.assert_array_equal(
            output['model_joints'],
            output['joints'][:, :smplx_wrapper.num_model_joints],
        )


class TestSMPLXParamsContainer:
    """Tests for SMPLXParamsContainer - no SMPL-X required."""

    def test_container_creation(self):
        """Test container can be created."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXParamsContainer

        params = {
            'body_pose': np.zeros((1, 63)),
            'left_hand_pose': np.zeros((1, 45)),
            'right_hand_pose': np.zeros((1, 45)),
            'betas': np.zeros((1, 10)),
        }

        container = SMPLXParamsContainer(params)
        assert container.batch_size == 1

    def test_container_get_joint_rotation(self):
        """Test getting individual joint rotations."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXParamsContainer

        body_pose = np.zeros((1, 63))
        body_pose[0, 0:3] = [0.1, 0.2, 0.3]  # First joint

        params = {
            'body_pose': body_pose,
            'left_hand_pose': np.zeros((1, 45)),
            'right_hand_pose': np.zeros((1, 45)),
        }

        container = SMPLXParamsContainer(params)
        rotation = container.get_joint_rotation(0)

        np.testing.assert_array_almost_equal(rotation[0], [0.1, 0.2, 0.3])

    def test_container_set_joint_rotation(self):
        """Test setting individual joint rotations."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXParamsContainer

        params = {
            'body_pose': np.zeros((1, 63)),
            'left_hand_pose': np.zeros((1, 45)),
            'right_hand_pose': np.zeros((1, 45)),
        }

        container = SMPLXParamsContainer(params)
        container.set_joint_rotation(0, np.array([[0.5, 0.6, 0.7]]))

        rotation = container.get_joint_rotation(0)
        np.testing.assert_array_almost_equal(rotation[0], [0.5, 0.6, 0.7])

    def test_container_clone(self):
        """Test cloning container."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXParamsContainer

        params = {
            'body_pose': np.ones((1, 63)),
            'left_hand_pose': np.zeros((1, 45)),
            'right_hand_pose': np.zeros((1, 45)),
        }

        container = SMPLXParamsContainer(params)
        clone = container.clone()

        # Modify original
        container.body_pose[0, 0] = 999

        # Clone should be unchanged
        assert clone.body_pose[0, 0] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
