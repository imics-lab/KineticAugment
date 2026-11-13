"""
Unit tests for the body_model module.

Tests cover:
- Joint mapping between SMPL-X and MediaPipe
- SMPL-X wrapper functionality (requires models)
- MediaPipe to SMPL-X fitting (requires models)
- SMPL-X to MediaPipe projection (requires models)
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

# Check if models are downloaded
SMPLX_ASSET_DIR = Path('models/smplx')
MODEL_PATH = 'models'
MODELS_DOWNLOADED = (SMPLX_ASSET_DIR / 'SMPLX_NEUTRAL.npz').exists()


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

        assert joints.shape == (1, smplx_wrapper.num_joints, 3)
        # In rest pose, joints should be centered around origin
        center = joints.mean(axis=1)
        assert np.abs(center).max() < 1.0  # Should be near origin


@pytest.mark.skipif(not SMPLX_AVAILABLE, reason="PyTorch/SMPL-X not installed")
@pytest.mark.skipif(not MODELS_DOWNLOADED, reason="SMPL-X models not downloaded")
class TestMediaPipeToSMPLX:
    """Tests for mp_to_smplx.py - requires SMPL-X models."""

    @pytest.fixture
    def fitter(self):
        """Create fitter for testing."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
        from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX

        wrapper = SMPLXWrapper(model_path=MODEL_PATH, gender='neutral')
        return MediaPipeToSMPLX(wrapper, num_iterations=10, verbose=False)

    def test_fitter_initialization(self, fitter):
        """Test fitter initializes correctly."""
        assert fitter.smplx is not None
        assert len(fitter.smplx_joint_indices) > 0

    def test_fit_random_landmarks(self, fitter):
        """Test fitting with random landmarks."""
        # Generate random landmarks (543 landmarks, 3D)
        landmarks = np.random.randn(543, 3).astype(np.float32) * 0.1

        # Add some structure (put hips and shoulders in reasonable positions)
        landmarks[11] = [0.0, 0.5, 0.0]   # left shoulder
        landmarks[12] = [0.0, 0.5, 0.3]   # right shoulder
        landmarks[23] = [0.0, 0.0, 0.0]   # left hip
        landmarks[24] = [0.0, 0.0, 0.3]   # right hip

        params = fitter.fit(landmarks)

        assert 'body_pose' in params
        assert 'global_orient' in params
        assert params['body_pose'].shape == (63,)


@pytest.mark.skipif(not SMPLX_AVAILABLE, reason="PyTorch/SMPL-X not installed")
@pytest.mark.skipif(not MODELS_DOWNLOADED, reason="SMPL-X models not downloaded")
class TestSMPLXToMediaPipe:
    """Tests for smplx_to_mp.py - requires SMPL-X models."""

    @pytest.fixture
    def projector(self):
        """Create projector for testing."""
        from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
        from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe

        wrapper = SMPLXWrapper(model_path=MODEL_PATH, gender='neutral')
        return SMPLXToMediaPipe(wrapper)

    def test_projector_initialization(self, projector):
        """Test projector initializes correctly."""
        assert projector.smplx is not None

    def test_project_default_pose(self, projector):
        """Test projecting default SMPL-X pose."""
        params = projector.smplx.get_default_params(batch_size=1)
        landmarks = projector.project(params)

        assert landmarks.shape == (1, 543, 3)

    def test_project_pose_only(self, projector):
        """Test projecting pose landmarks only."""
        params = projector.smplx.get_default_params(batch_size=1)
        landmarks = projector.project(params, return_full_frame=False)

        assert landmarks.shape == (1, 33, 3)


class TestSimpleFitter:
    """Tests for SimpleFitter - no SMPL-X required."""

    def test_simple_fitter_initialization(self):
        """Test SimpleFitter can be created without SMPL-X."""
        from kinetic_augment.body_model.mp_to_smplx import SimpleFitter
        fitter = SimpleFitter()
        assert fitter.mp_to_smplx is not None

    def test_simple_fitter_fit(self):
        """Test SimpleFitter produces valid output structure."""
        from kinetic_augment.body_model.mp_to_smplx import SimpleFitter

        fitter = SimpleFitter()

        # Create mock landmarks
        landmarks = np.random.randn(543, 3).astype(np.float32)
        # Set reasonable positions for key landmarks
        landmarks[11] = [0.0, 0.5, 0.0]   # left shoulder
        landmarks[12] = [0.3, 0.5, 0.0]   # right shoulder
        landmarks[23] = [0.0, 0.0, 0.0]   # left hip
        landmarks[24] = [0.3, 0.0, 0.0]   # right hip

        params = fitter.fit(landmarks)

        assert 'body_pose' in params
        assert 'global_orient' in params
        assert 'betas' in params
        assert params['body_pose'].shape == (1, 63)


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
