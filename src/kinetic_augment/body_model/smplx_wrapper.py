"""
SMPL-X Model Wrapper for KineticAugment.

This module provides a clean interface to the SMPL-X parametric body model,
handling model loading, forward kinematics, and parameter management.

SMPL-X Parameters:
    - betas (10): Body shape parameters
    - body_pose (63): Body joint rotations (21 joints × 3 axis-angle)
    - global_orient (3): Global body orientation
    - left_hand_pose (45): Left hand joint rotations (15 joints × 3)
    - right_hand_pose (45): Right hand joint rotations (15 joints × 3)
    - jaw_pose (3): Jaw rotation
    - leye_pose (3): Left eye rotation
    - reye_pose (3): Right eye rotation
    - expression (10): Facial expression parameters

Example:
    >>> wrapper = SMPLXWrapper(model_path="models", gender="neutral")
    >>> output = wrapper.forward(body_pose=my_pose)
    >>> joints = output['joints']          # (1, 127, 3) joints + landmarks
    >>> chain = output['model_joints']     # (1, 55, 3) kinematic chain only
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np

# Check for torch/smplx availability
try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    torch = None
    smplx = None


class SMPLXWrapper:
    """
    Wrapper class for SMPL-X body model.

    Provides a clean interface for:
    - Loading SMPL-X models (male, female, neutral)
    - Forward kinematics (parameters → joints/vertices)
    - Parameter management and validation
    - Batch processing support

    Attributes:
        model: The underlying SMPL-X model
        device: Torch device (cpu/cuda)
        gender: Model gender (male/female/neutral)
        num_betas: Number of shape parameters
        use_pca: Whether to use PCA for hand pose
        num_pca_comps: Number of PCA components for hands
    """

    # Default parameter dimensions
    NUM_BETAS = 10
    NUM_EXPRESSION = 10
    NUM_BODY_JOINTS = 21  # Excluding pelvis
    NUM_HAND_JOINTS = 15
    BODY_POSE_DIM = NUM_BODY_JOINTS * 3  # 63
    HAND_POSE_DIM = NUM_HAND_JOINTS * 3  # 45

    def __init__(
        self,
        model_path: Union[str, Path],
        gender: str = "neutral",
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
        use_pca: bool = False,
        num_pca_comps: int = 12,
        use_face_contour: bool = False,
        batch_size: int = 1,
        device: Optional[str] = None,
    ):
        """
        Initialize the SMPL-X wrapper.

        Args:
            model_path: Directory *containing* the smplx/ model folder
                (e.g. 'models', not 'models/smplx') -- smplx.create()
                appends the model type to this path
            gender: 'male', 'female', or 'neutral'
            num_betas: Number of body shape parameters
            num_expression_coeffs: Number of expression parameters
            use_pca: Use PCA representation for hand pose
            num_pca_comps: Number of PCA components (if use_pca=True)
            use_face_contour: Include face contour vertices
            batch_size: Default batch size for forward pass
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        if not SMPLX_AVAILABLE:
            raise ImportError(
                "SMPL-X requires torch and smplx packages. "
                "Install with: pip install torch smplx"
            )

        self.model_path = Path(model_path)
        self.gender = gender
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SMPL-X model path not found: {self.model_path}\n"
                "Download models from: https://smpl-x.is.tue.mpg.de/"
            )

        # Load the SMPL-X model
        self.model = smplx.create(
            model_path=str(self.model_path),
            model_type='smplx',
            gender=gender,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
            use_pca=use_pca,
            num_pca_comps=num_pca_comps,
            use_face_contour=use_face_contour,
            batch_size=batch_size,
        ).to(self.device)

        # Store model info
        self.faces = self.model.faces  # Face indices for mesh

        # SMPL-X exposes two different joint counts and conflating them is a
        # silent bug. J_regressor gives the kinematic chain (55 joints);
        # forward() then appends vertex-derived landmarks to the same tensor,
        # so its 'joints' output is wider (127 by default, 144 with
        # use_face_contour=True). Only the first num_model_joints entries are
        # posed by the kinematic chain -- the appended landmarks are read off
        # the mesh and are not reproducible by rigid forward kinematics.
        self.num_model_joints = self.model.J_regressor.shape[0]
        self.num_output_joints = self._measure_output_joints()

        # Hand pose dimension depends on PCA usage
        if use_pca:
            self.hand_pose_dim = num_pca_comps
        else:
            self.hand_pose_dim = self.HAND_POSE_DIM

    def _measure_output_joints(self) -> int:
        """
        Determine the width of the 'joints' tensor returned by forward().

        Measured with a single forward pass rather than derived arithmetically:
        the number of appended landmarks depends on use_face_contour and on the
        smplx version, so a hardcoded formula silently goes stale.
        """
        with torch.no_grad():
            return int(self.model(return_verts=False).joints.shape[1])

    @property
    def num_joints(self) -> int:
        """
        Deprecated: ambiguous. Use num_model_joints or num_output_joints.

        Kept as an alias for num_model_joints so existing callers keep the
        value they already had. Note this is NOT the width of
        forward()['joints'] -- sizing an array from it was the original bug.
        """
        warnings.warn(
            "SMPLXWrapper.num_joints is ambiguous and deprecated: it is the "
            "kinematic chain size, not the width of forward()['joints']. Use "
            "num_model_joints (kinematic chain) or num_output_joints "
            "(forward() output width) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_model_joints

    def forward(
        self,
        betas: Optional[np.ndarray] = None,
        body_pose: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
        left_hand_pose: Optional[np.ndarray] = None,
        right_hand_pose: Optional[np.ndarray] = None,
        jaw_pose: Optional[np.ndarray] = None,
        leye_pose: Optional[np.ndarray] = None,
        reye_pose: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
        transl: Optional[np.ndarray] = None,
        return_vertices: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass through SMPL-X model.

        Computes joint positions and optionally mesh vertices from parameters.

        Args:
            betas: Shape parameters (batch, 10)
            body_pose: Body joint rotations (batch, 63) in axis-angle
            global_orient: Global orientation (batch, 3) in axis-angle
            left_hand_pose: Left hand pose (batch, 45 or num_pca_comps)
            right_hand_pose: Right hand pose (batch, 45 or num_pca_comps)
            jaw_pose: Jaw rotation (batch, 3)
            leye_pose: Left eye rotation (batch, 3)
            reye_pose: Right eye rotation (batch, 3)
            expression: Expression parameters (batch, 10)
            transl: Translation (batch, 3)
            return_vertices: Whether to return mesh vertices

        Returns:
            Dictionary with:
                - joints: Kinematic joints followed by vertex-derived
                  landmarks (batch, num_output_joints, 3)
                - model_joints: The kinematic chain only, i.e. the leading
                  slice of 'joints' (batch, num_model_joints, 3)
                - vertices: Mesh vertices (batch, num_verts, 3) if return_vertices
                - full_pose: Full pose vector
        """
        # Convert numpy arrays to torch tensors
        kwargs = {}

        if betas is not None:
            kwargs['betas'] = self._to_tensor(betas)
        if body_pose is not None:
            kwargs['body_pose'] = self._to_tensor(body_pose)
        if global_orient is not None:
            kwargs['global_orient'] = self._to_tensor(global_orient)
        if left_hand_pose is not None:
            kwargs['left_hand_pose'] = self._to_tensor(left_hand_pose)
        if right_hand_pose is not None:
            kwargs['right_hand_pose'] = self._to_tensor(right_hand_pose)
        if jaw_pose is not None:
            kwargs['jaw_pose'] = self._to_tensor(jaw_pose)
        if leye_pose is not None:
            kwargs['leye_pose'] = self._to_tensor(leye_pose)
        if reye_pose is not None:
            kwargs['reye_pose'] = self._to_tensor(reye_pose)
        if expression is not None:
            kwargs['expression'] = self._to_tensor(expression)
        if transl is not None:
            kwargs['transl'] = self._to_tensor(transl)

        # Forward pass
        with torch.no_grad():
            output = self.model(return_verts=return_vertices, **kwargs)

        # Convert outputs to numpy
        joints = output.joints.cpu().numpy()
        result = {
            'joints': joints,
            # Expose the kinematic chain explicitly so callers never have to
            # hardcode the 55 boundary to separate it from the landmarks.
            'model_joints': joints[:, :self.num_model_joints],
        }

        if return_vertices:
            result['vertices'] = output.vertices.cpu().numpy()

        if hasattr(output, 'full_pose') and output.full_pose is not None:
            result['full_pose'] = output.full_pose.cpu().numpy()

        return result

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on the correct device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device).float()
        return torch.from_numpy(arr).to(self.device).float()

    def get_default_params(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """
        Get default (neutral) parameters.

        Args:
            batch_size: Number of instances

        Returns:
            Dictionary of default parameters (all zeros except shape)
        """
        params = {
            'betas': np.zeros((batch_size, self.num_betas), dtype=np.float32),
            'body_pose': np.zeros((batch_size, self.BODY_POSE_DIM), dtype=np.float32),
            'global_orient': np.zeros((batch_size, 3), dtype=np.float32),
            'left_hand_pose': np.zeros((batch_size, self.hand_pose_dim), dtype=np.float32),
            'right_hand_pose': np.zeros((batch_size, self.hand_pose_dim), dtype=np.float32),
            'jaw_pose': np.zeros((batch_size, 3), dtype=np.float32),
            'leye_pose': np.zeros((batch_size, 3), dtype=np.float32),
            'reye_pose': np.zeros((batch_size, 3), dtype=np.float32),
            'expression': np.zeros((batch_size, self.num_expression_coeffs), dtype=np.float32),
            'transl': np.zeros((batch_size, 3), dtype=np.float32),
        }
        return params

    def get_joint_positions(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get joint positions from parameters (convenience method).

        Args:
            params: Dictionary of SMPL-X parameters

        Returns:
            Joint positions (batch, num_output_joints, 3), i.e. the kinematic
            joints followed by the vertex-derived landmarks.
        """
        output = self.forward(**params, return_vertices=False)
        return output['joints']

    def get_rest_pose_joints(self) -> np.ndarray:
        """
        Get joint positions in rest pose (T-pose).

        Returns:
            Joint positions (1, num_output_joints, 3)
        """
        params = self.get_default_params(batch_size=1)
        return self.get_joint_positions(params)

    def validate_params(self, params: Dict[str, np.ndarray]) -> bool:
        """
        Validate parameter dimensions.

        Args:
            params: Dictionary of parameters

        Returns:
            True if valid, raises ValueError otherwise
        """
        expected_shapes = {
            'betas': (None, self.num_betas),
            'body_pose': (None, self.BODY_POSE_DIM),
            'global_orient': (None, 3),
            'left_hand_pose': (None, self.hand_pose_dim),
            'right_hand_pose': (None, self.hand_pose_dim),
            'jaw_pose': (None, 3),
            'leye_pose': (None, 3),
            'reye_pose': (None, 3),
            'expression': (None, self.num_expression_coeffs),
            'transl': (None, 3),
        }

        for name, arr in params.items():
            if name not in expected_shapes:
                continue

            expected = expected_shapes[name]
            actual = arr.shape

            if len(actual) != 2:
                raise ValueError(f"{name}: expected 2D array, got shape {actual}")

            if expected[1] is not None and actual[1] != expected[1]:
                raise ValueError(
                    f"{name}: expected dim 1 to be {expected[1]}, got {actual[1]}"
                )

        return True

    def params_to_dict(
        self,
        body_pose: np.ndarray,
        left_hand_pose: Optional[np.ndarray] = None,
        right_hand_pose: Optional[np.ndarray] = None,
        betas: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create a parameter dictionary, filling in defaults where needed.

        Args:
            body_pose: Body pose (required)
            left_hand_pose: Left hand pose (optional)
            right_hand_pose: Right hand pose (optional)
            betas: Shape parameters (optional)
            expression: Expression (optional)
            global_orient: Global orientation (optional)

        Returns:
            Complete parameter dictionary
        """
        batch_size = body_pose.shape[0]
        defaults = self.get_default_params(batch_size)

        # Override with provided values
        defaults['body_pose'] = body_pose

        if left_hand_pose is not None:
            defaults['left_hand_pose'] = left_hand_pose
        if right_hand_pose is not None:
            defaults['right_hand_pose'] = right_hand_pose
        if betas is not None:
            defaults['betas'] = betas
        if expression is not None:
            defaults['expression'] = expression
        if global_orient is not None:
            defaults['global_orient'] = global_orient

        return defaults

    @property
    def joint_names(self) -> list:
        """Get list of joint names."""
        from kinetic_augment.body_model.joint_mapping import SMPLX_JOINT_NAMES
        return SMPLX_JOINT_NAMES

    def __repr__(self) -> str:
        return (
            f"SMPLXWrapper(gender={self.gender}, device={self.device}, "
            f"use_pca={self.use_pca}, num_pca_comps={self.num_pca_comps})"
        )


class SMPLXParamsContainer:
    """
    Container for SMPL-X parameters with convenient access methods.

    This class wraps SMPL-X parameters and provides methods for:
    - Getting/setting individual joint rotations
    - Applying constraints
    - Converting between representations
    """

    def __init__(self, params: Dict[str, np.ndarray]):
        """
        Initialize with a parameter dictionary.

        Args:
            params: Dictionary of SMPL-X parameters
        """
        self._params = params.copy()

        # Validate structure
        required = ['body_pose', 'left_hand_pose', 'right_hand_pose']
        for key in required:
            if key not in self._params:
                raise ValueError(f"Missing required parameter: {key}")

        self.batch_size = self._params['body_pose'].shape[0]

    @property
    def body_pose(self) -> np.ndarray:
        """Get body pose parameters."""
        return self._params['body_pose']

    @body_pose.setter
    def body_pose(self, value: np.ndarray):
        """Set body pose parameters."""
        self._params['body_pose'] = value

    @property
    def left_hand_pose(self) -> np.ndarray:
        """Get left hand pose parameters."""
        return self._params['left_hand_pose']

    @left_hand_pose.setter
    def left_hand_pose(self, value: np.ndarray):
        """Set left hand pose parameters."""
        self._params['left_hand_pose'] = value

    @property
    def right_hand_pose(self) -> np.ndarray:
        """Get right hand pose parameters."""
        return self._params['right_hand_pose']

    @right_hand_pose.setter
    def right_hand_pose(self, value: np.ndarray):
        """Set right hand pose parameters."""
        self._params['right_hand_pose'] = value

    def get_joint_rotation(self, joint_idx: int) -> np.ndarray:
        """
        Get rotation for a specific body joint.

        Args:
            joint_idx: Joint index (0-20, excluding pelvis)

        Returns:
            Rotation vector (batch, 3) in axis-angle
        """
        start = joint_idx * 3
        end = start + 3
        return self._params['body_pose'][:, start:end]

    def set_joint_rotation(self, joint_idx: int, rotation: np.ndarray):
        """
        Set rotation for a specific body joint.

        Args:
            joint_idx: Joint index (0-20)
            rotation: Rotation vector (batch, 3) in axis-angle
        """
        start = joint_idx * 3
        end = start + 3
        self._params['body_pose'][:, start:end] = rotation

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export as dictionary."""
        return self._params.copy()

    def clone(self) -> 'SMPLXParamsContainer':
        """Create a deep copy."""
        cloned_params = {k: v.copy() for k, v in self._params.items()}
        return SMPLXParamsContainer(cloned_params)

    def apply_noise(
        self,
        joints: list,
        stddev: float = 0.1,
        respect_limits: bool = True
    ) -> 'SMPLXParamsContainer':
        """
        Apply Gaussian noise to specified joints.

        Args:
            joints: List of joint indices to perturb
            stddev: Standard deviation of noise (radians)
            respect_limits: Whether to clamp to anatomical limits

        Returns:
            New container with perturbed parameters
        """
        from kinetic_augment.body_model.joint_mapping import (
            clamp_joint_angles, SMPLX_JOINT_NAMES
        )

        result = self.clone()

        for joint_idx in joints:
            current = result.get_joint_rotation(joint_idx)
            noise = np.random.randn(*current.shape) * stddev
            new_rotation = current + noise

            if respect_limits and joint_idx < len(SMPLX_JOINT_NAMES):
                joint_name = SMPLX_JOINT_NAMES[joint_idx]
                for b in range(self.batch_size):
                    new_rotation[b] = clamp_joint_angles(joint_name, new_rotation[b])

            result.set_joint_rotation(joint_idx, new_rotation)

        return result
