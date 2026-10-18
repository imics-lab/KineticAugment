"""
Intrinsic (SMPL-X based) Augmentations for KineticAugment.

These augmentations operate on SMPL-X body model parameters, guaranteeing:
- Kinematic chain consistency (moving shoulder affects entire arm)
- Anatomical plausibility (joint limits from biomechanics)
- Limb length preservation

Key augmentations for Sign Language Recognition:
- JointAnglePerturbation: Add noise to joint angles within anatomical limits
- HandPosePerturbation: Perturb hand pose while preserving handshape category
- JointCoupledNoise: Correlated noise respecting inter-limb coordination

Example:
    >>> from kinetic_augment.augmentations.intrinsic import JointAnglePerturbation
    >>> aug = JointAnglePerturbation({'stddev': 0.1, 'joint_groups': ['left_arm', 'right_arm']})
    >>> augmented_params = aug.apply(smplx_params)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from kinetic_augment.augmentations.base import (
    SMPLXAugmentation,
    AugmentationRegistry,
)
from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    SMPLX_JOINT_LIMITS,
    LIMB_GROUPS,
    clamp_joint_angles,
    get_kinematic_chain,
)


@AugmentationRegistry.register("JointAnglePerturbation")
class JointAnglePerturbation(SMPLXAugmentation):
    """
    Perturb joint angles while respecting anatomical limits.

    This augmentation adds Gaussian noise to SMPL-X body pose parameters,
    automatically clamping results to biomechanically valid ranges.

    For Sign Language Recognition:
    - Focus perturbation on arm joints (shoulder, elbow, wrist)
    - Use lower stddev for hands to preserve handshapes
    - Higher stddev for torso creates signer variation

    Parameters:
        stddev (float): Standard deviation of noise in radians (default: 0.1)
        joint_groups (List[str]): Which joint groups to perturb
            Options: 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'spine'
            Default: all groups
        per_joint_stddev (Dict[str, float]): Override stddev for specific joints
        respect_limits (bool): Clamp to anatomical limits (default: True)

    Example:
        >>> aug = JointAnglePerturbation({
        ...     'stddev': 0.15,
        ...     'joint_groups': ['left_arm', 'right_arm'],
        ...     'per_joint_stddev': {'left_wrist': 0.05, 'right_wrist': 0.05}
        ... })
    """

    name = "JointAnglePerturbation"
    category = "intrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Extract configuration
        self.stddev = self.config.get('stddev', 0.1)
        self.joint_groups = self.config.get('joint_groups', list(LIMB_GROUPS.keys()))
        self.per_joint_stddev = self.config.get('per_joint_stddev', {})

        # Build set of joints to perturb
        self._build_joint_set()

    def _build_joint_set(self):
        """Build the set of joint indices to perturb."""
        self.joints_to_perturb: Set[int] = set()

        for group_name in self.joint_groups:
            if group_name in LIMB_GROUPS:
                for joint_name in LIMB_GROUPS[group_name]:
                    if joint_name in SMPLX_BODY_JOINTS:
                        joint_idx = SMPLX_BODY_JOINTS[joint_name]
                        if joint_idx > 0:  # Exclude pelvis (index 0)
                            self.joints_to_perturb.add(joint_idx)

    def _get_stddev_for_joint(self, joint_name: str) -> float:
        """Get standard deviation for a specific joint."""
        return self.per_joint_stddev.get(joint_name, self.stddev)

    def apply(self, params: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply joint angle perturbation to SMPL-X parameters.

        Args:
            params: SMPL-X parameter dictionary with 'body_pose'

        Returns:
            Modified parameters with perturbed body pose
        """
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        body_pose = result['body_pose'].copy()

        # Handle batch dimension
        if body_pose.ndim == 1:
            body_pose = body_pose[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = body_pose.shape[0]

        # Get joint name to index mapping (reverse lookup)
        idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}

        # Apply perturbation to each joint in the set
        for joint_idx in self.joints_to_perturb:
            joint_name = idx_to_name.get(joint_idx, f"joint_{joint_idx}")
            stddev = self._get_stddev_for_joint(joint_name)

            # Body pose indices (excluding pelvis, so joint_idx - 1)
            pose_idx = (joint_idx - 1) * 3

            for b in range(batch_size):
                # Add Gaussian noise
                noise = np.random.randn(3) * stddev
                body_pose[b, pose_idx:pose_idx + 3] += noise

                # Clamp to anatomical limits if enabled
                if self.respect_limits:
                    joint_angles = body_pose[b, pose_idx:pose_idx + 3]
                    clamped = clamp_joint_angles(joint_name, joint_angles)
                    body_pose[b, pose_idx:pose_idx + 3] = clamped

        if squeeze:
            body_pose = body_pose[0]

        result['body_pose'] = body_pose
        return result


@AugmentationRegistry.register("HandPosePerturbation")
class HandPosePerturbation(SMPLXAugmentation):
    """
    Perturb hand pose while attempting to preserve handshape category.

    Critical for Sign Language Recognition where handshape is semantic.
    Uses lower noise levels and optional per-finger constraints.

    Strategy:
    - Low noise for thumb (most discriminative for handshapes)
    - Medium noise for index/middle fingers
    - Higher noise for ring/pinky (less discriminative)
    - Option to couple finger joints (natural finger motion)

    Parameters:
        stddev (float): Base standard deviation in radians (default: 0.05)
        finger_weights (Dict[str, float]): Noise multiplier per finger
            Keys: 'thumb', 'index', 'middle', 'ring', 'pinky'
            Default: thumb=0.3, index=0.5, middle=0.7, ring=1.0, pinky=1.0
        couple_joints (bool): Apply correlated noise within fingers (default: True)
        left_hand (bool): Perturb left hand (default: True)
        right_hand (bool): Perturb right hand (default: True)

    Example:
        >>> aug = HandPosePerturbation({
        ...     'stddev': 0.03,
        ...     'finger_weights': {'thumb': 0.2, 'index': 0.4},
        ...     'couple_joints': True
        ... })
    """

    name = "HandPosePerturbation"
    category = "intrinsic"

    # SMPL-X hand joint structure (MANO format)
    # Each finger has 3 joints, thumb has different structure
    FINGER_JOINT_INDICES = {
        'index': [0, 1, 2],    # index1, index2, index3
        'middle': [3, 4, 5],   # middle1, middle2, middle3
        'pinky': [6, 7, 8],    # pinky1, pinky2, pinky3
        'ring': [9, 10, 11],   # ring1, ring2, ring3
        'thumb': [12, 13, 14], # thumb1, thumb2, thumb3
    }

    DEFAULT_FINGER_WEIGHTS = {
        'thumb': 0.3,   # Most discriminative - least noise
        'index': 0.5,
        'middle': 0.7,
        'ring': 1.0,
        'pinky': 1.0,   # Least discriminative - most noise
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)

        self.stddev = self.config.get('stddev', 0.05)
        self.finger_weights = {
            **self.DEFAULT_FINGER_WEIGHTS,
            **self.config.get('finger_weights', {})
        }
        self.couple_joints = self.config.get('couple_joints', True)
        self.perturb_left = self.config.get('left_hand', True)
        self.perturb_right = self.config.get('right_hand', True)

    def _perturb_hand(self, hand_pose: np.ndarray) -> np.ndarray:
        """
        Perturb a single hand pose.

        Args:
            hand_pose: (batch, 45) hand joint angles

        Returns:
            Perturbed hand pose
        """
        result = hand_pose.copy()

        if result.ndim == 1:
            result = result[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = result.shape[0]

        for finger_name, joint_indices in self.FINGER_JOINT_INDICES.items():
            weight = self.finger_weights.get(finger_name, 1.0)
            effective_stddev = self.stddev * weight

            if self.couple_joints:
                # Correlated noise: same base noise for all joints in finger
                for b in range(batch_size):
                    base_noise = np.random.randn(3) * effective_stddev

                    for i, joint_idx in enumerate(joint_indices):
                        pose_idx = joint_idx * 3
                        # Attenuate noise for distal joints (more constrained)
                        attenuation = 1.0 - (i * 0.2)  # 1.0, 0.8, 0.6
                        result[b, pose_idx:pose_idx + 3] += base_noise * attenuation
            else:
                # Independent noise per joint
                for b in range(batch_size):
                    for joint_idx in joint_indices:
                        pose_idx = joint_idx * 3
                        noise = np.random.randn(3) * effective_stddev
                        result[b, pose_idx:pose_idx + 3] += noise

        if squeeze:
            result = result[0]

        return result

    def apply(self, params: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply hand pose perturbation to SMPL-X parameters.

        Args:
            params: SMPL-X parameter dictionary with hand poses

        Returns:
            Modified parameters with perturbed hand poses
        """
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        if self.perturb_left and 'left_hand_pose' in result:
            result['left_hand_pose'] = self._perturb_hand(result['left_hand_pose'])

        if self.perturb_right and 'right_hand_pose' in result:
            result['right_hand_pose'] = self._perturb_hand(result['right_hand_pose'])

        return result


@AugmentationRegistry.register("JointCoupledNoise")
class JointCoupledNoise(SMPLXAugmentation):
    """
    Apply correlated noise respecting inter-limb coordination.

    Based on motor control research: limb movements are often coupled
    (e.g., arms swing together when walking, hands mirror during signing).

    Coupling modes:
    - 'mirror': Left and right limbs get opposite noise (signing, gesturing)
    - 'synchronous': Left and right get same noise (walking, running)
    - 'independent': No coupling (default)
    - 'kinematic': Noise propagates through kinematic chain

    Parameters:
        stddev (float): Base standard deviation (default: 0.1)
        coupling_mode (str): 'mirror', 'synchronous', 'independent', 'kinematic'
        coupling_strength (float): 0-1, how strongly coupled (default: 0.8)
        affected_limbs (List[str]): Which limb pairs to couple

    Example:
        >>> aug = JointCoupledNoise({
        ...     'stddev': 0.1,
        ...     'coupling_mode': 'mirror',
        ...     'coupling_strength': 0.7,
        ...     'affected_limbs': ['arms']
        ... })
    """

    name = "JointCoupledNoise"
    category = "intrinsic"

    LIMB_PAIRS = {
        'arms': ('left_arm', 'right_arm'),
        'legs': ('left_leg', 'right_leg'),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)

        self.stddev = self.config.get('stddev', 0.1)
        self.coupling_mode = self.config.get('coupling_mode', 'independent')
        self.coupling_strength = self.config.get('coupling_strength', 0.8)
        self.affected_limbs = self.config.get('affected_limbs', ['arms'])

    def _get_joint_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of joint indices that should be coupled."""
        pairs = []

        for limb_name in self.affected_limbs:
            if limb_name in self.LIMB_PAIRS:
                left_group, right_group = self.LIMB_PAIRS[limb_name]

                left_joints = LIMB_GROUPS.get(left_group, [])
                right_joints = LIMB_GROUPS.get(right_group, [])

                # Pair corresponding joints
                for left_name, right_name in zip(left_joints, right_joints):
                    left_idx = SMPLX_BODY_JOINTS.get(left_name)
                    right_idx = SMPLX_BODY_JOINTS.get(right_name)

                    if left_idx is not None and right_idx is not None:
                        pairs.append((left_idx, right_idx))

        return pairs

    def apply(self, params: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply coupled noise to SMPL-X parameters.

        Args:
            params: SMPL-X parameter dictionary

        Returns:
            Modified parameters with coupled noise
        """
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        body_pose = result['body_pose'].copy()

        if body_pose.ndim == 1:
            body_pose = body_pose[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = body_pose.shape[0]
        joint_pairs = self._get_joint_pairs()

        for left_idx, right_idx in joint_pairs:
            left_pose_idx = (left_idx - 1) * 3
            right_pose_idx = (right_idx - 1) * 3

            for b in range(batch_size):
                # Generate base noise
                base_noise = np.random.randn(3) * self.stddev

                if self.coupling_mode == 'mirror':
                    # Opposite noise for left/right (mirror Y and Z axes)
                    left_noise = base_noise
                    right_noise = base_noise * np.array([1, -1, -1])
                elif self.coupling_mode == 'synchronous':
                    # Same noise for both
                    left_noise = base_noise
                    right_noise = base_noise
                elif self.coupling_mode == 'kinematic':
                    # Propagate through chain (simplified)
                    left_noise = base_noise
                    right_noise = base_noise * 0.5  # Attenuated
                else:  # independent
                    left_noise = base_noise
                    right_noise = np.random.randn(3) * self.stddev

                # Apply with coupling strength
                independent_noise_left = np.random.randn(3) * self.stddev
                independent_noise_right = np.random.randn(3) * self.stddev

                final_left = (
                    self.coupling_strength * left_noise +
                    (1 - self.coupling_strength) * independent_noise_left
                )
                final_right = (
                    self.coupling_strength * right_noise +
                    (1 - self.coupling_strength) * independent_noise_right
                )

                body_pose[b, left_pose_idx:left_pose_idx + 3] += final_left
                body_pose[b, right_pose_idx:right_pose_idx + 3] += final_right

        # Clamp to limits
        result['body_pose'] = body_pose[0] if squeeze else body_pose
        result = self._clamp_to_limits(result)

        return result


@AugmentationRegistry.register("GlobalOrientPerturbation")
class GlobalOrientPerturbation(SMPLXAugmentation):
    """
    Perturb global body orientation (root rotation).

    Simulates different camera viewpoints or signer orientations.
    Useful for Sign Language Recognition to train viewpoint invariance.

    Parameters:
        max_angle_deg (Dict[str, float]): Maximum rotation per axis
            Keys: 'x' (pitch), 'y' (yaw), 'z' (roll)
        distribution (str): 'uniform' or 'gaussian'

    Example:
        >>> aug = GlobalOrientPerturbation({
        ...     'max_angle_deg': {'x': 10, 'y': 30, 'z': 5},
        ...     'distribution': 'uniform'
        ... })
    """

    name = "GlobalOrientPerturbation"
    category = "intrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)

        self.max_angle_deg = self.config.get('max_angle_deg', {'x': 10, 'y': 20, 'z': 10})
        self.distribution = self.config.get('distribution', 'uniform')

    def apply(self, params: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """Apply global orientation perturbation."""
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        if 'global_orient' not in result:
            return result

        global_orient = result['global_orient'].copy()

        if global_orient.ndim == 1:
            global_orient = global_orient[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = global_orient.shape[0]

        for b in range(batch_size):
            for i, axis in enumerate(['x', 'y', 'z']):
                max_rad = np.radians(self.max_angle_deg.get(axis, 0))

                if self.distribution == 'gaussian':
                    # Gaussian with 3-sigma = max_rad
                    delta = np.random.randn() * (max_rad / 3)
                    delta = np.clip(delta, -max_rad, max_rad)
                else:
                    # Uniform
                    delta = np.random.uniform(-max_rad, max_rad)

                global_orient[b, i] += delta

        if squeeze:
            global_orient = global_orient[0]

        result['global_orient'] = global_orient
        return result
