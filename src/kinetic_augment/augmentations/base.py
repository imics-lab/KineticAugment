"""
Base classes for KineticAugment augmentations.

This module defines the abstract base classes for all augmentation types:
- BaseAugmentation: Common interface for all augmentations
- LandmarkAugmentation: Operates directly on MediaPipe landmarks (fast)
- SMPLXAugmentation: Operates on SMPL-X parameters (anatomically correct)

The dual-mode design allows choosing between speed and accuracy:
- LandmarkAugmentation: ~1ms per frame, no FK/IK required
- SMPLXAugmentation: ~50ms per frame, full kinematic chain consistency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

REGIONS: Tuple[str, ...] = ("face", "body", "fingers", "temporal", "scene")


@dataclass
class AugmentationConfig:
    """Configuration for an augmentation."""

    name: str
    probability: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")


class BaseAugmentation(ABC):
    """
    Abstract base class for all augmentations.

    All augmentations must implement:
    - apply(): Transform input data
    - validate(): Check if augmentation is valid for input

    Attributes:
        name: Human-readable augmentation name
        category: 'extrinsic', 'intrinsic', or 'temporal'
        requires_smplx: Whether SMPL-X model is needed
    """

    name: str = "BaseAugmentation"
    category: str = "base"
    requires_smplx: bool = False
    regions: Tuple[str, ...] = ()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the augmentation.

        Args:
            config: Configuration dictionary with augmentation parameters
        """
        self.config = config or {}
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters. Override in subclasses."""
        pass

    @abstractmethod
    def apply(self, data: Any, **kwargs) -> Any:
        """
        Apply the augmentation to input data.

        Args:
            data: Input data (format depends on augmentation type)
            **kwargs: Additional arguments

        Returns:
            Augmented data in the same format as input
        """
        pass

    def validate(self, data: Any) -> bool:
        """
        Check if augmentation can be applied to this data.

        Args:
            data: Input data to validate

        Returns:
            True if augmentation can be applied
        """
        return data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


class LandmarkAugmentation(BaseAugmentation):
    """
    Base class for augmentations that operate on MediaPipe landmarks.

    These augmentations are fast but don't guarantee kinematic chain
    consistency. Use for:
    - Scene-level transformations (rotation, scaling, flipping)
    - Camera effects (trajectory jittering)
    - Time warping

    Input format: np.ndarray of shape (num_frames, num_landmarks, 3)
    Output format: np.ndarray of shape (num_frames, num_landmarks, 3)
    """

    category = "landmark"
    requires_smplx = False

    def validate(self, data: np.ndarray) -> bool:
        """Validate landmark data shape."""
        if not isinstance(data, np.ndarray):
            return False
        if data.ndim != 3:
            return False
        if data.shape[2] != 3:
            return False
        return True

    @abstractmethod
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply augmentation to landmark data."""
        pass


class SMPLXAugmentation(BaseAugmentation):
    """
    Base class for augmentations that operate on SMPL-X parameters.

    These augmentations are slower but guarantee:
    - Kinematic chain consistency (moving shoulder affects arm)
    - Anatomical plausibility (joint limits enforced)
    - Limb length preservation

    Input format: Dict with SMPL-X parameters:
        - body_pose: (batch, 63) axis-angle rotations for 21 body joints
        - left_hand_pose: (batch, 45) hand joint rotations
        - right_hand_pose: (batch, 45) hand joint rotations
        - betas: (batch, 10) shape parameters
        - global_orient: (batch, 3) global orientation
        - expression: (batch, 10) facial expression (optional)

    Output format: Same dict structure with modified parameters
    """

    category = "intrinsic"
    requires_smplx = True

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        respect_limits: bool = True,
    ):
        """
        Initialize SMPL-X augmentation.

        Args:
            config: Configuration dictionary
            respect_limits: Whether to enforce anatomical joint limits
        """
        super().__init__(config)
        self.respect_limits = respect_limits

    def validate(self, params: Dict[str, np.ndarray]) -> bool:
        """Validate SMPL-X parameter dictionary."""
        if not isinstance(params, dict):
            return False

        required_keys = ['body_pose']
        for key in required_keys:
            if key not in params:
                return False
            if not isinstance(params[key], np.ndarray):
                return False

        # Check body_pose shape
        body_pose = params['body_pose']
        if body_pose.ndim not in [1, 2]:
            return False
        if body_pose.shape[-1] != 63:
            return False

        return True

    @abstractmethod
    def apply(self, params: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """Apply augmentation to SMPL-X parameters."""
        pass

    def _ensure_batch_dim(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ensure all parameters have batch dimension."""
        result = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                result[key] = value[np.newaxis, ...]
            else:
                result[key] = value
        return result

    def _clamp_to_limits(
        self,
        params: Dict[str, np.ndarray],
        joint_limits: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Clamp body pose to anatomical joint limits.

        Args:
            params: SMPL-X parameters
            joint_limits: Optional custom joint limits

        Returns:
            Parameters with clamped body pose
        """
        if not self.respect_limits:
            return params

        from kinetic_augment.body_model.joint_mapping import (
            clamp_joint_angles,
            SMPLX_BODY_JOINTS,
        )

        result = params.copy()
        body_pose = result['body_pose'].copy()

        # Handle batch dimension
        if body_pose.ndim == 1:
            body_pose = body_pose[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = body_pose.shape[0]

        # Clamp each joint
        for joint_name, joint_idx in SMPLX_BODY_JOINTS.items():
            if joint_idx == 0:  # Skip pelvis (global orient)
                continue

            pose_idx = (joint_idx - 1) * 3  # -1 because body_pose excludes pelvis

            for b in range(batch_size):
                joint_angles = body_pose[b, pose_idx:pose_idx + 3]
                clamped = clamp_joint_angles(joint_name, joint_angles, joint_limits)
                body_pose[b, pose_idx:pose_idx + 3] = clamped

        if squeeze:
            body_pose = body_pose[0]

        result['body_pose'] = body_pose
        return result


class TemporalAugmentation(BaseAugmentation):
    """
    Base class for augmentations that operate on the temporal dimension.

    These augmentations modify the time axis of sequences:
    - Time warping (stretch/compress different parts)
    - Speed variation
    - Frame dropping/interpolation

    Input format: np.ndarray of shape (num_frames, ...) or sequence of SMPL-X params
    Output format: Same structure, potentially different number of frames
    """

    category = "temporal"
    requires_smplx = False

    def validate(self, data: Union[np.ndarray, List]) -> bool:
        """Validate temporal data."""
        if isinstance(data, np.ndarray):
            return data.ndim >= 1
        if isinstance(data, list):
            return len(data) > 0
        return False

    @abstractmethod
    def apply(
        self,
        data: Union[np.ndarray, List[Dict[str, np.ndarray]]],
        **kwargs
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """Apply temporal augmentation."""
        pass


class AugmentationRegistry:
    """
    Registry for augmentation classes.

    Allows looking up augmentations by name and creating instances
    from configuration dictionaries.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an augmentation class."""
        def decorator(aug_class: type):
            existing = cls._registry.get(name)
            if existing is not None and (existing.__module__, existing.__qualname__) != (
                aug_class.__module__,
                aug_class.__qualname__,
            ):
                raise ValueError(
                    f"Augmentation name {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
            cls._registry[name] = aug_class
            return aug_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get augmentation class by name."""
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseAugmentation:
        """Create augmentation instance from name and config."""
        aug_class = cls.get(name)
        if aug_class is None:
            raise ValueError(f"Unknown augmentation: {name}")
        return aug_class(config)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered augmentation names."""
        return list(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List augmentations by category."""
        return [
            name for name, aug_cls in cls._registry.items()
            if getattr(aug_cls, 'category', None) == category
        ]

    @classmethod
    def list_by_region(cls, region: str) -> List[str]:
        """List augmentations tagged with a body region."""
        return [
            name for name, aug_cls in cls._registry.items()
            if region in getattr(aug_cls, 'regions', ())
        ]


class ClipAugmentation(BaseAugmentation):
    """
    Base class for per-clip augmentations tagged by body region.

    Subclasses define DEFAULT_RANGES (and optionally DISTRIBUTIONS), and implement
    sample() (one draw per clip), apply() and identity_draw() (zero-strength draw).
    """

    key: str = ""
    stage: str = "clip"
    DEFAULT_RANGES: Dict[str, Any] = {}
    DISTRIBUTIONS: Dict[str, str] = {}

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        given = dict(ranges or {})
        for k in given:
            if k not in self.DEFAULT_RANGES:
                raise ValueError(f"Unknown range key {k!r} for {type(self).__name__}")
        merged = dict(self.DEFAULT_RANGES)
        merged.update(given)
        self.ranges = merged
        self.probability = probability
        self.defaulted = [k for k in self.DEFAULT_RANGES if k not in given]
        super().__init__({"ranges": given, "probability": probability})

    @abstractmethod
    def sample(self, rng: np.random.Generator, clip: Any, ctx: Any) -> Dict[str, Any]:
        """Draw this augmentation's parameters once for a clip."""

    @abstractmethod
    def apply(self, target: Any, draw: Dict[str, Any], ctx: Any) -> Any:
        """Apply a draw to the target."""

    @abstractmethod
    def identity_draw(self) -> Dict[str, Any]:
        """Return the zero-strength draw that reproduces the input exactly."""
