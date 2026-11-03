"""
KineticAugment Augmentations Package.

This package provides augmentation classes organized by type:

- **Extrinsic** (scene-level): Global transformations that don't require
  anatomical knowledge (rotation, scaling, flipping)

- **Intrinsic** (body-level): SMPL-X based augmentations that respect
  kinematic chains and anatomical limits (joint angle perturbation,
  limb length scaling)

- **Temporal** (time-level): Modifications to the time axis
  (time warping, speed variation)

Quick Start:
    >>> from kinetic_augment.augmentations import JointAnglePerturbation
    >>> aug = JointAnglePerturbation({'stddev': 0.1})
    >>> augmented_params = aug.apply(smplx_params)

For landmark-based augmentations (fast mode):
    >>> from kinetic_augment.augmentations import GlobalRotation
    >>> aug = GlobalRotation({'max_angle_deg': {'x': 10, 'y': 10, 'z': 10}})
    >>> augmented_landmarks = aug.apply(landmarks)
"""

from kinetic_augment.augmentations.base import (
    BaseAugmentation,
    LandmarkAugmentation,
    SMPLXAugmentation,
    TemporalAugmentation,
    AugmentationConfig,
    AugmentationRegistry,
)

# Import specific augmentations when modules are created
try:
    from kinetic_augment.augmentations.extrinsic import (
        GlobalRotation,
        GlobalScaling,
        PoseFlipping,
        TrajectoryJittering,
    )
except ImportError:
    pass

try:
    from kinetic_augment.augmentations.intrinsic import (
        JointAnglePerturbation,
        HandPosePerturbation,
        JointCoupledNoise,
    )
except ImportError:
    pass

try:
    from kinetic_augment.augmentations.temporal import (
        TimeWarping,
        SpeedVariation,
    )
except ImportError:
    pass

__all__ = [
    # Base classes
    "BaseAugmentation",
    "LandmarkAugmentation",
    "SMPLXAugmentation",
    "TemporalAugmentation",
    "AugmentationConfig",
    "AugmentationRegistry",
    # Extrinsic
    "GlobalRotation",
    "GlobalScaling",
    "PoseFlipping",
    "TrajectoryJittering",
    # Intrinsic
    "JointAnglePerturbation",
    "HandPosePerturbation",
    "JointCoupledNoise",
    # Temporal
    "TimeWarping",
    "SpeedVariation",
]
