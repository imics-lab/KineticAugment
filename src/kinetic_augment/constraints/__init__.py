"""
KineticAugment Constraints Package.

This package provides constraint enforcement for augmented motion:

- **Joint Limits**: Anatomical range-of-motion constraints
- **Velocity Limits**: Maximum angular velocities per joint
- **Collision Detection**: Self-collision and environment collision (requires PyBullet)

The ConstraintEngine orchestrates all constraints and can be used
to validate or correct augmented poses.

Example:
    >>> from kinetic_augment.constraints import ConstraintEngine
    >>> engine = ConstraintEngine()
    >>> valid_params = engine.enforce(augmented_params)
    >>> is_valid = engine.validate(params)

Collision Detection Example:
    >>> from kinetic_augment.constraints import ConstraintEngine
    >>> from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
    >>> wrapper = SMPLXWrapper(model_path='models')
    >>> engine = ConstraintEngine(collision_detection=True, smplx_wrapper=wrapper)
    >>> valid_params = engine.enforce(augmented_params)
"""

from kinetic_augment.constraints.engine import (
    ConstraintEngine,
    ConstraintViolation,
)
from kinetic_augment.constraints.joint_limits import (
    JointLimitConstraint,
    enforce_joint_limits,
)
from kinetic_augment.constraints.velocity_limits import (
    VelocityConstraint,
    enforce_velocity_limits,
)

# Collision detection (optional - requires PyBullet)
try:
    from kinetic_augment.constraints.collision import (
        CollisionConstraint,
        CollisionViolation,
        CollisionChecker,
        check_pybullet_available,
        PYBULLET_AVAILABLE,
    )
except ImportError:
    CollisionConstraint = None
    CollisionViolation = None
    CollisionChecker = None
    check_pybullet_available = lambda: False
    PYBULLET_AVAILABLE = False

__all__ = [
    # Core
    "ConstraintEngine",
    "ConstraintViolation",

    # Joint Limits
    "JointLimitConstraint",
    "enforce_joint_limits",

    # Velocity Limits
    "VelocityConstraint",
    "enforce_velocity_limits",

    # Collision Detection (may be None if PyBullet not installed)
    "CollisionConstraint",
    "CollisionViolation",
    "CollisionChecker",
    "check_pybullet_available",
    "PYBULLET_AVAILABLE",
]
