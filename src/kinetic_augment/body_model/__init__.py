"""
Body Model Module for KineticAugment.

This module provides SMPL-X integration for parametric human body representation,
enabling proper forward/inverse kinematics for anatomically plausible augmentations.

V1 converter (MediaPipe world landmarks <-> SMPL-X), see docs/V1_Findings.md:
    - SMPLXSkeleton: the SMPL-X rest skeleton and its shape space
    - fit_shape: one plausible body shape (betas) per clip
    - fit_body / reconstruct_body / BodyFit: per-frame parameters and the round trip

Example:
    >>> from kinetic_augment.body_model import (
    ...     SMPLXSkeleton, fit_shape, fit_body, reconstruct_body)
    >>> skeleton = SMPLXSkeleton(model_path="models")
    >>> betas = fit_shape(clip_landmarks, clip_validity, skeleton)
    >>> fit = fit_body(landmarks, validity, skeleton.shaped(betas))
    >>> rebuilt = reconstruct_body(fit, skeleton)

Also: joint_mapping correspondence tables, SMPLXWrapper, and the deprecated
image-space MediaPipeToSMPLX / SMPLXToMediaPipe classes (via the lazy getters).
"""


# Lazy imports to avoid requiring torch/smplx for basic functionality
def get_smplx_wrapper():
    """Get SMPLXWrapper class (lazy import)."""
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
    return SMPLXWrapper


def get_mp_to_smplx():
    """Get MediaPipeToSMPLX class (lazy import)."""
    from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX
    return MediaPipeToSMPLX


def get_smplx_to_mp():
    """Get SMPLXToMediaPipe class (lazy import)."""
    from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe
    return SMPLXToMediaPipe


# Always available
from kinetic_augment.body_model.joint_mapping import (  # noqa: E402
    SMPLX_TO_MEDIAPIPE_POSE,
    MEDIAPIPE_TO_SMPLX_POSE,
    SMPLX_JOINT_NAMES,
    MEDIAPIPE_POSE_NAMES,
    get_corresponding_joints,
)

# The V1 converter. None of these import torch or smplx until a skeleton is built.
from kinetic_augment.body_model.converter import BodyFit, fit_body, reconstruct_body  # noqa: E402
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

__all__ = [
    "SMPLXSkeleton",
    "fit_shape",
    "fit_body",
    "reconstruct_body",
    "BodyFit",
    "get_smplx_wrapper",
    "get_mp_to_smplx",
    "get_smplx_to_mp",
    "SMPLX_TO_MEDIAPIPE_POSE",
    "MEDIAPIPE_TO_SMPLX_POSE",
    "SMPLX_JOINT_NAMES",
    "MEDIAPIPE_POSE_NAMES",
    "get_corresponding_joints",
]
