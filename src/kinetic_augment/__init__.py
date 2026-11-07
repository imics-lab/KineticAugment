"""
KineticAugment: A geometry-aware data augmentation framework for human skeletal motion data.

This framework provides physically plausible augmentations for skeleton-based motion data,
respecting anatomical constraints, kinematic chain consistency, and semantic integrity.

Main Components:
    - body_model: SMPL-X integration for parametric body representation
    - augmentations: Core augmentation functions (extrinsic, intrinsic, temporal)
    - constraints: Physical constraint enforcement (joint limits, velocity, collisions)
    - pipeline: Unified augmentation pipeline with PyTorch Dataset support
    - evaluation: Metrics, datasets, models, and benchmarks for evaluation

Example:
    >>> from kinetic_augment import Pipeline
    >>> pipeline = Pipeline.from_preset('moderate')
    >>> augmented_data = pipeline.process(landmarks)

    >>> # Or from YAML config
    >>> pipeline = Pipeline.from_yaml('configs/slr_profile.yaml')

    >>> # With PyTorch Dataset
    >>> from kinetic_augment import AugmentedLandmarkDataset
    >>> dataset = AugmentedLandmarkDataset(
    ...     data_source=landmarks_list,
    ...     labels=labels,
    ...     pipeline=pipeline,
    ...     mode='train',
    ... )

    >>> # Evaluate augmentation quality
    >>> from kinetic_augment.evaluation import MetricAggregator
    >>> from kinetic_augment.evaluation.metrics import SmoothnessMetric, VarianceMetric
    >>> aggregator = MetricAggregator([SmoothnessMetric(), VarianceMetric()])
    >>> results = aggregator.evaluate(augmented_samples)
"""

__version__ = "0.4.0"
__author__ = "Vangelis"

# New pipeline module (Phase 4)
from kinetic_augment.pipeline import (
    Pipeline,
    PipelineConfig,
    ConstraintConfig,
    AugmentationChain,
    get_preset,
    list_presets,
    PRESET_NAMES,
)

# PyTorch datasets (optional)
try:
    from kinetic_augment.pipeline import (
        AugmentedLandmarkDataset,
        LandmarkSequenceDataset,
        create_data_loaders,
    )
except ImportError:
    AugmentedLandmarkDataset = None
    LandmarkSequenceDataset = None
    create_data_loaders = None

# Utilities
from kinetic_augment.utils.data_formats import load_and_reshape_json, save_to_json

# Backwards compatibility: AugmentationPipeline (deprecated)
from kinetic_augment.legacy.pipeline import AugmentationPipeline

# Evaluation module (Phase 5) - lazy import to avoid heavy dependencies
def _get_evaluation():
    """Lazy import for evaluation module."""
    from kinetic_augment import evaluation
    return evaluation

__all__ = [
    # New Pipeline API (Phase 4)
    "Pipeline",
    "PipelineConfig",
    "ConstraintConfig",
    "AugmentationChain",
    "get_preset",
    "list_presets",
    "PRESET_NAMES",
    # PyTorch Datasets
    "AugmentedLandmarkDataset",
    "LandmarkSequenceDataset",
    "create_data_loaders",
    # Utilities
    "load_and_reshape_json",
    "save_to_json",
    # Backwards compatibility (deprecated)
    "AugmentationPipeline",
    # Evaluation (Phase 5)
    "evaluation",
    # Metadata
    "__version__",
]
