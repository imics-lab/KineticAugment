"""
KineticAugment Evaluation Module.

Provides metrics, datasets, models, and benchmark tools for evaluating
augmentation quality and downstream SLR performance.
"""

# Base classes
from kinetic_augment.evaluation.base import (
    MetricResult,
    BaseMetric,
    ComparisonMetric,
    MetricAggregator,
)

# Metrics
from kinetic_augment.evaluation.metrics import (
    # Quality metrics
    LimbLengthConsistencyMetric,
    PoseValidityMetric,
    LandmarkDetectionRate,
    AnatomicalPlausibilityMetric,
    # Temporal metrics
    VelocityMetric,
    SmoothnessMetric,
    TemporalCoherenceMetric,
    SpectralSmoothnessMetric,
    # Diversity metrics
    VarianceMetric,
    DistributionShiftMetric,
    AugmentationDiversityMetric,
    CoverageMetric,
    # Constraint metrics
    ConstraintSatisfactionMetric,
    ViolationSeverityMetric,
    JointLimitComplianceMetric,
)

# Datasets
from kinetic_augment.evaluation.datasets import (
    EvaluationDataset,
    SyntheticDataset,
    WLASLDataset,
    WLASLPreprocessor,
)

# Models (optional - requires torch)
from kinetic_augment.evaluation.models import (
    BaseSLRClassifier,
    LSTMClassifier,
    GRUClassifier,
    FaceStreamClassifier,
    SLRTrainer,
    check_torch_available,
    TORCH_AVAILABLE,
)

# Benchmark
from kinetic_augment.evaluation.benchmark import (
    ExperimentConfig,
    ExperimentResult,
    BenchmarkRunner,
    BenchmarkReport,
    # Predefined experiments
    PRESET_COMPARISON,
    INDIVIDUAL_AUGMENTATIONS,
    CONSTRAINT_ABLATION,
    QUICK_TEST,
    create_augmentation_factor_experiments,
    get_standard_experiments,
)


__all__ = [
    # Base
    'MetricResult',
    'BaseMetric',
    'ComparisonMetric',
    'MetricAggregator',
    # Quality metrics
    'LimbLengthConsistencyMetric',
    'PoseValidityMetric',
    'LandmarkDetectionRate',
    'AnatomicalPlausibilityMetric',
    # Temporal metrics
    'VelocityMetric',
    'SmoothnessMetric',
    'TemporalCoherenceMetric',
    'SpectralSmoothnessMetric',
    # Diversity metrics
    'VarianceMetric',
    'DistributionShiftMetric',
    'AugmentationDiversityMetric',
    'CoverageMetric',
    # Constraint metrics
    'ConstraintSatisfactionMetric',
    'ViolationSeverityMetric',
    'JointLimitComplianceMetric',
    # Datasets
    'EvaluationDataset',
    'SyntheticDataset',
    'WLASLDataset',
    'WLASLPreprocessor',
    # Models
    'BaseSLRClassifier',
    'LSTMClassifier',
    'GRUClassifier',
    'FaceStreamClassifier',
    'SLRTrainer',
    'check_torch_available',
    'TORCH_AVAILABLE',
    # Benchmark
    'ExperimentConfig',
    'ExperimentResult',
    'BenchmarkRunner',
    'BenchmarkReport',
    'PRESET_COMPARISON',
    'INDIVIDUAL_AUGMENTATIONS',
    'CONSTRAINT_ABLATION',
    'QUICK_TEST',
    'create_augmentation_factor_experiments',
    'get_standard_experiments',
]
