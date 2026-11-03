"""
Baseline Models for KineticAugment Evaluation.

Simple SLR classifiers for evaluation and ablation studies.
"""

from kinetic_augment.evaluation.models.base import (
    BaseSLRClassifier,
    check_torch_available,
    TORCH_AVAILABLE,
)

from kinetic_augment.evaluation.models.lstm_classifier import (
    LSTMClassifier,
    GRUClassifier,
)
from kinetic_augment.evaluation.models.face_classifier import FaceStreamClassifier

from kinetic_augment.evaluation.models.trainer import SLRTrainer

__all__ = [
    'BaseSLRClassifier',
    'LSTMClassifier',
    'GRUClassifier',
    'FaceStreamClassifier',
    'SLRTrainer',
    'check_torch_available',
    'TORCH_AVAILABLE',
]
