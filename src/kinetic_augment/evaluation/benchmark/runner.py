"""
Benchmark Runner for KineticAugment Evaluation.

Runs augmentation experiments and collects metrics.

Limitation: the quality metrics (smoothness, variance, ...) are computed on SYNTHETIC
random-walk data (see `_generate_test_samples`), not on real clips, so they compare pipeline
configurations on a fixed stand-in signal and do not measure quality on real sign data.
Only the model-training stage uses the real dataset at `dataset_path`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np

from kinetic_augment.evaluation.base import MetricResult, MetricAggregator
from kinetic_augment.evaluation.metrics import (
    LimbLengthConsistencyMetric,
    SmoothnessMetric,
    VarianceMetric,
    JointLimitComplianceMetric,
)


@dataclass
class ExperimentConfig:
    """
    Configuration for a single benchmark experiment.

    Attributes:
        name: Experiment name
        description: Human-readable description
        pipeline_config: Preset name or PipelineConfig
        dataset: Dataset name (e.g., 'wlasl100')
        model: Model type ('lstm', 'gru')
        num_augmented_versions: Virtual dataset expansion factor
        epochs: Training epochs
        batch_size: Training batch size
        seed: Random seed
    """
    name: str
    description: str = ""
    pipeline_config: Union[str, Dict, Any] = 'none'
    dataset: str = 'wlasl100'
    model: str = 'lstm'
    num_augmented_versions: int = 1
    epochs: int = 50
    batch_size: int = 32
    seed: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """
    Result from a single experiment.

    Attributes:
        config: Experiment configuration
        quality_metrics: Augmentation quality metrics
        train_history: Training history
        test_metrics: Final test set metrics
    """
    config: ExperimentConfig
    quality_metrics: Dict[str, MetricResult]
    train_history: Optional[Dict[str, List[float]]] = None
    test_metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class BenchmarkRunner:
    """
    Run augmentation benchmark experiments.

    Workflow:
    1. Load dataset
    2. Create augmented versions using pipeline
    3. Compute augmentation quality metrics
    4. Train model with/without augmentation
    5. Evaluate on test set
    6. Generate comparison report

    Example:
        >>> runner = BenchmarkRunner(
        ...     experiments=[
        ...         ExperimentConfig(name='baseline', pipeline_config='none'),
        ...         ExperimentConfig(name='moderate', pipeline_config='moderate'),
        ...     ],
        ...     dataset_path='data/wlasl',
        ...     output_dir='results/benchmark',
        ... )
        >>> results = runner.run()
    """

    def __init__(
        self,
        experiments: List[ExperimentConfig],
        dataset_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        device: str = 'cpu',
        num_workers: int = 0,
    ):
        """
        Initialize benchmark runner.

        Args:
            experiments: List of experiment configurations
            dataset_path: Path to dataset (for WLASL)
            output_dir: Directory to save results
            device: Device for training
            num_workers: DataLoader workers
        """
        self.experiments = experiments
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = device
        self.num_workers = num_workers

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality metrics for all experiments
        self.quality_metrics = [
            LimbLengthConsistencyMetric(),
            SmoothnessMetric(),
            VarianceMetric(),
            JointLimitComplianceMetric(),
        ]

    def run(
        self,
        run_training: bool = True,
        verbose: bool = True,
    ) -> Dict[str, ExperimentResult]:
        """
        Run all experiments.

        Args:
            run_training: Whether to train models (False = quality metrics only)
            verbose: Print progress

        Returns:
            Dictionary mapping experiment names to results
        """
        results = {}

        for i, exp in enumerate(self.experiments):
            if verbose:
                print(f"\n[{i+1}/{len(self.experiments)}] Running: {exp.name}")
                print(f"  Description: {exp.description}")

            try:
                result = self._run_experiment(exp, run_training, verbose)
                results[exp.name] = result
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results[exp.name] = ExperimentResult(
                    config=exp,
                    quality_metrics={},
                    error=str(e),
                )

        # Save results
        if self.output_dir:
            self._save_results(results)

        return results

    def _run_experiment(
        self,
        config: ExperimentConfig,
        run_training: bool,
        verbose: bool,
    ) -> ExperimentResult:
        """Run a single experiment."""
        from kinetic_augment.pipeline import Pipeline

        # Create pipeline
        if isinstance(config.pipeline_config, str):
            pipeline = Pipeline.from_preset(config.pipeline_config, seed=config.seed)
        elif isinstance(config.pipeline_config, dict):
            pipeline = Pipeline.from_dict(config.pipeline_config)
        else:
            pipeline = Pipeline(config.pipeline_config)

        # Generate test samples for quality metrics
        test_samples = self._generate_test_samples(config.seed)

        # Compute quality metrics
        quality_results = self._compute_quality_metrics(
            test_samples, pipeline, verbose
        )

        # Training (optional)
        train_history = None
        test_metrics = None

        if run_training and self.dataset_path:
            train_history, test_metrics = self._run_training(
                config, pipeline, verbose
            )

        return ExperimentResult(
            config=config,
            quality_metrics=quality_results,
            train_history=train_history,
            test_metrics=test_metrics,
        )

    def _generate_test_samples(
        self,
        seed: int,
        num_samples: int = 20,
        num_frames: int = 30,
    ) -> List[np.ndarray]:
        """
        Generate SYNTHETIC test samples for quality evaluation.

        Each sample is a random offset plus a cumulative-sum random walk, shaped
        (num_frames, 543, 3). It is not real motion data, so quality metrics computed on it
        are only a relative comparison between pipeline configurations.
        """
        rng = np.random.RandomState(seed)

        samples = []
        for _ in range(num_samples):
            # Generate smooth motion-like data
            base = rng.randn(1, 543, 3) * 0.3
            motion = np.cumsum(rng.randn(num_frames, 543, 3) * 0.01, axis=0)
            sample = base + motion
            samples.append(sample.astype(np.float32))

        return samples

    def _compute_quality_metrics(
        self,
        samples: List[np.ndarray],
        pipeline: 'Pipeline',
        verbose: bool,
    ) -> Dict[str, MetricResult]:
        """Compute quality metrics on augmented samples."""
        if verbose:
            print("  Computing quality metrics...")

        aggregator = MetricAggregator(self.quality_metrics)

        # Augment samples
        augmented = []
        for sample in samples:
            aug = pipeline.process(sample, enforce_constraints=False)
            augmented.append(aug)

        # Compute metrics on augmented samples
        results = aggregator.evaluate(augmented)

        if verbose:
            for name, result in results.items():
                print(f"    {name}: {result.value:.4f}")

        return results

    def _run_training(
        self,
        config: ExperimentConfig,
        pipeline: 'Pipeline',
        verbose: bool,
    ) -> tuple:
        """Run model training for an experiment."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from kinetic_augment.pipeline import AugmentedLandmarkDataset
            from kinetic_augment.evaluation.datasets import WLASLDataset
            from kinetic_augment.evaluation.models import LSTMClassifier, GRUClassifier, SLRTrainer
        except ImportError as e:
            if verbose:
                print(f"  Skipping training: {e}")
            return None, None

        if verbose:
            print("  Loading dataset...")

        # Load dataset
        try:
            train_data = WLASLDataset(
                self.dataset_path,
                subset=config.dataset,
                split='train',
            )
            val_data = WLASLDataset(
                self.dataset_path,
                subset=config.dataset,
                split='val',
            )
            test_data = WLASLDataset(
                self.dataset_path,
                subset=config.dataset,
                split='test',
            )
        except Exception as e:
            if verbose:
                print(f"  Dataset loading failed: {e}")
            return None, None

        if verbose:
            print(f"  Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Create augmented dataset
        train_samples = [train_data[i][0] for i in range(len(train_data))]
        train_labels = [train_data[i][1] for i in range(len(train_data))]

        augmented_train = AugmentedLandmarkDataset(
            data_source=train_samples,
            labels=train_labels,
            pipeline=pipeline,
            mode='train',
            num_augmented_versions=config.num_augmented_versions,
            seed=config.seed,
        )

        # Validation and test without augmentation
        val_samples = [val_data[i][0] for i in range(len(val_data))]
        val_labels = [val_data[i][1] for i in range(len(val_data))]

        val_dataset = AugmentedLandmarkDataset(
            data_source=val_samples,
            labels=val_labels,
            pipeline=None,
            mode='val',
        )

        test_samples = [test_data[i][0] for i in range(len(test_data))]
        test_labels = [test_data[i][1] for i in range(len(test_data))]

        test_dataset = AugmentedLandmarkDataset(
            data_source=test_samples,
            labels=test_labels,
            pipeline=None,
            mode='val',
        )

        # Create data loaders
        train_loader = DataLoader(
            augmented_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Create model
        num_classes = train_data.num_classes
        if config.model == 'gru':
            model = GRUClassifier(num_classes=num_classes)
        else:
            model = LSTMClassifier(num_classes=num_classes)

        if verbose:
            print(f"  Model: {model.name} ({model.get_num_params():,} params)")

        # Train
        trainer = SLRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
        )

        if verbose:
            print(f"  Training for {config.epochs} epochs...")

        train_result = trainer.train(
            epochs=config.epochs,
            patience=10,
            verbose=verbose,
        )

        # Evaluate on test set
        test_metrics = trainer.evaluate(test_loader)

        if verbose:
            print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")

        return train_result['history'], test_metrics

    def _save_results(self, results: Dict[str, ExperimentResult]) -> None:
        """Save results to output directory."""
        # Convert to serializable format
        output = {}
        for name, result in results.items():
            output[name] = {
                'config': {
                    'name': result.config.name,
                    'description': result.config.description,
                    'pipeline_config': str(result.config.pipeline_config),
                    'dataset': result.config.dataset,
                    'model': result.config.model,
                    'num_augmented_versions': result.config.num_augmented_versions,
                    'epochs': result.config.epochs,
                    'seed': result.config.seed,
                },
                'quality_metrics': {
                    k: {'value': v.value, 'details': v.details}
                    for k, v in result.quality_metrics.items()
                },
                'test_metrics': result.test_metrics,
                'error': result.error,
            }

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)

    def get_comparison_table(
        self,
        results: Dict[str, ExperimentResult],
    ) -> str:
        """Generate comparison table as string."""
        lines = []
        lines.append("=" * 70)
        lines.append("Benchmark Results Comparison")
        lines.append("=" * 70)

        # Header
        headers = ['Experiment', 'Limb Cons.', 'Smooth.', 'Variance', 'Test Acc']
        lines.append(f"{'  '.join(h.ljust(12) for h in headers)}")
        lines.append("-" * 70)

        # Data rows
        for name, result in results.items():
            qm = result.quality_metrics
            row = [
                name[:12].ljust(12),
                f"{qm.get('LimbLengthConsistency', MetricResult('', 0)).value:.4f}".ljust(12),
                f"{qm.get('SmoothnessMetric', MetricResult('', 0)).value:.4f}".ljust(12),
                f"{qm.get('VarianceMetric', MetricResult('', 0)).value:.4f}".ljust(12),
            ]
            if result.test_metrics:
                row.append(f"{result.test_metrics['accuracy']:.4f}".ljust(12))
            else:
                row.append("N/A".ljust(12))
            lines.append("  ".join(row))

        lines.append("=" * 70)
        return "\n".join(lines)
