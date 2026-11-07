#!/usr/bin/env python3
"""
Phase 5 Test Script: Evaluation & Benchmarks

Tests all evaluation components:
- Quality metrics
- Temporal metrics
- Diversity metrics
- Constraint metrics
- Datasets (synthetic)
- Models (LSTM/GRU)
- Trainer
- Benchmark runner
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details and not passed:
        print(f"         {details}")


def generate_test_data(
    num_samples: int = 10,
    num_frames: int = 30,
    num_landmarks: int = 543,
    seed: int = 42,
) -> list:
    """Generate synthetic motion data for testing."""
    rng = np.random.RandomState(seed)

    samples = []
    for _ in range(num_samples):
        # Generate smooth motion-like data
        base = rng.randn(1, num_landmarks, 3) * 0.3
        motion = np.cumsum(rng.randn(num_frames, num_landmarks, 3) * 0.01, axis=0)
        sample = base + motion
        samples.append(sample.astype(np.float32))

    return samples


# =============================================================================
# Test 1: Base Classes
# =============================================================================
def test_base_classes():
    """Test MetricResult, BaseMetric, MetricAggregator."""
    print_header("Test 1: Base Classes")

    from kinetic_augment.evaluation import MetricResult, BaseMetric, MetricAggregator

    # Test MetricResult
    try:
        result = MetricResult(name="TestMetric", value=0.95, details={"key": "value"})
        assert result.name == "TestMetric"
        assert result.value == 0.95
        assert result.details == {"key": "value"}
        print_test("MetricResult creation", True)
    except Exception as e:
        print_test("MetricResult creation", False, str(e))
        return False

    # Test MetricResult repr
    try:
        repr_str = repr(result)
        assert "TestMetric" in repr_str
        assert "0.95" in repr_str
        print_test("MetricResult repr", True)
    except Exception as e:
        print_test("MetricResult repr", False, str(e))
        return False

    # Test BaseMetric is abstract
    try:
        # This should fail - can't instantiate abstract class
        try:
            metric = BaseMetric()
            print_test("BaseMetric is abstract", False, "Should not be instantiable")
            return False
        except TypeError:
            print_test("BaseMetric is abstract", True)
    except Exception as e:
        print_test("BaseMetric is abstract", False, str(e))
        return False

    # Test MetricAggregator
    try:
        from kinetic_augment.evaluation.metrics import VarianceMetric, SmoothnessMetric

        aggregator = MetricAggregator([VarianceMetric(), SmoothnessMetric()])
        samples = generate_test_data(num_samples=5)
        results = aggregator.evaluate(samples)

        assert isinstance(results, dict)
        assert len(results) == 2
        assert "VarianceMetric" in results
        assert "SmoothnessMetric" in results
        print_test("MetricAggregator evaluate", True)
    except Exception as e:
        print_test("MetricAggregator evaluate", False, str(e))
        return False

    return True


# =============================================================================
# Test 2: Quality Metrics
# =============================================================================
def test_quality_metrics():
    """Test quality metrics."""
    print_header("Test 2: Quality Metrics")

    from kinetic_augment.evaluation.metrics import (
        LimbLengthConsistencyMetric,
        PoseValidityMetric,
        LandmarkDetectionRate,
        AnatomicalPlausibilityMetric,
    )

    samples = generate_test_data(num_samples=5)

    # Test LimbLengthConsistencyMetric
    try:
        metric = LimbLengthConsistencyMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"LimbLengthConsistencyMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("LimbLengthConsistencyMetric", False, str(e))
        return False

    # Test PoseValidityMetric
    try:
        metric = PoseValidityMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"PoseValidityMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("PoseValidityMetric", False, str(e))
        return False

    # Test LandmarkDetectionRate
    try:
        metric = LandmarkDetectionRate()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"LandmarkDetectionRate: {result.value:.4f}", True)
    except Exception as e:
        print_test("LandmarkDetectionRate", False, str(e))
        return False

    # Test AnatomicalPlausibilityMetric
    try:
        metric = AnatomicalPlausibilityMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"AnatomicalPlausibilityMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("AnatomicalPlausibilityMetric", False, str(e))
        return False

    return True


# =============================================================================
# Test 3: Temporal Metrics
# =============================================================================
def test_temporal_metrics():
    """Test temporal metrics."""
    print_header("Test 3: Temporal Metrics")

    from kinetic_augment.evaluation.metrics import (
        VelocityMetric,
        SmoothnessMetric,
        TemporalCoherenceMetric,
        SpectralSmoothnessMetric,
    )

    samples = generate_test_data(num_samples=5, num_frames=50)

    # Test VelocityMetric
    try:
        metric = VelocityMetric(frame_rate=30)
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        assert 'max_velocity' in result.details
        print_test(f"VelocityMetric mean: {result.value:.4f}", True)
    except Exception as e:
        print_test("VelocityMetric", False, str(e))
        return False

    # Test SmoothnessMetric
    try:
        metric = SmoothnessMetric(frame_rate=30)
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        print_test(f"SmoothnessMetric (jerk): {result.value:.4f}", True)
    except Exception as e:
        print_test("SmoothnessMetric", False, str(e))
        return False

    # Test TemporalCoherenceMetric
    try:
        metric = TemporalCoherenceMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        # Coherence can be any value, just check it runs
        print_test(f"TemporalCoherenceMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("TemporalCoherenceMetric", False, str(e))
        return False

    # Test SpectralSmoothnessMetric
    try:
        metric = SpectralSmoothnessMetric(frame_rate=30)
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        print_test(f"SpectralSmoothnessMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("SpectralSmoothnessMetric", False, str(e))
        return False

    return True


# =============================================================================
# Test 4: Diversity Metrics
# =============================================================================
def test_diversity_metrics():
    """Test diversity metrics."""
    print_header("Test 4: Diversity Metrics")

    from kinetic_augment.evaluation.metrics import (
        VarianceMetric,
        DistributionShiftMetric,
        AugmentationDiversityMetric,
        CoverageMetric,
    )

    samples = generate_test_data(num_samples=10)
    original_samples = generate_test_data(num_samples=10, seed=0)
    augmented_samples = generate_test_data(num_samples=10, seed=123)

    # Test VarianceMetric
    try:
        metric = VarianceMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        print_test(f"VarianceMetric: {result.value:.6f}", True)
    except Exception as e:
        print_test("VarianceMetric", False, str(e))
        return False

    # Test DistributionShiftMetric
    try:
        metric = DistributionShiftMetric()
        # Convert lists to arrays
        orig_arr = np.stack(original_samples)
        aug_arr = np.stack(augmented_samples)
        result = metric.compute_comparison(orig_arr, aug_arr)
        assert isinstance(result.value, float)
        print_test(f"DistributionShiftMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("DistributionShiftMetric", False, str(e))
        return False

    # Test AugmentationDiversityMetric
    try:
        metric = AugmentationDiversityMetric()
        result = metric.compute(augmented_samples)
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        print_test(f"AugmentationDiversityMetric: {result.value:.6f}", True)
    except Exception as e:
        print_test("AugmentationDiversityMetric", False, str(e))
        return False

    # Test CoverageMetric
    try:
        metric = CoverageMetric()
        result = metric.compute(aug_arr)
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"CoverageMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("CoverageMetric", False, str(e))
        return False

    return True


# =============================================================================
# Test 5: Constraint Metrics
# =============================================================================
def test_constraint_metrics():
    """Test constraint metrics."""
    print_header("Test 5: Constraint Metrics")

    from kinetic_augment.evaluation.metrics import (
        JointLimitComplianceMetric,
        ViolationSeverityMetric,
    )

    samples = generate_test_data(num_samples=5)

    # Test JointLimitComplianceMetric (works on landmarks)
    try:
        metric = JointLimitComplianceMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0
        print_test(f"JointLimitComplianceMetric: {result.value:.4f}", True)
    except Exception as e:
        print_test("JointLimitComplianceMetric", False, str(e))
        return False

    # Test ViolationSeverityMetric (works on landmarks)
    try:
        metric = ViolationSeverityMetric()
        result = metric.compute(samples[0])
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        print_test(f"ViolationSeverityMetric: {result.value:.6f}", True)
    except Exception as e:
        print_test("ViolationSeverityMetric", False, str(e))
        return False

    return True


# =============================================================================
# Test 6: Datasets
# =============================================================================
def test_datasets():
    """Test dataset classes."""
    print_header("Test 6: Datasets")

    from kinetic_augment.evaluation.datasets import SyntheticDataset, EvaluationDataset

    # Test SyntheticDataset
    try:
        dataset = SyntheticDataset(
            num_samples=100,
            num_classes=10,
            num_frames=30,
            seed=42,
        )
        assert len(dataset) == 100
        assert dataset.num_classes == 10
        print_test(f"SyntheticDataset creation: {len(dataset)} samples", True)
    except Exception as e:
        print_test("SyntheticDataset creation", False, str(e))
        return False

    # Test __getitem__
    try:
        landmarks, label = dataset[0]
        assert landmarks.shape == (30, 543, 3)
        assert isinstance(label, int)
        assert 0 <= label < 10
        print_test(f"SyntheticDataset getitem: shape {landmarks.shape}, label {label}", True)
    except Exception as e:
        print_test("SyntheticDataset getitem", False, str(e))
        return False

    # Test get_class_names
    try:
        class_names = dataset.get_class_names()
        assert len(class_names) == 10
        assert class_names[0] == "class_0"
        print_test("SyntheticDataset get_class_names", True)
    except Exception as e:
        print_test("SyntheticDataset get_class_names", False, str(e))
        return False

    # Test EvaluationDataset is abstract
    try:
        try:
            dataset = EvaluationDataset()
            print_test("EvaluationDataset is abstract", False, "Should not be instantiable")
            return False
        except TypeError:
            print_test("EvaluationDataset is abstract", True)
    except Exception as e:
        print_test("EvaluationDataset is abstract", False, str(e))
        return False

    return True


# =============================================================================
# Test 7: Models
# =============================================================================
def test_models():
    """Test model classes."""
    print_header("Test 7: Models")

    from kinetic_augment.evaluation.models import (
        LSTMClassifier,
        GRUClassifier,
        check_torch_available,
        TORCH_AVAILABLE,
    )

    if not TORCH_AVAILABLE:
        print("  (Skipping model tests - PyTorch not available)")
        return True

    import torch

    # Test LSTMClassifier
    try:
        model = LSTMClassifier(num_classes=100, hidden_dim=128, num_layers=1)
        assert model.name == "LSTM"
        num_params = model.get_num_params()
        print_test(f"LSTMClassifier creation: {num_params:,} params", True)
    except Exception as e:
        print_test("LSTMClassifier creation", False, str(e))
        return False

    # Test forward pass
    try:
        batch = torch.randn(4, 30, 1629)  # batch, seq, features
        output = model(batch)
        assert output.shape == (4, 100)
        print_test(f"LSTMClassifier forward: output shape {tuple(output.shape)}", True)
    except Exception as e:
        print_test("LSTMClassifier forward", False, str(e))
        return False

    # Test GRUClassifier
    try:
        model = GRUClassifier(num_classes=100, hidden_dim=128, num_layers=1)
        assert model.name == "GRU"
        num_params = model.get_num_params()
        print_test(f"GRUClassifier creation: {num_params:,} params", True)
    except Exception as e:
        print_test("GRUClassifier creation", False, str(e))
        return False

    # Test GRU forward pass
    try:
        batch = torch.randn(4, 30, 1629)
        output = model(batch)
        assert output.shape == (4, 100)
        print_test(f"GRUClassifier forward: output shape {tuple(output.shape)}", True)
    except Exception as e:
        print_test("GRUClassifier forward", False, str(e))
        return False

    # Test predict
    try:
        preds = model.predict(batch)
        assert preds.shape == (4,)
        print_test(f"GRUClassifier predict: {tuple(preds.shape)}", True)
    except Exception as e:
        print_test("GRUClassifier predict", False, str(e))
        return False

    return True


# =============================================================================
# Test 8: Trainer
# =============================================================================
def test_trainer():
    """Test SLRTrainer."""
    print_header("Test 8: Trainer")

    from kinetic_augment.evaluation.models import (
        LSTMClassifier,
        SLRTrainer,
        TORCH_AVAILABLE,
    )

    if not TORCH_AVAILABLE:
        print("  (Skipping trainer tests - PyTorch not available)")
        return True

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create small synthetic dataset
    try:
        num_samples = 50
        num_classes = 10
        X = torch.randn(num_samples, 20, 1629)
        y = torch.randint(0, num_classes, (num_samples,))

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        print_test("DataLoader creation", True)
    except Exception as e:
        print_test("DataLoader creation", False, str(e))
        return False

    # Create model and trainer
    try:
        model = LSTMClassifier(num_classes=num_classes, hidden_dim=64, num_layers=1)
        trainer = SLRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )
        print_test("SLRTrainer creation", True)
    except Exception as e:
        print_test("SLRTrainer creation", False, str(e))
        return False

    # Test training (just 2 epochs)
    try:
        result = trainer.train(epochs=2, patience=5, verbose=False)
        assert 'history' in result
        assert 'best_val_acc' in result
        print_test(f"SLRTrainer train: best_val_acc={result['best_val_acc']:.4f}", True)
    except Exception as e:
        print_test("SLRTrainer train", False, str(e))
        return False

    # Test evaluation
    try:
        metrics = trainer.evaluate(val_loader)
        assert 'accuracy' in metrics
        assert 'loss' in metrics
        print_test(f"SLRTrainer evaluate: acc={metrics['accuracy']:.4f}", True)
    except Exception as e:
        print_test("SLRTrainer evaluate", False, str(e))
        return False

    return True


# =============================================================================
# Test 9: Benchmark Components
# =============================================================================
def test_benchmark():
    """Test benchmark runner and experiments."""
    print_header("Test 9: Benchmark Components")

    from kinetic_augment.evaluation.benchmark import (
        ExperimentConfig,
        ExperimentResult,
        BenchmarkRunner,
        BenchmarkReport,
        PRESET_COMPARISON,
        INDIVIDUAL_AUGMENTATIONS,
        QUICK_TEST,
        get_standard_experiments,
    )
    from kinetic_augment.evaluation import MetricResult

    # Test ExperimentConfig
    try:
        config = ExperimentConfig(
            name='test_experiment',
            description='Test experiment',
            pipeline_config='moderate',
            epochs=5,
        )
        assert config.name == 'test_experiment'
        assert config.pipeline_config == 'moderate'
        print_test("ExperimentConfig creation", True)
    except Exception as e:
        print_test("ExperimentConfig creation", False, str(e))
        return False

    # Test predefined experiments
    try:
        assert len(PRESET_COMPARISON) == 4
        assert len(INDIVIDUAL_AUGMENTATIONS) == 5
        assert len(QUICK_TEST) == 2

        standard = get_standard_experiments()
        assert len(standard) >= 4
        print_test(f"Predefined experiments: {len(standard)} standard", True)
    except Exception as e:
        print_test("Predefined experiments", False, str(e))
        return False

    # Test BenchmarkRunner initialization
    try:
        runner = BenchmarkRunner(
            experiments=QUICK_TEST,
            device='cpu',
        )
        assert len(runner.experiments) == 2
        print_test("BenchmarkRunner creation", True)
    except Exception as e:
        print_test("BenchmarkRunner creation", False, str(e))
        return False

    # Test BenchmarkReport
    try:
        # Create mock results
        mock_results = {
            'baseline': ExperimentResult(
                config=ExperimentConfig(name='baseline', description='No augmentation'),
                quality_metrics={
                    'LimbLengthConsistency': MetricResult('LimbLengthConsistency', 0.95),
                    'SmoothnessMetric': MetricResult('SmoothnessMetric', 0.02),
                    'VarianceMetric': MetricResult('VarianceMetric', 0.001),
                },
                test_metrics={'accuracy': 0.65, 'top5_accuracy': 0.85, 'loss': 0.5},
            ),
            'moderate': ExperimentResult(
                config=ExperimentConfig(name='moderate', description='Moderate augmentation'),
                quality_metrics={
                    'LimbLengthConsistency': MetricResult('LimbLengthConsistency', 0.92),
                    'SmoothnessMetric': MetricResult('SmoothnessMetric', 0.03),
                    'VarianceMetric': MetricResult('VarianceMetric', 0.002),
                },
                test_metrics={'accuracy': 0.72, 'top5_accuracy': 0.90, 'loss': 0.4},
            ),
        }

        report = BenchmarkReport(mock_results)
        summary = report.generate_summary()
        assert 'Benchmark Results' in summary
        assert 'baseline' in summary or 'moderate' in summary
        print_test("BenchmarkReport generate_summary", True)
    except Exception as e:
        print_test("BenchmarkReport generate_summary", False, str(e))
        return False

    # Test CSV generation
    try:
        csv = report.generate_csv()
        assert 'experiment' in csv
        assert 'baseline' in csv
        print_test("BenchmarkReport generate_csv", True)
    except Exception as e:
        print_test("BenchmarkReport generate_csv", False, str(e))
        return False

    # Test LaTeX generation
    try:
        latex = report.generate_latex_table()
        assert '\\begin{table}' in latex
        assert '\\end{table}' in latex
        print_test("BenchmarkReport generate_latex_table", True)
    except Exception as e:
        print_test("BenchmarkReport generate_latex_table", False, str(e))
        return False

    # Test Markdown generation
    try:
        markdown = report.generate_markdown()
        assert '# Benchmark Results' in markdown
        assert '|' in markdown
        print_test("BenchmarkReport generate_markdown", True)
    except Exception as e:
        print_test("BenchmarkReport generate_markdown", False, str(e))
        return False

    return True


# =============================================================================
# Test 10: End-to-End Integration
# =============================================================================
def test_integration():
    """Test full evaluation pipeline integration."""
    print_header("Test 10: End-to-End Integration")

    from kinetic_augment.evaluation import MetricAggregator
    from kinetic_augment.evaluation.metrics import (
        LimbLengthConsistencyMetric,
        SmoothnessMetric,
        VarianceMetric,
        JointLimitComplianceMetric,
    )
    from kinetic_augment.evaluation.datasets import SyntheticDataset
    from kinetic_augment.pipeline import Pipeline

    # Create pipeline
    try:
        pipeline = Pipeline.from_preset('moderate', seed=42)
        print_test("Pipeline creation", True)
    except Exception as e:
        print_test("Pipeline creation", False, str(e))
        return False

    # Create synthetic dataset
    try:
        dataset = SyntheticDataset(num_samples=20, num_classes=5, seed=42)
        print_test(f"SyntheticDataset: {len(dataset)} samples", True)
    except Exception as e:
        print_test("SyntheticDataset", False, str(e))
        return False

    # Augment samples
    try:
        original_samples = [dataset[i][0] for i in range(10)]
        augmented_samples = [pipeline.process(s) for s in original_samples]
        print_test("Augmentation of samples", True)
    except Exception as e:
        print_test("Augmentation of samples", False, str(e))
        return False

    # Compute metrics on augmented data
    try:
        aggregator = MetricAggregator([
            LimbLengthConsistencyMetric(),
            SmoothnessMetric(),
            VarianceMetric(),
            JointLimitComplianceMetric(),
        ])

        results = aggregator.evaluate(augmented_samples)

        print("  Metrics on augmented data:")
        for name, result in results.items():
            print(f"    {name}: {result.value:.4f}")

        print_test("MetricAggregator on augmented data", True)
    except Exception as e:
        print_test("MetricAggregator on augmented data", False, str(e))
        return False

    # Compare with original
    try:
        original_results = aggregator.evaluate(original_samples)

        print("  Metrics comparison (original → augmented):")
        for name in results:
            orig = original_results[name].value
            aug = results[name].value
            diff = aug - orig
            print(f"    {name}: {orig:.4f} → {aug:.4f} ({diff:+.4f})")

        print_test("Metrics comparison", True)
    except Exception as e:
        print_test("Metrics comparison", False, str(e))
        return False

    return True


# =============================================================================
# Main
# =============================================================================
def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  KineticAugment Phase 5: Evaluation & Benchmarks")
    print("  Test Suite")
    print("="*60)

    tests = [
        ("Base Classes", test_base_classes),
        ("Quality Metrics", test_quality_metrics),
        ("Temporal Metrics", test_temporal_metrics),
        ("Diversity Metrics", test_diversity_metrics),
        ("Constraint Metrics", test_constraint_metrics),
        ("Datasets", test_datasets),
        ("Models", test_models),
        ("Trainer", test_trainer),
        ("Benchmark Components", test_benchmark),
        ("End-to-End Integration", test_integration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} test groups passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
