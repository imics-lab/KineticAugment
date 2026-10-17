# KineticAugment: A Geometry-Aware Augmentation Framework for Human Motion

<p align="center">
  <img src="docs/media/banner.png" alt="KineticAugment Banner" width="70%">
</p>

**KineticAugment** is a production-ready framework for augmenting human motion data while preserving anatomical and semantic plausibility. It's designed for researchers and engineers working on Sign Language Recognition, Human Activity Recognition, and other motion analysis tasks who need to increase dataset size without introducing unrealistic or label-altering artifacts.

Traditional data augmentation techniques often fail on skeletal time-series data, creating impossible poses and unnatural movements. KineticAugment solves this by treating the human body as a structured kinematic system, governed by a set of core biomechanical and contextual principles.

---

## Key Features

*   **Anatomically Plausible:** Augmentations respect joint limits, bone lengths, and self-collision constraints via SMPL-X body model integration.
*   **Kinematically Consistent:** Transformations correctly propagate through the body's kinematic chains using forward kinematics.
*   **Temporally Coherent:** Generates smooth motion that respects human dynamics, with velocity limiting and jerk analysis.
*   **Semantically Aware:** A priority system prevents augmentations from changing the fundamental meaning of a motion.
*   **Dual-Mode Processing:** Fast landmark-based (extrinsic) or accurate SMPL-X parameter-space (intrinsic) augmentations.
*   **Physics-Based Constraints:** PyBullet integration for self-collision detection and resolution.
*   **Comprehensive Evaluation:** Built-in metrics for quality, smoothness, diversity, and constraint satisfaction.
*   **Task-Specific Profiles:** Easily configure the augmentation pipeline using YAML configuration files.

## Documentation

The complete documentation provides the theory, architecture, and practical guides for using the framework.

*   **[00 - Introduction](./docs/00_Introduction.md)**: Why KineticAugment? The problem with naive augmentation.
*   **[01 - Core Principles](./docs/01_Core_Principles.md)**: A deep dive into the 6 pillars that govern the framework.
*   **[02 - Framework Architecture](./docs/02_Framework_Architecture.md)**: The layered model: Core Engine, Constraint System, and Task Profiles.
*   **[03 - The Augmentation Catalogue](./docs/03_Augmentation_Catalogue.md)**: A comprehensive list of available augmentation techniques.
*   **[04 - Task-Specific Profiles](./docs/04_Task_Specific_Profiles.md)**: How to tailor augmentations for your specific application.
*   **[05 - The Validation Framework](./docs/05_Validation_Framework.md)**: Ensuring the quality and validity of generated data.
*   **[Glossary](./docs/GLOSSARY.md)**: Definitions of key terms.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/KineticAugment.git
cd KineticAugment

# Create conda environment
conda create -n kinetic python=3.10
conda activate kinetic

# Install dependencies
pip install -e .

# Download SMPL-X models (required for intrinsic augmentations)
# Place in models/smplx/ directory
```

## Quick Start

### Basic Usage with Presets

```python
from kinetic_augment import Pipeline

# Create pipeline with built-in preset
pipeline = Pipeline.from_preset('moderate')  # Options: none, conservative, moderate, aggressive

# Augment motion data (numpy array of shape [frames, landmarks, 3])
augmented_motion = pipeline.process(landmarks)
```

### Using YAML Configuration

```python
from kinetic_augment import Pipeline

# Load from YAML config file
pipeline = Pipeline.from_yaml('configs/slr_profile.yaml')

# Process with constraint enforcement
augmented = pipeline.process(landmarks, enforce_constraints=True)
```

### PyTorch Dataset Integration

```python
from kinetic_augment import Pipeline, AugmentedLandmarkDataset
from torch.utils.data import DataLoader

# Create pipeline and dataset
pipeline = Pipeline.from_preset('moderate')
dataset = AugmentedLandmarkDataset(
    data_source=landmarks_list,  # List of numpy arrays
    labels=labels,               # List of integer labels
    pipeline=pipeline,
    mode='train',
    num_augmented_versions=3,    # 3x dataset expansion
)

# Use with PyTorch DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Evaluating Augmentation Quality

```python
from kinetic_augment.evaluation import MetricAggregator
from kinetic_augment.evaluation.metrics import (
    SmoothnessMetric,
    VarianceMetric,
    LimbLengthConsistencyMetric,
)

# Create metric aggregator
aggregator = MetricAggregator([
    SmoothnessMetric(),
    VarianceMetric(),
    LimbLengthConsistencyMetric(),
])

# Evaluate augmented samples
results = aggregator.evaluate(augmented_samples)
for name, result in results.items():
    print(f"{name}: {result.value:.4f}")
```

### Running Benchmark Experiments

```python
from kinetic_augment.evaluation.benchmark import (
    BenchmarkRunner,
    PRESET_COMPARISON,
    BenchmarkReport,
)

# Run comparison of all presets
runner = BenchmarkRunner(
    experiments=PRESET_COMPARISON,
    output_dir='results/benchmark',
)
results = runner.run(run_training=False)  # Quality metrics only

# Generate report
report = BenchmarkReport(results)
report.save('results/', formats=['txt', 'csv', 'md'])
```

## Project Structure

```
KineticAugment/
├── src/kinetic_augment/
│   ├── body_model/          # SMPL-X integration & joint mapping
│   ├── augmentations/       # Extrinsic, intrinsic, temporal transforms
│   ├── constraints/         # Joint limits, velocity, collision detection
│   ├── pipeline/            # Unified pipeline & PyTorch datasets
│   └── evaluation/          # Metrics, datasets, models, benchmarks
├── configs/                 # YAML configuration files
├── scripts/                 # Test and utility scripts
├── docs/                    # Documentation
├── demo/round_trip/         # Round-trip web viewer (exported data not in git)
└── models/                  # SMPL-X models (not in git)
```

## Round-Trip Viewer

`demo/round_trip/` is an interactive web page for checking the MediaPipe → SMPL-X → MediaPipe round
trip (pose, hands and face mesh) clip by clip, with optional landmark smoothing. You can export
clips with `scripts/export_round_trip_demo.py`, or view a bundle someone shares with
`python3 -m http.server 8765 --directory round_trip`. See
[demo/round_trip/README.md](demo/round_trip/README.md) for both.

## Available Augmentations

| Category | Augmentations |
|----------|---------------|
| **Extrinsic** | GlobalRotation, GlobalScaling, GlobalTranslation, PoseFlipping, GaussianNoise, JointDropout |
| **Intrinsic** | JointAnglePerturbation, HandPosePerturbation, BodyShapeVariation |
| **Temporal** | TimeWarping, TemporalCrop, SpeedPerturbation, FrameDropout, TrajectoryJittering, MotionSmoothing |

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Quality** | LimbLengthConsistency, PoseValidity, AnatomicalPlausibility |
| **Temporal** | Velocity, Smoothness, TemporalCoherence, SpectralSmoothness |
| **Diversity** | Variance, DistributionShift, AugmentationDiversity, Coverage |
| **Constraints** | JointLimitCompliance, ViolationSeverity, ConstraintSatisfaction |

## Testing

```bash
# Run all phase tests
CUDA_VISIBLE_DEVICES="" python scripts/test_phase1.py  # SMPL-X integration
CUDA_VISIBLE_DEVICES="" python scripts/test_phase2.py  # Augmentations
CUDA_VISIBLE_DEVICES="" python scripts/test_phase3.py  # Collision detection
CUDA_VISIBLE_DEVICES="" python scripts/test_phase4.py  # Pipeline
CUDA_VISIBLE_DEVICES="" python scripts/test_phase5.py  # Evaluation
```

## Contributing

We welcome contributions from the community! Whether it's adding a new augmentation, improving documentation, or reporting a bug, your help is valued. Please read our **[Contributing Guide](./CONTRIBUTING.md)** to learn how you can get involved.

## License

This project is licensed under the [MIT License](./LICENSE).

## Citation

If you use KineticAugment in your research, please cite:

```bibtex
@software{kineticaugment2026,
  title = {KineticAugment: A Geometry-Aware Augmentation Framework for Human Motion},
  author = {Vangelis Metsis},
  year = {2026},
  url = {https://github.com/imics-lab/KineticAugment}
}
```
