# KineticAugment: A Geometry-Aware Augmentation Framework for Human Motion

<p align="center">
  <img src="docs/media/banner.png" alt="KineticAugment Banner" width="70%">
</p>

**KineticAugment** is a production-ready, general framework for augmenting human motion data while preserving anatomical and semantic plausibility. It is designed for researchers and engineers working on any task that learns from body, hand, or face motion (human activity and action recognition, gesture and sign language recognition, rehabilitation and sports analysis, human-robot interaction, motion generation) who need to increase dataset size without introducing unrealistic or label-altering artifacts.

Traditional data augmentation techniques often fail on skeletal time-series data, creating impossible poses and unnatural movements. KineticAugment solves this by treating the human body as a structured kinematic system, governed by a set of core biomechanical and contextual principles.

The framework is not tied to one application. Everything that depends on the task (which features carry meaning, how fast a hand may move, which augmentations are safe) lives in a [task profile](./docs/04_Task_Specific_Profiles.md), not in the core. Sign language recognition was the first case study, which is why the demo clips and several of the findings documents come from sign language data.

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

## Seeing the augmentations: the round-trip demo

<p align="center">
  <a href="docs/media/round_trip_demo.mp4">
    <img src="docs/media/round_trip_demo.gif" alt="Round-trip viewer playing a sign clip: source video, MediaPipe extraction, extraction vs round trip in 3-D, and the fitted SMPL-X body" width="100%">
  </a>
  <br>
  <em>Screen recording of the round-trip viewer (preview; <a href="docs/media/round_trip_demo.mp4">full 42 s recording</a>).</em>
</p>

This is how we check augmentations visually. Augmentations are not applied to the raw landmarks. Each frame's MediaPipe pose, hand and face landmarks are fitted to one SMPL-X body (MANO hands, head and jaw). The augmentation edits that body, and MediaPipe-format landmarks are read back off the mesh. If the round trip with no augmentation reproduces the original motion, then the differences seen after an augmentation come from the augmentation itself. The clips in this demo are sign language videos, the first case study; the same check applies to any recorded motion.

The viewer shows, for one clip at a time:

1. **Source video**: the clip being checked (the viewer labels this panel "Sign video").
2. **MediaPipe extraction (2-D)**: pose, hand and face-mesh landmarks drawn on the frame.
3. **Extraction vs round trip (3-D)**: the extracted landmarks against those read back off SMPL-X,
   coloured by error.
4. **SMPL-X body**: the fitted body with the round-trip points on it, which you can drag to orbit.

Below the panels, per-frame plots track the body, hand, finger-curl and face errors, the jitter,
and the gaps between streams. Those plots are what show whether a clip can be augmented safely.
To run it yourself, see [Round-Trip Viewer](#round-trip-viewer) below.

## Documentation

The complete documentation provides the theory, architecture, and practical guides for using the framework.

*   **[00 - Introduction](./docs/00_Introduction.md)**: Why KineticAugment? The problem with naive augmentation.
*   **[01 - Core Principles](./docs/01_Core_Principles.md)**: A deep dive into the 6 pillars that govern the framework.
*   **[02 - Framework Architecture](./docs/02_Framework_Architecture.md)**: The layered model: Core Engine, Constraint System, and Task Profiles.
*   **[03 - The Augmentation Catalogue](./docs/03_Augmentation_Catalogue.md)**: A comprehensive list of available augmentation techniques.
*   **[04 - Task-Specific Profiles](./docs/04_Task_Specific_Profiles.md)**: How to tailor augmentations for your specific application.
*   **[05 - The Validation Framework](./docs/05_Validation_Framework.md)**: Ensuring the quality and validity of generated data.
*   **[06 - Development Roadmap](./docs/06_Development_Roadmap.md)**: What is built and what is planned.
*   **[07 - Object-Aware Constraints](./docs/07_Object_Aware_Constraints.md)** (planned): Contact and penetration against object meshes, the floor, and the scene.
*   **[08 - Differentiable Losses and the Violation Report](./docs/08_Differentiable_Losses_and_Violation_Report.md)** (planned): The constraints as losses and as a per-frame, per-joint report.
*   **[09 - Plausibility Metrics and Validation](./docs/09_Plausibility_Metrics_and_Validation.md)** (planned): Floor and object metrics, and validation on recorded grasps.
*   **[Glossary](./docs/GLOSSARY.md)**: Definitions of key terms.

### Case study: sign language

These documents record the first application of the framework. They are specific to sign language data; the methods they test (the round trip between landmarks and the body model, hand and face fitting, meaning-preserving augmentation) are not.

*   **[V1 Problem Scope](./docs/V1_Problem_Scope.md)** and **[V1 Findings](./docs/V1_Findings.md)**: The MediaPipe to SMPL-X round trip.
*   **[Week 2 Findings](./docs/Week_2_Findings.md)**: The round trip on unseen footage.
*   **[V1.2 Findings](./docs/V1_2_Findings.md)**: Hands, from MediaPipe to MANO and back.
*   **[V1.3 Face Design](./docs/V1_3_Face_Design.md)**, **[V1.3 Findings](./docs/V1_3_Findings.md)**, and **[V1.3 Face Integration Results](./docs/V1_3_Face_Integration_Results.md)**: Head pose, jaw, hand-to-face geometry, and the face stream.
*   **[Augmentation Grounding](./docs/Augmentation_Grounding.md)**: Which augmentations preserve the meaning of a sign, and the evidence for each.

## Roadmap: from the body alone to the body with objects

The next phases extend the constraint system beyond the body itself: hands against object meshes, feet against the floor, and the same constraints offered as differentiable losses and as a dense violation report. This makes the framework usable for augmenting motion that involves objects and for checking and repairing motion produced by generative models. The companion project [KineticSynth](https://github.com/imics-lab/KineticSynth) builds on these extensions. See the [roadmap](./docs/06_Development_Roadmap.md) and documents 07 to 09.

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

`demo/round_trip/` is an interactive web page, in two pages:

- **Round trip** (`index.html`): checks the MediaPipe → SMPL-X → MediaPipe round trip (pose, hands
  and face mesh) clip by clip, with optional landmark smoothing.
- **Augmentations** (`augment.html`): shows a clip's Tier 1 augmentations as the original, the
  original and augmented bodies one over the other, and the augmented body, with a note on what
  each augmentation changes.

To run it for one clip, from the repository root, after the one-time setup (install and model
files):

```bash
python scripts/export_round_trip_demo.py --clips 20983 --video-dir path/to/videos
```

```bash
python3 -m http.server 8765 --directory demo/round_trip
```

Then open <http://localhost:8765/#clip=20983> (round trip) or
<http://localhost:8765/augment.html#clip=20983&a=combined> (augmentations).
[demo/round_trip/README.md](demo/round_trip/README.md#quick-start-run-the-viewer-for-one-clip)
gives every step, the model files needed, how to check each step worked, and fixes for common
problems.

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
