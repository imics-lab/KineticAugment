# 05: The Validation Framework

Applying a sequence of individually valid augmentations does not guarantee that the final combined result is plausible. Complex interactions can lead to unforeseen edge cases. Therefore, the final step in the KineticAugment pipeline is a comprehensive **Validation Framework**.

The purpose of validation is to provide a final, automated quality assurance check, ensuring that every sample added to the training dataset is of high quality and will help, not hinder, model performance.

The validation process is divided into two stages, mirroring the structure of our [Core Principles](./01_Core_Principles.md):
1.  **Physical Validation:** Checks for anatomical and physical plausibility.
2.  **Semantic Validation:** Checks for task-specific meaning and integrity.

> **Implementation Note:** The validation framework is implemented in `kinetic_augment.evaluation.metrics` with comprehensive metrics for quality assessment. See the [Evaluation Metrics](#implemented-metrics) section below.

---

## Stage 1: Physical Validation

This stage applies to *any* human motion data, regardless of the task. It ensures the final augmented sample represents a physically possible human in motion.

#### Key Physical Checks:

*   **Final Joint Angle Check:**
    *   **Question:** Are all joint angles within their pre-defined anatomical limits?
    *   **Method:** Iterate through all relevant joints and assert that their final calculated rotation is within the `[min, max]` range specified in the profile.

*   **Limb Length Stability Check:**
    *   **Question:** Have any limbs been stretched or compressed beyond a reasonable threshold?
    *   **Method:** Compare the final length of each "bone" (distance between two connected joints) to its original length. Assert that the deviation is less than a small tolerance (e.g., `5%`).

*   **Temporal Coherence Check (Velocity & Jerk):**
    *   **Question:** Is the motion smooth? Are there any instantaneous "teleports"?
    *   **Method:** Calculate the frame-to-frame displacement (velocity) for key landmarks. Assert that this velocity does not exceed a realistic maximum. A more advanced check can also limit the change in velocity (jerk).

*   **Self-Collision Check (Advanced):**
    *   **Question:** Are any body parts passing through each other?
    *   **Method:** Model limbs and the torso as simple geometric primitives (e.g., capsules or spheres) and check for intersections. This is computationally more expensive but crucial for preventing major artifacts like an arm passing through the chest.

---

## Stage 2: Semantic Validation

This stage is task-dependent and configured by the active [Task-Specific Profile](./04_Task_Specific_Profiles.md). It ensures the augmentation did not change the *meaning* of the motion.

#### Key Semantic Checks (Examples):

*   **Handshape Integrity (for SLR):**
    *   **Question:** Is the augmented handshape still classifiable as the original handshape?
    *   **Method:** Use a simple, pre-trained handshape classification model. Predict the handshape class for the original and augmented hand. If the predicted class changes (e.g., from 'A' to 'S'), reject the augmentation.

*   **Grammatical Marker Preservation (for SLR):**
    *   **Question:** If the original sign had a grammatical facial expression (e.g., "eyebrows up" for a question), is it still present?
    *   **Method:** Define a simple rule based on landmark positions (e.g., `eyebrow_y > forehead_y + threshold`). Assert that this rule holds true for both the original and augmented sample if the sign is a question.

*   **Trajectory Similarity (for SLR/Gesture):**
    *   **Question:** Has the core path of a critical limb (e.g., the dominant hand) deviated too much?
    *   **Method:** Calculate the [Dynamic Time Warping (DTW)](./GLOSSARY.md#dynamic-time-warping-dtw) or [Fréchet Distance](./GLOSSARY.md#fréchet-distance) between the original and augmented trajectories. Assert that the distance is below a threshold defined in the profile.

*   **Environmental Constraint Adherence (for HAR):**
    *   **Question:** Is the augmented person still interacting correctly with the environment?
    *   **Method:** If a `ground_plane` is defined, assert that the y-coordinates of the feet landmarks are not below the plane's coordinate.

---

## Conceptual Implementation

The validation process can be encapsulated in a single function that is called at the end of the augmentation pipeline.

```python
# Conceptual pseudocode for the validation pipeline

def validate_augmented_sample(original_motion, augmented_motion, profile):
    """
    Runs all validation checks on an augmented sample.
    Returns True if valid, False otherwise.
    """

    # --- Stage 1: Physical Validation ---
    if not check_all_joint_limits(augmented_motion, profile.joint_limits):
        print("Validation Failed: Joint limit exceeded.")
        return False

    if not check_limb_length_stability(original_motion, augmented_motion, tolerance=0.05):
        print("Validation Failed: Limb length distorted.")
        return False

    if not check_max_velocity(augmented_motion, profile.max_velocity):
        print("Validation Failed: Unrealistic velocity detected.")
        return False

    # --- Stage 2: Semantic Validation ---
    if profile.task_name == "SignLanguageRecognition":
        if not validate_handshape_consistency(original_motion, augmented_motion):
            print("Validation Failed: Semantic handshape changed.")
            return False
        if not validate_trajectory_similarity(original_motion, augmented_motion, threshold=0.1):
            print("Validation Failed: Trajectory deviated too much.")
            return False

    # ... other task-specific checks ...

    print("Validation Succeeded.")
    return True
```

---

## Implemented Metrics

The evaluation module (`kinetic_augment.evaluation`) provides a comprehensive set of metrics for validating augmented data:

### Quality Metrics
| Metric | Description |
|--------|-------------|
| `LimbLengthConsistencyMetric` | Checks limb length stability across frames |
| `PoseValidityMetric` | Validates pose is within reasonable bounds |
| `AnatomicalPlausibilityMetric` | Composite anatomical plausibility score |

### Temporal Metrics
| Metric | Description |
|--------|-------------|
| `VelocityMetric` | Measures velocity statistics and violations |
| `SmoothnessMetric` | Jerk-based smoothness analysis |
| `TemporalCoherenceMetric` | Frame-to-frame correlation and discontinuity detection |
| `SpectralSmoothnessMetric` | Frequency-domain smoothness analysis |

### Constraint Metrics
| Metric | Description |
|--------|-------------|
| `JointLimitComplianceMetric` | Percentage of frames within joint limits |
| `ViolationSeverityMetric` | Severity of constraint violations |
| `ConstraintSatisfactionMetric` | Overall constraint satisfaction rate |

### Diversity Metrics
| Metric | Description |
|--------|-------------|
| `VarianceMetric` | Data variance and spread |
| `DistributionShiftMetric` | Distribution shift from original data |
| `CoverageMetric` | Coverage of the pose space |

### Usage Example

```python
from kinetic_augment.evaluation import MetricAggregator
from kinetic_augment.evaluation.metrics import (
    LimbLengthConsistencyMetric,
    SmoothnessMetric,
    JointLimitComplianceMetric,
)

# Create aggregator with desired metrics
aggregator = MetricAggregator([
    LimbLengthConsistencyMetric(),
    SmoothnessMetric(),
    JointLimitComplianceMetric(),
])

# Evaluate augmented samples
results = aggregator.evaluate(augmented_samples)

# Check results
for name, result in results.items():
    print(f"{name}: {result.value:.4f}")
    if result.value < 0.9:  # Threshold for rejection
        print(f"  Warning: {name} below threshold")
```

---

### Planned Additions

Metrics for the body's relation to the floor and to objects (ground penetration, floating, foot skating, object penetration, contact precision and recall), and a validation of the constraint engine against recorded grasps, are specified in [09 - Plausibility Metrics and Validation](./09_Plausibility_Metrics_and_Validation.md).

### Next Up: Roadmap and Glossary

This concludes the documentation of the framework's philosophy and current structure. The [Development Roadmap](./06_Development_Roadmap.md) describes what is built and what is planned, documents 07 to 09 specify the planned extensions, and the final document provides definitions for key terms used throughout.

**Next: [06 - Development Roadmap](./06_Development_Roadmap.md)** or **[Glossary](./GLOSSARY.md)**