# 04: Task-Specific Profiles

The true power of the KineticAugment framework lies in its ability to be tailored to specific tasks. A one-size-fits-all augmentation strategy is suboptimal because the features that are critical to one task may be noise in another. For example, the precise shape of the hand is paramount in Sign Language Recognition but less important for recognizing a "running" action.

This is where **Task-Specific Profiles** come in. A profile is a configuration file (e.g., a `.yaml` file) that defines a complete augmentation plan by specifying:
*   **Which** augmentations from the [catalogue](./03_Augmentation_Catalogue.md) to apply.
*   The **probability** and **intensity** of each augmentation.
*   The **constraints** needed to preserve the meaning of the motion for that task.

The key concept that enables this is the **Augmentation Priority System**.

---

## The Augmentation Priority System

To preserve **Semantic Integrity**, we must first identify which features of a motion are most important for its classification. We categorize features into three priority tiers:

*   **HIGH Priority (Minimal/No Augmentation):**
    *   **Definition:** These are the core, defining features of the motion. Altering them is highly likely to change the action's label.
    *   **Action:** Apply only minimal, tightly controlled noise, or no augmentation at all.
    *   *SLR Example:* The specific handshape of a fingerspelled letter; the "eyebrows up" facial expression for a question.

*   **MEDIUM Priority (Constrained Augmentation):**
    *   **Definition:** These features contribute to the action's identity but have a natural range of variation.
    *   **Action:** Augment within a carefully defined, constrained range.
    *   *SLR Example:* The overall path of the hand during a sign's movement; the location of a sign relative to the chest.

*   **LOW Priority (Heavy Augmentation):**
    *   **Definition:** These features are incidental to the action's meaning and can be varied widely.
    *   **Action:** Augment freely to increase data diversity.
    *   *SLR Example:* The signer's overall position in the frame; subtle swaying of the torso.

---

## Example Profile 1: Sign Language Recognition (SLR)

*   **Goal:** To simulate natural variations between different signers and performances of the same sign, without altering the sign's linguistic meaning.
*   **Key Challenge:** Preserving handshapes, grammatical facial expressions, and critical sign locations.

#### Feature Priorities for SLR:
*   **HIGH:** Handshapes, finger configurations, grammatical non-manual markers (NMMs), contact points between hands.
*   **MEDIUM:** Movement path and velocity, sign location in "sign space" (e.g., near face vs. chest), palm orientation.
*   **LOW:** Overall body position and orientation, non-grammatical facial expressions, torso posture.

#### Example `slr_profile.yaml`:
```yaml
# configs/slr_profile.yaml
# Augmentation profile for Sign Language Recognition

# --- Global Settings ---
kinematic_chains:
  # Define bone connections for kinematic consistency
  - [shoulder, elbow, wrist]
  - [hip, knee, ankle]

joint_limits:
  # Define anatomical joint angle limits (in radians)
  elbow: [0.0, 2.6] # ~0 to 150 degrees
  knee: [0.0, 2.4]

# --- Augmentation Plan (A list of operations to apply) ---
plan:
  - operation: "GlobalRotation"
    probability: 0.5
    params:
      max_angle: 0.08 # ~5 degrees

  - operation: "JointAnglePerturbation"
    probability: 0.7
    params:
      joint_group: "arms"
      noise_stddev: 0.05 # Moderate noise on arm joints

  - operation: "JointCoupledNoise"
    probability: 0.4
    params:
      joint_group: "fingers"
      noise_stddev: 0.005 # VERY low noise on fingers
      # Use a correlation matrix to preserve handshape structure
      correlation_source: "anatomical_hand"
      # HIGH PRIORITY: Do not apply if handshape is critical
      # (This logic would be handled by the pipeline)

  - operation: "TimeWarping"
    probability: 0.6
    params:
      max_warp_factor: 0.15 # Moderate time warping
```

*   **Rationale:**
    *   `JointCoupledNoise` on fingers has a very low standard deviation (`0.005`) to avoid breaking the handshape.
    *   `JointAnglePerturbation` on the arms is more generous (`0.05`) to vary the signing style.
    *   `GlobalRotation` is used to simulate small shifts in the signer's orientation to the camera.

---

## Example Profile 2: Human Activity Recognition (HAR) - "Running"

*   **Goal:** To make the model robust to different running styles, speeds, and terrains.
*   **Key Challenge:** The core reciprocating motion of the limbs must be preserved, but almost everything else can be varied significantly.

#### Feature Priorities for Running:
*   **HIGH:** The fundamental anti-phase coordination of arms and legs.
*   **MEDIUM:** Torso lean, head position, ground-plane contact.
*   **LOW:** Stride length, cadence (speed), arm swing magnitude, absolute position.

#### Example `har_running_profile.yaml`:
```yaml
# configs/har_running_profile.yaml
# Augmentation profile for HAR of "Running"

# --- Global Settings ---
kinematic_chains:
  - [shoulder, elbow, wrist]
  - [hip, knee, ankle]

joint_limits:
  elbow: [0.0, 2.6]
  knee: [0.0, 2.4]

environmental_constraints:
  # Enforce the "Environmental Interaction" principle
  ground_plane:
    enabled: True
    y_coordinate: 0.0

# --- Augmentation Plan ---
plan:
  - operation: "RateAlteration"
    probability: 0.8
    params:
      # Aggressively augment speed
      rate_range: [0.8, 1.3] # 20% slower to 30% faster

  - operation: "JointAnglePerturbation"
    probability: 0.7
    params:
      # Allow for significant variation in arm swing and knee bend
      joint_group: "all"
      noise_stddev: 0.1

  - operation: "GlobalTranslation"
    probability: 0.9
    params:
      # Allow the runner to be anywhere in the frame
      max_offset_x: 0.3
      max_offset_y: 0.1
```

*   **Rationale:**
    *   `RateAlteration` is applied with high probability and a wide range to create variations in running speed.
    *   `JointAnglePerturbation` is much more aggressive (`0.1`) than in the SLR profile, as running styles vary greatly.
    *   A `ground_plane` constraint is enabled to prevent the runner's feet from being augmented to go through the floor.

---

### Next Up: The Validation Framework

After applying a series of augmentations based on a profile, how can we be certain the final result is valid? The next section describes the crucial final step in the pipeline.

➡️ **Next: [05 - The Validation Framework](./05_Validation_Framework.md)**