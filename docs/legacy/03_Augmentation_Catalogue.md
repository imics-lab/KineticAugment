> **Legacy:** This is the original wish-list of augmentations. It is out of date: it lists operations that were never implemented (e.g. Limb Length Scaling) and contradicts the research.
> The registered operations are listed in the [README](../../README.md#available-augmentations); the evidence-based design is in [Augmentation_Grounding](../Augmentation_Grounding.md).

# 03: The Augmentation Catalogue

This document serves as a comprehensive catalogue of the augmentation techniques available in the KineticAugment framework. These are the fundamental "verbs" that can be combined and constrained within a [Task-Specific Profile](../04_Task_Specific_Profiles.md) to create a full augmentation strategy.

The techniques are divided into three categories:
*   **A. Extrinsic (Scene-Level) Augmentations:** Modifying the subject's relationship to the camera/environment without altering the pose itself.
*   **B. Intrinsic (Anatomy-Level) Augmentations:** Modifying the subject's body configuration.
*   **C. Temporal Augmentations:** Modifying the motion along the time axis.

---

## A. Extrinsic (Scene-Level) Augmentations

These transformations simulate changes in camera position, angle, or the subject's location within the scene. They are generally safe as they do not alter the pose's geometry.

| Augmentation              | Description & Purpose                                                                                                 | Key Parameters                                        | Constraints & Considerations                                                                                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Global Translation**    | Shifts the entire skeleton by a random vector in 3D space. **Purpose:** Simulates the subject being in a different location in the camera's view. | `translation_vector`: `(dx, dy, dz)`                  | **Temporal Coherence:** The translation vector should be applied smoothly over time to avoid jerky motion. The skeleton should remain within a virtual camera frustum.                                                      |
| **Global Rotation**       | Rotates the entire skeleton around a central pivot point (e.g., the pelvis). **Purpose:** Simulates the subject turning slightly or the camera moving around them. | `pivot_point`, `rotation_axis`, `angle` (or `quaternion`) | **Implementation:** Use quaternions for rotation to avoid gimbal lock and ensure smooth interpolation. **Temporal Coherence:** Rotations should be small and applied smoothly across frames.                                    |
| **Global Scaling**        | Scales the entire skeleton uniformly. **Purpose:** Simulates the subject moving closer to or further from the camera.                | `scale_factor` (e.g., `0.9` to `1.1`)                 | **Plausibility:** The scale factor should be kept close to 1.0 to be realistic. Drastic scaling can affect downstream models that rely on absolute size. **Temporal Coherence:** Apply scaling changes gradually.         |
| **Perspective Transform** | Applies a subtle perspective warp to the skeleton. **Purpose:** Simulates changes in camera focal length or being viewed from a slightly different angle. | `warp_matrix`                                         | This is a more complex transformation. It must be implemented carefully to preserve the relative depth ordering of joints and avoid creating anatomical distortions. Often, a combination of scaling and rotation is a simpler, safer alternative. |

---

## B. Intrinsic (Anatomy-Level) Augmentations

These transformations directly modify the pose and configuration of the body. They are highly powerful but require strict adherence to the **Core Principles** to remain valid.

| Augmentation                      | Description & Purpose                                                                                                         | Key Parameters                                     | Constraints & Considerations                                                                                                                                                                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Joint Angle Perturbation**      | Adds small, random noise to the rotation angles of one or more joints. **Purpose:** Introduces natural variation in pose execution. | `joint_group`, `noise_stddev`                      | **Anatomical Plausibility:** **Crucial.** The resulting angle *must* be clamped to the joint's pre-defined anatomical range of motion. **Kinematic Chain Consistency:** The transformation must be propagated down the limb's kinematic chain. |
| **Limb Length Scaling**           | Applies a small, non-uniform scaling factor to the length of specific bones. **Purpose:** Simulates variation in body proportions or soft-tissue artifacts. | `limb_name`, `scale_factor`                        | **Anatomical Plausibility:** This should be used sparingly, with very small scale factors (e.g., 1.0 ± 0.05). Large changes will look unnatural and violate the assumption of a rigid skeleton.                                                         |
| **Pose Flipping (Bilateral Mirroring)** | Mirrors the entire pose across a sagittal plane (usually the Y-Z plane). **Purpose:** A powerful way to double the dataset, especially useful if there is a dominant-hand bias (e.g., right-handed signers). | `axis_of_symmetry`                                 | **Semantic Integrity:** **Crucial.** After mirroring the coordinates, the *labels* for left/right landmarks must be swapped (e.g., `left_wrist` data becomes `right_wrist` data and vice-versa). Fails for semantically chiral actions (e.g., writing). |
| **Joint-Coupled Noise**           | Adds noise to a group of joints while respecting their anatomical coupling. **Purpose:** Realistically augments complex structures like hands or faces. | `joint_group`, `noise_stddev`, `correlation_matrix` | **Inter-Limb Coordination:** Essential for hands, where fingers do not move independently (e.g., ring and pinky fingers are strongly coupled). The correlation matrix defines these dependencies. Used to preserve handshapes while adding noise. |
| **Facial Expression Jittering**   | Applies small, localized deformations to facial landmarks. **Purpose:** Introduces natural variation in non-manual markers and facial expressions. | `landmark_group` (e.g., mouth, eyebrows), `strength` | **Semantic Integrity:** Must not alter grammatical markers (e.g., question eyebrows). **Inter-Limb Coordination** (bilateral symmetry) should be respected for expressions. Works well with Joint-Coupled Noise.                    |

---

## C. Temporal Augmentations

These transformations manipulate the motion sequence along the time axis. They are critical for building models robust to variations in speed and rhythm.

| Augmentation                   | Description & Purpose                                                                                                  | Key Parameters                                | Constraints & Considerations                                                                                                                                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Time Warping**               | Non-linearly stretches and compresses the time axis of the motion sequence. **Purpose:** Simulates variations in cadence, where a person speeds up or slows down during an action. | `warp_control_points`                         | **Temporal Coherence:** The warping function must be smooth and monotonic (i.e., time cannot go backward). The overall duration should not be changed too drastically. Preserves the order of events.              |
| **Dynamic Trajectory Jittering** | Adds a smooth, low-frequency random noise vector to landmark trajectories over time. **Purpose:** Simulates natural human tremor and slight deviations in movement paths. | `frequency`, `amplitude`                        | **Temporal Coherence:** Using a smooth noise function (like Perlin or Simplex noise) is key to avoid jerky motion. The amplitude should be small to avoid significant path deviation that could alter meaning. |
| **Rate Alteration (Resampling)** | Uniformly speeds up or slows down the entire sequence by resampling. **Purpose:** A simple method to simulate faster or slower execution of a whole action. | `rate_factor` (e.g., `0.8` for slower, `1.2` for faster) | **Temporal Coherence:** A simpler form of Time Warping. Extreme factors can create unrealistic motion. The number of frames will change, which may require padding or trimming.                                     |
| **Motion Slicing (Cropping)**    | Extracts a random sub-sequence from a longer motion clip. **Purpose:** Creates more, shorter samples from a long recording. Useful for training on the most relevant part of an action. | `slice_length`, `start_frame`                 | **Semantic Integrity:** The slice must be long enough to contain the core semantic action. Random slicing can easily cut off the preparation or retraction phase, which may or may not be desirable.            |

---

### Next Up: Task-Specific Profiles

This catalogue lists the available tools. The next section explains how to intelligently select, combine, and constrain these tools to build a robust augmentation strategy for your specific problem.

➡️ **Next: [04 - Task-Specific Profiles](../04_Task_Specific_Profiles.md)**