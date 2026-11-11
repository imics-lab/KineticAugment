# 08: Differentiable Losses and the Violation Report

**Status: planned (Phase 8 of the [roadmap](./06_Development_Roadmap.md)).** Nothing in this document is implemented yet.

## 1. Why

The constraint engine answers two questions today: "is this frame valid?" and "what is the nearest valid frame?". Both are computed in NumPy, one frame at a time, and the result is a list of `ConstraintViolation` objects.

Two further forms of the same knowledge are needed, by this framework and by projects that build on it:

- **Losses.** A function of the body-model parameters that is zero when a constraint holds, grows with the violation, and has a gradient. With it, an optimizer or a neural network can be pushed toward valid motion instead of being corrected after the fact.
- **A dense report.** For a whole sequence, per frame and per joint, how far the motion is from satisfying each constraint, as one array in physical units. It is what a metric aggregates, what a learned model of the constraints is trained to predict, and what a person looks at to see where a sequence goes wrong.

The rule that governs this document: **one definition of each constraint, three forms.** The hard check, the loss, and the report must be computed from the same quantities and must agree.

## 2. What has to change

| Today | Needed |
|---|---|
| `SMPLXWrapper.forward` runs without gradients and returns NumPy arrays | A path through the body model that keeps gradients and works on batches of frames |
| Checks take one frame as a dictionary of parameters | Functions that take a whole sequence as arrays of shape `(T, ...)` |
| Velocity checks default to a time step of 1/30 s | The frame rate is always passed explicitly; data at 20 frames per second must not be judged by 30 |
| Violations are a list of objects | A dense array with a fixed layout |

## 3. Design

### 3.1 Losses

One function per constraint, in PyTorch, each returning a value per frame and per joint so that the caller decides how to weight and sum:

| Loss | Quantity penalized |
|---|---|
| Joint limits | Angle beyond the allowed range, with the same limits and the same angle definition as `JointLimitConstraint` |
| Velocity limits | Angular speed above the limit, with the same limits as `VelocityConstraint` |
| Object penetration | Penetration depth of body points, from the signed distance grid ([07](./07_Object_Aware_Constraints.md)) |
| Contact | Distance between a contact target and the object's surface |
| Floor | Depth below the floor |

Self-collision as checked in PyBullet has no gradient. It stays a hard check and a channel of the report.

Do not introduce a second definition of any limit. The loss for a constraint reads the same tables and thresholds as its hard check.

### 3.2 The violation report

For a sequence of `T` frames, a `ViolationReport` holds:

- `values`: array of shape `(T, J, K)`, in physical units, zero where a constraint is satisfied;
- `valid`: boolean array of the same shape, false where a channel could not be computed (no object in the sequence, the first frame for velocity, a channel that was switched off);
- `joints`: the names of the `J` joints, in order;
- `channels`: the names and units of the `K` channels, in order;
- `engine_version` and the configuration used, so that a stored report can be traced to the code that made it.

Proposed channels:

| Channel | Unit | Meaning |
|---|---|---|
| `joint_limit_excess` | radians | Angle beyond the allowed range |
| `velocity_excess` | radians per second | Angular speed above the limit |
| `self_penetration` | meters | Depth of penetration into another body part |
| `object_penetration` | meters | Depth of penetration into an object or the static scene |
| `contact_distance` | meters | Distance from a contact target to the surface it should touch |
| `floor_height` | meters | For the feet: positive when floating above the floor, negative when below it |

Quantities measured on the surface of the body (penetration, contact, floor) are attributed to joints: each mesh vertex belongs to the joint with the largest skinning weight, and a joint's value is the largest violation among its vertices.

### 3.3 Consistency

For every constraint with both a loss and a report channel: where the report is zero the loss is zero, and the loss does not decrease when the reported excess increases. Where a hard check exists, a frame it accepts has a zero channel. These statements are tests, not intentions.

## 4. Tasks

**L1. Differentiable forward pass.** An option of the body-model wrapper, or a thin separate function, that returns joints and vertices as tensors with gradients, for a batch of frames.
*Done when* a test compares it with the existing forward pass (same output) and checks a gradient numerically.

**L2. Sequence-level inputs with an explicit frame rate.** The public functions added in this phase take arrays of shape `(T, ...)` and a frame rate. Find every place where a time step of 1/30 s is assumed and make the frame rate an argument.
*Done when* the same motion sampled at 20 and at 30 frames per second gives the same velocity violations.

**L3. Losses.** As in section 3.1. Start with joint limits and velocity, which need no object.
*Done when* gradient descent on each loss alone, from a violating sequence, reaches a sequence that the corresponding hard check accepts.

**L4. Violation report.** As in section 3.2, built on the existing checks and on the queries of document 07.
*Done when* the report of a recorded sequence is near zero everywhere, and a sequence with one injected violation shows it in the right frame, joint, and channel.

**L5. Consistency tests.** As in section 3.3.

**L6. Throughput.** Batch the report over frames and, where possible, run it on the GPU. Report frames per second per channel. Projects that label large sets of sequences with the report need this number, and need to be able to skip the slowest channels (the `valid` mask then says so).

**L7. Release.** Tag a version once L1 to L5 are merged, and state in the release notes which channels are implemented. Downstream projects pin to the tag.

## 5. Open decisions

| ID | Question |
|---|---|
| L-1 | The joint set `J`: the body-model joints as they are, or a reduced set. |
| L-2 | Whether penalties are linear or squared in the excess, and whether that is fixed or left to the caller. |
| L-3 | How velocity limits that depend on the task (for example the higher hand speeds allowed for signing) are passed in: through the task profile, as today, or as an argument. |
| L-4 | Whether PyTorch becomes a required dependency of the constraint package or stays optional, with the NumPy path kept for users who only augment. |

---

### Next Up

➡️ **Next: [09 - Plausibility Metrics and Validation](./09_Plausibility_Metrics_and_Validation.md)**
