# V1 Problem Scope: MediaPipe ↔ SMPL-X Round-Trip Conversion

**Status:** Steps 0–10 complete; V1 result accepted 2026-10-05 (Step 10). **Date:** 2026-10-01,
revised through 2026-10-05. **Source snapshot:** `45472c6`. Proven and disproven assumptions are
collected in [Section 12](#12-assumptions-ledger); a presentation-ready summary is in
[`V1_Findings.md`](V1_Findings.md).

This document scopes the first deliverable for the MediaPipe ↔ SMPL-X conversion task:
convert MediaPipe landmarks into genuine SMPL-X parameters and back, losing as little
information as the representation allows, and measure precisely how much is lost.

*Plans, ledgers, one-off scripts, tests and evidence files named here without a link are kept on
the author's research branches and are not part of this repository.*

Evidence labels:
**Code present**, **Verified now**, **Historical claim**, **Inference**, **Proposed work**.

---

## 1. The ask, stated precisely

Go from MediaPipe landmarks to SMPL-X parameters and back without losing information.

Restated as an engineering problem:

> Given MediaPipe Holistic landmarks `P` for a frame, produce SMPL-X parameters `θ` that are
> **valid inputs to the real SMPL-X model**, such that running SMPL-X forward on `θ` and
> reading off the corresponding joints reconstructs `P` to within a stated tolerance.

Two requirements, and both must hold:

1. **Round-trip fidelity.** `P → θ → P'` where `P' ≈ P`.
2. **Parameter validity.** `θ` must be real axis-angle rotations and shape coefficients that
   drive the actual SMPL-X mesh. A `(1, 63)` array of the right shape is not enough.

Requirement 2 is the one that makes this hard, and the one the current code skips.

---

## 2. Why "lossless" must be defined before it can be delivered

Bit-exact round-trip through real SMPL-X is **not achievable**, for three independent reasons.
This is a property of the representations, not a limitation of any implementation.

| Reason | Consequence |
| --- | --- |
| **Bone lengths are constrained.** SMPL-X limb lengths are determined by 10 shape coefficients plus the template. MediaPipe's observed bone lengths are whatever the detector reports, and they drift frame to frame on a rigid limb. | No `betas` vector reproduces arbitrary per-frame bone lengths. Residual length error is structural. |
| **MediaPipe's `z` is not metric.** It is a relative pseudo-depth in units of image width, not a calibrated Euclidean coordinate. | The target is not a metric 3D point cloud. Fitting treats it as one only by convention. |
| **Joint centres do not coincide.** MediaPipe's `left_shoulder` is a surface-adjacent keypoint; SMPL-X's `left_shoulder` is an internal rotation centre. | A systematic offset exists between corresponded points even for a perfect fit. |

Therefore "lossless" is operationalised as:

> **Round-trip error is at or below MediaPipe's own frame-to-frame noise floor.**

If the reconstruction differs from the input by less than the detector's own jitter on an
unmoving limb, the conversion has discarded nothing the input reliably carried. The noise
floor is **measured** (Step 1), not assumed.

#### ⚠️ Correction (2026-10-04): the noise floor is a reference, not a threshold

The definition above conflates two different statistics and must not be used as a pass/fail gate
in its current form:

- **Temporal jitter** is frame-to-frame variation. A conversion with a *systematic* bias — every
  wrist displaced 2 cm the same way every frame — has a jitter of zero.
- **Reconstruction error** is per-frame deviation from the target, systematic parts included.

A reconstruction can therefore sit far above the jitter floor while discarding nothing random,
and the two numbers are not commensurable.

The estimators are also undermined at the source: MediaPipe Holistic defaults to
`smooth_landmarks=True` (**verified now**), which deliberately correlates consecutive frames.
Method B's second-difference estimator assumes independent per-frame noise, so smoothing makes
the measured floor an **underestimate** of true per-frame uncertainty.

Treat **0.18% as a reference scale**, not an acceptance threshold. A defensible threshold needs
to be derived from the same statistic as the reconstruction score — for example, the spread
between repeated detections of the same pose, or a task-level criterion.

**Resolved in Step 9 (2026-10-05).** The threshold is the **rigid-skeleton floor**: the same
per-frame position error, computed by the same round trip, through the best constant-shape
skeleton for each clip. It is the most "lossless" any rigid body model can be on this input. See
Step 9, Consequence 1.

**Rejected alternative.** Exactness is reachable by storing a per-landmark residual alongside
`θ`, making reconstruction exact by construction. This was considered and declined for V1: it
guarantees the number without earning it, and the residual would silently absorb exactly the
error this study exists to quantify. It is retained as a documented fallback (Section 9).

---

## 3. What exists today, and why it does not satisfy the ask

**Verified now.** The current converter round-trips to a maximum absolute error of
`3.7e-09` (05-project-status.md:17). That number is real
and reproducible. It also does not mean what it appears to mean.

[`mp_to_smplx.py:1-30`](../src/kinetic_augment/body_model/mp_to_smplx.py) states in its own module
docstring that it stores the raw bone displacement `child − parent` directly into the axis-angle
slots of `body_pose`, and that the result is **"NOT standard SMPL-X axis-angle, so you cannot
feed these params directly into an SMPL-X model and expect meaningful output."**

[`smplx_to_mp.py`](../src/kinetic_augment/body_model/smplx_to_mp.py) then reconstructs by adding
that displacement back onto the parent position. It accepts an `smplx_wrapper` argument and
**never calls it** ([smplx_to_mp.py:14](../src/kinetic_augment/body_model/smplx_to_mp.py)).

So the round-trip is exact because it is a subtraction followed by an addition. The SMPL-X model
is never evaluated, no rotation is ever computed, and `betas` and `global_orient` have no effect
on the output. It satisfies requirement 1 by entirely bypassing requirement 2.

Consequences that follow directly from this, and that V1 exists to fix:

- Intrinsic augmentations treat displacements as angles. Zero-noise `JointAnglePerturbation`
  changes projected landmarks by `0.56` — a no-op operation that is not a no-op
  (**Verified now**, 05-project-status.md:18).
- Joint-limit clamping applies radian bounds to displacement values, which has no anatomical meaning.
- No mesh, no collision geometry, and no body-shape variation is reachable through this path.

**Historical claim, relevant.** A real Adam-based fitting loop existed before commit `f9c6d7b`
and was replaced by the current encoding. Recover it for reference with:

```bash
git show f9c6d7b^:src/kinetic_augment/body_model/mp_to_smplx.py
```

It is prior art to learn from, not a working baseline — it was never verified, and its
`smoothing_weight` argument was accepted and never used.

---

## 4. Scope boundaries

### In scope for V1

- **Body only.** The 12 reliably corresponded MediaPipe pose landmarks:
  shoulders (11, 12), elbows (13, 14), wrists (15, 16), hips (23, 24), knees (25, 26),
  ankles (27, 28).
- Fitting genuine `body_pose` rotations and `global_orient` for those limbs.
- A measured noise floor, and a round-trip error decomposed by cause.

> **As delivered (Finding 5).** Legs are not observed in this footage, so V1 fits the **6 arm
> landmarks** (shoulders, elbows, wrists): 4 constrained joints plus the root, oriented by the
> nose, or by the hip centre when the nose is missing. The landmark tables below describe the
> original plan.

### Explicitly out of scope for V1

| Deferred | Why |
| --- | --- |
| Hands (MANO, 21 landmarks/hand) | Roughly triples the difficulty: wrist coupling, finger twist ambiguity, and MediaPipe's separate hand coordinate frame. V2. |
| Face (468 landmarks) | Moved into scope by V1.3 ([design](V1_3_Face_Design.md)): head, neck and jaw from FaceLandmarker; expression stays out. |
| Mesh vertices, collision geometry | Downstream of valid parameters. Nothing to do until V1 lands. |
| Rewiring the augmentation pipeline | V1 delivers a verified converter. Integration is a separate change with its own risks. |
| Temporal smoothing, real-time performance | Correctness first. Optimisation against an unverified result is wasted work. |

### What the 12 landmarks actually constrain

**Inference.** Of SMPL-X's 21 `body_pose` joints, the in-scope landmarks constrain **8**:

| Constrained (8) | Driven by bone |
| --- | --- |
| `left_hip`, `right_hip` | hip → knee |
| `left_knee`, `right_knee` | knee → ankle |
| `left_shoulder`, `right_shoulder` | shoulder → elbow |
| `left_elbow`, `right_elbow` | elbow → wrist |

The remaining 13 (`spine1/2/3`, `neck`, `head`, `left/right_collar`, `left/right_foot`,
`left/right_wrist`, and others) have **no MediaPipe landmark constraining them**. MediaPipe
provides no spine landmarks at all.

V1 policy, stated explicitly rather than left implicit:

- **Spine, neck, collars:** held at zero. Torso orientation is absorbed into `global_orient`,
  derived from the shoulder–hip frame.
- **Wrists:** held at zero. Wrist rotation requires hand landmarks (out of scope).
- **Feet, head:** held at zero. The existing `head → nose` and `foot → foot_index` entries in
  [`joint_mapping.py:175-187`](../src/kinetic_augment/body_model/joint_mapping.py) are marked
  `# Approximate` in the source and are **excluded** from the V1 correspondence set.

Reconstruction of unconstrained joints therefore reflects a documented default, not a fit.

---

## 5. Definition of done

**Revised 2026-10-05; all met.** The original criteria (struck below) named the legacy
`MediaPipeToSMPLX` / `SMPLXToMediaPipe` classes, which the Step 7 decision deprecated, and asked
for rotation recovery, which joint positions cannot support because twist is unobservable
(Step 5). V1 as delivered:

| # | Criterion | Status |
| --- | --- | --- |
| 1 | `fit_shape` and `fit_body` return genuine SMPL-X parameters (`betas`, `body_pose`, `global_orient`, `transl`) that run through the real SMPL-X forward pass, for a plausible body | ✅ The forward pass reproduces `reconstruct_body` within 1e-5 m with a fitted shape (test). The body is upright in 409/409 frames, with or without the nose, and its shape lies inside SMPL-X's 95% region |
| 2 | `reconstruct_body` rebuilds landmarks from the parameters alone, shape included, and reports a landmark it could not place as missing | ✅ Tests. Before the fix, 61/409 real frames returned invented positions |
| 3 | Synthetic ground truth recovers known joint **positions** to machine precision; rotations are not claimed | ✅ ≤ 4e-16 m, including rotated roots |
| 4 | The real-clip round trip is reported as direct error, per joint, per clip and held out, against the rigid-skeleton floor | ✅ 3.52% (3.99% held out), against 2.36% (3.41%) |
| 5 | The result is written down **whether or not it passes** | ✅ [`V1_Findings.md`](V1_Findings.md); accepted 2026-10-05 |

"Plausible" is checked as posture and as typicality of shape. Mesh contact and
self-intersection are not measured.

Original criteria (2026-10-01), superseded:

1. ~~`MediaPipeToSMPLX.fit()` returns parameters that run through the real SMPL-X forward pass
   without error and produce a plausible mesh.~~
2. ~~`SMPLXToMediaPipe.project()` reconstructs landmarks **by evaluating the SMPL-X model**,
   not by adding stored displacements.~~
3. ~~A synthetic ground-truth test recovers known rotations to near machine precision (Step 5a).~~
4. ~~Round-trip error on the five cached real clips is reported with the decomposition in Step 5b,
   and compared against the Step 1 noise floor.~~
5. The result is written down **whether or not it passes**. A measured failure is a valid V1
   outcome; an unmeasured success is not.

Point 5 is deliberate. The prior `0.5974 RMSE` figure in the roadmap is uninterpretable because
no one recorded what it measured (05-project-status.md:36).

---

## 6. Approach

Analytic swing-twist inverse kinematics, with shape refinement added only if measurement
demands it.

**Core idea.** For each bone, the rotation carrying the rest-pose bone direction onto the
observed bone direction decomposes into a *swing* (the minimal rotation between two directions)
and a *twist* (rotation about the bone's own axis). Joint positions constrain the swing exactly
and constrain the twist not at all. So the swing is computed in closed form and the twist is set
by explicit policy — zero for V1 — rather than left to an optimiser to resolve arbitrarily.

Walking the kinematic chain parent-to-child and composing these rotations yields local
axis-angle values for each constrained joint.

**Why this over gradient-based fitting:** deterministic, fast, has no local minima, is
unit-testable bone by bone, and makes the twist ambiguity an explicit documented decision
instead of an implicit one. Gradient fitting remains available as a fallback (Section 9).

**What this leaves behind.** Every bone points in the correct direction; bone *lengths* remain
those of SMPL-X's default-shape skeleton. If the subject's proportions differ from the template,
each joint lands on the correct ray at the wrong distance, and the error compounds down the chain.

**Shape refinement, if needed.** Solving for `betas` and a global scale closes that gap. This is
not a patch over residual error — those coefficients exist precisely to match a specific person's
proportions, and the core step simply held them at zero. It breaks a chicken-and-egg dependency:
rotations need the skeleton's proportions, proportions need an approximate pose. The analytic
solve supplies the approximate pose, shape is solved against it, and rotations are optionally
re-solved against the corrected skeleton.

Shape is solved **once per sequence** from median bone lengths, not per frame. Shape does not
change during a clip, and the median rejects per-frame detector jitter.

**This step is evidence-gated.** It is built only if Step 6 shows length error is what pushes
the round-trip outside the noise floor. If the core step already lands inside, it is not built.

---

## 7. Steps

Step 0 is complete, so every step below is unblocked. The `SkeletonSource` seam in Step 2 is
still worth building — not to work around missing weights, but because the synthetic skeleton in
Step 5a is what makes the math testable against known ground truth.

### Step 0 — Unblock the environment — ✅ **COMPLETE (2026-10-01)**

**Verified now:** the weights were **never** in this repository.
[`.gitignore:219-224`](../.gitignore) excludes `models/smplx/*.npz`, `*.pkl`, and
`models/vposer/*.pt`; there is no Git LFS; and the only files ever committed under `models/`
were `version.txt`, `README.md`, and one vertex-ID array. **A fresh clone cannot produce them.**
They are licensed assets requiring registration at `smpl-x.is.tue.mpg.de`.

- [x] `SMPLX_NEUTRAL.npz` (103.7 MB) installed at `models/smplx/SMPLX_NEUTRAL.npz`, from the
      `models_smplx_v1_1.zip` release. Confirmed ignored by git.
- [x] `torch 2.14.1` and `smplx 0.1.28` installed into `.venv` (Python 3.11.15).
- [x] Forward pass verified. Three findings recorded below.

#### Finding 1 — `model_path` must be `"models"`, not `"models/smplx"`

**Verified now.** `smplx.create()` appends the model type to the given directory, so
`"models/smplx"` resolves to `models/smplx/smplx/` and raises
`AssertionError: Path models/smplx/smplx does not exist!`

The docstrings in
[smplx_wrapper.py:19](../src/kinetic_augment/body_model/smplx_wrapper.py) and
[body_model/\_\_init\_\_.py:15](../src/kinetic_augment/body_model/__init__.py) are **wrong**;
the runnable scripts ([test_phase1.py:324](../scripts/test_phase1.py),
[test_phase3.py:473](../scripts/test_phase3.py),
[render_landmarks.py:170](../scripts/render_landmarks.py)) are right.

#### Finding 2 — the forward pass returns **127** joints, not 55

**Verified now.** `output.joints` has shape `(1, 127, 3)`, while `J_regressor` is `(55, 10475)`.
SMPL-X returns its 55 model joints **plus 72 extra landmarks** appended after them.

The wrapper docstring's `(1, 55, 3)` claim
([smplx_wrapper.py:21](../src/kinetic_augment/body_model/smplx_wrapper.py)) is wrong, and
`SMPLXWrapper.num_joints` reports `55` — the `J_regressor` row count — while `forward()`
returns 127. **Index into the first 55 only.** Anything that trusts `num_joints` to describe the
`forward()` output is reading the wrong axis length.

#### Finding 3 — the repo's joint mapping is correct

**Verified now.** The model's `parents` array `(55,)` begins:

```
[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
```

This matches [`SMPLX_BODY_JOINTS`](../src/kinetic_augment/body_model/joint_mapping.py) exactly
for all 22 body joints. That table can be trusted; Step 3 should read `model.parents` directly
rather than re-deriving the tree.

#### Rest-pose reference skeleton (neutral, `betas=0`, metres)

The numbers `SkeletonSource` must supply, and the baseline Step 7 adjusts away from:

| Bone | Length | | Joint | Index |
| --- | --- | --- | --- | --- |
| Upper arm | 0.2676 | | `left_shoulder` | 16 |
| Forearm | 0.2531 | | `left_elbow` | 18 |
| Thigh | 0.3828 | | `left_wrist` | 20 |
| Shin | 0.4066 | | `left_hip` / `left_knee` / `left_ankle` | 1 / 4 / 7 |
| Shoulder width | 0.3159 | | | |

#### Note on `scripts/download_models.py --check`

It exits `1` on a correct V1 install. Its `REQUIRED_FILES` list
([download_models.py:25-30](../scripts/download_models.py)) demands `SMPLX_MALE.npz` and
`SMPLX_FEMALE.npz`, which **no code path in this repository requests** — every call site
defaults to `gender="neutral"`. Treat its failure as a false negative, or narrow the list.

### Step 1 — Measure the noise floor — ✅ **COMPLETE (2026-10-01)**

Reproduce with `scripts/measure_noise_floor.py`.

Two independent estimators were used, because each fails differently and agreement between
them is the evidence that the number is real:

- **Method A** — 10th-percentile frame-to-frame displacement. On near-stationary frames the
  observed step is mostly jitter.
- **Method B** — second-difference estimator. For `p[t] = s[t] + n[t]` with `s` smooth,
  `D[t] = p[t-1] − 2p[t] + p[t+1]` annihilates constant velocity and leaves
  `Var(D) = 6σ²`, so `σ ≈ std(D)/√6`. Restricted to the slowest 25% of frames, since rapid
  direction changes violate local linearity.

#### Result

| Band | Method A | Method B |
| --- | --- | --- |
| x/y | 0.08% | 0.10% |
| z | 0.48% | 0.46% |
| 3-D | 0.73% | 0.46% |

> ~~V1 error budget: 0.73% of shoulder width~~ — **superseded.** Measured on the distorted
> image-space data (Finding 9).

**Re-measured on world landmarks (2026-10-04):**

| Band | Method A | Method B |
| --- | --- | --- |
| x/y | 0.07% | 0.06% |
| z | 0.06% | 0.07% |
| 3-D | 0.18% | 0.10% |

> **V1 error budget: 0.18% of shoulder width (≈ 0.6 mm at SMPL-X's 0.3159 m scale).**
>
> *Now a reference scale, not a gate (Section 2 correction). The gate is the rigid-skeleton
> floor (Step 9).*

Landmark validity now comes from MediaPipe's own per-landmark `visibility` rather than the
image-bounds heuristic, which does not apply to world coordinates. It confirms Finding 5
independently and more directly — the detector rates knees at **0.02** and ankles at **0.00**,
stating plainly that those are its prior and not an observation. Shoulders read 1.00, elbows
0.98, wrists 0.96.

#### Finding 4 — ~~MediaPipe's `z` is ~4.8× noisier than `x`/`y`~~ **SUPERSEDED**

Originally measured as `0.46%` versus `0.10%`, with the conclusion that a fitting objective
should down-weight `z`. **That conclusion was wrong and was never implemented.**

The 4.8× ratio was an artefact of the image-space anisotropy described in Finding 9, not a
property of MediaPipe's depth estimate. Re-measured on world landmarks, `z` is **1.2×** `x`/`y`
— close to isotropic. No axis weighting is needed.

A useful caution: this looked like a clean, actionable result and would have led to a fitting
objective tuned against a coordinate-system bug.

#### Finding 5 — half the in-scope landmarks are not observed in this data

**Verified now.** These are upper-body-framed signing videos. MediaPipe reports leg coordinates
by extrapolating past the image boundary, so they are model prior, not observation:

| Landmark | Observed in frame | 3-D noise |
| --- | --- | --- |
| `left_shoulder` / `right_shoulder` | 100% / 100% | 0.37% / 0.29% |
| `left_elbow` / `right_elbow` | 100% / 100% | 0.47% / 0.49% |
| `left_wrist` / `right_wrist` | 63% / 82% | 1.17% / 1.01% |
| `left_hip` / `right_hip` | **32% / 32%** | excluded |
| `left_knee` / `right_knee` | **0% / 0%** | excluded |
| `left_ankle` / `right_ankle` | **0% / 0%** | excluded |

**This changes Section 4.** Two consequences:

1. **Only 4 joints are constrainable on this data**, not 8: `left/right_shoulder` (via
   shoulder→elbow) and `left/right_elbow` (via elbow→wrist). The hip and knee rotations have no
   observed evidence behind them. Fitting legs here would be fitting to MediaPipe's guesses, and
   any round-trip error reported for them would be meaningless.
2. **`global_orient` cannot use the shoulder–hip frame** as Section 4 assumed — hips are observed
   in only 32% of frames. Derive the torso frame from **shoulders and nose** instead, as
   [`canonicalization.py:35-50`](../src/kinetic_augment/canonicalization.py) already does.

Legs remain in the method's scope; they are simply unverifiable until a full-body clip exists.
Record that as a limitation rather than quietly fitting them.

### Steps 2 & 3 — ✅ **COMPLETE (2026-10-04)**

Implemented test-first in [`skeleton.py`](../src/kinetic_augment/body_model/skeleton.py) and
[`kinematics.py`](../src/kinetic_augment/body_model/kinematics.py), with 15 tests in
`tests/test_kinematics.py`, all passing.

#### Result — forward kinematics matches SMPL-X

> Across 20 random body poses, our FK agrees with `model(...).joints[:, :22]` to a maximum
> absolute error of **1.96e-07 m (0.0002 mm)** — float32 round-off against float64 arithmetic.

This confirms the assumption Step 3 was built to test: SMPL-X's posed joints are a rigid
transform chain over the shaped rest skeleton, so pose-corrective blendshapes move mesh vertices
without displacing joints. **Requirement 2 of Section 1 now has a test behind it.** Parameters
we fit will mean to us exactly what they mean to the model.

#### Finding 6 — `flat_hand_mean=False` silently poisons the zero pose

**Verified now.** The first FK/SMPL-X comparison failed on exactly the finger joints
(26, 27, 29, 30, 32, 33, …) with a maximum error of 0.047 m. The cause was not the kinematics:
SMPL-X defaults to `flat_hand_mean=False`, which **adds a mean hand pose of up to 0.85 rad** to
whatever hand parameters it is given. Passing zeros does not produce a zero pose.

[`SMPLXSkeleton`](../src/kinetic_augment/body_model/skeleton.py) therefore constructs the model
with `flat_hand_mean=True`, so a zero rotation means no rotation and the rest pose is genuinely
the rest pose. Note that
[`SMPLXWrapper`](../src/kinetic_augment/body_model/smplx_wrapper.py) does **not** pass this
flag and so still carries the offset — harmless for V1, which touches no hand parameters, but it
must be settled before V2 fits hands, and any hand pose is only comparable between the two paths
once both agree on the flag.

#### Finding 7 — installing the weights un-skipped 10 broken tests

**Verified now.** `tests/test_body_model.py` now reports 2 failures and 10 errors. Neither is
caused by this work, and both were previously hidden:

- **10 errors** — `AssertionError: Path models/smplx/smplx does not exist!`
  [`test_body_model.py:40`](../tests/test_body_model.py) sets `MODEL_PATH = models/smplx`,
  the same wrong spelling as Finding 1. These tests were skipped while the weights were absent,
  so the bug sat undetected behind a skip marker.
- **2 failures** — `ImportError: cannot import name 'SimpleFitter'`, already recorded in
  05-project-status.md:36.

**Resolved 2026-10-04.** `MODEL_PATH` was serving two incompatible purposes: the asset check
needs `models/smplx`, while `SMPLXWrapper` needs `models`. Splitting them into `MODEL_PATH` and
`SMPLX_ASSET_DIR` fixed 8 of the 10. The suite is now **43 passed, 7 xfailed, 0 failures**.

The 7 remaining are marked `xfail` with reasons rather than deleted or rewritten, because they
fall into two groups that must not be conflated:

**Group 1 — one real production bug (1 test).** `test_rest_pose_joints` asserts
`joints.shape == (1, num_joints, 3)` and gets `(1, 127, 3) == (1, 55, 3)`. This is Finding 2
reproduced as a test failure: the **test is correct and the wrapper is wrong**.
`SMPLXWrapper.num_joints` reports the `J_regressor` row count while `forward()` returns the
appended-landmark count. This needs a production fix, not a test edit.

**Group 2 — six tests written against the API deleted in `f9c6d7b`.** They are a precise
inventory of what the bone-vector encoding gave up:

| Failure | Capability it describes |
| --- | --- |
| `__init__() got an unexpected keyword argument 'num_iterations'` | a fitter that optimises |
| `project() missing 1 required positional argument: 'target_mp'` | a projector that reconstructs from parameters **alone** |
| `cannot import name 'SimpleFitter'` | a model-free fitter |

The middle row is the diagnostic one. A genuine converter does not need the original landmarks
handed back to it; the current signature *requires* `target_mp` precisely because it cannot
reconstruct anything, only offset what it is given. These six tests already encode V1's target
contract, so rewriting them to match the current hack would cement the broken design and force
them to be un-written at Step 4. **When Step 4 lands they should flip to XPASS** — that is the
signal the capability has been restored.

---

### Step 2 — `SkeletonSource` interface

A narrow seam that decouples all fitting work from the blocked weights.

- Interface: rest-pose joint positions `(J, 3)`, parent index array `(J,)`, joint names
- `SyntheticSkeleton`: hardcoded plausible proportions. No weights required.
- `SMPLXSkeleton`: wraps the existing [`SMPLXWrapper`](../src/kinetic_augment/body_model/smplx_wrapper.py). Requires Step 0.

Fitting code depends on the interface only, so swapping in real SMPL-X changes no fitting logic.

### Step 3 — Forward kinematics and projection (SMPL-X → MediaPipe)

- Read the parent tree from `model.parents` directly (Finding 3), not a re-derived table
- Compose rotations down the parent chain to produce world joint positions
- Index into the **first 55 joints only** — `forward()` returns 127 (Finding 2)
- Select corresponded joints and emit a MediaPipe-shaped array
- **Test:** zero rotations reproduce the rest pose exactly

**Store and invert the normalisation.** Fitting happens in SMPL-X's metric, pelvis-centred space;
MediaPipe lives in image-normalised coordinates. The similarity transform between them — roughly
7 numbers (scale, rotation, translation) — is **round-trip state that must be carried alongside
the parameters**. Omit it and reconstruction lands in metres around the hip while the input sits
in `[0,1]` image coordinates, looking catastrophically wrong despite a perfect fit.

This is *not* the residual channel rejected in §2. The distinction is dimensionality and meaning:
a similarity transform is 7 numbers with a physical interpretation (effectively where the camera
stood), whereas the residual would be 99 unexplained numbers per frame absorbing exactly the
error this study exists to measure.

Build this before the inverse. It is the thing the inverse is checked against.

### Step 4 — ✅ **COMPLETE (2026-10-04)**

Implemented test-first in [`swing_ik.py`](../src/kinetic_augment/body_model/swing_ik.py), with 11
tests in `tests/test_swing_ik.py`. Full suite: **54 passed,
7 xfailed**.

#### Result — bone directions are reproduced essentially exactly

Run over all 410 frames of the five cached clips:

| Bone | Solved | Median angular error | Max |
| --- | --- | --- | --- |
| Left shoulder | 100% | 0.0045° | 0.026° |
| Left elbow | 100% | 0.0036° | 0.025° |
| Right shoulder | 100% | 0.0050° | 0.026° |
| Right elbow | 100% | 0.0036° | 0.026° |

No NaNs, no unsolved frames. The residual is float round-off through the
matrix → axis-angle → matrix conversion, not fitting error.

**This is the decomposition Section 6 predicted, now confirmed.** Direction error is
effectively zero *by construction*, because the swing rotation solves it in closed form. So any
round-trip position error Step 5 finds must come from **bone length** — SMPL-X's default-shape
skeleton versus the subject's actual proportions — and not from the fit missing. That is exactly
the condition under which Step 7's shape solve is the right remedy.

#### ~~Deliberate deferral — `global_orient` is not estimated~~ **RETRACTED**

The original claim was that leaving `global_orient` unestimated "provably cannot affect the V1
gate," because with spine and collar at zero the swing solve places each bone along its observed
direction regardless of root rotation.

**That claim is false.** It conflates bone *direction* with joint *position*. Directions are
rotation-invariant; positions are not. Shoulder rotations move elbows and wrists but **cannot
relocate the shoulders themselves**, so a rotated root leaves the shoulders at rest position with
no parameter able to correct them. Procrustes alignment cannot absorb it either: the rotation
that would carry the shoulders into place simultaneously rotates the arms *away* from directions
that were already correct.

**Measured counterexample** — same skeleton for truth and fit, so bone lengths are identical by
construction and root rotation is the only variable:

| Root rotation | Mean position error | Max |
| --- | --- | --- |
| 0° | 0.00 mm | 0.00 mm |
| 10° | 9.87 mm | 81 mm |
| 20° | 16.55 mm | 119 mm |
| 45° | 47.04 mm | 307 mm |

The claim holds only in the degenerate 0° case, which is what made it look true.

Root orientation is therefore a **real error term**, quantified in the revised Step 6 below.

#### Finding 8 — the all-zeros sentinel is a live hazard

**Verified now.** MediaPipe marks undetected landmarks with exact zeros, which is
indistinguishable from a landmark legitimately at the origin. This caused two separate false
results while building Step 4, both in test fixtures that placed a joint at the origin: a test
that passed for the wrong reason, and a test that failed against correct code.

Real image-space landmarks are never exactly at `(0, 0, 0)`, so the sentinel is safe for actual
MediaPipe input. It is **not** safe for synthetic or canonicalised data, where the origin is a
natural place for a joint to sit. The same hazard is already noted for canonicalisation in
03-math-and-geometry.md:142. Any future code
generating synthetic landmarks must keep joints off the origin, or carry an explicit validity
mask instead.

---

### Step 4 — Analytic swing IK (MediaPipe → SMPL-X) *(core deliverable)*

- Derive the torso frame from shoulders and hips → `global_orient`
- Per bone: compute the swing rotation from rest direction to observed direction
- Apply the twist policy (zero for V1)
- Convert world rotations to parent-relative local rotations, in chain order
- Write unconstrained joints as zero per the Section 4 policy

### Step 5 — ✅ **COMPLETE (2026-10-04)**

Re-run on world landmarks extracted by
[`scripts/extract_world_landmarks.py`](../scripts/extract_world_landmarks.py) (409/410 frames
detected). The blocking data problem is recorded as Finding 9 below.

#### Result

| | Value |
| --- | --- |
| Part A — synthetic ground truth (**positions**) | **2.99e-16 m** (machine precision) |
| Part B — round-trip error | **5.94%** of shoulder width |
| Step 1 noise floor | 0.18% |
| Verdict | **33× the jitter reference**, which is not a gate (Section 2 correction) |

Per joint (median / p95, % shoulder width): shoulders 6.2 / 11.6 and 6.5 / 10.6, elbows
5.7 / 9.1 and 4.2 / 8.9, wrists 6.9 / 15.5 and 6.5 / 14.8. Error is spread evenly rather than
accumulating down the chain, which already argues against a kinematics fault.

**Part A measures positions, not rotations.** It must not be described as recovering the
original rotations: twist is underdetermined, so many rotation vectors produce identical joint
positions. Exact *position* recovery is the correct test, and rotation recovery should only be
asserted for cases where the ambiguity is explicitly pinned
(`test_recovers_a_known_single_rotation` uses a pure-swing rotation
for exactly that reason). Analytic swing is well-founded — HybrIK
([arXiv:2011.14672](https://arxiv.org/abs/2011.14672)) uses the same decomposition, and resolves
twist from additional image evidence rather than from positions.

**The reported number is an *aligned* residual.** The harness fits a fresh rotation, translation
and scale per frame before scoring
([`measure_round_trip.py`](../scripts/measure_round_trip.py)). That is a useful diagnostic, but
the transform is **not part of the converter's output**, so this is not yet the error a consumer
of the converter would see. Direct and aligned error must be reported separately, and the
converter's contract must say what coordinate state it returns.

#### The decomposition answers Step 6 cleanly

| Component | Value | Reading |
| --- | --- | --- |
| Bone **length** error | 6.66% | **dominant** |
| Bone **direction** error | exact by construction | fit is not missing — *exact in the IK, but 2–3° in the output until Step 8 removed the final rotation (now 0.00°)* |

Observed bone lengths against the SMPL-X neutral rest skeleton:

| Bone | SMPL-X | Observed | Ratio |
| --- | --- | --- | --- |
| L upper arm | 0.2676 | 0.2388 | 0.893 |
| L forearm | 0.2531 | 0.2677 | 1.058 |
| R upper arm | 0.2749 | 0.2429 | 0.884 |
| R forearm | 0.2498 | 0.2458 | 0.984 |

The ratios are **not uniform** — spread 0.071. That is the decisive detail. A single global
scale cannot reconcile them, and the Procrustes alignment already absorbs uniform scale, so
what remains is a *proportion* difference rather than a size difference.

#### Finding 10 — the five clips are different people, and the lengths are partly a prior

**Verified now.** The table above pools five clips. They are **not one subject**: the videos
come from at least three sources and show at least four different people. Per clip:

| Clip | Shoulder W | L upper | L fore | R upper | R fore | fore/upper |
| --- | --- | --- | --- | --- | --- | --- |
| 01986 | 0.3327 | 0.2433 | 0.2765 | 0.2487 | 0.2487 | 1.068 |
| 01987 | 0.3337 | 0.2406 | **0.1909** | 0.2506 | 0.2207 | **0.837** |
| 01988 | 0.3253 | 0.2569 | 0.2715 | 0.2362 | 0.2569 | 1.072 |
| 01991 | 0.3467 | 0.2285 | 0.2388 | 0.2598 | 0.2718 | 1.046 |
| 01992 | 0.3296 | 0.2291 | 0.2748 | 0.2339 | 0.2353 | 1.103 |
| **spread** | **2.1%** | 4.4% | 13.1% | 3.9% | 7.1% | 9.3% |

`01987` is the one male signer and is also the proportion outlier, so world landmarks do carry
real individual signal. But shoulder width varies by only **2.1%** across one man and three
women (four people over five clips — `01991` and `01992` show the same signer), which is far
tighter than adult anthropometry allows. `pose_world_landmarks` are model
output fitted to the image, carrying MediaPipe's own body prior — not an anthropometric
measurement.

**Two consequences for Step 7:**

1. **Fit `betas` per clip, not globally.** This answers the open question in Section 10 for V1:
   the clips are different bodies, so a shared shape vector would be fitting an average of
   strangers.
2. **State the objective accurately.** Step 7 reconciles SMPL-X's template with MediaPipe's
   *estimate* of each body. That is valid for the round-trip gate, which asks only whether the
   conversion preserves what the input carried. It is **not** recovery of a person's true
   proportions, and no write-up should claim it is.

#### ~~Step 6 verdict: length error dominates → proceed to Step 7~~ **RETRACTED**

That verdict was reached without ever testing the root contribution, so it was unfounded. The
decomposition below runs each candidate fix as an **oracle** — the best any estimator of that
kind could do — to measure what each actually explains.

| Configuration | Pooled error | Reduction |
| --- | --- | --- |
| Baseline (current) | 5.94% | — |
| Root orientation only | 5.13% | 13.7% |
| Bone length only | 3.86% | 35.0% |
| **Both** | **2.64%** | **55.5%** |

Shape is the larger single contributor, so the original verdict was right in direction. But two
things it asserted are wrong:

1. **Root is a real term, not zero.** It explains 13.7% on its own.
2. **Neither fix reaches the gate, nor do both together.** With *oracle* shape and *oracle* root,
   **44.5% of the error remains unexplained**. Step 7 alone was never going to close this.

The residual needs its own investigation before more parameters are added. The leading candidate
is the joint-centre offset named in Section 2: MediaPipe's `left_shoulder` is a surface-adjacent
keypoint while SMPL-X's is an internal rotation centre, and nothing in the current pipeline
models that difference.

Corroborating evidence: the per-frame Procrustes fits a median scale of **0.9726** (5th–95th
percentile 0.91–1.02) when both spaces are metric and the true scale is 1.0. A free scale
parameter absorbing ±5% per frame is a symptom of systematic correspondence error, not noise.

**Disproven (2026-10-05).** With each clip's fitted shape (Step 10) the median per-frame scale is
**0.9997** (template body: 0.978 on the clean data). The deficit was body shape, not
correspondence error — consistent with E7 (Step 9), where constant offsets explain about 4%.

---

### Step 5 — the data problem that had to be fixed first

Harness built: [`scripts/measure_round_trip.py`](../scripts/measure_round_trip.py), with
[`alignment.py`](../src/kinetic_augment/body_model/alignment.py) (6 tests) for the
Umeyama similarity used to compare SMPL-X's metric space to MediaPipe's.

#### Part A — synthetic ground truth: ✅ PASS

> 200 random poses, worst joint position error: **2.99e-16 m**.

~~Rotations we generated are recovered to machine precision.~~ Joint *positions* are recovered to
machine precision; rotations are not claimed, because twist is unobservable (Step 5, above).
**The mathematics is correct.**

#### Part B — real clips: 50.05% of shoulder width, 68× the Step 1 budget

Before treating that as a result, note it contradicts Step 4, which measured bone-direction
error at 0.004°. Here the same bones read ~5° after alignment. A fit that is exact in direction
cannot produce 5° of direction error unless the alignment is fighting the data.

#### Finding 9 — the cached landmarks are not in a Euclidean space

**Verified now.** The forearm-to-upper-arm length ratio, which anatomy fixes at roughly 0.95:

| Source | fore/upper |
| --- | --- |
| SMPL-X rest skeleton | 0.946 |
| Cached landmarks, xyz | 1.228 |
| Cached landmarks, xy only | 0.614 |

The cached values bracket the truth instead of approximating it, because the axes are not on a
common scale. MediaPipe normalises **x by image width and y by image height**, so any non-square
video is anisotropically stretched — and the five clips have **three different aspect ratios**
(320×240 and 640×480 at 1.333, 1280×720 at 1.778, 288×192 at 1.500). There is no single
correction, because the distortion differs per clip.

**Root cause.** [`test_phase1.py:146-147`](../scripts/test_phase1.py) captures
`results.pose_landmarks` — image-normalised coordinates with a predicted pseudo-depth.
MediaPipe also exposes `results.pose_world_landmarks`: metric, hip-centred, in real units.
That is the correct input for fitting a metric body model.

Re-extracting clip `01986` both ways:

| | fore/upper | upper-arm length |
| --- | --- | --- |
| `pose_landmarks` (cached) | 1.567 | n/a — not metric |
| `pose_world_landmarks` | **1.066** | **0.2433 m** |
| SMPL-X rest | 0.946 | 0.2676 m |

World landmarks were present in **83/83 frames** and give anatomically plausible proportions.
~~The residual 13% is ordinary individual variation — precisely what `betas` exists to absorb, and
exactly the case Step 7 was scoped for.~~ **Disproven in Step 10 (Finding 14):** a forearm/upper-arm
ratio of 1.066 lies 5.9σ outside SMPL-X's population, and no plausible body reaches it.

#### Consequence

A metric body model cannot be fitted to anisotropically distorted image coordinates. This is a
correctness requirement, not a preference. Two things follow:

1. **Part B's 50% is not a round-trip result** and must not be reported as one. It measures
   coordinate distortion, and would have been published as a finding about SMPL-X.
2. **The Step 1 noise floor (0.73%) is also measured on distorted data** and must be re-measured
   once the input is fixed. Treat that number as provisional.

Steps 2, 3 and 4 are unaffected — swing IK reads directions and is scale-invariant, and the
synthetic verification never used this data.

---

### Step 5 — Round-trip harness and error decomposition

**5a — Synthetic ground truth** *(no weights needed; the rigorous test)*

Generate random valid rotations → forward kinematics → landmarks → fit → compare against the
rotations you started from. The true answer is known exactly, which is never true of real data.

**Pass condition:** recovered rotations and positions match to near machine precision. Failure
here is a bug in the math, isolated from every question about real-world data quality.

**5b — Real data** *(needs Step 0)*

Run the five cached clips through `P → θ → P'` and report three separate quantities:

| Metric | Question it answers |
| --- | --- |
| **Bone direction error** (angular, per bone) | Did the fit miss? |
| **Bone length error** (per bone) | What can SMPL-X structurally not represent? |
| **Joint position error** (MPJPE, % shoulder width) | The headline round-trip number. |

Reporting these separately is the point. A single RMSE conflates a fitting bug with a
representational limit, and they call for opposite responses.

**Compute every metric over the 12 fitted landmarks only.** `project()` copies the other 531
landmarks — face, hands, and the unconstrained pose points — straight through from the input
frame. An error averaged over all 543 would be measuring copied data against itself, and could
report an impressively small number while the fit is completely wrong. This is the most likely
way to manufacture an unfalsifiable success, and plausibly how the historical `0.5974 RMSE`
became uninterpretable: nobody recorded which landmarks it covered.

### Step 6 — Decision gate — ✅ **RUN (2026-10-04)**

The original gate assumed a single dominant cause. The oracle decomposition in Step 5 shows
there are at least three, so the gate is replaced by its result: **shape 35%, root 13.7%,
unexplained 44.5%.** See the retraction above.

---

## Revised plan (2026-10-04)

Steps 7–9 replace the original conditional Step 7. The change is driven by two measured facts:
root orientation is a real error term, and shape plus root together leave nearly half the error
unexplained.

### Step 7 — ✅ **COMPLETE (2026-10-04)**

**Wiring decision: keep the paths separate and mark the legacy one.** The new converter is the
V1 deliverable. `MediaPipeToSMPLX` and `SMPLXToMediaPipe` remain, because image-space callers
depend on them, but now emit a `DeprecationWarning` naming what they actually do and pointing to
`converter.py`. This keeps to the original Section 4 boundary, which put pipeline integration
outside V1.

The six legacy `xfail` tests are retired in favour of
`tests/test_converter.py`, which encodes the correct contract. They
described an optimisation-based fitter that was removed in `f9c6d7b` and deliberately not
rebuilt, so keeping them as permanent `xfail`s would have recorded a rejected design as pending
work. Two tests replace them, asserting that each legacy class warns and names the converter.

Suite: **72 passed, 2 xfailed.** The two are intentional: the `SMPLXWrapper.num_joints` bug
(Finding 2, being fixed separately) and the `strict=True` rotated-root case that is Step 8's
target.

Pipeline integration remains open and needs a camera model to project world-space fits back to
the classifier's image-space format — recorded as V2 scope.

---

### Step 7 — implementation record

Implemented test-first in [`converter.py`](../src/kinetic_augment/body_model/converter.py) with
8 tests in `tests/test_converter.py`, plus a rigid mode for
[`alignment.py`](../src/kinetic_augment/body_model/alignment.py). Full suite: **71 passed,
8 xfailed.**

#### The contract, as built

```python
fit_body(landmarks, validity, skeleton) -> BodyFit   # body_pose, global_orient, transl, betas
reconstruct_body(fit, skeleton)         -> (33, 3)   # NaN where V1 does not model
```

- **Reconstruction takes only the parameters.** A test asserts its signature is exactly
  `(fit, skeleton)`, so the displacement-encoding failure mode cannot return.
- **The output is pure SMPL-X.** The rigid frame transform folds exactly into SMPL-X's own
  `global_orient` and `transl`, compensating for `global_orient` pivoting about the root's rest
  position rather than the origin. No side-channel coordinate state.
- **Verified against the real model:** SMPL-X's forward pass on the returned parameters
  reproduces `reconstruct_body` to within 1e-5 m.
- **Rigid, not similarity.** Scale is fixed at 1.0 because both spaces are metric.
- Unmodelled landmarks return NaN rather than invented values; invalid landmarks never influence
  the fit; fewer than three valid correspondences raises rather than guessing.

#### Result — direct versus aligned

| | Value |
| --- | --- |
| Synthetic, root unrotated | **5.3e-15 m** — exact |
| Synthetic, root rotated 30° | **70 mm worst** — the Step 8 gap, now visible in the harness |
| **Real clips, direct** | **6.55%** of shoulder width — the number a consumer gets |
| Real clips, aligned | 5.94% — the figure previously reported |

**The previously reported 5.94% was the aligned figure.** A free scale was hiding 0.61
percentage points. Every earlier comparison to 5.94% should be read as optimistic by that much.

Per joint, direct median / p95, with coverage: shoulders 6.43 / 12.56 and 7.80 / 12.62 (100%),
elbows 6.04 / 9.58 and 4.98 / 10.73 (100%), wrists 7.85 / 17.33 (87%) and 7.01 / 16.63 (89%).

The rotated-root case is captured as a `strict=True` xfail in `test_converter.py`: it fails today
and will fail the suite *loudly* the moment Step 8 makes it pass, forcing the marker's removal.

#### Blocked — wiring into the public classes

The review is right that the verified maths is not yet reachable through `MediaPipeToSMPLX` and
`SMPLXToMediaPipe`. But this is not a drop-in swap, for three reasons:

1. **Every existing caller passes image-space data.** `scripts/render_landmarks.py` and
   `scripts/test_phase2.py` call `fit(frame)` on 543-landmark image-space frames — exactly what
   Finding 9 shows cannot support a metric fit. Rewiring would make them silently wrong.
2. **The framework's interchange format is image space.** Augmented landmarks feed a classifier
   as `(T, 543, 3)` image-space arrays. Fitting in world space produces world-space output, and
   no camera model exists to project it back.
3. **The six legacy `xfail` tests describe a rejected design.** They expect a
   `num_iterations` optimiser argument and random `(543, 3)` input; analytic IK was chosen
   deliberately and has no iterations.

This is an architecture decision about the framework's data format, not a V1 implementation
detail, and is raised for a decision rather than resolved unilaterally.

**Decided (2026-10-04): keep the paths separate** and deprecate the legacy classes (the Wiring
decision above). Since 2026-10-05 the V1 converter is exported from `kinetic_augment.body_model`.

---

### Step 7 (original text) — Define and enforce the converter's contract

Nothing further should be measured until the deliverable's interface is pinned down, because the
current headline is an aligned residual whose transform the converter does not return.

> **Contract.** World-coordinate body landmarks + per-landmark validity → SMPL-X parameters
> **and the coordinate state needed to reconstruct** → world-coordinate body landmarks.

- Decide explicitly what coordinate state is returned. Both spaces are metric, so a **rigid**
  transform should suffice; the fitted scale of 0.9726 is evidence of correspondence error, and
  a free scale parameter hides it. Prefer rigid, and report any residual scale as a finding.
- Report **direct** reconstruction error (the number a consumer sees) **and** aligned error
  (diagnostic) side by side, per joint, with median, p95 and observation coverage.
- **Wire the verified maths into the public classes.** `MediaPipeToSMPLX` and `SMPLXToMediaPipe`
  still use the displacement encoding of Section 3; the new modules are parallel to them, not
  connected. The six `xfail` tests in `tests/test_body_model.py` flip when this lands and are the
  completion signal.

### Step 8 — ✅ **COMPLETE (2026-10-05)**

Implemented test-first in [`converter.py`](../src/kinetic_augment/body_model/converter.py)
(`torso_rotation`, and a rewritten `fit_body`), with `SMPLXSkeleton.rest_nose` added in
[`skeleton.py`](../src/kinetic_augment/body_model/skeleton.py). Full suite: **81 passed,
2 xfailed** — the `num_joints` bug (Finding 2) and the strict collar case below.

#### What changed

Fitting now runs in an order where nothing fights:

1. **Root from the torso.** The shoulder axis sets two rotational degrees of freedom; the nose,
   when visible, sets the third.
2. **Arms by swing IK solved *relative to* that root**, so each bone points along its observed
   direction in world space.
3. **Translation by least squares** over the valid correspondences. No rotation is fitted after
   step 2, because that is what turned correct arm directions away from the observations.

The shoulders now anchor the frame, so `fit_body` requires both and raises without them. This
replaces the earlier "three correspondences" rule, which belonged to the whole-body rigid fit.

#### Synthetic cases added — the ones that could have caught the retracted claim

| Case | Result |
| --- | --- |
| Root rotated 30° (the reviewer's counterexample) | **exact**, 2.9e-16 m (was 70 mm) |
| Spine bend at all three spine joints | exact |
| Root rotation recovered from shoulders + nose | exact, < 1e-9 rad |
| Missing wrist with a rotated root | remaining joints exact |
| Different body shape (`betas` ≠ 0) | **bone directions exact**; positions differ, honestly |
| Collar motion (a shrug) | **not representable** — `strict=True` xfail, see below |
| Both shoulders required | raises without them |

The body-shape case failed under the old converter: its whole-body rigid fit bent bone directions
to compensate for length mismatch. That directly confirms the review's point that a final
rotation fights the solved arms.

#### Finding 11 — torso pitch is invisible to the arm landmarks (a gauge freedom)

**Verified now, by test and on real data.** Rotation about the shoulder axis moves neither
shoulder, and the arms are solved in world directions, so it cannot change any reconstructed arm
landmark. Only the shoulder-axis *direction* — two of the root's three degrees of freedom — is
observable from V1's correspondences.

- **Test:** `test_torso_pitch_does_not_affect_arm_reconstruction` fits the same frame with and
  without the nose. The roots disagree by more than 5°; the arm reconstructions agree to 1e-9 m.
- **Real data:** the earlier decomposition estimated the root using the SMPL-X *head joint* as the
  nose's counterpart, which points **29.9°** away from the true nose vertex about the shoulder
  axis. Despite that 30° pitch error, its "root only" result was **5.13%** — identical to the
  corrected converter's aligned figure. A 30° error in the gauge direction changed nothing.

Consequences. The nose makes `global_orient` anatomically plausible (an upright, correctly facing
torso) but **cannot improve round-trip error**. And it explains why fixing the root helped only
13.7%: signers mostly face the camera squarely, and for a frontal torso the shoulder axis of
MediaPipe's world frame and SMPL-X's rest frame already agree — the ~180° difference between them
is a rotation *about* that axis, which is gauge.

`SMPLXSkeleton.rest_nose` uses SMPL-X's own nose vertex (vertex 9120, which the model also emits
as output joint 55) rather than the head joint, so the pitch that is fitted is the anatomically
correct one.

#### ⚠️ Found in Step 10 (2026-10-05): the nose-less fallback turns the body upside down

**Verified now, on real data.** With the nose, the fitted body is upright on all 409 real frames:
the head joint is above the shoulder midpoint and the pelvis below it. This resolves the inverted
body the review assessment found before Step 8 (its §4.1). With the nose masked, the fallback — the
minimal rotation between shoulder axes — puts the head *below* the shoulders on **409/409**
frames: MediaPipe's world frame is y-down, so "minimal" means "inverted".

The arms are unaffected (pitch is gauge, Finding 11), and the nose is visible on every real frame,
so no measured number changes. It is **latent, not fixed**: a sign with the hand over the face
would flip `global_orient` by ~180° for those frames. The fix is a convention-aware fallback, for
example the hip centre as apex, which MediaPipe's world origin provides.

**Fixed (2026-10-05).** Without the nose, the apex is now the hip centre: observed when both hips
are valid, and otherwise MediaPipe's hip-centred origin.
- **On real data,** with the nose masked, and with nose and hips masked, the body stays upright on
  409/409 frames.
- **The root moves** a median 7.0° (max 18.4°) from the nose-based one. That change is pitch only,
  so the arms are unaffected (`measure_round_trip.py`).
- **Tests:** `test_hips_recover_the_root_when_the_nose_is_missing` and
  `test_without_nose_or_hips_a_mediapipe_body_is_not_turned_upside_down`.

#### Collar motion is a real V1 limitation

A shrug rotates one shoulder about its collar joint, changing the shoulder geometry in a way no
root rotation reproduces. V1 holds the collars at zero, so it cannot represent this, and shoulder
movement is meaningful in some signs. Recorded as a `strict=True` xfail so the marker is removed
the moment collars are fitted.

#### Result on real data

| | Step 7 | Step 8 |
| --- | --- | --- |
| **Direct round-trip error** | 6.55% | **6.02%** |
| Aligned (diagnostic) | 5.94% | 5.13% |
| Arm direction error | 2–3° | **0.00°** |

#### The `01988` question, resolved

The plan required understanding why root refinement made `01988` slightly worse before
accepting an estimator. A **paired, per-frame comparison with bootstrap confidence intervals**
separates two different effects:

| Clip | Direct, new − old | 95% CI | Frames improved |
| --- | --- | --- | --- |
| 01986 | −0.96 | [−1.19, −0.68] | 82% |
| 01987 | −0.35 | [−1.08, +0.16] | 81% |
| 01988 | **+0.35** | **[+0.09, +0.67]** | 44% |
| 01991 | +0.27 | [−2.66, +0.45] | 47% |
| 01992 | −0.28 | [−0.41, −0.13] | 68% |

1. **The originally flagged regression was noise.** It was measured under the *aligned* metric,
   where the paired difference for `01988` is +0.07 with CI [−0.23, +0.15].
2. **A different, real regression appears under the *direct* metric** — and the harness's median
   over all joint samples reports `01988` *improving* (5.57% → 5.40%). When two reasonable
   aggregations disagree on the sign, the effect is mixed across joints, and the per-joint split
   shows how: `01988`'s shoulders improve while its wrists worsen by 1.21 and 0.71 points.

The mechanism is a trade-off, not an estimator fault. The old converter fitted one extra rotation
over all six points, and that rotation bent every arm bone **1.9–3.2°** away from what MediaPipe
observed, which spread bone-length mismatch across the body. The new converter keeps directions
exact (0.00° on every clip), so the same length mismatch accumulates at the wrists instead.
`01988` is tied for the most frontal torso (3.8° yaw), so it gains least from the corrected root,
and the trade-off shows; `01986`, the most rotated (7.9°), gains most.

**Decision: accept the new estimator.** The old converter's lower error on `01988` was error
moved out of positions and into joint angles, where the round-trip metric cannot see it. For a
framework whose purpose is to perturb joint angles, a 2–3° distortion of every elbow and
shoulder is not acceptable. The wrist error that replaced it is honest length mismatch — the
quantity Step 10 targets.

---

### Step 8 (original text) — Fix root and shoulder-anchor handling

The largest *correctly understood* gap, and a prerequisite for interpreting any shape fit.

- Estimate `global_orient` from the shoulders-and-nose frame. Nose visibility is **1.00** in all
  five clips (**verified now**), so the frame is available on every usable frame.
- Extend the synthetic suite to cover what it currently cannot distinguish: **rotated roots**,
  torso motion, differing body shapes, and missing landmarks. The retracted claim above survived
  precisely because no synthetic case had a rotated root.
- Note that root-only refinement made clip `01988` slightly *worse* (5.06% → 5.11%). Understand
  that before accepting any estimator.

### Step 9 — ✅ **COMPLETE (2026-10-05)**

Reproduce with `scripts/diagnose_residual.py` (add
`--unsmoothed` for E6, which reruns MediaPipe); the floor itself is reported by
[`scripts/measure_round_trip.py`](../scripts/measure_round_trip.py). **No library code changed**:
the investigation found nothing in the converter to fix.

The "unexplained 44.5%" came from the pre-Step 8 converter under the aligned metric, so it no
longer described anything. The ladder was rerun on the current converter, direct metric.

#### ⚠️ Re-baselined (2026-10-05): the data were order-dependent

**Verified now.** `extract_world_landmarks.py` ran one MediaPipe Holistic instance across all
clips. Video mode is stateful — a tracked region of interest and landmark smoothing — so each
clip inherited the previous clip's state. Flagged by the review assessment (§4.3) and confirmed
by re-extracting every clip with its own instance:

| Clip | Arm landmarks, stored vs fresh | Median arm-length change |
| --- | --- | --- |
| 01986 (processed first) | identical | 0% |
| 01987 | median 32.8 mm, max 114 mm | +3.3 to +4.9% |
| 01988 | median 3.6 mm, max 79 mm | −1.2 to +0.7% |
| 01991 | median 0.7 mm, max 24 mm | ≤ 0.4% |
| 01992 | median 22.4 mm, max 103 mm | up to −4.5% |

Fixed test-first: one instance per clip (`tests/test_extract_world_landmarks.py`). Extraction is
now deterministic — forward and reverse processing orders give bit-identical files — and the
stored `*_world.npz` were regenerated. Step 10 fits each clip's shape to these lengths, so the fix
came first.

| | Before | After |
| --- | --- | --- |
| Direct, template body | 6.02% | **5.92%** |
| Rigid-skeleton floor | 2.63% | **2.36%** |
| Gap | 3.39 pts | **3.55 pts** |
| Floor per clip | 01986 2.33, 01987 4.09, 01988 3.19, 01991 2.71, 01992 1.44 | 2.33, **2.26**, 2.84, 2.78, 1.41 |

Much of `01987`'s apparent non-rigidity was inherited tracker state. Every Step 9 conclusion
survives a rerun of `diagnose_residual.py` on the clean data: bone-length CV 2.0–14.6% (E1); the
per-frame skeleton is exact, 2.5e-14% (E2); joint angles R² 0.55 against a null of 0.43, depth
0.42 against 0.16 (E4); rescaling is not depth-specific, x 5.4%, y 15.8%, z 16.0% (E5);
constant offsets explain 4.4% (E7); with smoothing off, slowness falls only from 4.7 to 3.3,
still partial (E6, `--unsmoothed`). The noise reference is unchanged at 0.18%. The numbers in
Steps 5–9 are left as measured.

#### Result — the residual is identified, and it lies outside the converter

| Skeleton forced to | Direct error | Explains |
| --- | --- | --- |
| Neutral SMPL-X (current converter) | 6.02% | — |
| Per-clip bone lengths | 3.59% | 40.3% |
| Per-clip shoulder width | 5.95% | 1.2% |
| **Per-clip lengths + width** | **2.63%** | **56.4%** |
| Per-frame lengths + width | 2.5e-14% | 100% |

The last row is decisive. The six arm landmarks are 18 numbers, fully determined by the two
shoulder positions and the four bone vectors, so a skeleton re-shaped to every frame reproduces
them to machine precision. **Nothing else contributes.** Everything below the best constant
shape — 2.63% — is frame-to-frame variation in MediaPipe's own output.

#### Finding 12 — MediaPipe's world landmarks are not a rigid body

**Verified now.** Within a single clip, bone lengths vary with a coefficient of variation of
**3–12%** (forearms up to 12.2%). Shoulder width is steadier at 0.6–4.6%, consistent with the
detector's body prior (Finding 10). Real bones do not change length, so no rigid body model —
SMPL-X or any other — can reproduce this.

#### What drives it: five explanations tested

| Exp. | Hypothesis | Result |
| --- | --- | --- |
| E4 | Joint-angle effect: length follows the elbow and shoulder angles | R² 0.54 against a **chance level of 0.36**. Weak. |
| E4 | Depth ambiguity: length follows the bone's angle to the camera | R² 0.48 against a chance level of 0.16 — stronger, but the sign flips between clips (12 positive, 8 negative) |
| E5 | Per-clip depth-scale error | Best per-clip rescaling: z 15.2%, y 13.8%, x 5.6% — **not depth-specific**. Strong only in `01986` (k_z = 0.59) |
| E6 | Per-frame noise stretched into drift by MediaPipe's smoothing | Smoothing off: slowness 4.4 → 3.3, not the ~1 independent noise gives. **Partial** |
| E7 | **The plan's leading hypothesis:** constant joint-centre offsets | Best constant offset per joint, fixed in its segment's frame, explains **5.0%** by RMS. **Refuted** |

The variation is a slowly drifting, image-dependent error in MediaPipe's 3-D predictions, with a
depth component and some contribution from temporal smoothing. No single geometric cause
dominates, and none is a property of SMPL-X or of the conversion.

Three methodological notes, because each changed a conclusion:

- **The null in E4 matters.** An earlier pass reported "pose R² 0.54" as evidence of pose
  dependence. Regressing each bone on *another clip's* angles showed slowly varying series
  correlate at R² 0.36 by chance, so most of that was spurious.
- **E5's objective has degenerate optima.** `01991`'s best y-factor sits at the search boundary
  (4.0); its per-clip factors should not be read as calibrations.
- **Motion jitter dwarfs the Step 1 floor.** With smoothing off, bone-length jitter during motion
  is ~1.6% of bone length (3–4 mm), against Step 1's 0.6 mm measured on still frames. A further
  reason the 0.18% figure was never a usable threshold.

#### Consequence 1 — the V1 gate is the rigid-skeleton floor

Section 2's correction asked for a threshold built from the same statistic as the reconstruction
score. The rigid-skeleton floor is exactly that, and it is the best any constant-shape body can
achieve with exactly fitted bone directions.

> **V1 gate: direct round-trip error should reach the rigid-skeleton floor.**
> Today: **6.02%** against a floor of **2.63%** — a gap of 3.39 points, all of it shape.

Floor per clip: `01986` 2.33%, `01987` 4.09%, `01988` 3.19%, `01991` 2.71%, `01992` 1.44%.

The floor uses per-clip *median* lengths, so a shape tuned to minimise error could sit marginally
below it. It also bypasses `betas`, so SMPL-X's shape space may not reach it. The 3.39-point gap is
therefore the *most* Step 10 can recover.

#### Consequence 2 — what "lossless" means for the original question

The conversion preserves everything a physical body can carry: bone directions exactly (0.00°),
and lengths and width up to their per-clip values once shape is fitted. What it discards is
MediaPipe's frame-to-frame violation of bone-length constancy — variation no real body exhibits.
In that sense the round trip denoises rather than loses information.

That has a consequence for augmentation which is **inference, not yet tested**. Samples generated
through SMPL-X will have perfectly rigid bones, while real MediaPipe data carries 3–12% length
drift. A classifier trained on both sees two different distributions of bone-length behaviour,
and is tested on the non-rigid one. If that matters, the residual channel rejected in Section 2
gains a legitimate role: not to fake an exact round trip, but to carry the detector's signature
onto augmented samples. Open question for V2.

#### Consequence 3 — no new parameters

- **No correspondence offsets.** E7 bounds what any constant-offset model could recover at ~5%.
- **No per-clip depth calibration.** E5 found nothing depth-specific, and its optima can be
  degenerate.
- **Shape is the only lever left,** with a measured ceiling.

---

### Step 9 (original text) — Investigate the unexplained 44.5%

Do not add parameters before the dominant unexplained term is identified. Adding `betas` now
risks absorbing a correspondence bias into a shape estimate and reporting it as anatomy.

Leading hypothesis: the **joint-centre offset** of Section 2 — MediaPipe's shoulder keypoint sits
near the body surface, SMPL-X's is an internal rotation centre. Testable by estimating a
per-joint offset and measuring what it explains, exactly as the Step 5 oracles did.

### Step 10 — ✅ **COMPLETE (2026-10-05)**

Implemented test-first: a new [`shape.py`](../src/kinetic_augment/body_model/shape.py)
(`fit_shape`, `round_trip_quadratic`, `arm_proportions`); the shape space and `shaped()` in
[`skeleton.py`](../src/kinetic_augment/body_model/skeleton.py); `betas` carried through
[`converter.py`](../src/kinetic_augment/body_model/converter.py). The headline comes from
[`scripts/measure_round_trip.py`](../scripts/measure_round_trip.py), and every table below from
`scripts/study_shape_fit.py` (sections S0–S6). Full suite:
**98 passed, 2 xfailed**.

The data were re-extracted first (Step 9, re-baseline box), so the gap to close is **3.55
points** (5.92% → 2.36%), not 3.39.

#### The contract, extended to shape

```
betas = fit_shape(clip_landmarks, clip_validity, skeleton)      # once per clip
fit   = fit_body(landmarks, validity, skeleton.shaped(betas))   # per frame; records betas
out   = reconstruct_body(fit, skeleton)                         # shape comes from fit.betas
```

Reconstruction takes the body shape from the parameters, so Step 7's claim — everything needed
lives in the parameters — now holds for shaped bodies. Before, `fit.betas` was always zero and
never read: with a shaped skeleton, SMPL-X's own forward pass missed `reconstruct_body` by
42–112 mm (review assessment §4.5). It now agrees to within 1e-5 m with all 300 coefficients in
use (`test_smplx_forward_pass_reproduces_the_reconstruction[fitted]`). A consumer must build
SMPL-X with `num_betas=len(fit.betas)`.

#### Finding 13 — the round trip sees a skeleton through five numbers, linearly

**Verified now.** Bone directions are reproduced exactly (Step 8), so a skeleton enters the round
trip only through **shoulder width and the four arm bone lengths**. Within a frame each landmark's
error is linear in those five: laid out from the left shoulder, a landmark is a sum of lengths
along directions the frame itself supplies, and the translation absorbs the mean. A clip's summed
squared round-trip error is therefore an **exact quadratic** in five numbers.

- It equals the converter's own error on all five real clips to **4e-14** (S0). The test pins it
  on a non-rigid clip with a missing wrist and an unfittable frame.
- So the shape fit minimises the converter's actual error, not a proxy for it.

#### How the fit is built, and why

| Choice | Alternative | Fitted / held out |
| --- | --- | --- |
| Minimise the round-trip error itself | Match median proportions (the floor's own definition) | **3.52 / 3.99%** vs 3.68 / 4.08% (S6) |
| SMPL-X's full shape space, **300 betas** | The conventional 10 | **3.52 / 3.99%** vs 3.90 / 4.26%; saturates by about 100 (S5) |
| Plausibility bound, \|betas\| ≤ √χ²(0.95, df) = **3.33** | No bound | Fitted bodies 1.60–1.70 m tall. Unbounded fits do better (2.45 / 3.43%) with non-human bodies (Finding 15) |
| Validity mask only | `visibility` as weights (the original plan) | Not adopted: visibility does not track positional noise (ρ ≈ 0, review assessment). Not re-measured here |

- **Why a chi-square bound on |betas|.** At the optimum the betas lie in the span the five
  proportions can see. |betas| is therefore the Mahalanobis distance of the fitted proportions from
  SMPL-X's population mean. Its df is the number of observable proportions: five, fewer if a bone
  is never seen, and never more than the number of betas.
  - At 0.95, the body's arm proportions may be no more unusual than 95% of SMPL-X's bodies.
  - The prior was checked, not assumed. Under betas ~ N(0, I), the population stands 1.73 m
    ± 9.6 cm (S1), a plausible adult spread, so the coefficients really are unit-variance.
- **Why 300 coefficients.** Truncating to ten makes the prior understate how bodies vary in exactly
  the directions MediaPipe pulls on. The weakest observable direction moves 0.045 mm per unit beta
  with ten coefficients, against 0.34 mm with 300 (S2).
- **Ties.** Among bodies with equal error, the most typical is returned.

#### Result

| Skeleton | Fitted | Held out |
| --- | --- | --- |
| Template body, `betas` = 0 (Steps 8–9) | 5.92% | 5.92% |
| Shape fit, 10 betas | 3.90% | 4.26% |
| **Shape fit, 300 betas — V1** | **3.52%** | **3.99%** |
| Best symmetric skeleton, bypassing `betas` | 3.29% | 3.62% |
| Rigid-skeleton floor, bypassing `betas` | 2.36% | 3.41% |

"Held out" fits shape on one contiguous half of each clip and scores the other.

- **Shape recovers 2.40 of the 3.55-point gap (68%)** on the clip it is fitted to.
- **Held out, it recovers 77% of the gap:** 5.92 → 3.99%, against a held-out floor of 3.41%. The
  gain survives held-out evaluation. The review assessment expected about 10%, but it measured
  10 betas, the Step 7 converter and the contaminated data, so that expectation is superseded.
- **In millimetres,** at a 0.33 m shoulder width: 19.5 → 11.6 mm, against a floor of 7.8 mm.
- **Scale is no longer hidden.** An evaluation-time similarity now hides only 0.03 points (3.52%
  → 3.48% aligned), against 0.57 for the template body. The fitted shape has absorbed the scale
  mismatch the free scale used to hide (Step 7).
- **Per clip,** template → fitted (floor): `01986` 4.97 → 3.48% (2.33), `01987` 6.05 → 2.67% (2.26),
  `01988` 5.42 → 4.40% (2.84), `01991` 7.51 → 3.71% (2.78), `01992` 6.85 → 3.27% (1.41).
- **Every clip** fits on the plausibility bound, |betas| = 3.33.

#### Finding 14 — MediaPipe's arm proportions are not a human body's

**Verified now** (S1), against SMPL-X's own population (300 betas, N(0, I)):

| | Forearm / upper arm | z | Upper arm, L − R | z |
| --- | --- | --- | --- | --- |
| SMPL-X population | 0.927 ± 0.024 | | −7.3 ± 1.8 mm | |
| 01986 | 1.068 | +5.9 | −5.5 mm | +1.0 |
| 01987 | 0.834 | −3.9 | −8.5 mm | −0.7 |
| 01988 | 1.084 | +6.6 | +23.6 mm | +17.6 |
| 01991 | 1.046 | +5.0 | −31.1 mm | −13.6 |
| 01992 | 1.078 | +6.3 | −1.3 mm | +3.4 |

Four clips have forearms longer than upper arms, 5–7σ outside the population. Two have left–right
differences no SMPL-X body has: the shape space is nearly symmetric, with left and right rows of
the proportion Jacobian at cosine 0.996–0.9995 (S2). This confirms the review assessment's §4.10
and extends it to the full shape space. It is why every clip sits on the bound.

It also accounts for what remains. These are ladder rows of pooled medians, so the parts are not
additive:

- **Left–right asymmetry.** The best *symmetric* skeleton reaches 3.29%; only an asymmetric one
  reaches 2.36%.
- **Plausibility.** The fitted body sits 0.23 points above the best symmetric skeleton, mostly the
  forearm/upper-arm ratio it is not allowed to reach.

The floor's remaining gap is MediaPipe skeleton geometry outside the human shape space. It is not
a converter fault, and no plausible SMPL-X body reaches it.

#### Finding 15 — the prior is not optional

**Verified now** (S4). Loosening the bound keeps lowering the error, to 2.45% with no bound,
because the data always prefer MediaPipe's proportions exactly. The bodies that get there are not
human: with 10 betas, |betas| reaches **337** and one body stands **3.81 m** tall. Held-out error
also falls, nearly monotonically, as the bound loosens (4.29% at 0.5, 3.99% at 0.95, 3.43%
unbounded). The data therefore cannot choose the level: 0.95 is a policy about what counts as a human body, fixed in
advance rather than tuned. This answers review §6, step 6, which asked for "an explicit
prior-strength policy".

#### Finding 16 — the in-sample floor overstates what a constant shape achieves

**Verified now** (S3). Fitted to one half of a clip and scored on the other, the rigid-skeleton
floor is **3.41%**, not 2.36%. MediaPipe's proportions drift within a clip (Finding 12), so part of
the in-sample floor is a skeleton tracking that drift. The gate keeps its definition, since it is
the right ceiling for the clip a shape is fitted to. Quote the held-out figures beside it.

#### Bound the claim

- **`betas` are a convention.** Five proportions cannot identify 300 coefficients. Of all bodies
  with the fitted proportions, the most typical is returned
  (`test_prefers_the_most_typical_body_with_those_proportions`). Bodies 12 cm apart in height can
  share all five (review assessment, A006).
- **The fitted body is the plausible human closest to MediaPipe's skeleton,** not the signer's
  anatomy. MediaPipe's proportions lie outside the human range (Finding 14), so they carry
  detector convention as well as anatomy.

#### Decision (2026-10-05): accept — the remaining 1.15 points

**Decided: option 1.** V1's result is **3.52%** of shoulder width (held out 3.99%), through a
plausible SMPL-X body. The options considered, since shape was the last lever inside SMPL-X
(Step 9, Consequence 3):

1. **Accept** *(chosen)*. Report 3.52% (held out 3.99%) as V1's round trip through a plausible
   SMPL-X body, and the remainder as MediaPipe skeleton geometry outside the human shape space. It
   answers the original question with a measured, bounded number.
2. **Relax plausibility.** This lowers the error with non-human bodies (Finding 15). It would harm
   anything that uses the mesh: contact checks, rendering, the collision constraints in Phase 3.
3. **Carry a per-clip skeleton correction.** Store five numbers: MediaPipe's proportions minus the
   fitted body's, applied after reconstruction.
   - The round trip depends on nothing else (Finding 13), so this reaches the floor exactly while
     the SMPL-X body stays plausible.
   - It is Section 9's residual channel in its smallest, reported form: per clip, not per landmark.

Option 3 probably matters more for augmentation than for V1. A sample generated through option 1
has SMPL-X's arm proportions, not the signer's MediaPipe proportions. A classifier would then see
two skeleton conventions, just as it would see two rigidity regimes (Step 9, Consequence 2). Both
are **inference, not yet tested**, and both can only be tested once the converter is wired into
augmentation.

#### Context: MediaPipe against itself

**Verified now** (`measure_detector_agreement.py`).
Each clip was re-extracted with one setting changed and compared with the stored extraction,
using the round trip's own statistic with no alignment:

| Change | Median | p95 |
| --- | --- | --- |
| Smoothing off | 2.41% | 5.67% |
| Static-image mode | 9.22% | 20.04% |
| Heavy model | 25.47% | 88.46% |

The V1 round trip (3.52%) loses about as much as turning smoothing off. These are disagreements,
not errors, but they are the same statistic as the round-trip score, which the 0.18% jitter
figure never was. The smoothing-off row is the better reference scale (review assessment §6,
step 9). The heavy-model row reproduces the assessment's 25.7%.

#### Contract fixes after acceptance (2026-10-05)

From the review assessment's remaining items, each verified on current code before it was fixed.
None changes a headline number.

| Defect | Before | After |
| --- | --- | --- |
| **Invented landmarks.** An unobserved elbow or wrist leaves its bone at the rest pose, and `reconstruct_body` returned that position as if fitted | 61/409 real frames (14.9%); the invented points sit a median 46% of shoulder width from MediaPipe's own low-visibility estimate | `BodyFit.placed` marks what the fit determined. Anything else comes back NaN (`test_landmarks_it_could_not_place_come_back_missing`) |
| **Orphaned landmarks.** A valid wrist below an invalid elbow entered the translation and shifted every correctly placed landmark | 60 mm on a synthetic case; no real frames | Only placed landmarks set the translation (`test_an_orphaned_landmark_does_not_disturb_the_rest`). The shape fit's quadratic already excluded them |
| **The nose-less root** | Upside down on 409/409 frames | Hip-centre apex (Step 8 note) |
| **Public API.** The package advertised the deprecated classes | `fit_body` and `reconstruct_body` not exported | The V1 converter is exported from `kinetic_augment.body_model` |
| **The noise script** printed "PROPOSED V1 ERROR BUDGET" | — | Labelled a jitter reference, not a gate |

The harnesses score only landmarks the fit placed. Re-run, the headline is unchanged: 3.52%
fitted, 3.99% held out.

---

### Step 10 (original text) — Shape, with a prior and a bounded claim *(only after Step 9)*

As built, two points changed:
- `visibility` weights were not adopted, for the reason in the table above.
- Scale was not fitted separately, because betas already reach a uniform scale. The review
  assessment measured about 0.9σ per 1%.

- **Measured ceiling (Step 9):** shape can recover at most **3.39 points** (6.02% → 2.63%), and
  only if `betas` can reach each clip's median arm lengths and shoulder width. Width alone is
  worth 1.2%; lengths carry most of it. Success means approaching the floor, not reaching zero.
- Four arm lengths plus shoulder width cannot determine 10 `betas` plus scale. The problem is
  **underdetermined** and needs regularisation toward the shape prior; an unregularised fit will
  find a shape that matches five numbers and means nothing.
- Keep shape **shared across a sequence** but **per clip** — Finding 10 establishes the clips are
  different people.
- Compare against analytic IK as baseline, and consider a small confidence-weighted optimisation
  using `visibility` as weights, initialised from the analytic solution.
- **Bound the claim.** A better reconstruction does not establish that a signer has the inferred
  proportions. Step 9 narrows the alternatives: constant correspondence offsets explain only ~5%,
  but detector bias remains live — MediaPipe's body prior (Finding 10) and its non-rigid output
  (Finding 12) both shape the "observed" lengths a shape fit would be matching.

---

## 8. Deliverables

**Revised 2026-10-05; all delivered.**

1. ✅ **The V1 converter**, exported from `kinetic_augment.body_model`: `SMPLXSkeleton`,
   `fit_shape`, `fit_body`, `reconstruct_body` and `BodyFit`. It replaces original deliverables
   1–2, the legacy classes, which the Step 7 decision deprecated.
2. ✅ **A round-trip test suite**, covering synthetic exactness, the parameter contract and the
   shape fit. Real-data harnesses: [`measure_round_trip.py`](../scripts/measure_round_trip.py),
   `diagnose_residual.py` and
   `study_shape_fit.py`.
3. ✅ **Results documents:** this scope, the full record, and [`V1_Findings.md`](V1_Findings.md),
   the summary. The Step 6 verdict they were to record was retracted and superseded by Steps 9–10.

---

## 9. Fallbacks

Ordered by what the measurement would be telling us.

| Trigger | Fallback |
| --- | --- |
| Length error survives Step 7 — `betas` cannot span the subject's proportions | Accept and report it. This is a genuine finding about SMPL-X's shape space, not a failure. **Triggered in Step 10:** no plausible `betas` span MediaPipe's proportions (Finding 14). Reported; the choice of response is open (Step 10). |
| Analytic IK accumulates error down long chains | **Approach A:** gradient-based fitting (SMPLify-X style), optimising all parameters jointly against a global objective. Adam loop recoverable from `f9c6d7b^` as a starting reference. |
| Direction error persists — joint centres offset systematically | Calibrate a per-joint correspondence offset from data, then re-fit. |
| Round-trip cannot reach the noise floor by any route | **Residual channel:** store `θ` plus a per-landmark residual, making reconstruction exact by construction, and report residual magnitude as the measure of what SMPL-X cannot represent. Declined for V1 (Section 2); legitimate as a last resort *provided the residual is reported, not hidden*. Its smallest form is a per-clip, five-number skeleton correction (Step 10, option 3). |
| ~~Weights never arrive~~ | Resolved 2026-10-01 — see Step 0. |

---

## 10. Open questions

- Which twist policy for V1 — zero, or minimal-twist propagated along the chain? Zero is assumed
  above; it does not affect joint positions, so it does not affect the round-trip gate. It does
  affect mesh plausibility, which V1 does not measure.
- Does the shoulder–hip torso frame suffice for `global_orient`, or does depth noise in
  MediaPipe's `z` make it unstable? Testable in Step 1.
- Should `betas` be shared across all clips from one signer rather than per clip? Relevant to V2;
  V1 has too few subjects to tell. Finding 16 shows MediaPipe's proportions drift even *within* a
  clip, so sharing would need held-out evidence across clips.
- ~~Accept the plausible-body round trip, relax plausibility, or carry a per-clip skeleton
  correction?~~ **Decided 2026-10-05: accept** (Step 10). The correction stays a candidate for
  augmentation, pending the next question.
- Do SMPL-X's rigid bones and human proportions on augmented samples hurt a classifier trained
  and tested on MediaPipe's non-rigid, non-human skeleton? Testable only after integration
  (Step 9, Consequence 2; Step 10, option 3).

---

## 11. Known downstream defect — temporal incoherence in intrinsic augmentation

**Verified now (source).** Outside V1 scope, since §4 defers pipeline integration. Recorded here
because it is the first thing that will break *after* V1 lands: a correct round-trip feeding a
temporally incoherent augmenter still produces unusable training data.

**The classifier is a sequence model.**
[`lstm_classifier.py:114`](../src/kinetic_augment/evaluation/models/lstm_classifier.py) reads
`(batch, T, 543, 3)` and takes **only the final hidden state** (line 128), emitting one label for
the whole clip. A training example is a clip, not a frame, and
[`dataset.py`](../src/kinetic_augment/pipeline/dataset.py) multiplies the dataset by
`num_augmented_versions` copies of each clip, all sharing its label.

An augmentation must therefore be **temporally coherent**: one perturbation sampled per clip and
applied to all `T` frames.

**Extrinsic augmentations do this correctly.** `GlobalRotation` samples a single rotation and
applies it to the flattened `(T*J, 3)` clip, so every relative motion inside the clip survives.

**Intrinsic augmentations do not.**
[`intrinsic.py:133-136`](../src/kinetic_augment/augmentations/intrinsic.py) draws fresh noise
*inside* the batch-row loop:

```python
for b in range(batch_size):
    noise = np.random.randn(3) * stddev     # fresh noise every row
    body_pose[b, pose_idx:pose_idx + 3] += noise
```

and [`mp_to_smplx.py:96`](../src/kinetic_augment/body_model/mp_to_smplx.py) sets
`B = landmarks.shape[0]`, turning a clip's `T` frames into `B` independent batch rows. Every
frame therefore receives an unrelated perturbation.

**Consequence.** Instead of *"the same sign, performed with the elbow held 8° differently"*, the
augmenter produces *"a sign in which the elbow twitches randomly every frame"* — high-frequency
noise in exactly the temporal dimension an LSTM exists to read. `BodyProportionVariation` has the
same defect, varying body shape per frame within a single clip.

**Root cause is structural, not a typo.** No parameter dictionary marks that a batch is one
temporal sequence, and `fit()` has no concept of time, so an intrinsic operation cannot
distinguish 32 unrelated poses from a 32-frame clip.

**Requirement for whoever integrates V1:** sample augmentation parameters **once per clip** and
apply them across all `T` frames. Per-frame is correct for the *geometry*; once-per-clip is
correct for the *augmentation*. These are different time constants and the current code conflates
them — the same distinction §6 already makes for shape.

---

## 12. Assumptions ledger

Every assumption this work has relied on or tested, with its status and where the evidence lives.
Update it whenever a step proves or retracts something.

### Proven

| # | Assumption | Evidence |
| --- | --- | --- |
| P1 | The SMPL-X weights were never in git; they are installed locally, and the license forbids redistributing them | Step 0 |
| P2 | `smplx.create` takes `model_path="models"`, not `"models/smplx"` | Finding 1 |
| P3 | SMPL-X's forward pass returns 127 joints; the kinematic chain is the first 55 | Finding 2 |
| P4 | The repo's SMPL-X ↔ MediaPipe joint mapping is correct | Finding 3 |
| P5 | Our forward kinematics is SMPL-X's, to ~2e-7 m, provided `flat_hand_mean=True` | Steps 2–3, Finding 6 |
| P6 | Legs are not observed in this data: knee and ankle visibility ≤ 0.03 | Finding 5 |
| P7 | Zeros cannot mark missing data in hip-centred coordinates; NaN does | Finding 8 |
| P8 | `pose_world_landmarks` are metric; image-space `pose_landmarks` are anisotropic | Finding 9 |
| P9 | The five clips show at least four different people, so shape is fitted per clip | Finding 10 |
| P10 | Swing IK solved relative to the torso root reproduces every bone direction exactly (0.00°) | Steps 4, 8 |
| P11 | Torso pitch is a gauge freedom for the arm landmarks | Finding 11 |
| P12 | MediaPipe's world skeleton is not rigid: bone lengths vary 2–15% within a clip | Finding 12, Step 9 box |
| P13 | Below the rigid-skeleton floor lies only MediaPipe non-rigidity (per-frame skeleton: 2.5e-14%) | Step 9 |
| P14 | The round trip sees a skeleton only through five proportions, as an exact quadratic (4e-14) | Finding 13 |
| P15 | SMPL-X's joints are linear in `betas`, and its shape prior is N(0, I): population 1.73 m ± 9.6 cm | Step 10, S1 |
| P16 | A plausible per-clip shape recovers 68% of the gap to the floor, and 77% held out | Step 10 |
| P17 | MediaPipe's arm proportions lie outside SMPL-X's human shape space: ratio 4–7σ, asymmetry up to 18σ | Finding 14 |
| P18 | With the nose, the fitted body is upright on all 409 real frames | Step 8, Step 10 note |
| P19 | A shared MediaPipe instance made stored landmarks depend on clip order, by up to 114 mm | Step 9 box |
| P20 | MediaPipe disagrees with itself by more than the conversion loses: smoothing off 2.41%, static mode 9.22%, heavy model 25.47% | Step 10, context |
| P21 | The body round trip holds on 20 randomly drawn unseen clips (seed 20261007, from the same 334-clip WLASL pool): 3.13% direct, 3.37% held out, against the pilot's 3.52% and 3.99%; clip-level 95% CI on the median difference [-1.21, +1.16] pp; upright on 1234/1234 frames, with the nose and masked | `scripts/evaluate_round_trip.py`, `data/round_trip_eval/round_trip_report.json` (2026-10-07) |
| P22 | The MANO hand round trip holds on the same 20 clips (26 tracks): in-sample 5.10 mm mean, 2.53 tips, 14.59 p95; held out (time halves) 5.62 and 16.50. The pilot's hands (7.38 / 3.28 / 20.89) were harder than typical, not easier | Same run |
| P23 | The MediaPipe→FLAME embedding transfers to SMPL-X: eye corners 1.25-1.48 mm and nose tip 3.89 mm from smplx's own landmarks (F1; five points compared; the rest rely on the triangle check) | V1.3 Task 1, [V1.3 findings](V1_3_Findings.md) §6 |
| P24 | FaceLandmarker's facial-transformation-matrix convention and the OpenGL-to-image axis flip match the fit's: head-rotation agreement median 0.58° near / 0.57° far; 3.2-10.6° without the flip and 7.8-12.6° inverted on three clips. Not the face-to-pose-world axis match, and agreement is not independence (both rotations come from the same landmarks) (F2, matrix part) | V1.3 findings §3.2, §6 |

### Disproven, retracted or superseded

| # | Claim | What replaced it | Evidence |
| --- | --- | --- | --- |
| D1 | The existing converter round-trips at 3.7e-9 | It never evaluated SMPL-X, so it never produced valid parameters | Section 3 |
| D2 | MediaPipe's `z` is 4.8× noisier than x/y | An artifact of image-space landmarks | Finding 4 |
| D3 | "Lossless" means below the 0.18% jitter floor | A different statistic; the rigid-skeleton floor is the gate | Section 2, Step 9 |
| D4 | `global_orient` cannot affect the gate | Retracted after the review's counterexample (47 mm at 45°) | Steps 4, 8 |
| D5 | Length error dominates, so fit shape next (first Step 6 verdict) | Retracted: root first, then the residual investigation | Step 6 |
| D6 | "This signer has shorter upper arms" | The figures pooled at least four people | Finding 10 |
| D7 | An aspect-ratio correction rescues image-space landmarks | It made things worse; world landmarks replaced them | Finding 9 |
| D8 | The aligned error (5.94%) is the converter's error | The direct error is the real number | Step 7 |
| D9 | Synthetic Part A recovers rotations | It recovers positions only; twist is unobservable | Step 5 |
| D10 | `01988` regressed under the new root | Noise under the aligned metric; a real, accepted trade-off under the direct one | Step 8 |
| D11 | Joint-centre offsets explain the residual (Step 9's leading hypothesis) | They explain 4–5% | Step 9, E7 |
| D12 | Joint angles drive length drift (R² 0.54) | Mostly chance: the null R² is 0.36–0.43 | Step 9, E4 |
| D13 | A per-clip depth-scale error | Not depth-specific | Step 9, E5 |
| D14 | Smoothing-stretched noise explains the drift | Only partly | Step 9, E6 |
| D15 | Stored world landmarks depend on the clip alone | They depended on clip order until re-extraction | Step 9 box |
| D16 | `betas` can reach each clip's median proportions | Only with non-human bodies: \|betas\| 337, 3.81 m tall | Findings 14–15 |
| D17 | Ten `betas` suffice | 300 recover more, held out too | Step 10, S5 |
| D18 | `visibility` can weight the fit | It does not track positional noise (ρ ≈ 0, review assessment; not re-measured) | Step 10 |
| D19 | The floor is what a constant shape achieves | Only in-sample: 2.36% fitted, 3.41% held out | Finding 16 |
| D20 | The nose-less fallback root is acceptable | It inverted the body on 409/409 frames. Fixed with a hip-centre apex | Step 8 note |
| D21 | MediaPipe's proportions differ from SMPL-X's by ordinary individual variation that `betas` absorb | Four clips lie 5–7σ outside the population | Finding 9 note, Finding 14 |
| D22 | The 0.97 evaluation-time scale is evidence of correspondence error | With fitted shape it is 0.9997: the deficit was shape | Step 6 note |
| D23 | Every modelled landmark that `reconstruct_body` returns was fitted | 14.9% of real frames returned invented elbows or wrists. Now NaN | Step 10, contract fixes |
| D24 | Adaptive IK would meaningfully lower the error | With fitted shape, 3.52% → 3.43% for 1.2° of direction error; worse after 3 passes | Findings doc, study S7 |

### Open

| # | Question | Status |
| --- | --- | --- |
| O1 | Accept the plausible-body round trip, relax plausibility, or carry a per-clip skeleton correction? | **Decided 2026-10-05: accept**, 3.52% (held out 3.99%) (Step 10) |
| O2 | Do SMPL-X's rigid bones and human proportions on augmented samples hurt a classifier? | Untested; needs integration |
| O3 | Should shape be fitted per signer rather than per clip? | V2; needs cross-clip held-out evidence (Finding 16) |
| O4 | How should shrugs (collar motion) be represented? | Not representable in V1 (strict xfail) |
| O5 | The nose-less fallback root, and orphan correspondences (review assessment §4.6) | **Both fixed 2026-10-05** (Step 10, contract fixes) |
| O6 | Are the elbow and shoulder joint-limit axes wrong (review assessment §4.2)? | Not re-verified; latent until augmentation is wired |
| O7 | Is `z` noise 1.2× or about 1.9× that of x/y per axis (review assessment §4.7)? | Not re-verified; reference figures only |
| O8 | The per-clip body tail is wider than the pilot showed: direct 1.44–5.52% (pilot 2.67–4.40%), gap to floor up to 3.85 pp (`08917`, `08918`, `57949`), where `betas` cannot reach the clip's proportions (P17). Accept, or flag such clips before augmenting? | Open (2026-10-07) |
| O9 | Does the round trip hold on another dataset, camera set-up or signer pool? The P21–P22 sample shares WLASL's sources and some signers | Untested |
| O10 | F1: the MediaPipe→FLAME embedding transfers to SMPL-X correctly (V1.3 design §8) | **Proven**, now P23 |
| O11 | F2: FaceLandmarker pixel axes (x·W, y·H, z·W) match the pose world axes | Matrix convention and axis flip **proven**, now P24. The face-to-pose-world axis match is untested (metric 2 never involves the pose-world frame); the depth gap in O16 bears on it (V1.3 findings §3.2, §6) |
| O12 | F3: the 17-point rigid set gives head rotation under expression and partial occlusion | Partly supported: residual and agreement do not degrade on near frames (2.27% vs 2.14%; p95 2.66° vs 1.40°), but only 30 of 143 near frames lost any rigid point (18 lost two or more, 1 unreliable) and expression is unmeasured. Independent accuracy under real occlusion untested (V1.3 findings §6) |
| O13 | F4: the neck swing to the observed ear midpoint fixes head placement without touching the arms | Partly supported: ear residual far 7.61% → 2.82%, near 7.15% → 4.56% of shoulder width (p95 ≈ 9-10%). Arms-unchanged only by a synthetic test; not re-measured on real data (V1.3 findings §3.4) |
| O14 | F5: lip gap is a usable proxy for jaw pitch with expression at zero | Partly supported: Spearman median 0.92 over 8 of 20 clips; 69364 (0.496) and 63665 (0.687) below 0.7; 12 clips not scorable (V1.3 findings §3.3) |
| O15 | F6: landmark rules on the rest template give sensible ASL-LEX regions | Tests pass and `docs/media/face_regions.png` rendered; **awaits the user's review of the render**. Eye 750 vertices after dropping eyeballs; Ear radius 3.5 cm is generous |
| O16 | Why do fitted hands sit about 0.18 m from the head along the camera axis on near-face frames, with contact never observed? | Already present in MediaPipe's own pose-world observations: fingertip-to-nose depth is 0.217 m (median, 152 near hand-frames), before any fitting. Real distance versus pose-world depth error unresolved. Penetration is unmeasured (V1.3 findings §5) |
