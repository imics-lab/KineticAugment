# V1.2 plan: MediaPipe hands ↔ MANO and SMPL-X

**Status:** P0–P3 pilot implemented; see [V1.2 findings](V1_2_Findings.md).

**Date:** 2026-10-05.

**Priority:** Low landmark roundtrip error; measurable targets proposed below, as requested.

**Naming:** “MONO” is interpreted as **MANO**, the hand model already present in this repository.

**Companion:** [Assumption ledger, dependencies, and failure traces](V1_2_Assumptions.md).

## 1. Outcome and scope

Given either hand's 21 MediaPipe landmarks, estimate genuine MANO or SMPL-X pose and shape
parameters, evaluate the actual model, and reconstruct the 21 corresponding landmarks with a
small, explicitly measured error. Preserve which hand, which frame, and which observations were
valid. Support both hands independently and together in a sequence.

The principal products are two independently measured routes:

```text
MediaPipe hand observations → MANO parameters → MANO forward → 21 model-derived landmarks
MediaPipe hand observations → SMPL-X parameters → SMPL-X forward → 21 model-derived landmarks
```

Also test model-generated landmarks → fitted parameters → model-generated landmarks, and measure
MANO ↔ SMPL-X transfer separately. Do not require the MediaPipe → SMPL-X route to pass through
MANO: that would compound transfer error and obscure which model limits the fit.

V1.2 covers world-hand observations, hand-local reconstruction, real parameter serialization,
missing observations, sequence consistency, and an explicit SMPL-X wrist-attachment contract.
Image reprojection is a separately scored extension. Full replacement of the 543-point image
pipeline, facial modeling, classifier gains, new learned RGB estimators, and real-time performance
are outside the initial release gate. The project version in `pyproject.toml` need not become
`1.2`; V1.2 is this research milestone's name.

**A small roundtrip error means agreement with MediaPipe's observations.** It does not establish
true anatomy, calibrated physical accuracy, unique rotations, or preservation of sign meaning.
Arbitrary detector landmarks need not lie on either model's representable hand surface/skeleton.
No bit-exact model-only roundtrip is promised.

## 2. Starting point: what carries over from V1

This plan follows the final, accepted V1 scope and findings, rather than treating the older review
snapshot as the current implementation. Planning used the working tree at HEAD
`920d4bf71f0c562d21b58ffefb736b8b07d61bcb`, which already contained substantial uncommitted V1 work.
HEAD alone therefore does not identify the inspected source. P0 must freeze a file/hash manifest.

| Evidence | Implication for V1.2 |
| --- | --- |
| [V1 scope, Steps 9–10](V1_Problem_Scope.md), [accepted findings](V1_Findings.md): 3.52% shoulder-width body error, 3.99% held out | These are arm results, not a hand baseline or hand tolerance. Rebuild hand-specific measurements. |
| [V1 scope, sections 2–3](V1_Problem_Scope.md): the original near-zero result bypassed SMPL-X; jitter was later rejected as an accuracy threshold | Decode only genuine parameters through the real forward model. Never use temporal jitter as the reconstruction pass/fail criterion. |
| [World extractor](../scripts/extract_world_landmarks.py): pose-world extraction only | Collect synchronized hand-world observations; do not treat the existing world caches as hand data. |
| [Body converter](../src/kinetic_augment/body_model/converter.py): genuine `BodyFit` and six arm outputs; hands remain unmodeled | Add a separate hand fit/composite contract while preserving body-only behavior. |
| [Legacy hand input](../src/kinetic_augment/legacy/mp_to_smplx.py), [legacy output](../src/kinetic_augment/legacy/smplx_to_mp.py) | Their displacement encoding and dependence on original landmarks cannot be the V1.2 model roundtrip. |
| [Skeleton](../src/kinetic_augment/body_model/skeleton.py) sets `flat_hand_mean=True`; [wrapper](../src/kinetic_augment/body_model/smplx_wrapper.py) does not pass it | Reconcile pose means, PCA mode, and beta dimensions before any accuracy comparison. |
| [Frozen E11 evidence](../scripts/review_checks/direction-and-blocker/e11_out.txt) | The pre-Step-8 image test doubled the wrist/hand gap after body replacement. Retain the failure case; do not report its numbers as current V1.2 performance. |
| [V1 findings](V1_Findings.md): shape is underdetermined, priors matter, extraction order once changed results | Fit shape across a sequence, reset tracking state per clip, and split calibration/evaluation data. |

Current source evidence is distinct from a newly executed test: this planning pass inspected code,
data, and dependencies but did not rerun the V1 suite or execute a hand fitter.

The local read-only data probe found five clips and 410 cached image frames. Hand detection
coverage in the legacy arrays is below; a present hand means the group contains nonzero values,
not independently verified landmark accuracy.

| Clip | Frames | Left hand present | Right hand present |
| --- | --- | --- | --- |
| 01986 | 83 | 52 | 52 |
| 01987 | 75 | 51 | 46 |
| 01988 | 144 | 102 | 94 |
| 01991 | 49 | 0 | 49 |
| 01992 | 59 | 59 | 58 |

The five world NPZ files have only `landmarks[T,33,3]` and `visibility[T,33]`; 409 of 410
body frames are finite. The 543-point caches have image hands at `[501:522]` and `[522:543]`,
with no hand-confidence arrays. Their provenance must not be assumed identical to a future
paired extraction.

The available `.venv` reports Python 3.11.15, NumPy 1.26.4, Torch 2.14.1 (CUDA unavailable),
`smplx` 0.1.28, MediaPipe 0.10.21, and OpenCV 4.11.0.86; `chumpy` is absent. MANO left/right
PKL and converted NPZ files and all three SMPL-X gender NPZ files exist locally. SMPL-X metadata
disagrees: the asset index says v1.1 while `version.txt`/model setup guidance says v1.0. P0 must
resolve provenance by actual asset metadata/checksums; a filename or stale label is insufficient.

## 3. Observation and parameter contracts

### 3.1 Capture a versioned hand observation record

Prefer a separate, synchronized Hands/Hand Landmarker extraction beside the established V1 body
extractor. Keep the existing MediaPipe pin until a small compatibility probe selects legacy Hands
versus Tasks. A Tasks migration is an explicit change, not a prerequisite assumed by the plan.

Save the following together, with a schema version and source clip hash:

- `hand_world[T,2,21,3]`: raw detector world-hand coordinates in metres, one origin per hand.
- `hand_image[T,2,21,3]`: normalized image x/y and the detector's relative z; label its semantics.
- `hand_valid[T,2,21]`, `hand_present[T,2]`, and identity/association status. Missing values are NaN
  plus masks; valid zero coordinates remain valid.
- Handedness label and classification score, mirroring state, and a stable left/right assignment.
  A handedness score is not a positional confidence score. Preserve any actually available
  detector scores with their precise meaning; do not fabricate per-landmark confidence.
- Paired pose-world/image observations, pose visibility, timestamps/frame indices, width, height,
  frame rate, detector version/settings, model asset hash, and extraction-state policy.

Google documents hand-world coordinates as hand-centered and in metres; image x/y use separate
width/height normalizations and image z has wrist-relative, approximately width-based scale.
These are different data types. [Official Hand Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

Use one detector state per clip; prove clip-order independence. Associate hand detections using
handedness, paired image wrist position, and temporal continuity, with an explicit ambiguous
state when hands cross. Do not use detection-list order as identity. Verify mirroring on labeled
left/right footage: legacy Hands documents a selfie-image assumption, which must not be
silently generalized to every API. [Legacy Hands contract](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)

### 3.2 Frames and scale

Define a fixed basis conversion `B` and use column-vector notation throughout. For each hand:

```text
P_local[j] = B (P_world[j] - P_world[0])
```

This wrist-centering is the declared local data contract. It removes location, not rotation or
scale. Score predicted and observed wrist-relative vectors in these same axes. Never independently
rotate or resize the output at evaluation time to improve the primary score.

Keep the raw hand-origin/wrist translation as provenance if an inverse change of coordinates is
needed; it is not an estimate of the hand's absolute position in a shared room/body frame. Treat
MediaPipe “metres” as detector-reported metric estimates, not independently calibrated truth.

Verify whether the chosen hand-world axes agree with pose-world axes. Until verified, an explicit
`R_hand_to_body` is required; matching timestamps and metric units do not prove a shared frame.
Any fixed reflection used in a basis change is recorded separately from rotations, which must
remain proper `SO(3)` rotations. Handedness is not repaired with an arbitrary negative scale.

### 3.3 Fit result and decoder

Use proposed records `HandObservation`, `HandFit`, and `BodyHandsFit` rather than packing unrelated
values into the existing `BodyFit` fields. Exact naming can follow repository conventions.

`HandFit` records backend, side, 45 full local axis-angle values, MANO root orientation/translation
when applicable, shape coefficients, frame and unit convention, model/adapter versions, fitting
status, losses, and validity/uncertainty flags. Shape is sequence-shared; pose is per frame.
SMPL-X results retain full body/global/translation/beta state and both hand poses, including wrist
rotations in `body_pose`. Store `use_pca=False` and `flat_hand_mean=True` explicitly.
For standalone MANO, verify its shape-dependent wrist pivot: to place the wrist at `p_w`, compute
`transl = p_w - wrist(model_forward_without_translation)`. Do not assume the model's untransformed
wrist is the coordinate origin for every shape.

The decoder accepts only parameters, model assets, the fixed landmark adapter, and declared frame
metadata. It must work in a new process after the original landmark arrays are unavailable.
Per-frame solver diagnostics may be retained but must not feed the decoder. No cached landmark
array, per-point residual, or original-frame reconstruction shortcut counts toward model-only
accuracy. If residual storage is explored later, label and score it as a separate representation.

## 4. The actual 21-point mapping

Use one named, versioned observation operator `H_backend(parameters) → (21,3)` per side. It combines
the wrist, 15 articulated model joints, and five mesh-derived fingertips. These are candidate
anatomical correspondences, not proof that MediaPipe predicts exactly the same internal centers.

| MediaPipe indices / meaning | MANO skeletal indices, candidate standard order | SMPL-X skeletal indices, left / right |
| --- | --- | --- |
| 0 wrist | 0 | 20 / 21 |
| 1, 2, 3 thumb CMC, MCP, IP | 13, 14, 15 | 37, 38, 39 / 52, 53, 54 |
| 5, 6, 7 index MCP, PIP, DIP | 1, 2, 3 | 25, 26, 27 / 40, 41, 42 |
| 9, 10, 11 middle MCP, PIP, DIP | 4, 5, 6 | 28, 29, 30 / 43, 44, 45 |
| 13, 14, 15 ring MCP, PIP, DIP | 10, 11, 12 | 34, 35, 36 / 49, 50, 51 |
| 17, 18, 19 little MCP, PIP, DIP | 7, 8, 9 | 31, 32, 33 / 46, 47, 48 |
| 4, 8, 12, 16, 20 fingertips | mesh vertices below | mesh vertices below |

The SMPL-X joint names agree with the repository's [mapping](../src/kinetic_augment/body_model/joint_mapping.py)
and the [upstream joint names](https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py).
P1 must validate the MANO order against the installed assets and kinematic tree, both sides.

| Tip order | MANO candidate vertex | SMPL-X left vertex | SMPL-X right vertex |
| --- | --- | --- | --- |
| Thumb | 744 | 5361 | 8079 |
| Index | 320 | 4933 | 7669 |
| Middle | 443 | 5058 | 7794 |
| Ring | 554 | 5169 | 7905 |
| Little | 671 | 5286 | 8022 |

These are the upstream [vertex selectors](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py),
to be pinned and checked visually/numerically on the actual assets. Do not mix indices from
different MANO libraries. The upstream MANO forward implementation can return 16 joints because
tip appending is commented out; explicitly sample vertices instead of assuming `output.joints`
contains 21 points. [Model implementation](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)

Start with this fixed adapter. If systematic anatomical offsets dominate, compare a fixed
surface/barycentric adapter calibrated on training data only. Version it and score held-out data;
never learn an arbitrary per-frame offset that simply copies each target landmark.

## 5. Fitting approach

### 5.1 Establish MANO as the isolated hand baseline

1. Load the side-specific model with full 45D pose, flat hand mean, explicit units and beta count.
   Verify existing converted NPZ assets and left-hand shape correction provenance before changing
   anything. Model presence alone is not a successful runtime/forward check.
2. With neutral shape, estimate palm orientation from wrist and several MCPs. Use a proper rigid
   fit with degeneracy checks; a near-collapsed palm must return a flagged fallback.
3. Initialize fingers using chain directions. Account for the thumb's different base geometry.
   Analytic swing is initialization, not the final criterion.
4. Refine wrist orientation and all finger rotations through differentiable real-model forward
   evaluation, using masked robust position loss. Use previous-frame initialization within a
   track and several deterministic starts for difficult first/reacquired frames.
5. Fit a sequence-shared MANO shape on diverse calibration frames, with a declared prior; freeze
   it for the held-out evaluation frames. Compare neutral, fixed calibrated, and jointly fitted
   shapes. Do not optimize free scale and size-bearing betas without fixing their gauge.
6. Add temporal regularization only after framewise errors are understood. Penalize rotation
   changes on `SO(3)`, respect timestamps, and break continuity across long missing intervals.

The default shape policy is one MANO beta vector per anatomical side per clip, shared across its
frames and reacquired track segments. Record calibration counts and capacity separately for each
side. Do not assume the same coefficients describe both sides; compare a bilateral consistency
prior as an ablation after verifying side-specific shape conventions. Cross-clip/person sharing
requires known signer identity and its own held-out evidence.

Use a loss of the form:

```text
L = masked robust landmark error
  + λ_tip × additional fingertip emphasis
  + λ_pose × pose/twist regularization
  + λ_shape × sequence shape prior
  + λ_time × temporal regularization
  + λ_limit × validated anatomical penalties
```

Normalize losses consistently by valid weight and hand scale. Start with uniform valid-point
weights and isotropic residuals; introduce statistical weights only if calibration supports them.
Handedness confidence must never stand in for coordinate uncertainty. All-zero/NaN observations,
insufficient palm geometry, and failed convergence get explicit status, not a plausible-looking
default reported as a successful reconstruction.

For partially observed fingers, optimization can retain a latent chain and fit available
endpoints, with uncertainty/regularization. Analytic initialization must not treat a missing
parent as an observed origin. Keep imputed frames and observed frames separately scored.

Start with Adam plus a deterministic local refinement such as L-BFGS, subject to P2 evidence;
record tolerances, iteration counts, seed, and termination reason. Settle accuracy before speed.
Tune weights on calibration/validation data and freeze them before reporting test results.
Define anatomical constraints in verified bone-local frames, not by assuming each raw axis-angle
component is an anatomical hinge. Calibrate the shape prior to the installed MANO version;
V1's body-specific shape prior and its bound do not automatically apply to hands. Inspect and
report finger collapse, extreme twists, and mesh intersections before accepting a low-error fit;
add collision penalties only if these failures require them.

### 5.2 Direct SMPL-X hand fitting and attachment

Run the same landmark objective through SMPL-X. Initially freeze the accepted V1 body shape and
non-wrist body pose, and optimize hand poses plus wrist orientation. Unlike standalone MANO,
SMPL-X has no independent MANO beta vector or independent translation for each attached hand.
Its global shape changes body and hands together.

With a declared desired world wrist rotation `R_w` and the model parent's world rotation `R_p`,
the wrist's local rotation is `R_pᵀ R_w` under the model's verified frame convention. Convert
through any necessary fixed rest-basis mapping first. Use FK to obtain `R_p`; do not assign a
MANO global rotation to the whole body's `global_orient`. Setting the wrist rotation changes
descendant fingers while the wrist joint position remains determined by the arm.

For translation, distinguish two outputs:

- **Local hand reconstruction:** wrist-relative error isolates hand modeling.
- **Body-attached reconstruction:** the wrist is the actual SMPL-X body wrist. Report the
  body-wrist error separately; do not move a detached hand to the observed wrist and call it
  native SMPL-X output.

If the true hand-world/body-world orientation transform cannot be established, release local
hand results with that limitation and keep attachment validation open. An arbitrarily chosen
attachment is not measured global accuracy.

If fixed-body shape limits hand fidelity, compare a constrained sequence-level SMPL-X shape
refit that also includes V1 arm residuals. Preserve the full body beta dimension; do not silently
truncate the accepted 300-coefficient V1 fit to a wrapper's 10-coefficient default. Report the
body/hand tradeoff. Independent hand scaling is a different, nonstandard model and excluded from
the strict SMPL-X claim.

### 5.3 MANO ↔ SMPL-X transfer

Use matched anatomical joints/vertices and wrist frames to initialize transfer, then refit the
target backend. Match pose order and hand-mean conventions explicitly. Never copy standalone
MANO betas into SMPL-X betas. Even similarly named pose parameters need a forward-output check.
Upstream supplies MANO/SMPL-X vertex correspondences, but correspondence is not a guarantee of
equal geometry. [Official correspondence guidance](https://github.com/vchoutas/smplx#mano-and-flame-correspondences)

Measure MANO → SMPL-X → MANO and the reverse on synthetic poses, separately from the two
MediaPipe routes. Quantify incremental transfer error. If transferring is worse than fitting
the target model directly, retain direct fitting as the default; the core deliverable remains
supported by both backends. The general upstream transfer tooling also warns that shape spaces
are not interchangeable. [Transfer documentation](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md)

## 6. Metrics, provisional targets, and what a failure means

For valid non-wrist points `j=1…20`, compute:

```text
e[t,j] = 1000 × ||(P_pred[t,j] - P_pred[t,0]) - (P_obs[t,j] - P_obs[t,0])||₂  [mm]
```

This is a direct score in the declared wrist-relative frame, with no rotation/scale registration.
Report all-21 error too, but never let the necessarily zero wrist error dilute the primary result.
For original-frame or body-attached output, additionally score absolute landmark and wrist errors.

| Measure | Proposed target / interpretation |
| --- | --- |
| Adapter/forward parity | Maximum discrepancy ≤0.1 mm against direct model output with identical parameters, both sides. |
| Noiseless, same-backend synthetic inversion | Mean non-wrist error ≤0.5 mm and p95 ≤1 mm on predefined well-conditioned fixtures; stress/missing cases reported separately. Pose parameters need not be identical. |
| Real held-out local roundtrip, each backend and side | Mean non-wrist error ≤5 mm; mean of five fingertips ≤8 mm; p95 of valid point errors ≤15 mm. Provisional engineering targets, not measured feasibility or anatomical accuracy. |
| Hand-scale reporting | Also divide errors by one calibration-derived clip/hand median wrist-to-middle-MCP length. Record the denominator; reject degenerate scale. Do not rescale per frame. |
| Coverage | Report detected, eligible, fitted, failed, and imputed counts over all input frames. Target successful fits ≥95% of detector-present, predeclared eligible frames. Never exclude a high-error solved frame post hoc. |
| Native attachment | Hand wrist equals body wrist within 0.1 mm by construction and test; report observed body-wrist discrepancy separately without assuming V1 makes it zero. |
| V1 regression | With frozen body state, six-arm outputs unchanged within 0.1 mm. If shape is refitted, publish its arm error change; proposed allowed increase ≤0.1 percentage point of shoulder width, frozen before evaluation. |
| Sequence stability | On predefined static segments, mean non-wrist temporal RMS displacement about each point's temporal mean must be ≤ the framewise baseline +0.5 mm. On predefined motion events, added timing lag must be ≤1 native frame and ≤50 ms. Positional gates still apply. Freeze these provisional tolerances at the end of P3, before P5 tuning. |
| Runtime | Report hardware, seconds/frame and batch cost. No real-time release promise. |

Require targets for both sides and both backends, plus per-clip/per-finger tables and worst cases;
an aggregate cannot hide a failing thumb or hand. Use equal-clip summaries alongside pooled point
statistics, and clip/signer bootstrap uncertainty when sample size allows. The five V1 clips alone
are a pilot, not evidence of population-wide performance.
Report per-finger p95 and maximum errors separately from the pooled p95, plus velocity and
acceleration residuals. Define motion-event timing from synthetic known trajectories and labeled
real events before tuning; do not select events after seeing which ones smoothing preserves.

Maintain three distinct partitions: shape/pose calibration, development validation, and a sealed
final test set. P3 may inspect validation frames held out from shape calibration; those are
development results, not release test results. Freeze every numerical release threshold at the
end of P3 using only calibration/validation evidence. P4 and P5a may tune fitting/temporal settings
on validation data, but cannot revise those thresholds or inspect the final test targets/results.
Freeze all settings, adapters, priors, eligibility rules, and event definitions before P5b opens
the final test set. Any later change requires a new untouched final test set or an explicitly
exploratory result; never relabel the inspected test set as still held out.

Freeze provisional thresholds after the pilot, before any final held-out result is available.
If they prove infeasible,
write a decision record with achieved values and the limiting factor. Do not quietly loosen them.
A completed investigation with a negative result is useful; it is not a passing low-loss release.

### Separate optimizer error from representational mismatch

Construct an error ladder: neutral fixed shape → sequence-calibrated shape → stronger multi-start
offline fit under the same priors → relaxed diagnostic skeleton/shape fit. Compare on identical
frames, masks, adapters, and statistics. Use synthetic known-feasible samples to catch optimizer
failures. The multi-start result is a best-found reference, not a certified global minimum; a
relaxed hand skeleton is not automatically a valid mathematical lower bound on skinned fingertip
error. Do not transfer V1's exact five-proportion quadratic floor to hands.

Static recordings and controlled image perturbations can characterize detector sensitivity;
repeated execution of the same deterministic frame is not independent measurement noise. Neither
this reference nor temporal jitter replaces reconstruction error. Low real-data residual can still
fit detector bias. Test anatomical correctness only with independently justified evidence.

## 7. Work packages and gates

Execute in this order. Each package has a concrete output, tests, and a decision; implementation
uses fresh Luna implementers and independent Luna reviewers with file-backed briefs/reports as
required by project instructions. Do not run overlapping edits concurrently. Independent data
inspection and adapter review may run in parallel after interfaces are fixed.

| Package | Depends on | Work and deliverables | Acceptance / stop condition |
| --- | --- | --- | --- |
| **P0 — freeze evidence and capture pilot** | none | Record source/asset/package hashes and V1 baseline; select hand extraction API; collect paired raw hand-world/image data and masks; label mirroring and clip splits. | Both hands detected in suitable pilot frames; units/origin/side/timestamps auditable; clip order independent. If metric hands are unavailable, keep image-only work a separately scoped track. |
| **P1 — backend contracts and 21-point adapters** | P0 | MANO loader and SMPL-X configuration parity; mapping manifest; actual forward checks; neutral/one-finger renders; parameter serialization. | Mapping, tip indices, left/right shape behavior, means/PCA, beta dimensions, decoder independence and gradients pass. Do not begin accuracy tuning on a broken adapter. |
| **P2 — synthetic inversion and observability** | P1 | Deterministic rigid/palm initialization, analytic finger seed, real-model refinement; synthetic corpus and reports for both backends. | Meet synthetic position gates; finite gradients; chirality/rotation/translation equivariance; document null spaces and degenerate cases. Fix convention/solver failures before fitting real-data shape. |
| **P3 — MANO real-hand baseline and shape study** | P0, P2 | Per-frame fit, multi-start reference, shared-shape calibration, prior ablation, development-validation reconstruction, error ladder. | Local target feasibility assessed on validation only; freeze release thresholds before final test access. State which failures are data, adapter, optimization, or model mismatch. Stop increasing capacity if anatomy collapses. |
| **P4 — direct SMPL-X and wrist attachment** | P2, P3 | Fit hands with frozen V1 body; explicit parent-relative wrist rotations; body/local metrics; MANO transfer comparison; constrained shared-shape experiment only if needed. | Local SMPL-X targets and native seam tests pass; frozen-body regressions absent. Unverified hand/body frame relation leaves attachment gate open. |
| **P5 — temporal robustness and final held-out evaluation** | P3, P4 | P5a: tune sequence regularization and reset/reacquisition on validation data, then freeze all settings. P5b: run independent final evaluator, occlusion/crossing fixtures, and error/coverage report. | Untouched final test meets frozen positional/stability gates; no silent swaps, zero-default successes, or tuning from test results. |
| **P6 — release documentation and optional image bridge** | P5 | V1.2 findings, model cards/contracts, replayable evaluation command, before/after renders, reviewed public hand API. Optional paired camera/reprojection experiment has its own report. | Independent whole-change review; asset-backed checks really run, not all skipped; documentation states limits. Image pipeline remains gated on its own evidence. |

P0–P2 form the first feasibility slice. It should produce real hand data plus a verified synthetic
roundtrip before committing to the rest of the design. Duration is intentionally left unestimated
until asset/runtime and pilot availability are checked.
If metric hand observations cannot be obtained, an image-only fallback does not pass this 3D
feasibility slice or satisfy the requested millimetre-level objective; record `revise-plan` for
that objective while keeping any useful 2D experiment separately labeled.

### Candidate repository changes during implementation

| Area | Proposed change |
| --- | --- |
| `scripts/extract_hand_landmarks.py` (new) | Paired hand observations and manifest; preserve existing V1 caches. |
| `body_model/hand_observations.py`, `hand_mapping.py` (new under `src/kinetic_augment/`) | Versioned records, coordinate/side checks, and named 21-point adapters. |
| `body_model/hand_converter.py` (new) | Backend-specific fitting and parameter-only reconstruction with shared losses. |
| `body_model/mano_wrapper.py` (new if useful) | Thin side-aware MANO interface; actual tip vertices and model metadata. |
| Existing `smplx_wrapper.py`, `skeleton.py`, `converter.py` | Explicit model conventions and composite body/hands entry point while retaining body-only API. |
| `scripts/measure_hand_round_trip.py`, `configs/hand_roundtrip.yaml` (new) | Fixed splits, masks, seeds, priors, thresholds, and machine-readable metrics. |
| `tests/test_hand_mapping.py`, `test_hand_converter.py`, `test_hand_attachment.py` (new) | Unit, synthetic inversion, asset-backed parity, serialization, and seam tests. |
| `docs/V1_2_Findings.md` (future) | Measured result, rejected hypotheses, known limitations, and acceptance decision. |

Update each affected `index.md`; add any new folder index to `AGENTS.md` and preserve its
`CLAUDE.md` symlink. Keep licensed models out of Git. Prefer existing dependencies before adding
an optimizer, learned prior, or rendering stack.

## 8. Validation matrix and experimental discipline

- **Geometry:** left and right open palm, fist, thumb opposition, pinch, finger spread, asymmetric
  articulation, near-straight/collinear chains, large wrist rotations, nonzero shape, root
  translations, and mirrored input conventions. Synthetic fixtures are generated from each real
  backend, plus independent adapter checks so shared indexing mistakes cannot validate themselves.
- **Observation failures:** absent hand, zeros that are valid, NaNs, isolated missing joints,
  visible fingertip with missing parent, low palm area, hand crossing, swapped identities,
  long occlusion, reacquisition, dropped timestamps, and detector resets.
- **Data splits:** use the five V1 clips as pilot only. Add a practical initial target of at least
  20 clips across at least five identified people, with both sides, hand sizes, poses, occlusion,
  and aspect ratios represented. Split by signer where possible, otherwise label a clip-held-out
  claim; never silently infer signer identity. Freeze test IDs before solver/adapter tuning.
- **Calibration:** when per-person/clip shape calibration is permitted, disclose the calibration
  frames and evaluate disjoint time blocks. Also report cold-start neutral-shape performance.
  Do not smooth or optimize test targets jointly with training/calibration targets across splits.
- **Ablations:** root initialization, tip loss, uniform versus justified uncertainty weights,
  neutral versus calibrated shape, priors on/off as a diagnostic, time regularization on/off,
  direct SMPL-X versus transferred MANO initialization, fixed versus alternate landmark adapter.
- **Evaluation independence:** decoder-only replay in a clean process; no ground-truth parameters
  given to the synthetic fitter; finite-difference gradient checks; known-bad permutation/mean
  fixtures that the metrics must reject. Report original masks even when solving fails.
- **Asset testing:** lightweight tests can use synthetic fixtures without weights. A separate
  locally licensed asset-backed lane is mandatory before accepting MANO/SMPL-X fidelity; skips
  are not passes. Re-run relevant V1 body tests after integration.

## 9. Optional image-space return

Returning to MediaPipe-style **ordering and world-hand coordinates** is in scope. Returning exact
normalized image x/y/z is a different claim: the detector's image z is not a calibrated projection
of its world output. Preserve these two meanings in API names and evaluation reports.

For an image bridge, fit and validate a camera from paired observations (calibrated perspective
when available; explicitly labeled weak perspective as a tested fallback), report x/y pixel error,
and keep relative-z conventions separate. Test identity, known wrist transforms, and synthetic
3D perturbations. A wrist-following image offset is a useful baseline but does not reconstruct
finger articulation through MANO/SMPL-X. Camera fitting to the same 2D targets must be disclosed;
held-out tests are needed for predictive claims.

Do not integrate intrinsic sequence augmentation until its once-per-clip perturbation policy is
fixed; V1 already records per-frame random perturbations as a downstream defect. Conversion,
projection, temporal augmentation, and classifier evaluation have distinct acceptance gates.

## 10. Current decision

**Gate: go-with-risks for P0–P2; the low-loss release is unproven.** The architectures and hand
observation APIs exist, but the current repository has not established metric hand extraction,
21-point adapter parity, genuine hand inversion, or the attainable error on these clips.

The highest-risk dependencies are anatomical correspondence (A007), representational fit (A011),
hand/body frame alignment (A005), and shared SMPL-X shape (A013). The
[assumption ledger](V1_2_Assumptions.md) defines their verification steps and the exact downstream
claims that must be withdrawn if they fail. Begin with evidence and synthetic correctness; let
those measurements decide how much fitting complexity is justified.
