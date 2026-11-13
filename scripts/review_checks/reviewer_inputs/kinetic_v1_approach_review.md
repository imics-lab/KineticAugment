# Review of the KineticAugment V1 approach

Reviewed 2026-10-04. Source: `/Users/bemnetbeshah/Workspace/cv-research/KineticAugment/docs/V1_Problem_Scope.md` and current working-tree code/data. The scope changed during the review; this assessment includes the world-landmark result of 5.94% and the proposed Step 7. Three Luna agents performed read-only code exploration. No repository files were edited by this review.

## Outcome and decision

Deliver real SMPL-X parameters and an explicit reconstruction contract, with measured error over observed body landmarks and honest limits on anatomical inference.

**Decision: revise-plan before treating Step 7 as the established next remedy.** Keep the analytic IK, skeleton abstraction, real-model forward validation, observed-landmark masking, and separate error metrics. Correct the treatment of root/torso pose, uncertainty, and the acceptance gate. My recommended architecture is analytic initialization plus a small, regularized fitting stage, evaluated against the unchanged analytic baseline.

## Assumption ledger

| ID | Claim | Type / dependency | Evidence | Confidence / status | Verification and failure signal |
| --- | --- | --- | --- | --- | --- |
| A001 | The new skeleton/FK path corresponds to real SMPL-X joints. | Integration / none | Local-weight check: first 55 joints agree within 1.37e-7 m. | Confirmed / verified | Compare FK with actual forward for random poses and nonzero shape/root. A systematic mismatch invalidates fitted parameter semantics. |
| A002 | Exact arm directions leave only length error in reconstructed positions. | Geometry / A001 | Same-skeleton root-rotation counterexample has identical limb lengths but 44.7 mm mean error after similarity alignment. | Risky / broken | Run saved counterexample. Moving anchors with unchanged lengths disproves length-only attribution. |
| A003 | Synthetic position recovery establishes recovery of the original rotations. | Observability / A001 | The 200-pose harness measures six joint positions only; terminal-segment twist is not identifiable. | Risky / broken | Generate different twists with identical endpoints. Require position equivalence; require rotation equivalence only in an explicitly constrained, identifiable test family. |
| A004 | The 0.18% statistic is a calibrated pass/fail noise floor for reconstruction. | Measurement / none | Method A uses a p10 step; Method B assumes iid noise, while extraction enables temporal smoothing by default. Neither equals the median residual statistic. | Risky / open | Calibrate the same statistic on controlled stationary/repeated observations and account for temporal correlation. Threshold changing with smoothing or frame rate signals a repeatability proxy. |
| A005 | The 5.94% score measures the delivered converter's direct round trip. | Integration / A001 | The harness fits a fresh similarity transform to each frame, outside the public converter's serialized output. | Risky / broken | Reconstruct from saved fitted parameters and explicit frame metadata alone. Any dependency on unrecorded post-hoc alignment breaks this contract. |
| A006 | Four observed arm lengths identify ten betas plus scale and establish signer anatomy. | Data / A002, A004 | The length-only inverse problem has fewer independent scalar observations than unknowns; world landmarks are model estimates. | Risky / broken as a unique-identification claim | Inspect length Jacobian rank and fit stability. Use a shape prior/reduced model and held-out frames; a regularized solution is a convention, not recovered true anatomy. |
| A007 | Public converter classes now use the verified analytic path. | Integration / A001 | Current mp_to_smplx and smplx_to_mp still encode/decode raw displacements. | Confirmed / broken | End-to-end fit -> real model -> selected joints test. Genuine helper math alone does not change public behavior. |

Dependency chain: valid data/coordinate contract -> A001 model semantics -> root/torso and identifiable pose handling (A002/A003) -> complete reconstruction state (A005/A007) -> defensible scoring (A004) -> evidence for shape refinement (A006).

## Most consequential findings

1. **Root orientation affects position error.** The document's claim that it cannot affect the gate is false. A shoulder rotation moves the elbow and wrist, but cannot relocate the shoulder itself. Rotate the whole synthetic body by 0.7 rad, leave fitted root at zero, and arm directions still match within 1.21e-6 degrees. Maximum position error is 319.15 mm before alignment and 79.18 mm afterward; mean aligned error is 44.70 mm. Bone lengths match exactly. This establishes an additional error source, not its numerical contribution to the real clips.
2. **Validity, fidelity, and plausibility are distinct.** Valid axis-angle parameters can generate implausible elbows/shoulders. Joint positions do not uniquely identify all rotations, shape, or torso pose. The scope currently requires a plausible mesh while excluding its assessment. Include a small mesh/pose sanity check or narrow that promise.
3. **The real-data numbers reproduce, but their interpretation is too strong.** Scripts reproduce 5.94% median aligned position residual and 6.66% aligned bone-length error. Post-alignment direction errors are around 2.1–2.8 degrees, even though the conclusion calls them negligible. Use a common coordinate convention when decomposing errors. Per-frame Procrustes is a valid supplementary fit diagnostic; it is not a substitute for a complete saved conversion state.
4. **World coordinates are the right baseline input, not ground truth.** MediaPipe documents metric, hip-centered outputs. Its visibility field represents visibility, not calibrated 3-D coordinate uncertainty. A stable biased prediction can have very little temporal jitter. Therefore the estimated proportions do not establish the person's true proportions, and the approximately 0.6 mm budget does not establish detector accuracy.
5. **The document needs one current contract.** It mixes normalized-image and world-coordinate input, 12-landmark scope and six observed arm landmarks, planned and completed Step 4/5 sections, and position recovery with rotation recovery. The seven-number similarity discussion cannot serve as an exact camera model for arbitrary image-normalized pseudo-depth. Make V1 explicitly world-body -> SMPL-X -> world-body; any later image-space return requires its own projection contract.

## Recommended next sequence

1. **Define the input/output and evaluation contract.** Input: world body landmarks, explicit validity/confidence, and sequence/frame identity. Output: SMPL-X parameters, any additional scale/coordinate state, and which quantities are observed versus defaulted. Keep image-space hands/face outside this metric body result. Public reconstruction must work without consulting original body coordinates.
2. **Correct root/anchor handling and expand synthetic checks.** Test nonzero root, torso motion, nonzero shape, missing joints, and valid points at the origin. Use real SMPL-X forward to validate the final path. Compare positions; compare rotations only where they are identifiable under the stated twist convention. Sparse upper-body data require a prior for ambiguous torso orientation; shoulders plus nose are a heuristic, since the head can move independently.
3. **Retain analytic IK as baseline and initialization.** Estimate per-frame root/pose within a fixed coordinate convention. Then compare a small confidence-weighted robust fit with pose/shape priors against that baseline. Allow weak torso/collar adjustment only as justified by anchor residuals. Shape and any calibration scale belong at sequence level, with scale fixed or strongly constrained. A regularized ten-beta estimate may be usable, but cannot be advertised as uniquely recovered from four lengths.
4. **Evaluate the refinements independently.** Report direct reconstruction errors in mm and percent shoulder width, per-joint median/p95, coverage, direction and length errors in the same frame. Report Procrustes-aligned error separately. Compare root-only, shape-only, and root-plus-shape refinements so causation is testable. Use held-out frames/clips for shared shape/calibration and new full-body data before claiming legs. Keep temporal repeatability as a reference until its threshold is calibrated.

This need not become a new learned system. HybrIK is useful prior art for analytic swing plus additional information for twist; SMPLify-X illustrates fitting with pose priors. The existing analytic work is reusable.

## Repeatable checks and evidence

From the project root:

```sh
rtk proxy .venv/bin/pytest -q tests/test_swing_ik.py tests/test_kinematics.py
rtk proxy .venv/bin/python scripts/measure_noise_floor.py
rtk proxy .venv/bin/python scripts/measure_round_trip.py
rtk proxy .venv/bin/python /private/tmp/kinetic_v1_root_counterexample.py
```

Targeted tests: 27 passed. Synthetic harness: maximum position error 2.993e-16 m across 200 poses. These are scoped checks, not a claim that the full project suite passes.

Detailed local reports: [IK audit](/private/tmp/kinetic_v1_ik_audit.md), [measurement audit](/private/tmp/kinetic_v1_measurement_audit.md), [SMPL-X integration audit](/private/tmp/kinetic_v1_smplx_audit.md). Counterexample: [script](/private/tmp/kinetic_v1_root_counterexample.py), [captured output](/private/tmp/kinetic_v1_root_counterexample.txt).

Primary external references: [MediaPipe coordinate definitions](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python), [MediaPipe temporal smoothing](https://chuoling.github.io/mediapipe/solutions/pose.html#smooth_landmarks), [HybrIK paper](https://arxiv.org/abs/2011.14672), [SMPL-X / SMPLify-X project](https://smpl-x.is.tue.mpg.de/index.html).
