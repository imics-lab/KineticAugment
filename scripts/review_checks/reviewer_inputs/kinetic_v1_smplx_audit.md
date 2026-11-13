# KineticAugment V1 SMPL-X integration audit

Read-only audit of the current worktree and local neutral SMPL-X weights. No repository files changed. The worktree already contained user changes; this report is the only output.

## Findings

### Existing converters remain a raw-vector bypass

- `src/kinetic_augment/body_model/mp_to_smplx.py:80-121` still returns zero `global_orient` and zero `betas`, then stores observed child-minus-parent displacements directly in body pose slots. Its module docstring says these are not standard axis-angle values.
- `smplx_to_mp.py:65-150` still requires `target_mp`, adds those stored vectors to original parent positions, and never calls the wrapper. It copies face and other unconstrained landmarks through from that input.
- The new solver is not wired into either converter. Therefore the public fit/project path still does not satisfy “genuine params in, real SMPL-X forward, model-derived projection out.” Old hand slots are also raw segment vectors, not rotations.

### The new kinematics do map to real SMPL-X

- `skeleton.py:36-70` loads the SMPL-X model, computes rest joints from the shaped template and joint regressor, reads `model.parents`, and limits the skeleton to the first 55 model joints. The first 22 parent indices match the established body joint map.
- `kinematics.py:14-55` composes local axis-angle rotations down that tree. `swing_ik.py:32-37,80-135` solves shoulder joints 16/17 and elbow joints 18/19 from MP shoulder/elbow/wrist directions. Mapping its 55x3 output into SMPL-X is root row 0 -> `global_orient`; rows 1:22 -> flattened `body_pose`. Wrist joints 20/21 remain at zero; elbow rotations orient the forearms and place wrist endpoints.
- Fresh check with local weights: FK vs actual SMPL-X `output.joints[:, :55]` differed by at most **1.37e-7 m** on a random pose. A real SMPL-X-generated arm pose with zero root orientation was fit by swing IK, then passed through actual SMPL-X forward; the six shoulder/elbow/wrist positions matched within **2.2e-8 m**. Forward output shape was `(1,127,3)`: use only first 55 for model joints.
- Existing synthetic IK test builds and checks landmarks using helper FK. The separate FK-vs-model test composes with it; my direct real-model fit/forward check closes this for a zero-root arm pose. A permanent public converter-to-model regression test is still warranted once integrated.
- The wrapper accepts axis-angle arrays. `body_pose` covers joints 1..21; pelvis/root is `global_orient`. But `SMPLXWrapper.num_joints` is 55 from the regressor while forward emits 127, and its docstring incorrectly claims output 55.
- Hand conventions differ: `SMPLXSkeleton` uses `flat_hand_mean=True`; `SMPLXWrapper` leaves default `False`. Irrelevant to body-only V1 but must be unified before interpreting hand zeros or fitting hands.

### Root, coordinates, shape and the error gate remain open

- `solve_swing_ik()` accepts optional `global_orient` but does not estimate it, and `measure_round_trip.py:126` calls it without one. Fresh nonzero-root experiment: fitting landmarks generated with nonzero root and forwarding the zero-root result caused up to **0.289 m** driven-joint error before alignment. Bone directions can be absorbed into shoulder rotations while shoulder anchors and torso anatomy stay wrong. The documented deferral is acceptable for a direction-only baseline, but does not establish a plausible full-body pose.
- The scope correctly moved from image-normalized `pose_landmarks` to metric, hip-centered `pose_world_landmarks`. `measure_round_trip.py` applies a per-frame Umeyama similarity after fitting, but that scale/rotation/translation is only evaluation state; neither converter stores or applies it. A parameter-only projection to the same input frame needs an explicit coordinate-transform contract and carried transform metadata (or a clear fixed convention). This alignment state is meaningful, unlike per-landmark residuals.
- Current evidence reports **5.94% shoulder-width** median error against a **0.18%** world-landmark noise floor; length dominates. So analytic IK validates direction and parameter semantics, not the stated fidelity gate.
- Observed arm segment ratios vary by side and segment; global scale cannot explain all. However ten `betas` inferred from four arm lengths are underconstrained and confounded with scale, joint-center correspondence offsets, and root pose. Shape refinement should be sequence-level and regularized; report identifiability. For these upper-body signing clips, legs are prior-only (knees/ankles visibility near zero), hips observed only about one-third of frames. Exclude them from claims and metrics.

## Minimal robust architecture

1. Keep analytic swing IK as deterministic initialization and direction solve; it is validated against real SMPL-X.
2. Replace the legacy converter implementation with one layer accepting metric world landmarks plus explicit validity/confidence masks. Package the 55x3 solve as root/body slots above, with documented neutral priors for unconstrained joints, hands, face and betas.
3. Define root/camera similarity consistently. Do not present post-hoc Procrustes as a solved pose unless the transform is preserved and projection applies it. Estimate root orientation only where landmarks constrain it; otherwise make the ambiguity explicit.
4. Validate the converter output through the actual wrapper/model and compare selected first-55 joints in a declared coordinate frame. Keep synthetic truth tests and add direct fit -> actual model forward tests.
5. If fidelity remains above noise, refine the analytic initialization with regularized least squares over visible landmarks, joint priors and sequence-level shape. Hold shape fixed per clip, weight by confidence, and compare against the analytic baseline. A learned model cannot resolve missing observations or make root/shape identifiable.
6. Make projection parameter-only for fitted body correspondences. Face/hands/unfitted joints should be explicitly unavailable or documented passthrough; requiring original `target_mp` makes projection dependent on the answer and can hide errors.

## Evidence locations

`docs/V1_Problem_Scope.md`; `src/kinetic_augment/body_model/{mp_to_smplx.py,smplx_to_mp.py,skeleton.py,kinematics.py,swing_ik.py,smplx_wrapper.py,alignment.py}`; `scripts/measure_round_trip.py`; `tests/test_kinematics.py`; `tests/test_swing_ik.py`.
