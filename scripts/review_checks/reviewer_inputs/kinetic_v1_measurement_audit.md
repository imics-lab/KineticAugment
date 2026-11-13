# KineticAugment V1 measurement audit

Read-only audit of the current working tree on 2026-10-04. The repository has numerous pre-existing modifications and untracked files, including the scope note, scripts, world archives, and tests. No repository files were changed.

## Findings

### 1. Current world-coordinate inputs exist and have the expected schema

There are five current untracked `data/test_output/*_world.npz` files: `01986`, `01987`, `01988`, `01991`, and `01992`. Each contains `landmarks` float32 `(T, 33, 3)` and `visibility` float32 `(T, 33)`. Frame counts are 83, 75, 144, 49, and 59 (410 total); arrays contain finite coordinates for all 33 landmarks in every frame in the current archives. The files open and the extraction's schema matches `scripts/extract_world_landmarks.py:13-18, 43-63, 81-91`. The extractor writes NaN/zero visibility when no pose is detected (`:53-60`) and `np.savez_compressed` to the fixed output directory (`:73-87`); I did not run extraction because it would overwrite user data.

The noise script reads the current archives and says observed fractions (visibility >= .5, averaged across clips) are 100% for shoulders/elbows, 89% left wrist, 91% right wrist, 55% hips, and 0% knees/ankles. This differs from the scope's earlier per-landmark observation percentages (docs/V1_Problem_Scope.md:335-342), so those percentages need refreshing. It appears the scope/research note may have changed during the audit; use the current files as source of truth.

### 2. Reproducible current run

Ran `.venv/bin/python scripts/measure_noise_floor.py` and `.venv/bin/python scripts/measure_round_trip.py` (both read-only). The former reproduces pooled Method A/B xyz values 0.18%/0.10%; the latter reproduces the scope's headline:

- Synthetic recovery: worst position error `2.993e-16 m` across 200 generated poses.
- Real data: 5.94% median landmark position residual after per-frame similarity alignment; per-clip medians 6.31%, 6.38%, 5.06%, 9.20%, 6.19%.
- Median length error after alignment 6.66%; per-bone 4.38–8.21%.
- The harness reports post-alignment angular discrepancies of 2.1–2.8 degrees, so the printed “direction error: negligible” (`scripts/measure_round_trip.py:205-206`) contradicts its own table (`:192-196`) and the scope's “exact by construction” phrasing (docs/V1_Problem_Scope.md:468-486). Direction fit is near-exact before alignment; the post-fit Procrustes rotation changes directions.

These results establish that the implementation is deterministic and that the current scoring setup yields a large residual. They do not establish that true round-trip accuracy is 5.94%, that noise is 0.18%, or that subject shape is the cause.

### 3. The real-data “round-trip” score is an in-sample, per-frame Procrustes residual

`scripts/measure_round_trip.py:123-157` fits the IK from the observed frame, then estimates scale, 3-D rotation, and translation from the same observed correspondences that it scores (`:129-140`). This is six points when all arm landmarks are visible; minimum is only three (`:60-61, 129-136`). The 7-parameter similarity is free on every frame. This is a valid in-sample reconstruction-fit diagnostic under a per-frame similarity objective, but it is not a complete serialized conversion/project round trip: the estimated transforms are not carried by the converter as state. The per-frame similarity can remove global orientation, position, and scale separately for each frame, while simultaneously rotating bone directions and rescaling bone lengths. The fit residual should therefore be labeled as the residual after per-frame Procrustes alignment, not direct end-to-end round-trip fidelity.

The sample does not guard the similarity fit against degeneracy/conditioning. `alignment.py:38-43, 45-66` checks shape and N>=3 only; it does not check rank, singular values, or condition. `tests/test_alignment.py:22-68` covers exact known transforms, reflection, and input shapes, but not collinear/near-collinear inputs, noise behavior, or held-out scoring. Planar points (rank 2) do not by themselves make a proper rigid transform ambiguous; collinear or near-collinear point sets (rank 1) are the relevant rotational degeneracy. Report singular values/conditioning to detect that case.

For per-bone length comparisons, a free global scale is unnecessary because bone-length ratios and joint-to-joint Euclidean lengths are invariant to rotation/translation. Report raw world-space observed bone lengths against model lengths first, then separately report any fixed clip-level scale-adjusted comparison. Do not re-fit a transform separately on every scored frame.

### 4. Noise “floor” measures short-term repeatability proxies, not accuracy

`scripts/measure_noise_floor.py:97-103` Method A is the 10th percentile of adjacent-frame displacement. This is neither a standard deviation nor a conservative upper bound: under stationary independent isotropic noise it is a low quantile of a difference norm (and therefore depends on dimension); real motion and sampling frequency also affect it.

Method B (`:105-126`) assumes iid additive noise, divides second-difference coordinate SD by `sqrt(6)`, and first selects the slowest 25% of observed adjacent speeds (`:115-123`). The extractor sets `static_image_mode=False` and leaves MediaPipe's temporal landmark smoothing at its default (`scripts/extract_world_landmarks.py:73-75`); temporally correlated/smoothed errors invalidate the iid `Var(D)=6 sigma^2` conversion, and selecting low observed speeds conditions on the same noisy values and biases the estimate downward. Both methods are diagnostics of temporal variation in these five clips, not detector accuracy against truth.

The gate is additionally mismatched: `measure_noise_floor.py:97-126` returns a component SD norm (or a p10 displacement), then normalizes by shoulder width; the round-trip headline is a median per-landmark Euclidean residual (`measure_round_trip.py:138-140, 186-200`). Taking `max(0.18%, 0.10%)` as a budget (`measure_noise_floor.py:209-215`) does not make these the same statistic or supply uncertainty. Five clips from one capture setup do not justify a universal detector-noise threshold. The approximate “0.6 mm” conversion at SMPL-X's canonical 0.3159-m width (scope line 310) is just a unit conversion, not evidence of sub-mm MediaPipe accuracy.

### 5. Bone-length discrepancy does not identify SMPL-X shape as the cause

The present 4 arm-segment observations are sparse and may be asymmetric/noisy; they cannot uniquely identify ten `betas` plus a global scale. A clip has many frames, but the same four subject-specific segments are repeatedly measured, not new independent shape dimensions. A regularized, low-dimensional shape solve might be testable, but only after specifying a prior or fixing scale and adding independent measurements. Also, MediaPipe world coordinates being nominally metric does not certify anatomical accuracy; pose-model bias, occlusion, and smoothing can explain segment discrepancies. The current data contain no motion-capture or marker ground truth, so wording like “this signer has shorter upper arms” is stronger than the evidence supports.

## Recommendation on approach / acceptance criteria

Keep the analytic swing IK and synthetic known-pose validation: the synthetic test is a strong check of the math. Change the real-data experiment before accepting the 5.94% number or starting an unconstrained 10-beta fit.

1. Define distinct goals: (a) internal parameter/kinematic correctness, (b) reconstruction fidelity in the chosen MediaPipe coordinate convention, and (c) anatomical accuracy. The current setup can test (a), gives a provisional residual for (b), and cannot establish (c).
2. Use the same named estimand in both sides of the gate (e.g. median 3-D landmark Euclidean residual / one fixed shoulder-width normalization), and report clip-level median plus spread/bootstrap interval. Treat temporal-noise estimates as a repeatability reference, explicitly labeled as such. Better noise data: static repeated captures, multiple takes, or synthetic perturbations validated against repeated detections; do not call any of these absolute accuracy without external 3-D ground truth.
3. Declare the coordinate convention and scale policy separately from pose. A constant convention transform and, if needed, one per-sequence scale can be calibrated independently; per-frame root pose may legitimately vary with motion and should be estimated and carried as part of the fitted parameter state. Score held-out frames/landmarks after applying the serialized state. Keep the current per-frame Procrustes residual as a labeled in-sample fitting diagnostic, and report raw world-space bone lengths and pose-direction error before alignment separately. If root pose is not estimated/carried by the converter, do not call the aligned residual end-to-end round-trip fidelity.
4. Before shape optimization, run an identifiability check on the beta-to-observable-length Jacobian for the selected landmarks, with scale included. If rank is insufficient (expected with four arm segments), fix scale and restrict beta dimensions using a documented prior, or obtain additional independent body measurements / full-body clips. Compare fits across clips and against a default-shape baseline.
5. Carry an explicit observed mask through IK. `solve_swing_ik._observed` in `src/kinetic_augment/body_model/swing_ik.py:66-78` rejects an exactly-zero point even though its own documentation says the hip-centered world origin is legitimate; current frame calls use NaN masking (`measure_round_trip.py:123-127`) but exact valid-origin landmarks would be silently treated as absent.

## Key evidence paths

- Scope and current claims: `docs/V1_Problem_Scope.md:302-316, 330-355, 468-499, 525-567`.
- Noise definitions and reduction: `scripts/measure_noise_floor.py:57-70, 82-126, 129-215`.
- World extraction and defaults: `scripts/extract_world_landmarks.py:43-63, 66-95`.
- Round-trip alignment/metrics: `scripts/measure_round_trip.py:104-162, 165-210`.
- Umeyama implementation: `src/kinetic_augment/body_model/alignment.py:17-66`.
- Zero-origin issue: `src/kinetic_augment/body_model/swing_ik.py:66-78`.
