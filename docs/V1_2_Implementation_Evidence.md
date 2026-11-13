# V1.2 implementation evidence

**Date:** 2026-10-06. **Cited by:** [V1.2 findings](V1_2_Findings.md) and the
[V1.2 assumption ledger](V1_2_Assumptions.md). **Plan:** [V1.2 implementation plan](V1_2_Implementation_Plan.md).

This appendix keeps the measurements that the findings rely on but that no committed data file
holds. They were taken while Tasks 1–6 of the implementation plan were built and reviewed, on
2026-10-05 and 2026-10-06. The implementers made them with one-off scripts and wrote them into
working reports, which are not kept in the repository. The numbers below are copied exactly from
those reports. Where a review made the measurement (§2, §6), they come from the review's notes.
§2, §6, §8 and §9 also record checks made for the findings in Task 7, and §2 and §8 record checks
made by the final whole-branch review.

Each section names the task, what was run, the commit it ran on and the date. It also names the
committed check or file that covers the same ground. The tests assert the gates; they do not pin
these exact values. The commit hashes refer to branch `feat/v1.2-hands` and stay valid only if it
is merged without squashing.

## 1. Synthetic gate: refined fits (Task 3)

The Task 3 implementer measured this on commit `896d73f` (2026-10-06), with the data of
`test_recovers_hands_the_model_can_represent`:

- Each case has 12 frames from `synthetic_track` (seed 21).
- The random-pose cases draw finger poses from N(0, 0.3).
- The training-pose cases take 12 real poses from MANO's training set (`training_poses`, seed 20).
  The same poses are used for both backends.

The script ran `fit_hands`, then `reconstruct_hand`, and scored the result with
`landmark_errors_mm`. *Start only* scores the analytic start on the same data. Every status was
`ok`.

| Backend | Side | Poses | Mean (mm) | p95 (mm) | Max (mm) | Start only: mean / p95 / max (mm) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MANO | left | random | 0.0089 | 0.0305 | 0.0791 | 0.140 / 0.786 / 3.686 |
| MANO | left | training | 0.0073 | 0.0255 | 0.0687 | 0.211 / 1.406 / 3.725 |
| MANO | right | random | 0.0115 | 0.0359 | 0.1725 | 0.142 / 0.830 / 3.047 |
| MANO | right | training | 0.0074 | 0.0251 | 0.0693 | 0.211 / 1.406 / 3.725 |
| SMPL-X | left | random | 0.0120 | 0.0437 | 0.1114 | 0.250 / 1.382 / 3.842 |
| SMPL-X | left | training | 0.0092 | 0.0310 | 0.0790 | 0.313 / 1.658 / 3.996 |
| SMPL-X | right | random | 0.0158 | 0.0509 | 0.1814 | 0.229 / 1.199 / 3.213 |
| SMPL-X | right | training | 0.0094 | 0.0298 | 0.0744 | 0.301 / 1.851 / 4.279 |

All eight cases clear the gate (mean ≤ 0.5 mm, p95 ≤ 1 mm) by at least 30 times on the mean and
19 times on the p95. The start alone has a p95 above 1 mm in 6 of the 8 cases, and 0.79 and
0.83 mm in the other two. So the p95 gate tests the refinement.

**Committed check:** `test_recovers_hands_the_model_can_represent` asserts the gate on this data
for both backends and both sides. It passed on 2026-10-06 (§9).

## 2. Batch coupling and the iteration cap (Task 3; checked again in Task 7 and the final review)

**Batch coupling.** The Task 3 implementer measured this on commit `896d73f` (2026-10-06), on the
right-hand, random-pose gate data of §1 (12 frames, seed 21):

- A frame fitted alone converges essentially exactly: mean error 0.0000 mm, p95 0.0001 mm.
- Fitted in a shared batch of 12, it ends at 0.0115 mm (MANO) and 0.0158 mm (SMPL-X) mean.
- A frame's reconstruction differs by at most 0.134 mm (MANO) and 0.139 mm (SMPL-X) between the
  batch of 12 and fitting it alone. This is the largest difference in any single coordinate. It is
  not the distance a landmark moves (see below).
- Results are deterministic for identical input and batching.

**Per coordinate and as a distance.** The final whole-branch review (2026-10-06, on commit
`a4a8ed0`) re-measured the coupling on the same data: the right hand, random poses, seed 21, 12
frames. For each landmark of each frame, the shift is the difference between its position rebuilt
from the batch-of-12 fit and from a fit of the frame alone. The review reported it two ways:

| Shift | MANO | SMPL-X |
| --- | ---: | ---: |
| Largest difference in a single coordinate | 0.1344 mm | 0.1394 mm |
| Largest Euclidean shift of a landmark | 0.1715 mm | 0.1810 mm |
| Mean Euclidean shift of a landmark | 0.011 mm | 0.015 mm |

The single-coordinate row agrees with the implementer's 0.134 and 0.139 mm above. The Euclidean
rows are the figures to quote for how far a landmark moves: up to 0.172 mm (MANO) and 0.181 mm
(SMPL-X). They are close to the batch fit's own largest and mean landmark errors in §1 (0.1725 and
0.1814 mm; 0.0115 and 0.0158 mm), which is expected, since a frame fitted alone reproduces its
observations almost exactly (first bullet above).

**Iteration cap.** The Task 3 review (2026-10-06) found that the batched L-BFGS ends at the
1,000-iteration cap, so status `ok` does not certify convergence and `rms_mm` is the quality
signal. The Task 3 report itself had not confirmed why the runs stop.

Task 7 checked this on 2026-10-06, on commit `b0d7b55`, whose converter code is the code that
produced the measurement. A read-only script wrapped `torch.optim.LBFGS.step` and recorded the
iterations and function evaluations of each run. Every run used all 1,000 iterations and 1,018–1,072
of its 1,250 allowed evaluations. So each run stopped at the iteration cap, not at a tolerance.

| Run | Iterations | Function evaluations |
| --- | ---: | ---: |
| Synthetic gate data (§1): 8 batches of 12 frames | 1,000 each | 1,054–1,072 |
| 01991 right, MANO: first 32 frames, neutral `fit_hands` | 1,000 | 1,018 |
| 01986 left, SMPL-X: first 32 frames, neutral `fit_hands` | 1,000 | 1,024 |
| 01987 right, MANO: `fit_hand_shape` on 27 frames, neutral batch then shared-shape run | 1,000 and 1,000 | 1,024 and 1,024 |

**Committed checks:** `MAX_ITERATIONS = 1000` in `src/kinetic_augment/body_model/hand_converter.py`
carries the comment from the pre-implementation dry run: 200 iterations stop short of convergence
(0.07 mm), and 1,000 reach 0.02 mm. `fit_hands` gives status `ok` to any finite result.

## 3. Synthetic gate: shape fits, and what the betas recover (Task 4)

The Task 4 implementer measured this on commit `21bbcf4` (2026-10-06), with the tests' own data
and seeds. Errors are `landmark_errors_mm` of the rebuilt hand against the observed one.

| Test (data) | Backend | Mean (mm) | p95 (mm) | Fitted \|betas\| |
| --- | --- | ---: | ---: | --- |
| `test_a_fitted_shape_reproduces_a_plausible_hand` (8 frames, true \|betas\| = 2.0) | MANO | 0.069 | 0.151 | 2.376, at 1.06 from the true betas |
| | SMPL-X | 0.108 | 0.255 | 4.126, at 4.50 from the true betas |
| `test_per_frame_shapes_follow_a_hand_that_changes_size` (4 frames, 4 shapes) | MANO | 0.111 | 0.306 | 3.94 / 3.14 / 3.95 / 2.04 |
| | SMPL-X | 0.055 | 0.147 | 2.13 / 3.66 / 3.99 / 1.84 |

On the same 8-frame data, the neutral shape gives a mean of 1.490 mm and a p95 of 2.556 mm on
MANO, and 2.110 and 3.849 mm on SMPL-X, so only a real shape fit passes. With the true shape
given, the floor is 0.010 / 0.030 mm (MANO) and 0.011 / 0.034 mm (SMPL-X). A single shape for the
four-shape track gives 1.851 / 4.317 mm (MANO) and 2.507 / 5.252 mm (SMPL-X).

The landmarks are reproduced, but the betas are not recovered. SMPL-X's fitted betas lie 4.50 from
the true ones, near the bound (radius 4.279), and MANO's lie 1.06 from them. Fitted from hand
landmarks, the betas are a plausible shape that reproduces the landmarks, not an estimate of the
true shape.

**Committed checks:** `test_a_fitted_shape_reproduces_a_plausible_hand` and
`test_per_frame_shapes_follow_a_hand_that_changes_size` assert the gate, on both backends and on the
right hand only. Both passed on 2026-10-06 (§9).

## 4. Per-frame-shape diagnostic: synthetic floor against track length (Task 4)

The Task 4 implementer measured this on commit `21bbcf4` (2026-10-06), on synthetic tracks:

- *One shape per track*: one plausible shape (|betas| = 2.0) for the whole track, with random
  poses, fitted with `fit_hand_shape` and then `fit_hands`.
- *Per-frame shape*: a different plausible shape on every frame, fitted with
  `fit_hands_per_frame_shape`.

The report's timings are left out.

| Backend | Frames (T) | One shape per track: mean / p95 / max (mm) | Fitted \|betas\| | Per-frame shape: mean / p95 / max (mm) |
| --- | ---: | --- | ---: | --- |
| MANO | 8 | 0.017 / 0.042 / 0.057 | 1.92 | 0.100 / 0.338 / 0.549 |
| MANO | 32 | 0.020 / 0.065 / 0.131 | 1.99 | 0.139 / 0.388 / 0.716 |
| MANO | 100 | 0.020 / 0.060 / 0.246 | 2.00 | 0.164 / 0.457 / 1.044 |
| MANO | 250 | 0.023 / 0.067 / 0.348 | 2.00 | 0.174 / 0.476 / 1.120 |
| SMPL-X | 100 | 0.053 / 0.130 / 0.339 | 4.04 | 0.165 / 0.492 / 1.247 |

The per-frame diagnostic fits all of a track's frames in one L-BFGS run. Its synthetic floor
therefore rises from 0.100 to 0.174 mm mean as the track grows. So on real data, a gap between one
shape per track and one per frame that is smaller than this floor says nothing about changes in
hand size.

## 5. Check 2: summed squared error against mean distance (Task 6)

The Task 6 implementer measured this on commit `15db091` (2026-10-06), with a read-only diagnostic
that makes the same library calls on the same inputs as the measurement script's `ladder`. It
reproduced the JSON means to three decimals. It covers the two track-backends where the in-sample
shape raises the mean distance above the neutral `refined` fit.

| Track-backend | Summed squared error, neutral → in-sample | Mean distance, neutral → in-sample | Frames worse | Worst-frame RMS |
| --- | --- | --- | ---: | --- |
| 01986 right, MANO | 174,580 → 132,538 mm² (−24.1%) | 8.385 → 8.403 mm | 15 of 52 | 21.6 → 17.6 mm |
| 01988 right, SMPL-X | 554,237 → 423,502 mm² (−23.6%) | 9.758 → 10.001 mm | 36 of 101 | 39.0 → 32.8 mm |

The fitter minimises the summed squared error, and the shape lowers it by about a quarter. The
mean distance is a different statistic.

**Committed alternative:** the round-trip JSON (`data/test_output/hand_round_trip.json`, `tracks`,
rows `refined` and `in-sample shape`) shows the same trade on these two track-backends. The mean
rises while the p95 falls, from 31.23 to 24.00 mm and from 42.37 to 33.05 mm.

## 6. Unassigned detections (Task 5 review)

The clips store how many detections each frame left without a side, but not the detections
themselves or the reason. The Task 5 review (2026-10-06) analysed the extraction of commit
`3be7fa3` and recorded:

- Nine are a second detection at a side already taken: 01991 ×7 and 01988 ×2.
- Three lie beyond the 10% wrist cap: 01988 frames 126–128, with gaps of 0.103, 0.115 and 0.138
  of the image height.
- All 535 assigned detections satisfy the rule. Their median gap to the own pose wrist is 2.1%,
  the maximum is 8.1%, and the minimum margin over the other wrist is 4.7% of the image height.

**Committed alternatives:**

- The frames with an unassigned detection can be read from the clips' `unassigned` array:
  - 01991 frames 14, 15, 23, 25, 27, 32 and 34, each with the right hand assigned;
  - 01988 frames 28 and 30, with the right hand assigned;
  - 01988 frames 126 and 127, with the left hand assigned;
  - 01988 frame 128, with no hand assigned.
- Task 7 recomputed the gaps and margins of the assigned detections from the clips' image
  landmarks: median 2.10%, maximum 8.12%, minimum margin 4.72% of the image height.
- The [probes](V1_2_Feasibility_Probes.md) ("Duplicates") describe the same-side pairs as duplicate
  detections of one hand in 01991, and as overlapping hands at the frame edge in 01988.

## 7. Extraction: clip order and determinism (Task 5)

On commit `3be7fa3` (2026-10-06), the Task 5 implementer re-extracted all five clips in reverse
order into a scratch directory. The arrays came out bit-identical, with the same dtypes and
metadata, and the `.npz` file bytes were identical to the committed files.

On commit `3238469` (2026-10-06), the Task 5 fix round ran altered copies of the extractor, kept
outside the repository, against the previous and the corrected clip-order test:

| Extractor under test | Previous test (01991 then 01992, compares 01992) | Corrected test (01992 then 01991, compares 01991) |
| --- | --- | --- |
| Unmodified copy | passed | passed |
| One `Hands` shared by all clips | passed (blind) | failed: `world` NaN locations differ |
| One `Holistic` shared by all clips | failed: `pose_world`, 5841 of 5841 elements differ | failed: `pose_world`, 4851 of 4851 elements differ |
| Both shared | failed: `pose_world`, 5841 of 5841 elements differ | failed: `world` NaN locations differ |

**Committed check:** `test_a_clip_does_not_depend_on_the_clip_before_it` is the corrected test. It
passed on 2026-10-06 (§9).

## 8. Measurement repeatability and run time (Task 6; checked again in Task 7)

On commit `6e1b594` (2026-10-06), the Task 6 fix round ran the measurement again, with two extra
rows. Against the JSON of the first run (commit `15db091`), every previously reported number was
bit-identical: the largest relative difference was 0.0. That covers the environment hashes, frame
counts, hand scales, every row of every track, the shapes, the chirality values and the pooled
table. The run took 895 s by the script's own timer, against 681 s in the first run, which fitted
fewer shapes.

Task 7 checked this against git history, and the final whole-branch review repeated the comparison
(both 2026-10-06). All 1,417 values that the first JSON
(`git show 15db091:data/test_output/hand_round_trip.json`) shares with the current one are equal,
under one row mapping: the first run's row `held-out shape` is the current row
`held-out shape (time halves)`, in the JSON `mano: held-out shape` and `smplx: held-out shape`,
now `mano: held-out shape (time halves)` and `smplx: held-out shape (time halves)`. The later run
added the interleaved row and gave the two held-out rows distinct names. Without this mapping, the
first run's `held-out shape` rows have no counterpart in the current file and count as missing. The
1,417 are 1,381 single numbers and the 36 elements of 18 two-element lists of shape norms. With the
mapping, the only keys of the first run now missing are its two replaced observability fields,
`median_weak_directions` and `weakest_direction_joint`.

**Committed alternative:** both JSON files are in git history, so anyone can repeat the comparison.

## 9. Checks run for the findings (Task 7, 2026-10-06)

- The four hand test modules (`tests/test_hand_model.py`, `tests/test_hand_converter.py`,
  `tests/test_hand_observations.py`, `tests/test_extract_hand_landmarks.py`): 68 passed in
  57.08 s, and none was skipped.
- The L-BFGS stop-reason check (§2) and the JSON history comparison (§8).
- The coverage, side-assignment and wrist-gap figures in findings §5 were recomputed from the
  committed clips with `load_hand_clip`.

These were read-only one-off checks; their scripts are not kept. The measurement and the extraction
were not re-run.
