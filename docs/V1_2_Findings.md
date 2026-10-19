# V1.2 findings: MediaPipe hands → MANO and SMPL-X → landmarks

**Date:** 2026-10-06. **Scope:** the feasibility slice P0–P2 and the P3 pilot measurement of the
V1.2 plan, built by the implementation plan
on branch `feat/v1.2-hands`. **Ledger:** V1.2 assumptions. **Earlier
measurements:** V1.2 feasibility probes.

*Plans, ledgers, one-off scripts, tests and evidence files named here without a link are kept on
the author's research branches and are not part of this repository.*

The round trip fits genuine MANO or SMPL-X parameters to each hand MediaPipe detects (21 world
landmarks, in metres), rebuilds the 21 landmarks from those parameters alone, and scores the
rebuilt landmarks against the observed ones. This document records what was measured on the five
pilot clips, what limits the error, and which assumptions the evidence settles.

**Sources.** Every number names where it comes from:

- *Round-trip JSON*: `data/test_output/hand_round_trip.json`, written by
  `scripts/measure_hand_round_trip.py` at commit `6e1b594`. Committed.
- *Hand clips*: `data/test_output/<clip>_hands.npz`, written by `scripts/extract_hand_landmarks.py`
  and read with `load_hand_clip`. Committed.
- *Tests*: `tests/test_hand_model.py`, `tests/test_hand_converter.py`,
  `tests/test_hand_observations.py` and `tests/test_extract_hand_landmarks.py`. On 2026-10-06 all
  68 passed and none was skipped (implementation evidence §9).
- *Probes*: V1.2 feasibility probes, measured before implementation.
- *Implementation evidence*: V1.2 implementation evidence, which
  keeps the measurements taken while the code was built: the synthetic gates, batch coupling, the
  check-2 diagnosis, extraction determinism and the checks run for these findings. Each comes with
  its task, procedure, commit and date. Cited as "implementation evidence §n". Where a test or the
  round-trip JSON covers the same ground, that is cited too.

Millimetre values from the measurement are rounded to two decimals, as the script prints them. No
value is rounded onto or across a target.

## 1. The answer

On the five pilot clips, one plausible hand shape per track misses the provisional mean and p95
targets on both backends: fitted on some frames of a track and scored on the others, it gives a
pooled mean error of 7.39–10.92 mm for MANO and 7.48–11.18 mm for SMPL-X (target ≤ 5 mm) and a
p95 of 20.93–33.25 mm and 23.01–35.16 mm (target ≤ 15 mm), while the fingertips meet their ≤ 8 mm
target at 3.28–6.83 mm and 3.56–7.33 mm. The fitter recovers model-generated hands to a mean of
0.007–0.016 mm against the 0.5 mm gate (refined fits, on both backends and both sides; the shape
fits pass the same gate on the right hand only), so as far as single-start fits can show, it is
unlikely to be what limits the real-data error. No multi-start reference was run on real hands, so
local minima there are not excluded (§4.1). The per-frame-shape diagnostic, which gives every frame
its own plausible shape and is not an achievable pipeline result, gets inside all three targets:
4.41 mm mean, 2.11 mm tips and 11.67 mm p95 for MANO, and 4.49, 2.16 and 12.31 mm for SMPL-X.

Sources: round-trip JSON, `pooled` (§2). The synthetic values are in implementation evidence §1,
and `test_recovers_hands_the_model_can_represent` asserts the gate they pass (§4.1).

## 2. Error ladder

**Metric.** The *wrist-relative error* is the plan's primary metric (plan §6): for each frame and
each of landmarks 1–20, the distance in millimetres between the predicted and the observed
position, each measured from its own wrist, with no rotation or scale alignment. The wrist is zero
by construction and is left out.

**Terms.** A *track* is one hand of one clip; there are nine, because 01991 has no left hand. A
*track-backend* is one track fitted with one backend; there are 18. A *frame-side* is one hand in
one frame; there are 535, and every row below is scored on the same 535. *Mean* and *p95* (the
95th percentile) are taken over all 535 × 20 landmark errors. *Tips* is the mean over the five
fingertips. *% of hand* divides each error by its track's median wrist-to-middle-MCP length
(68.9–99.9 mm, from the track's own frames) before averaging. The *equal-track mean* is the mean
of the nine per-track means, so each track counts once (`track_mean_mm`).

**Rows.**

| Row | What it is |
| --- | --- |
| swing start | The analytic start only: a palm rotation fitted to the five palm landmarks (the thumb CMC and the four finger MCPs), then each finger bone swung onto its observed direction. Neutral shape. |
| refined | From the swing start, L-BFGS (a quasi-Newton optimiser) on all 48 rotation values (3 wrist, 45 finger) through the real model. Neutral shape. |
| held-out shape (time halves) | One shape is fitted on the first half of the track in time and scored on the second half, then the reverse. *Pessimistic*: MediaPipe's drift in hand size between the halves counts as error. |
| held-out shape (interleaved frames) | One shape is fitted on the even-indexed frames and scored on the odd-indexed ones, then the reverse. *Optimistic*: every scored frame neighbours frames the shape was fitted on. |
| in-sample shape | One shape is fitted on the frames it is scored on. |
| per-frame shape (diagnostic) | Every frame gets its own shape: 10 shape values on top of 48 rotation values, against 60 observed values per frame. *Not an achievable pipeline result.* |
| rigid reference | Model-free: a rigid palm template per track, anchored at the wrist, with each finger bone at its median length along its observed direction. A reference, not a lower bound. |
| rigid reference, per-frame scale | The same, scaled in each frame by the least-squares factor against that frame's observations. *An oracle*: the scale is chosen from the observations it is scored on, so this is not an achievable result either. |

A shape is the model's 10 shape coefficients, its *betas*. Every fitted shape stays inside the
*plausibility bound*: |betas| ≤ 4.2787, the radius that holds 95% of the shape prior N(0, I) for
10 betas (`plausible_radius`; `test_plausible_radius_is_the_chi_square_quantile`).

Source for both tables: round-trip JSON, `pooled`. Values are in millimetres, except % of hand.
The rigid references do not use a model, so both tables show the same rigid-reference values.

### MANO

| Row | Mean | Tips | p95 | % of hand | Equal-track mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Provisional target* | *≤ 5* | *≤ 8* | *≤ 15* | | |
| swing start | 19.65 | 22.45 | 46.56 | 23.56 | 19.48 |
| refined | 8.25 | 3.85 | 24.15 | 9.79 | 8.13 |
| held-out shape (time halves) | 10.92 | 6.83 | 33.25 | 13.14 | 10.44 |
| held-out shape (interleaved frames) | 7.39 | 3.28 | 20.93 | 8.84 | 7.24 |
| in-sample shape | 7.38 | 3.28 | 20.89 | 8.83 | 7.24 |
| per-frame shape (diagnostic; not achievable) | 4.41 | 2.11 | 11.67 | 5.22 | 4.44 |
| rigid reference (model-free) | 14.69 | 16.79 | 35.90 | 17.63 | 14.02 |
| rigid reference, per-frame scale (oracle) | 5.27 | 6.67 | 13.07 | 6.31 | 5.34 |

### SMPL-X

| Row | Mean | Tips | p95 | % of hand | Equal-track mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Provisional target* | *≤ 5* | *≤ 8* | *≤ 15* | | |
| swing start | 24.06 | 26.52 | 58.34 | 29.07 | 23.95 |
| refined | 8.59 | 2.43 | 34.93 | 10.32 | 8.60 |
| held-out shape (time halves) | 11.18 | 7.33 | 35.16 | 13.43 | 10.62 |
| held-out shape (interleaved frames) | 7.48 | 3.56 | 23.01 | 8.94 | 7.32 |
| in-sample shape | 7.47 | 3.55 | 22.87 | 8.93 | 7.31 |
| per-frame shape (diagnostic; not achievable) | 4.49 | 2.16 | 12.31 | 5.31 | 4.50 |
| rigid reference (model-free) | 14.69 | 16.79 | 35.90 | 17.63 | 14.02 |
| rigid reference, per-frame scale (oracle) | 5.27 | 6.67 | 13.07 | 6.31 | 5.34 |

How to read the tables:

- The two held-out rows bracket how a fitted shape generalises. For the mean, the bracket is
  7.39–10.92 mm (MANO) and 7.48–11.18 mm (SMPL-X). Both ends miss the mean and p95 targets on
  both backends, and both ends meet the tips target.
- The interleaved row lands within 0.04 mm of the in-sample row on every track (§4.4), so in
  practice the lower end of the bracket is the in-sample result.
- The neutral `refined` fit (8.25 mm and 8.59 mm) lies inside the bracket.
- Only the per-frame-shape diagnostic gets inside all three targets. The per-frame-scale oracle
  meets the tips and p95 targets but not the mean (5.27 mm).

## 3. Per finger and per track

Pooled mean per finger, in mm, for both held-out rows. A finger's mean is taken over its four
landmarks. For the four fingers these are the knuckle at the base (MCP), the middle and the end
joint (PIP, DIP) and the tip; for the thumb they are its base joint (CMC), MCP, end joint (IP) and
tip. Source: round-trip JSON, `pooled`, `<finger>_mm` and `tips_mm`.

| Finger | MANO, time halves | MANO, interleaved | SMPL-X, time halves | SMPL-X, interleaved |
| --- | ---: | ---: | ---: | ---: |
| Thumb | 10.78 | 7.88 | 11.15 | 8.05 |
| Index | 11.51 | 7.73 | 11.73 | 7.74 |
| Middle | 11.53 | 7.56 | 11.91 | 8.02 |
| Ring | 11.51 | 7.57 | 11.22 | 7.15 |
| Pinky | 9.28 | 6.21 | 9.87 | 6.44 |
| *Fingertips (all five)* | *6.83* | *3.28* | *7.33* | *3.56* |

No finger stands out. In each held-out row, the thumb, index, middle and ring means lie within
1 mm of one another, and the pinky is the lowest. The worst finger is the **thumb** on
interleaved frames (MANO 7.88 mm, SMPL-X 8.05 mm) and the **middle finger** on time halves (MANO
11.53 mm, SMPL-X 11.91 mm). In each case the margin over the next finger is 0.02–0.17 mm. The
fingertips are fitted much better than the landmarks as a whole: 3.28 mm against 7.39 mm for MANO
on interleaved frames.

Per-track mean, in mm, for both held-out rows. *Hand size* is the track's median distance from
the wrist to the middle-finger MCP. *half_span_ratio* is that median over the second half of the
track in time, divided by the same median over the first half; 1.00 means MediaPipe kept the
hand's size. Source: round-trip JSON, `tracks`.

| Track | Frame-sides | Hand size (mm) | half_span_ratio | MANO, time halves | MANO, interleaved | SMPL-X, time halves | SMPL-X, interleaved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01986 left | 48 | 96.9 | 1.05 | 4.96 | 4.92 | 5.12 | 5.06 |
| 01986 right | 52 | 68.9 | 0.66 | 12.25 | 8.41 | 12.33 | 8.36 |
| 01987 left | 49 | 87.5 | 1.55 | 17.55 | 10.70 | 17.96 | 10.99 |
| 01987 right | 27 | 82.7 | 1.14 | 8.33 | 7.55 | 8.21 | 7.53 |
| 01988 left | 92 | 92.7 | 1.29 | 8.05 | 6.15 | 8.32 | 6.23 |
| 01988 right | 101 | 82.5 | 0.61 | 16.02 | 9.74 | 16.86 | 10.00 |
| 01991 right | 49 | 99.9 | 1.04 | 4.21 | 3.62 | 4.44 | 3.62 |
| 01992 left | 59 | 81.0 | 0.84 | 8.35 | 5.75 | 8.56 | 5.88 |
| 01992 right | 58 | 80.1 | 0.78 | 14.23 | 8.38 | 13.78 | 8.16 |
| *Equal-track mean* | | | | *10.44* | *7.24* | *10.62* | *7.32* |

The worst track is **01987 left**, on both held-out rows and both backends: 17.55 and 17.96 mm on
time halves, and 10.70 and 10.99 mm on interleaved frames. MediaPipe's estimate of that hand grows
by 55% between the halves of the clip (half_span_ratio 1.55), the largest drift of any track. The
next worst is 01988 right (16.02 and 16.86 mm; 9.74 and 10.00 mm), whose hand shrinks by 39%
(0.61). The best is 01991 right (4.21 and 4.44 mm; 3.62 and 3.62 mm), with a ratio of 1.04. It is
the only track under the 5 mm mean on both rows and both backends. 01986 left meets that target
for MANO only (4.96 and 4.92 mm).

## 4. What limits the error

The pooled mean falls in steps down the ladder. The steps are differences between rows, not
additive causes (round-trip JSON, `pooled`):

| Step | MANO | SMPL-X | What the step reflects |
| --- | ---: | ---: | --- |
| held-out shape (time halves) | 10.92 | 11.18 | The pessimistic end of the bracket. |
| → held-out shape (interleaved frames) | −3.53 | −3.70 | The cost of fitting the shape on the other half of the track in time. It tracks MediaPipe's drift in hand size (§4.2). |
| → in-sample shape | −0.01 | −0.01 | Nothing. A shape fitted on neighbouring frames is as good as an in-sample one (§4.4). |
| → per-frame shape (diagnostic) | −2.97 | −2.98 | What one shape per frame absorbs: frame-to-frame size change, and more (§4.2). |
| per-frame shape (diagnostic) | 4.41 | 4.49 | Not separated: MediaPipe's non-rigid hand, model mismatch and detector noise (§4.3). |

### 4.1 Optimiser: not the limit on synthetic evidence

- On model-generated hands the same fitter meets the synthetic gates (mean ≤ 0.5 mm, p95 ≤ 1 mm).
  Refined fits reach a mean of 0.007–0.016 mm and a p95 of at most 0.051 mm, on both backends and
  both sides (implementation evidence §1; `test_recovers_hands_the_model_can_represent` asserts
  the gate). Shape fits reach a mean of 0.069 mm on MANO and 0.108 mm on SMPL-X, on the right hand
  only (implementation evidence §3; `test_a_fitted_shape_reproduces_a_plausible_hand`). The
  per-frame-shape fit passes the same gate on the right hand only, on both backends
  (`test_per_frame_shapes_follow_a_hand_that_changes_size`). On the real hands the `refined` row is
  8.25 mm and 8.59 mm, more than 500 times larger. All of these are single-start fits.
- Refinement lowers the swing start's mean on all 18 track-backends (round-trip JSON, `tracks`).
- The fitter minimises the summed squared wrist-relative error, plus a pose prior of
  1e-4 mm² per rad² on the finger rotations. A fitted shape includes the neutral one (zero betas
  lie inside the bound), so fitting a shape should never raise that summed squared error; the
  check is stated in those terms. The mean distance can still rise, and it did on 2 of 18
  track-backends: 01986 right on MANO (8.38 → 8.40 mm) and 01988 right on SMPL-X
  (9.76 → 10.00 mm). On those two, the summed squared error fell by 24.1% and 23.6%. The shape
  lowered the worst frames (worst-frame RMS 21.6 → 17.6 mm and 39.0 → 32.8 mm) and made 15 of 52
  and 36 of 101 frames worse (implementation evidence §5). The round-trip JSON shows the same
  trade: on those two, the p95 fell from 31.23 to 24.00 mm and from 42.37 to 33.05 mm while the
  mean rose (`tracks`). These are not optimiser failures.
- Batched L-BFGS couples the frames of a batch. On the synthetic gate data (right hand, random
  poses), a frame's reconstruction in a batch of 12 differs from its reconstruction in a fit on its
  own by at most 0.134 mm (MANO) and 0.139 mm (SMPL-X) in any single coordinate. A landmark moves
  by up to 0.172 mm (MANO) and 0.181 mm (SMPL-X) as a Euclidean distance, with a mean of 0.011 and
  0.015 mm (implementation evidence §2, which records where each figure comes from). The runs end
  at the 1,000-iteration cap: every run checked, synthetic and real, used all 1,000 iterations
  (implementation evidence §2). Status `ok` only means the result is finite (`fit_hands`), so it
  is not a convergence certificate; each fit's `rms_mm` is the quality signal. These effects are
  small next to the 7–11 mm measured. No multi-start reference was run on the real hands, so local
  minima on real frames are not excluded.

### 4.2 MediaPipe's change of hand size: within a clip and from frame to frame

**Within a clip.** MediaPipe's estimate of a hand's size drifts during a clip: half_span_ratio
runs from 0.61 to 1.55, and five of the nine tracks drift by more than 20%. Define the
*time-halves penalty* as the time-halves mean minus the neutral `refined` mean. Over the nine
tracks, the penalty correlates with the drift, |half_span_ratio − 1|, at r = 0.89 for MANO and
r = 0.73 for SMPL-X. Measured against the interleaved row instead of the neutral fit, the
correlation is 0.84 and 0.85. The penalty therefore tracks drift and is not a sign of shape
overfitting: on interleaved frames, whose neighbours share their size, the shape scores within
0.04 mm of the in-sample shape on every track (§4.4). Source: round-trip JSON, `tracks`; the
correlations are computed from it.

| Track (by drift) | half_span_ratio | MANO penalty (mm) | SMPL-X penalty (mm) |
| --- | ---: | ---: | ---: |
| 01991 right | 1.04 | −2.17 | +0.51 |
| 01986 left | 1.05 | −1.84 | −0.74 |
| 01987 right | 1.14 | +0.45 | −1.76 |
| 01992 left | 0.84 | +1.37 | −1.01 |
| 01992 right | 0.78 | +5.35 | +5.61 |
| 01988 left | 1.29 | +1.27 | +0.89 |
| 01986 right | 0.66 | +3.87 | +1.04 |
| 01988 right | 0.61 | +5.62 | +7.10 |
| 01987 left | 1.55 | +6.84 | +6.56 |

MediaPipe's size estimate also differs between the two hands of one clip. In 01986 it gives the
left hand a median wrist-to-middle-MCP length of 96.9 mm and the right hand 68.9 mm, a ratio of
1.41 (round-trip JSON, `hand_scale_mm`).

**From frame to frame.** The probes measured a per-frame palm scale with a 5th–95th percentile
range of 0.69–1.31, and a median within-clip coefficient of variation of 15.9% for bone lengths.

- Only the 1-DoF (one value per frame) scale reference isolates size change. Scaling the
  model-free rigid reference by one number per frame cuts its pooled mean from 14.69 to 5.27 mm,
  a 64% reduction. That scale is an oracle, chosen from the observations it is scored on.
- Giving each frame its own shape cuts the model error from 7.38 to 4.41 mm (MANO) and from 7.47
  to 4.49 mm (SMPL-X). The cut happens on all 18 track-backends. This includes frame-to-frame size
  change, but also anything else that 10 more values per frame can absorb: 58 free values against
  60 observed ones. So it is not a measure of size change alone. The model ladder has no
  scale-only row, so the part due to size alone is not measured for the models.
- The per-frame diagnostic's floor on synthetic hands is 0.10–0.17 mm (implementation
  evidence §4), far below these differences.

### 4.3 Model mismatch and MediaPipe's non-rigid hand: the remaining 4–5 mm, not separable

- MediaPipe's hand is not rigid. After a rigid fit, the palm keeps a 6.6 mm mean residual. Bone
  lengths vary within a clip by a median coefficient of variation of 15.9%; the worst bone is the
  ring finger's MCP→PIP, at 43% (Probes). A model with one shape per track cannot follow
  proportions that change from frame to frame, and a shape per frame follows them only as far as
  10 bounded shape values allow.
- Per-frame freedom leaves 4–5 mm: 4.41 mm (MANO) and 4.49 mm (SMPL-X) with a shape per frame,
  and 5.27 mm on the rigid reference with the per-frame-scale oracle.
- The models do better than the rigid reference, both per track and per frame. The in-sample
  shape scores 7.38 and 7.47 mm against the rigid reference's 14.69 mm, and the per-frame shape
  4.41 and 4.49 mm against the scale oracle's 5.27 mm. So the hand models cost no more than a
  rigid hand built from each track's own proportions. The rigid reference is not a lower bound,
  though, so this does not bound the mismatch.
- The backend barely matters. On every shape-fitted row, MANO and SMPL-X differ by 0.08–0.25 mm in
  pooled mean, although SMPL-X's template palm vectors are 1.13–1.32 times as long as MANO's
  (Probes). The remaining error is therefore not specific to one model's template.
- The fingertips are fitted far better than the landmarks as a whole: 3.28 against 7.39 mm for
  MANO and 3.56 against 7.48 mm for SMPL-X, on interleaved frames. This is consistent with the
  structure of the fit, though not proven by it: every rotation along a finger moves its tip,
  while the five palm landmarks share only the wrist rotation and the shape, so a palm that changes
  shape cannot be followed.
- With a shape per frame, the thumb is the worst finger on both backends. MANO's thumb is at
  5.18 mm and its other fingers at 4.15–4.31 mm; SMPL-X's thumb is at 5.99 mm and its other
  fingers at 3.67–4.73 mm (round-trip JSON, `pooled`). The cause could be a thumb correspondence
  offset or a thumb the models cannot represent. Per-landmark residuals were not analysed, so
  assumption A007 (landmark correspondence) stays open.
- With no ground truth, no renders and no alternative landmark adapter, this evidence cannot
  separate model mismatch from detector noise.

### 4.4 Shape: bound-limited, and only "a plausible shape"

- Every fitted shape sits at 99.57–99.97% of the plausibility bound. That covers all 90 norms:
  9 tracks × 2 backends × 5 fits (two time halves, two interleaved halves and one in-sample fit).
  Source: round-trip JSON, `shape`. The observations pull every shape to the edge of the
  plausible region, so the shape fits are bound-limited.
- On every track, the interleaved row lands within 0.04 mm of the in-sample row: the difference is
  +0.00 to +0.03 mm on MANO and −0.02 to +0.04 mm on SMPL-X. A shape fitted on half the frames
  serves their neighbours as well as one fitted on all of them, so adjacent frames show no sign of
  overfitting. Across the time halves the shape does worse, and that loss tracks drift (§4.2).
- A fitted shape improves on the neutral one. The pooled in-sample mean is 7.38 mm against the
  `refined` 8.25 mm on MANO, and 7.47 against 8.59 mm on SMPL-X. By mean distance the shape
  improves 16 of 18 track-backends; on the other two it improves the summed squared error (§4.1).
- SMPL-X's 10 betas shape the whole body. Fitted from hand landmarks alone, they give a plausible
  shape that reproduces the landmarks, not an estimate of body shape. SMPL-X betas are not
  identifiable from hand landmarks. On synthetic hands generated with a shape of norm 2.0, the
  SMPL-X fit lands at |betas| = 4.13, 4.50 from the true vector, while it reproduces the landmarks
  to a mean of 0.108 mm. MANO's fit lands at 2.38, 1.06 from the true vector, so MANO's betas are
  not recovered either (implementation evidence §3).

### 4.5 Observability: the rotations are not unique; the landmark error is unaffected

A rotation direction is *weak* at threshold t when its singular value, in the Jacobian of the
20 wrist-relative landmarks with respect to the 48 rotation values, is below t times the largest.
The Jacobian was sampled on every fifth fitted frame: 111 frames per backend, 6–21 per track. The
table gives the median over tracks of each track's median weak count, with the range over tracks
in brackets. Source: round-trip JSON, `observability` and `observability_settings`.

| Threshold (s / s_max) | MANO | SMPL-X |
| --- | --- | --- |
| 1e-6 | 10 (10–10) | 10 (10–10) |
| 1e-3 | 14 (12.5–14) | 14 (13.5–15) |
| 1e-2 | 15 (15–15) | 15 (15–15) |
| 3e-2 | 25 (22–26) | 25 (23.5–26.5) |

- Exactly 10 directions are numerically null on every track and both backends: the 21 landmarks
  do not determine those rotations at all. Beyond them, the weak count depends on the threshold,
  so there is no natural number of "weak" directions.
- At 1e-2, the weak subspace spreads over all 15 finger joints. Weighted by frames, each joint
  holds 0.64–1.28 (MANO) or 0.63–1.28 (SMPL-X) of the roughly 15 weak directions
  (`joint_weak_weight`). The five heaviest joints are, on MANO, index DIP (1.28), ring DIP (1.25),
  middle DIP (1.22), pinky PIP (1.21) and thumb IP (1.16), and on SMPL-X, index DIP (1.28), ring
  DIP (1.23), middle DIP (1.19), thumb IP (1.18) and pinky DIP (1.18). The three lightest are the
  index, ring and pinky MCPs on both backends (0.64–0.65 on MANO, 0.63–0.66 on SMPL-X). Each
  value is the mean of the joint's per-track `joint_weak_weight` over the nine tracks, each track
  weighted by the number of frames its Jacobian was sampled on (`observability.frames`: 6–21 per
  track, 111 in all); the ranges above are the smallest and largest of these means. The wrist
  holds none: its weight is below 4e-5 on every track. The most loaded joint changes from track to
  track, so no single joint is "the weakest".
- Null directions do not move the landmarks, so they add no landmark error. The pose prior
  settles them. The fitted rotations are therefore one solution among many: the round trip does
  not recover unique joint rotations (plan §1; assumption A010).

### 4.6 Side assignment: fit quality cannot check it

- Each track was also fitted with the other side's model. On 16 of 18 track-backends, that raises
  the neutral `refined` mean only by a factor of 1.02–1.31. Only 01991 right separates clearly: on
  MANO, 6.38 mm with its own side against 9.98 mm with the other (×1.56), and on SMPL-X, 3.94
  against 7.29 mm (×1.85). Source: round-trip JSON, `chirality_mean_mm`.
- So a swapped side would barely change the error, and fit quality cannot verify which hand is
  which. The sides rest on the geometric assignment (§5). That assignment is consistent on the
  pilot, but it has no labelled ground truth.

## 5. Data

Per-clip coverage, recomputed from the hand clips with `load_hand_clip`. *No hand* counts frames
where `Hands` detected no hand. *Left* and *Right* count the frames in which a detection was
assigned to that side. *Label agrees* counts assigned detections whose handedness label, read as
naming the opposite hand, matches the assigned side.

| Clip | Video | Frames | No hand | Left | Right | Unassigned | Label agrees |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01986 | 320×240, 25.00 fps | 83 | 31 | 48 | 52 | 0 | 100.0% (100/100) |
| 01987 | 1280×720, 29.97 fps | 75 | 24 | 49 | 27 | 0 | 90.8% (69/76) |
| 01988 | 640×480, 30.00 fps | 144 | 40 | 92 | 101 | 5 | 100.0% (193/193) |
| 01991 | 288×192, 29.97 fps | 49 | 0 | 0 | 49 | 7 | 100.0% (49/49) |
| 01992 | 288×192, 29.97 fps | 59 | 0 | 59 | 58 | 0 | 100.0% (117/117) |
| All | | 410 | 95 | 248 | 287 | 12 | 98.7% (528/535) |

- **Detector.** Hand world landmarks come from legacy `mp.solutions.hands.Hands` (MediaPipe
  0.10.21): video mode, at most two hands, model complexity 1, detection and tracking confidence
  0.5. A fresh instance runs per clip, next to a fresh `Holistic` for the pose (hand clips,
  `meta`). It finds no hand in 95 of the 410 frames (23%), one hand in 83 and two in 232: 547
  detections. `Holistic` sees the left hand in 265 frames and the right in 304, but it gives no
  hand world landmarks (Probes).
- **Side rule.** A detection takes the side of the nearest pose wrist (pose landmark 15 is the
  left wrist, 16 the right) under three conditions. Its image wrist must be within 10% of the
  image height of that pose wrist. It must be at least 3% of the height nearer to that wrist than
  to the other. And that side must not be taken yet. Otherwise the detection stays unassigned.
  Detection order and the detector's label play no part.
- **Assigned: 535** (248 left, 287 right). Each assigned detection's wrist sits a median 2.1% of
  the image height from its pose wrist, with a maximum of 8.1%. Each is at least 4.7% of the
  height nearer to its own pose wrist than to the other (hand clips). 01991 has no left hand in
  any frame.
- **Unassigned: 12.** Nine are a second detection at a side already taken. The seven in 01991 are
  duplicate detections of one hand. The other two are in 01988, at frames 28 and 30 by the clips;
  the probes describe that pair as overlapping hands at the frame edge
  (probes, "Duplicates"). The extraction does not store unassigned
  detections, so the 01988 pair is not re-checked here. The last three lie beyond the 10% cap:
  01988 frames 126–128, with gaps of 10.3%, 11.5% and 13.8% of the height. The clips record which
  frames hold an unassigned detection, but not why; the reasons come from the Task 5 review and the
  probes (implementation evidence §6). The rule's constants come from the implementation plan and
  were not tuned after extraction.
- **Label.** The legacy label assumes mirrored (selfie) input, so on this unmirrored video it names
  the opposite hand. Read that way, it agrees with the geometric side on 528 of 535 assigned
  detections. All seven disagreements are in 01987.
- **Coverage accounting (plan §6).** Detected: 547. Eligible (assigned a side): 535. Fitted: 535.
  Failed: 0. Imputed: 0. *Fitted* means the neutral fit returned a finite result (status `ok`);
  *failed* counts the frames whose neutral fit had any other status (`failed_neutral_fits` is 0 on
  all 18 track-backends). Neither is a convergence count: every run checked stopped at the
  iteration cap, and `rms_mm` is the quality signal (§4.1). Every ladder row is scored on the same
  535 frame-sides; no frame was dropped.
- **Units.** The landmarks are in metres. The per-clip median wrist-to-middle-MCP length is
  80.3–99.9 mm (hand clips). These "metres" are MediaPipe's estimates, not calibrated
  measurements.
- **Hand frame against pose frame.** The probes found that hand-world axes equal pose-world axes
  (x right, y down, z away from the camera), so no basis change is applied. The origins differ:
  each hand's world origin is its own, near the centre of the knuckle line and 79–98 mm from the
  wrist (Probes), while the pose's is at the hips (`HandClip`). The implementation relies on the
  axes finding without testing it again, and nothing was attached to a body (P4).
- **Determinism.** Extraction is independent of clip order
  (`test_a_clip_does_not_depend_on_the_clip_before_it`, which fails when a detector is shared
  between clips; implementation evidence §7). Re-extracting all five clips in reverse order
  reproduced the committed files byte for byte (implementation evidence §7).

## 6. Assumptions

Status changes in the ledger, all dated 2026-10-06. *Partly settled* means
part of the claim was tested; the evidence column names that part.

| ID | Was | Now | Evidence |
| --- | --- | --- | --- |
| A002 assets run | likely / open | confirmed / **verified** (gradients: right hand only) | `test_points_equal_the_full_model_forward` (both backends and sides, within 1e-9 m of the full smplx forward) passes on CPU, and so does `test_gradients_are_exact` (both backends, right hand only); the asset SHA-256 are in the round-trip JSON. |
| A003 metric hand data | likely / open | confirmed / **verified** (legacy `Hands`) | 547 detections on the five clips, and every assigned hand has 21 finite world points in metres (hand clips); `test_hands_are_metric_sided_and_missing_as_nan` passes. |
| A004 sides, masks, state | unknown / open | likely / **open** (partly settled) | Settled: clip independence (`test_a_clip_does_not_depend_on_the_clip_before_it`), NaN masks and the side rule (`TestAssignSides`), and consistent sides on the pilot (§5). Open: no labelled sides, no crossing-hand fixture, and fit quality cannot check sides (§4.6). |
| A005 hand-to-body frame | unknown / open | likely / **open** (partly settled) | Settled by the probe: hand-world axes equal pose-world axes (Probes, "Axes"). Untested: attachment (P4). |
| A006 conventions | risky / open | confirmed / **verified** (numerical; no renders) | `test_points_equal_the_full_model_forward`, `test_zero_pose_is_the_flat_template_hand`, `test_landmarks_follow_hand_anatomy`, `test_tips_extend_their_distal_bone`, `test_left_hand_is_the_mirror_of_the_right` and `test_one_joint_moves_only_its_own_finger` pass on both backends. |
| A008 parameter-only decoder | unknown / open | confirmed / **verified** (fresh-process test: MANO, right hand) | `test_a_fresh_process_rebuilds_the_hands_from_saved_parameters`: a new process given only the saved parameters and the model rebuilds the hands to within 1e-12 m. That test runs on MANO's right hand only. `reconstruct_hand` takes only a fit and a model for every backend and side, but no fresh-process test covers SMPL-X or the left hand. |
| A009 synthetic inversion | unknown / open | confirmed / **verified** (synthetic; single start; shape fits on the right hand only) | `test_recovers_hands_the_model_can_represent` (mean 0.007–0.016 mm; both backends and sides) and `test_a_fitted_shape_reproduces_a_plausible_hand` (0.069 and 0.108 mm; both backends, right hand only) pass (implementation evidence §1, §3). |
| A010 observability | risky / open | risky / **open** (partly settled) | Rank measured: 10 of 48 rotation directions are null on every track (round-trip JSON, `observability`), so the rotations are not unique and a pose prior settles them. Untested: multi-start and occlusion. |
| A011 real fit ≤ 5 mm | unknown / open | **broken** on the pilot | Held-out mean 7.39–10.92 mm (MANO) and 7.48–11.18 mm (SMPL-X) against ≤ 5 mm, and p95 20.93–35.16 mm against ≤ 15 mm (round-trip JSON, `pooled`). No sealed test set and no multi-start reference. |
| A012 shared shape generalises | unknown / open | risky / **open** (partly settled) | Every fitted shape is at the bound (99.57–99.97%). The time-halves shape is worse than the neutral one on 7 of 9 (MANO) and 6 of 9 (SMPL-X) tracks, tracking drift (r = 0.89 and 0.73). The interleaved shape is within 0.04 mm of in-sample (round-trip JSON). |
| A018 corpus coverage | risky / open | **broken** for the current five clips | 5 clips of 4 people ([V1 scope, Finding 10](V1_Problem_Scope.md)), 9 hand tracks, 535 frame-sides, and no left hand in 01991: below the plan §8 minimum of 20 clips and 5 people. |
| A019 reproducibility | likely / open | likely / **open** (partly settled) | Package versions and the SHA-256 of the assets and hand clips are in the round-trip JSON, and re-runs reproduce bit for bit on this machine (implementation evidence §7, §8). Not done: a clean-environment replay. |

The evidence also bears on A007, the 21-point landmark correspondence, but its status stays risky
/ open. The adapter matches each model's skeleton (`test_landmarks_follow_hand_anatomy`,
`test_tips_extend_their_distal_bone`). On the pilot, the thumb is the worst finger once each frame
has its own shape (§4.3). Per-landmark residuals and alternative adapters were not examined. This
slice did not test A001 or A013–A017, and their status is unchanged.

## 7. Limits

- **Corpus.** The pilot is five clips showing four people: 01991 and 01992 show the same signer
  ([V1 scope, Finding 10](V1_Problem_Scope.md)). The data carries no signer labels. There are
  9 hand tracks and 535 frame-sides, and 01991 has no left hand. There is no sealed test set, so
  every number here is a development result. The held-out rows hold out frames within a track,
  not clips or signers.
- **No wrist attachment (P4).** Only the local, wrist-relative reconstruction is measured. SMPL-X
  hands were fitted with every other body joint at rest, so nothing here speaks to the SMPL-X body
  or the V1 arm.
- **No temporal model (P5).** Every frame is fitted on its own.
- **Untested fitting aspects.** There is no multi-start best-found reference, so local minima on
  real frames are not excluded. There are no renders, so the fitted hands were not inspected for
  collapsed fingers, extreme twists or mesh intersections (plan §5.1 asks for this). The loss is
  squared error only, with no robust loss.
- **Reporting gaps.** The measurement reports no per-finger p95, no maximum errors, no per-side
  pooled values and no bootstrap intervals, although plan §6 asks for them.
- **What a small round-trip error means (plan §1).** It means agreement with MediaPipe's
  observations. It does not establish the true anatomy, a calibrated physical accuracy, unique
  rotations (§4.5 shows that they are not unique), or that the meaning of a sign is preserved.

## 8. Next steps

1. **P4: attach the hands to the V1 arm.** Set the SMPL-X wrist's local rotation to
   `R_parentᵀ R_wrist`, where `R_parent` comes from forward kinematics of the frozen V1 body. Then
   test the seam as plan §6 specifies: the hand's wrist equals the body's wrist within 0.1 mm, and
   the V1 arm outputs stay unchanged within 0.1 mm. P4 needs the V1 body converter, with its shape
   fit, to be on the same branch as this hand code. P4 would settle the attachment half of A005
   and test A013.
2. **Grow the corpus, as plan §8 sets out.** Collect at least 20 clips across at least five
   identified people, covering both sides, a range of hand sizes, poses, occlusion and aspect
   ratios. Split by signer, freeze the test IDs before any tuning, and keep a sealed final test
   set (P5b). A018 and A016 need this, and so does any general claim about A011.
3. **Only then, augmentation and training.** No augmentation or classifier experiment should rely
   on these hand fits before P4 and the larger corpus exist.

Plan §6 also asks for the release thresholds to be frozen at the end of P3, before any P4 or P5
tuning, with a decision record if the thresholds are infeasible. This document supplies the
achieved values and the limiting factors. It changes no target.

## 9. Reproduce

From the repository root, with the licensed model files in place. These files are not in git:
`models/mano/MANO_LEFT.pkl` and `models/mano/MANO_RIGHT.pkl` (from the official MANO release,
`mano_v1_2.zip`, where they are `models/MANO_{LEFT,RIGHT}.pkl`) and
`models/smplx/SMPLX_NEUTRAL.npz`. The code loads MANO from `.npz` files, which the release does not
contain: step 1 builds `models/mano/MANO_LEFT.npz` and `models/mano/MANO_RIGHT.npz` from the
`.pkl` files. The SHA-256 of the three `.npz` files are recorded in the round-trip JSON
(`environment.assets`).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,smplx]'    # dev pins mediapipe 0.10.21; smplx installs torch and smplx

# 1. MANO assets: the official .pkl files -> models/mano/MANO_{LEFT,RIGHT}.npz
python scripts/convert_mano.py

# 2. Hand landmarks: data/sign_videos/*.mp4 -> data/test_output/<clip>_hands.npz
PYTHONPATH=src python scripts/extract_hand_landmarks.py

# 3. The error ladder -> data/test_output/hand_round_trip.json (about 15 minutes on CPU)
PYTHONPATH=src python scripts/measure_hand_round_trip.py

# 4. The tests (about a minute; the extractor tests run MediaPipe on two clips)
PYTHONPATH=src python -m pytest tests/test_hand_model.py tests/test_hand_converter.py \
    tests/test_hand_observations.py tests/test_extract_hand_landmarks.py -q

# The pooled table
PYTHONPATH=src python -c "import json; r = json.load(open('data/test_output/hand_round_trip.json')); print(json.dumps(r['pooled'], indent=1))"
```

Step 1 needs only numpy and scipy (it reads the `.pkl` files without chumpy). Run it from the
repository root: it reads and writes `models/mano/` relative to the working directory, and it
leaves the `.pkl` files unchanged. It also applies the left-hand `shapedirs` fix
(vchoutas/smplx#48). The x shape component of `MANO_LEFT.pkl` equals the right hand's instead of
being mirrored, so the script negates it in the converted left model and prints
`Applied the MANO_LEFT shapedirs x-sign fix (vchoutas/smplx#48).` The left-hand results and
`test_left_hand_is_the_mirror_of_the_right` depend on this fix: with the same pose and shape, the
fixed left hand mirrors the right to within 0.006 mm, and the unfixed one is off by 32.4 mm
(Probes). Without step 1, every MANO test skips silently (`tests/hand_assets.py` finds no `.npz`
files, and pytest still exits with code 0) and the measurement script fails when it builds the MANO
model. Compare the SHA-256 of the three asset files with `environment.assets` in the round-trip
JSON. If they differ, the files are not byte for byte those behind these results, and the numbers
may differ.

Steps 2 and 3 overwrite committed files. On the machine that recorded them, the extraction
reproduced the committed clip files byte for byte, and a re-run of the measurement reproduced every
earlier number bit for bit; the measurement took 895 s (implementation evidence §7, §8). The
recorded environment was Python 3.11.15, numpy 1.26.4, scipy 1.17.1, torch 2.14.1 on CPU and smplx
0.1.28 (round-trip JSON), with MediaPipe 0.10.21 (hand clips, `meta`). The SMPL-X asset is version
1.1 by content, although `models/smplx/version.txt` says 1.0 (Probes).
