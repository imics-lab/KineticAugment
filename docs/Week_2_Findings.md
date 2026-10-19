# Week 2 findings: the round trip on unseen footage, and what its error is made of

**Date:** 2026-10-07. **Branch:** `feat/kinematics-and-ik`. **Ledger:** the new rows P21, P22, O8
and O9 in [V1 problem scope §12](V1_Problem_Scope.md#12-assumptions-ledger). **Earlier results:**
[V1 findings](V1_Findings.md) (body), [V1.2 findings](V1_2_Findings.md) (hands).

This week asked whether the current pipeline (MediaPipe → SMPL-X → MediaPipe for the arms,
MediaPipe → MANO → MediaPipe for the hands) holds up on clips it has never seen, and what its
remaining error actually consists of. It ends with an agreed plan for better error metrics and
for temporal smoothing.

**Sources.** Every number names where it comes from:

- *Evaluation report*: `data/round_trip_eval/round_trip_report.json`, written by
  `scripts/evaluate_round_trip.py` on 20 random clips. Not committed: it is derived from WLASL
  footage outside the repo (`~/Downloads/raw_videos`).
- *Pilot report*: the same script run on the five pilot clips (`data/sign_videos`). It reproduces
  the recorded pilot figures exactly: body 3.52% direct, 3.99% held out (ledger O1), and the hand
  rows of `data/test_output/hand_round_trip.json` (for example, clip 01988 left hand in-sample
  6.14 / 2.78 / 21.08 mm).
- *Videos*: `data/round_trip_eval/videos/<clip>_body_hand_round_trip.mp4`, rendered by
  `scripts/view_body_hand_round_trip.py`.
- *Decomposition analysis*: a one-off scratch analysis (not committed), described in §4. It is
  preliminary, and the metrics in §6 are meant to replace it.

## 1. New tools

| Script | What it does |
| --- | --- |
| `scripts/view_body_hand_round_trip.py` | Renders one clip: MediaPipe's 2-D output, body + hands as input vs round trip (front and side views), and the full SMPL-X mesh posed from the round-trip parameters alone, plus an error strip. `--video` renders footage stored anywhere. |
| `scripts/evaluate_round_trip.py` | One script for any footage. It extracts landmarks in a single MediaPipe pass, scores the body round trip as `measure_round_trip.py` does and the hands as `measure_hand_round_trip.py` does, writes a JSON report, and can render each clip with `--render`. |

**Stitching the body and hands.** MediaPipe reports each hand in its own metric frame, with axes
aligned to the pose's world frame. Each hand is attached by putting its wrist on the body's wrist:
the observed hand on the observed wrist, and the reconstructed hand on the reconstructed wrist.

**The body figure.** The mesh uses the body fit's orientation, pose and shape, and the MANO finger
pose as SMPL-X hand pose. The wrist rotation is solved so that the SMPL-X palm lines up with the
MANO reconstruction's palm. The mesh's wrists land on the round-trip wrists to 0.000 mm.

The figure's hands are about 12% too large: about 106 mm from wrist to middle knuckle, against
83–95 mm for the MANO fit and MediaPipe. This is because the body's shape is fitted from arm
lengths alone. It shows as a 15 mm median offset in the finger joints on clip 01988. That offset
is a size difference, not a pose error.

## 2. Does the pipeline generalise? Yes, within WLASL

**Sample.** 20 clips drawn at random (seed 20261007) from the 329 non-pilot clips in
`~/Downloads/raw_videos`. They range from 288×192 to 1920×1080 and from 24 to 106 frames, 1,236
frames in total. Before looking at the results, "matches" was defined as the clip-level bootstrap
95% CI on the difference of medians, together with the share of new clips that fall inside the
pilot's range.

**Body round trip** (% of shoulder width, medians):

| | Pilot (5 clips) | Random 20 |
| --- | --- | --- |
| Direct error, pooled | 3.52% | 3.13% |
| Held out, pooled | 3.99% | 3.37% |
| Rigid-skeleton floor, pooled | 2.36% | 1.91% |
| Per-clip range of the direct error | 2.67–4.40% | 1.44–5.52% |
| Difference of per-clip medians | | −0.22 pp, CI [−1.21, +1.16] |
| Upright, with the nose and with it masked | 409/409 | 1234/1234 |

**MANO hand round trip** (mm; the provisional targets are mean ≤ 5, tips ≤ 8, p95 ≤ 15):

| | Pilot (9 tracks) | Random 20 (26 tracks) |
| --- | --- | --- |
| In-sample shape: mean / tips / p95 | 7.38 / 3.28 / 20.89 | 5.10 / 2.53 / 14.59 |
| Held out (time halves): mean / p95 | 10.92 / 33.25 | 5.62 / 16.50 |
| Hand detected, share of side-frames | 65% | 42% |

**Verdict.** The round trip transfers: the body medians match the pilot within noise, and the
hands do *better* than on the pilot. The pilot hands were the hard case. On new footage the hands
sit at the target: tips and p95 pass, and the mean misses by 0.1 mm.

**Limits of the verdict.**

- The sample comes from the same WLASL pool, and some clips share a signer: 08917 and 08918 are the
  same signer doing the same sign twice. Other datasets are untested (ledger O9).
- The per-clip spread is wider than the pilot showed (ledger O8).
- Hand coverage is lower than on the pilot. Only 6 of the 20 clips have a scored left hand, so the
  hand result mostly describes the dominant hand.

## 3. The "worst" clip depends on the metric

| Ranked by | Worst clips |
| --- | --- |
| Body direct error, median | 05631 (5.52%), 28111 (5.34%), 08918 (5.16%) |
| Body direct error, p95 (spikes) | 63229 (20.3%), 05631 (20.0%), 12328 (18.4%) |
| Body gap to floor | 08917 (3.85 pp), 08918 (3.84 pp), 57949 (3.61 pp) |
| Hand mean (in-sample) | 69257 (9.31 mm), 09849 (8.33 mm) |

What the videos show:

- **08917 and 08918:** a steady offset of the right elbow, with a flat error strip at about 6–7%.
  The reconstruction looks good to the eye. Visually it is not a failure.
- **05631:** about 5% while the arm is down, and 9–10% while the hand is at the cheek. In the side
  view the elbow is displaced in depth.
- **63229:** about 1–2% while the arm is down (frames 1–17), and 9–10% while the hand is at the
  chin (frames 20–35). Its p95 comes from that whole phase, not from single-frame spikes.
- **69257:** the body is fine (about 3%). Both hands sit near 7–8 mm, with short spikes to about
  17 mm (frames 17 and 49).

A ranking by a single number therefore mixes up three different things: a body that cannot take
on the signer's proportions, an error that depends on the pose, and detector noise.

## 4. What the error is made of (preliminary)

The scratch analysis split each clip's error vectors (world frame, per joint) into a constant part
(the mean over the clip) and the remaining varying part. The table gives the share of squared error
that is constant:

| Clip | Body: constant share | Hand (right): constant share |
| --- | --- | --- |
| 05631 | 74% | 14% |
| 28111 | 72% | 18% |
| 63229 | 69% | 33% |
| 08917 | 61% | 45% |
| 69257 | 11% | 27% |
| 09849 | 18% | 31% |
| Pilot clips | 26–69% | 15–59% |

**Findings.**

1. **The highest-ranked body clips are mostly systematic error.** It is not instability, and it is
   why they look fine on video.
2. **"Constant" is the wrong word for it: the error depends on the pose.** On 63229 and 05631 it
   switches on when the elbow bends and the hand rises to the face. The scratch split counted it as
   constant only because these signers hold one pose for long stretches. The cause is the known
   mismatch between MediaPipe's arm proportions and the ones SMPL-X can represent (ledger P17).
   That mismatch shows more in some poses than in others. Metrics must separate *systematic*
   (shape-explained) error from *varying* (detector) error, not only constant from varying.
3. **Hand error mostly varies** (14–59% constant). The hands are limited by frame-to-frame
   detector change, not by a fixed offset.
4. **The conversion adds almost no jitter.** Measured as the median size of the second
   difference, the reconstruction moves frame to frame only 0.00–0.25 pp more than MediaPipe's
   input on every clip tested. Glitches in the output are the detector's, passed straight through.
5. **Out-of-range frames are rare.** 0–9 frames per clip lie more than 3 robust standard
   deviations above the clip's median error.

## 5. Consequences

- **Smoothing denoises MediaPipe; it does not repair the conversion.** Once it is added, the error
  against the raw input will *rise* by the noise it removes. Smoothing therefore has to be judged
  visually and by its own metrics (jitter removed, motion kept), not by the round-trip error alone.
- **Whether a clip is safe to augment** should depend on the varying error and on outliers, not on
  the total. A clip with a large systematic offset reconstructs as a plausible body with slightly
  different proportions (08917, 08918).

## 6. Agreed plan

The metrics are for both purposes (option C): first diagnostics that point at a cause, then
thresholds set from what those diagnostics show.

1. **Diagnostic metrics and a per-video deviation view, built together.**
   - Split each clip's error into systematic (shape-explained) error, varying (detector) error and
     conversion-added error.
   - Measure how far, and for how long, the error leaves its normal range over the clip.
2. **Temporal smoothing** of the output, applied to the fitted parameters or the vertices so that
   detector glitches do not reach augmented samples.
3. **Evaluation.**
   - Check visually whether smoothing looks better, with the stitched videos side by side.
   - Use the new metrics to decide whether they can tell when a clip is safe to augment.

## 7. Open

- **O8:** accept clips with a large gap to the floor, or flag them before augmenting?
- **O9:** does the round trip hold on another dataset, camera set-up or signer pool?
- **O6, carried over:** the elbow and shoulder joint-limit axes. They must be checked before any
  augmentation clamps poses to joint limits.
- The temporal-incoherence defect in intrinsic augmentation
  ([V1 problem scope §11](V1_Problem_Scope.md)) is still open and interacts with step 2.
