# V1.3 findings: the phase-A gate (head, neck, jaw and hand-to-face geometry)

**Date:** 2026-10-07. **Branch:** `feat/face-v1.3`. **Scope:** the phase-A verification gate of the
[V1.3 face design](V1_3_Face_Design.md) (§7), run on the 20 clips of `data/round_trip_eval`.
**Plan:** [V1.3 implementation plan](V1_3_Implementation_Plan.md). **Ledger:**
[V1 problem scope §12](V1_Problem_Scope.md#12-assumptions-ledger), rows O10 to O16 and P23 to P24.

Phase A fits a SMPL-X neck, head and jaw to MediaPipe FaceLandmarker output, places the fitted
hands in the same frame, and measures signed distances from hand points to the head surface. The
gate asks whether that is accurate enough for phase C (the face stream) to proceed. The user
decides that; this document only reports what was measured.

**Sources.**

- *Face report*: `data/round_trip_eval/face_report.json`, written by
  `scripts/evaluate_face.py`. The folder is untracked (it derives from WLASL footage outside the
  repo), so the report is not committed. Every number below is copied from it.
- *Commands*, run in this order. The report is from the second evaluation run, which added the
  distance, contact and gap rows:

  ```sh
  .venv/bin/python scripts/extract_face_landmarks.py --clips-from data/round_trip_eval \
    --video-dir ~/Downloads/raw_videos --output-dir data/round_trip_eval
  .venv/bin/python scripts/evaluate_face.py --landmark-dir data/round_trip_eval
  ```

- *Caches*: `<clip>_face.npz` (FaceLandmarker output) and `<clip>_face_features.npz` (per-frame
  fit and geometry), both in `data/round_trip_eval`.
- *Diagnostic*: `.superpowers/sdd/V1_3_Implementation_Plan/scratch/mediapipe_depth_gap.py` (§5),
  uncommitted, reading the `*_hands.npz` and `*_face_features.npz` files.
- *Task reports*: Task 1 (F1 distances), Task 6 (synthetic recovery), Task 8 (region counts),
  Task 10a (gate run), in `.superpowers/sdd/V1_3_Implementation_Plan/`.

**Terms.** A frame is *hand-near-face* ("near") when either hand's padded outline overlaps the
bounding box of the 478 face points; all other frames are "far". The 20 clips hold 1,236 frames:
143 near and 1,093 far. Near frames come from 12 of the 20 clips. Clip 09849 has no face detected
in any of its 81 frames.

## 1. Result in brief

- **Met:** head pose agreement (metric 2), median 0.57 / 0.58 degrees against a 5 degree target,
  p95 1.40 / 2.66 degrees against 10. Jaw rank correlation (metric 3), median 0.92 against 0.7,
  on the 8 clips that could be scored.
- **Not met:** the jaw target holds for 6 of those 8 clips, not all of them. Clips 69364 (0.496)
  and 63665 (0.687) fall below 0.7. The other 12 clips could not be scored.
- **Unmeasured, not passed:** metric 5 (penetration). No hand point ever came closer than
  0.137 m to the head and contact is 0 in every frame, so a penetration fraction of 0 says
  nothing (§4).
- **Reported only:** metrics 1, 4 and 6 have no pass/fail target. Metric 1 has a floor, but the
  floor is above the measured medians (§3.1).
- **New open question (O16):** on near-face frames the hand is about 0.18 m from the head along
  the camera axis. MediaPipe's own pose-world output shows the same gap before any fitting (§5).
- **Caution on metric 2:** it measures agreement between two rotations that both come from the
  same face landmarks. It is not an independent accuracy check (§3.2).

## 2. Setup notes that affect every number

- The near-face invalid fraction is 0 by construction: `hand_near_face` is only computed on
  frames where a face was detected. The near-face unreliable fraction is not: it is 0.007 (1 of
  143 frames, one with only 7 rigid points left). The far-frame invalid and unreliable
  fractions (8.7 %) include every frame with no detected face, 81 of which are clip 09849.
- Near frames are defined by bounding-box overlap, not by occlusion. Of the 143, 113 (79 %) kept
  all 17 rigid points; 30 lost at least one, 18 lost two or more (counted from `points_used` in
  the feature files).
- Only frames with a posed hand count for penetration, minimum distance and contact: 676 of the
  1,093 far frames, and all 143 near frames.
- Metric 2 uses the clip's mean rotation (a scipy chordal mean) as the reference for each side,
  not the median that design §7 names. Both rotations are taken relative to their own clip mean.

## 3. The six gate metrics

Each row gives the median, 95th percentile (p95) and mean over frames, near and far side by
side. Source for all rows: `face_report.json`, `metrics`.

### 3.1 Metric 1: rigid-set residual, % of outer-eye-corner distance

| Frames | n | Median | p95 | Mean |
| --- | --- | --- | --- | --- |
| Near | 143 | 2.270 | 3.653 | 2.371 |
| Far | 998 | 2.136 | 3.234 | 2.160 |

**Floor** (`floor_rigid_rms_pct`): 2.725 %. It is the same residual for MediaPipe's
`canonical_face_model.obj` fitted to the SMPL-X neutral template, with the points unweighted.

The floor is not a lower bound. The measured medians (2.14 and 2.27 %) are below it. It compares
the neutral template with MediaPipe's canonical face, unweighted, which is a different comparison
from the per-frame, weighted fit; it indicates the size of the correspondence error but does not
bound the fit's residual. There is no target.

### 3.2 Metric 2: head rotation against FaceLandmarker's, degrees (target median at most 5, p95 at most 10)

| Frames | n | Median | p95 | Mean | Target |
| --- | --- | --- | --- | --- | --- |
| Near | 143 | 0.580 | 2.660 | 0.856 | met |
| Far | 998 | 0.571 | 1.405 | 0.641 | met |

Both targets are met by a wide margin, in both splits. Three qualifications:

- **Agreement, not independence.** The fitted head rotation and FaceLandmarker's facial
  transformation matrix are both computed from the same landmarks. Small angles show that the
  conventions line up, not that either rotation is correct.
- **The matrix convention is tested, though.** A reviewer scored metric 2 on three real clips
  (63665, 69364, 57949) with the conventions wrong. As coded: 0.58 to 0.65 degrees median.
  Without the axis flip: 3.2 to 10.6 degrees. With the rotation inverted: 7.8 to 12.6 degrees.
  The heads' own motion in those clips was 3.9 to 6.5 degrees. A wrong convention is therefore
  distinguishable from the 0.6 degrees measured, although it would not necessarily fail the gate
  (the no-flip median on the best clip, 3.2 degrees, is under the 5 degree target). This tests
  FaceLandmarker's matrix convention and the OpenGL-to-image axis flip. It never involves the
  pose-world frame, so the face-to-pose-world axis match in F2's wording is not tested (§6).
- **Mean, not median.** The reference rotation per clip is the chordal mean (§2), not
  the design's median.

### 3.3 Metric 3: jaw pitch against the `jawOpen` blendshape, Spearman rank correlation per clip (target at least 0.7)

Eight clips had at least 10 jaw-reliable valid frames and a `jawOpen` spread of at least 0.02.
The other twelve are `null` in the report.

| Clip | Spearman | | Clip | Spearman |
| --- | --- | --- | --- | --- |
| 13630 | 0.948 | | 57949 | 0.892 |
| 17083 | 0.950 | | 63665 | 0.687 |
| 21871 | 0.767 | | 69364 | 0.496 |
| 21942 | 0.951 | | 28111 | 0.969 |

Summary over the 8 scored clips: median 0.920, p95 0.963, mean 0.832. Not scorable: 05631, 07389,
08917, 08918, 08938, 09849, 12328, 57641, 63229, 63669, 69257, 69411.

The median meets the target. Two of the eight clips do not (69364 and 63665), and 60 % of the
clips were not scorable, so the figure describes the clips with visible jaw motion.

### 3.4 Metric 4: observed against fitted ear midpoint, % shoulder width

| Frames | n | Swing median | Swing p95 | Swing mean | Rest median | Rest p95 | Rest mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Near | 143 | 4.558 | 9.822 | 4.402 | 7.148 | 15.671 | 7.768 |
| Far | 998 | 2.816 | 9.191 | 3.456 | 7.609 | 14.545 | 8.026 |

"Swing" is the fit with the neck swung toward the observed ear midpoint; "rest" leaves the neck
at its rest pose. The swing lowers the median residual in both splits (far 7.61 to 2.82 %, near
7.15 to 4.56 %), and the near-face residual stays higher than the far one. In synthetic recovery
(Task 6) the ear midpoint was recovered to 0.58 mm with the swing and 23.0 mm with the neck at
rest.

### 3.5 Metric 5: hand points inside the head, and how close hands come

| Row | Frames | n | Median | p95 | Mean |
| --- | --- | --- | --- | --- | --- |
| Fraction with a point more than 1 cm inside | Near | 143 | 0 | 0 | 0 |
| | Far | 676 | 0 | 0 | 0 |
| Minimum signed distance, m | Near | 143 | 0.208 | 0.289 | 0.225 |
| | Far | 676 | 0.294 | 0.445 | 0.290 |
| Contact fraction (a point within 2 cm) | Near | 143 | 0 | 0 | 0 |
| | Far | 676 | 0 | 0 | 0 |
| Depth gap, m (near only) | Near | 143 | 0.181 | 0.253 | 0.188 |
| Image-plane gap, m (near only) | Near | 143 | 0.111 | 0.167 | 0.115 |

The last two rows are the new ones. For each near frame they take the hand point with the
smallest absolute distance and decompose its offset from its closest head-surface point into the
camera-axis part (`gap_depth_m`) and the image-plane part (`gap_plane_m`).

**Penetration is unmeasured, not passed.** Across all 20 clips the smallest hand-to-head distance
is 0.137 m and contact is 0 in every frame. A zero penetration fraction is what you get when no
hand ever reaches the head, so it cannot show that hands and head are placed consistently. The
real-data question this metric was meant to answer is therefore still open. It is also blind to a
fingertip a few millimetres inside the mouth opening, because a point nearest an open edge is
forced positive (§7).

### 3.6 Metric 6: unreliable fractions

| Flag | Frames | n | Median | p95 | Mean |
| --- | --- | --- | --- | --- | --- |
| Invalid | Near | 143 | 0 | 0 | 0 |
| | Far | 1,093 | 0 | 1 | 0.087 |
| Unreliable | Near | 143 | 0 | 0 | 0.007 |
| | Far | 1,093 | 0 | 1 | 0.087 |
| Jaw unreliable | Near | 143 | 0 | 1 | 0.259 |
| | Far | 1,093 | 0 | 1 | 0.087 |

The near-face invalid figure is 0 by construction (§2), so near and far are not comparable for
that row; the near-face unreliable figure (0.007, 1 of 143 frames) is a measurement, but near
frames are only a subset in which a face was found. The far 8.7 % is dominated by clip 09849 (81 of
1,093 far frames, no face detected), plus partial detection in 07389 (87.8 % of frames) and 57641
(93.1 %). That is a detector coverage issue, not a fit failure. Jaw reliability is the one flag
that is meaningful near the face: 25.9 % of near frames are jaw-unreliable, which is consistent
with a hand covering the mouth.

## 4. What the near-face numbers say about hands and head

On near frames the hand is about 0.21 m from the head surface (median minimum distance), never in
contact, and the gap is mostly along the camera axis (median depth 0.181 m against 0.111 m in the
image plane). In the image the hands overlap the face; in the fitted 3-D scene they hover in front
of it. Section 5 checks whether the fitting is responsible.

## 5. Diagnostic: is the depth gap already in MediaPipe's observations?

**Question.** Does the 0.18 m camera-axis gap appear in MediaPipe's own pose and hand landmarks,
before any SMPL-X fitting?

**Method.** For each clip, hand-near-face frames from `<clip>_face_features.npz` and observations
from `<clip>_hands.npz`. Each observed hand is placed the way the pipeline places hands: its
wrist (hand landmark 0) is translated onto the pose wrist (pose 15 left, 16 right). Then the index
fingertip (hand landmark 8) is measured against the observed pose nose (pose 0) and against the
ear midpoint (poses 7 and 8). Medians over hand-frames of the absolute depth difference (|dz|),
the image-plane difference (|dxy|) and the 3-D distance, in metres. Restricting to wrist and nose
visibility of at least 0.5 changes no median by more than 0.008 m.

| Reference | Split | Hand-frames | Clips | Median abs dz | Median abs dxy | Median 3-D |
| --- | --- | --- | --- | --- | --- | --- |
| Nose | Near | 152 | 12 | 0.217 | 0.137 | 0.253 |
| Ear midpoint | Near | 152 | 12 | 0.339 | 0.158 | 0.372 |
| Nose | Far | 892 | 20 | 0.152 | 0.298 | 0.352 |
| Ear midpoint | Far | 892 | 20 | 0.283 | 0.315 | 0.440 |

Per clip, the near-face nose |dz| ranges from 0.099 m (12328) to 0.300 m (63665); 11 of 12 clips
are between 0.15 and 0.30 m.

**Result.** The gap is already present before fitting, in the raw pose-world observations as the
pipeline consumes them. The diagnostic reuses the pipeline's own wrist-to-pose-wrist translation,
so it cannot separate a pose-world depth error from an error in that placement rule. On near frames the observed fingertip is
0.217 m from the observed nose along the camera axis, slightly more than the fitted scene's
0.181 m (the two are not the same quantity: the fitted figure is from the nearest hand point to the
closest head-surface point; this one is the index fingertip to the nose landmark). The ear
midpoint reads further because the ears sit behind the nose; that offset is real anatomy and says
nothing about the hand. The far-frame nose |dz| (0.152 m) is given for comparison only: hands in
signing space naturally sit in front of the nose, so it supports no inference about an offset.
This table counts every present hand on a near frame (152 hand-frames over the near frames), while the
fitted gap takes the minimum over the hands' points per frame, so the 0.217 and 0.181 m figures
are further apart in definition than they look.

**Conclusion.** The gap is already present in MediaPipe's own pose-world observations, before
any SMPL-X fitting. The fitting does not create it. Whether it comes from pose-world depth for the
wrists relative to the head, or from hands genuinely held in front of the face, is not decided.

**What this does not settle.** "Near" is defined by overlap in the image, so a hand held 20 cm in
front of the face and not touching it is near. The diagnostic cannot say how much of the 0.2 m is
real distance and how much is depth error in the pose-world wrists. Hands that touch the face (a
chin sign, say) should put the fingertip within a few centimetres of the head surface; in the
fitted scene no hand point on any frame came closer than 0.137 m to the head. Whether the data
holds no contact signs, or contact is lost to depth error, is the new open question O16.

## 6. Assumptions F1 to F6

| ID | Verdict | Evidence |
| --- | --- | --- |
| F1 | Supported (Proven, P23) | Eye corners 1.25, 1.48, 1.48, 1.25 mm and nose tip 3.89 mm from SMPL-X's own landmarks (Task 1; limits were 6 and 8 mm). The FLAME triangle check also passed. Only these five points were compared with SMPL-X's landmarks; the other points of the 105 rest on the triangle check. |
| F2 | Split. Matrix convention and axis flip: Proven (P24). Face axes against the pose-world axes: still open (O11) | Metric 2 on real data: 0.58 degrees near, 0.57 far. With the axis flip removed the reviewer's clips read 3.2 to 10.6 degrees; with the rotation inverted, 7.8 to 12.6. That settles FaceLandmarker's matrix convention and the flip. Metric 2 compares the fitted rotation with FaceLandmarker's own matrix and never involves the pose-world frame, so the (x·W, y·H, z·W) match to pose-world axes (the design's wording) is untested. The depth gap in O16 bears on it. Both rotations also come from the same landmarks, so agreement is not independent accuracy. |
| F3 | Partly supported, still open (O12) | Rigid residual and agreement do not degrade on near frames: 2.27 % against 2.14 %; head pose p95 2.66 against 1.40 degrees (143 near frames, 12 clips). But near frames are bounding-box overlap, not occlusion: 113 of 143 (79 %) kept all 17 rigid points, 30 lost at least one, 18 lost two or more, and 1 was flagged unreliable. Agreement with MediaPipe's matrix cannot see damage that both estimates share, and expression is not measured. Independent accuracy under real occlusion and expression is untested. |
| F4 | Partly supported, still open (O13) | The swing lowers the ear-midpoint residual: far 7.61 to 2.82 %, near 7.15 to 4.56 % (p95 about 9 to 10 %). The synthetic arms-unchanged test passes, but it is structurally trivial; arm metrics were not re-measured on real data in this gate. |
| F5 | Partly supported, still open (O14) | Median 0.92 over 8 clips, but 69364 (0.496) and 63665 (0.687) are below 0.7, and 12 of 20 clips could not be scored. |
| F6 | Not yet judged, still open (O15) | Region labelling is covered by tests and rendered in `docs/media/face_regions.png`. **The user still has to review the render.** The Eye count was inflated before eyeball meshes were dropped from the head surface; it is now 750. The Ear radius of 3.5 cm is generous (1,199 vertices) and takes some temple and jaw-hinge area. |

## 7. Decisions and deviations during implementation

**User decisions.**

- No matplotlib in library code. `points_in_polygon` is a numpy function in `head_converter`;
  matplotlib stays in scripts.
- Eyeball meshes are dropped from the head surface. They flipped signed distances for points
  inside the head behind the eyes.

**Controller decisions.**

- Mouth region is "inside the outer-lip outline and in front of the ears". The inner-lip vertices
  had been labelled CheekNose.
- Neck is every vertex not dominated by the head, jaw or eye joints. Spine3 vertices had been
  mislabelled.
- A signed distance is forced positive when the closest point lies on an open boundary edge: the
  neck rim and the mouth opening (not the eye sockets, although an earlier docstring said so; the
  wording is fixed). Far hands had been reading -0.35 to -0.6 m. The side effect is in §3.5.

**Measured along the way.** F1 in Task 1 (above). Synthetic recovery in Task 6: head orientation
0.415 degrees, jaw 0.32 degrees, ear midpoint 0.58 mm with the swing against 23.0 mm with the neck
at rest.

**Reviewer's convention check.** Three real clips (63665, 69364, 57949) were rescored with wrong
conventions (§3.2).

## 8. Which targets were met

| Target | Verdict |
| --- | --- |
| Metric 2 median at most 5 degrees | Met: 0.58 near, 0.57 far |
| Metric 2 p95 at most 10 degrees | Met: 2.66 near, 1.40 far |
| Metric 3, 0.7 or better per clip | Median 0.92; 6 of 8 scored clips meet it, 2 do not, 12 not scorable |
| Metric 1 against the floor | Reported; medians are below the floor, so the floor is not a bound |
| Metrics 4 and 6 | Reported, no target |
| Metric 5, penetration | Unmeasured: no hand came within 0.137 m, contact is 0 everywhere |

## 9. Open items for the phase-C decision

1. **Hand-to-head depth gap (O16).** About 0.18 m on near frames, present in MediaPipe's own pose
   world. Phase C inherits it.
2. **Contact never observed.** Metric 5 is unmeasured until a clip where a hand touches the head
   appears, or the depth gap is corrected.
3. **Jaw on 12 of 20 clips.** F5 is evidenced on 8 clips, with two below target.
4. **Region render.** The user reviews `docs/media/face_regions.png` (F6).
5. **Metric 2 independence.** No second head-orientation source exists in this data.
