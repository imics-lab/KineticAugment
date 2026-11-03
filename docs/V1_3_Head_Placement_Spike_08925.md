# V1.3 spike: head placement and hand-to-face proximity on clip 08925

**Date:** 2026-10-18. **Code:** branch `exp/head-placement-08925` (cut from `feat/face-v1.3` at
c818dd9; commits 7ae0d27..53595a0, not merged). **Ledger:**
[V1 problem scope §12](V1_Problem_Scope.md#12-assumptions-ledger), rows O11 and O16.

**Question.** Can MediaPipe's pose face landmarks (0 nose, 1-6 eyes, 7-8 ears, 9-10 mouth) place
the SMPL-X head better than the ear-midpoint neck swing, and does that help hand-to-face
proximity?

**Clip.** 08925, 41 frames, 288×192 px, right hand only (the left hand is off-frame). By eye, the
right fingers rest on the cheek beside the nose in every frame, so contact is the expected answer
throughout. One clip at low resolution: treat every number below as an example, not a rate.

## 1. Pose landmarks alone

Closest pose hand point (15-22) to pose face point (0-10), per frame, using only points with
visibility ≥ 0.5 that are inside the frame:

| Measure | Median | Min |
| --- | --- | --- |
| 3-D distance, pose world | 24.0 cm | 22.2 cm |
| Depth component \|dz\| | 21.1 cm | 19.2 cm |
| Image-plane distance, same pair (body-plane 247 px/m) | 10.4 cm | 10.3 cm |
| Image distance, closest of any pair | 6.9 cm | 6.7 cm |

No frame comes within 2 cm in 3-D or in the image. There are two reasons. The depth gap is already
in the pose output. Pose hand points mark the wrist and knuckles, not the fingertips, so even in
the image they stay about 7 cm from the face. A pose-only extraction is identical (0.00 cm, 0.00 px)
to the Holistic pose already stored in `08925_hands.npz`, so the pipeline sees exactly these
points.

## 2. Head placement from 11 pose face points

The head's world rotation stays the FaceLandmarker fit. Only the neck swing changes: it now aims
the head joint at the weighted mean of (observed pose point − rotated rest offset) over the 11
points, instead of at the ear midpoint. SMPL-X counterparts are the `nose`, `leye`, `reye`, `lear`
and `rear` vertices, plus transferred face-mesh points 362/263, 133/33 and 291/61 for the eye and
mouth corners. A side check put each mapped pair 1.0-3.8 px apart in the image against
7.4-20.7 px mirrored, so no pair was swapped.

| Median over 41 frames | Neck at rest | Ear-midpoint swing | 11-point swing |
| --- | --- | --- | --- |
| 11-point RMS residual | 2.49 cm | 2.27 cm | 2.01 cm |
| Ear-midpoint residual | 1.92 cm | 1.85 cm | 2.06 cm |
| Minimum hand-to-head-surface distance | 19.5 cm | 19.2 cm | 18.1 cm |

- The head joint moves 1.04 cm from the ear placement (max 1.26 cm). The swing reaches its target
  to within 0.16 cm.
- The lower 11-point residual holds by construction: that residual is what the swing minimises,
  and about 2 cm of it cannot be removed by moving the neck. The ear-midpoint residual gets
  slightly worse.
- Occlusion weighting never triggered. The nearest pose face point was 3.4 px from a hand outline
  against a 3.1 px pad, so the case it exists for is untested here.

**Conclusion.** Head placement moves the head by about 1 cm. The proximity error is about 20 cm in
depth, so head placement is not the lever.

## 3. Depth correction (task 14, `hand_depth.py`) on the same clip

This ran the existing depth correction (`run_clip(depth_corrected=True)`, via
`scripts/view_proximity_round_trip.py`), which places the wrist from the hand's apparent size in
the image:

- Before correction, no hand point came within 17.9 cm of the head.
- After correction, 23 of 41 frames register contact (under 2 cm), and the SMPL-X wrist reached
  its image-derived target in all 23.
- In the other 18 frames the wrist did not reach the target, so contact there is untrusted.
- The image overlap measure flags NEAR in all 41 frames.

Of everything tried on this clip, depth correction is the only approach that recovers contact.

## 4. A roll in the pose face points, not in the frame

- A weighted Kabsch rotation from the SMPL-X rest points to the 11 observed pose face points
  differs from the FaceLandmarker head rotation by 10.2 ± 0.8° in every frame. Almost all of it is
  roll about the camera axis (axis ≈ (0.08, −0.09, −0.99)).
- The 2-D similarity from pose-world (x, y) to pose image pixels over the same face points
  rotates by a constant 7.5 ± 0.8°.
- **Corrected by the cross-clip measurement** (`docs/O11_Frame_Alignment.md`, branch
  `claude/keen-almeida-22d42e`). Fitted over the whole body, the same similarity is +1.1° on
  08925, and its per-clip median is 1.3° over 21 clips. So the pose-world frame is not rolled
  against the image. The face points are: pose-world's eye line is 5.7° off the image (median,
  p95 15°), while R_head's matches FaceLandmarker within 0.3°. The 10° comes from pose-world
  face-point distortion, and R_head remains the better head rotation.
- The Kabsch angle also depends on which points enter it (9.0° with eyes and ears, 22.3° with eyes
  and mouth only). Pose face points are not metrically rigid: the eye centres are 3.6 cm apart
  against 6.6 cm on SMPL-X.
- Swapping in the Kabsch rotation moves the head-joint target by only 0.23 cm (median), so §2's
  result does not depend on it.

## Reproduce

From the spike worktree, with `PYTHONPATH=src:scripts .venv/bin/python` (the shared venv's
editable install points at another checkout):

```sh
V=/Users/bemnetbeshah/Workspace/cv-research/KineticAugment/data/metsis_new_unseen_sign_videos
O=data/test_output/exp_08925
PYTHONPATH=src:scripts .venv/bin/python scripts/extract_pose_only.py $V/08925.mp4 --output-dir $O
PYTHONPATH=src:scripts .venv/bin/python scripts/view_pose_face_proximity.py 08925 \
  --landmark-dir $O --video-dir $V --output-dir $O
PYTHONPATH=src:scripts .venv/bin/python scripts/exp_head_placement.py 08925 \
  --data-dir $O --video-dir $V --output-dir $O
```

The hand and face caches come from `extract_hand_landmarks.extract_all` and
`extract_face_landmarks.py` on the same video. Outputs (`*_pose_only.npz`,
`*_head_placement.npz`, three mp4s and stills) are untracked in `data/test_output/exp_08925/`.
