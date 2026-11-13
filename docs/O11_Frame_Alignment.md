# O11: pose-world vs image frame alignment (2026-10-18)

The V1.3 face pipeline takes FaceLandmarker's head rotation R_head, which is fitted in the image
(pixel) frame, and applies it unchanged in MediaPipe's pose-world frame. Ledger item O11 says the
two frames are assumed aligned. The 08925 head-placement spike pointed the other way. It found a
7.5° world→image similarity rotation and a ~10° roll between the pose-face Kabsch rotation and
R_head. This note measures both quantities on every clip with extracted hands and face landmarks:
the 20 clips in `data/round_trip_eval` plus 08925, 1,277 frames in total.

Script: `scripts/exp_frame_alignment.py`. It uses the first pass of `face_pipeline.run_clip`
(no depth correction) and the spike's correspondence, weights and Kabsch code. Reproduce with:

```
PYTHONPATH=src:scripts .venv/bin/python scripts/exp_frame_alignment.py --model-dir models
```

## What was measured (per frame)

- **sim_all**: signed rotation of the 2-D least-squares similarity from pose-world (x, y) to
  pose pixels, fitted over all visible (≥ 0.5) in-frame pose points 0-32. A 2-D similarity can
  only rotate about the camera axis, so this is pure roll. It is positive from +x towards +y
  (clockwise on screen).
- **sim_face**: the same fit restricted to face points 0-10. This is the spike's 7.5° number.
- **ortho**: a scaled-orthographic camera fitted from pose-world (x, y, z) to pose pixels (x, y),
  split into roll and swing (pitch + yaw). It does not use MediaPipe's noisy image z. The 3-D
  Umeyama fit to (x W, y H, z W) is also in the script (`cam3d`).
- **Kabsch vs R_head**: angle of D = R_head Kᵀ. K is the weighted Kabsch rotation from the
  SMPL-X rest face points to pose face points 0-10 in pose-world, with weights as in the spike:
  visible, finite, and outside the padded hand outlines. D is split into its roll (twist about
  camera z) and its swing.
- **after roll fix**: the angle between K and Rz(−sim_all) R_head, i.e. what remains once R_head
  is rotated back by the measured frame roll.
- **shoulder tilt** of the 12 → 11 line in the image, and its frame-by-frame correlation with
  sim_all.
- **eye-line tilt** (outer corners, pose 6 → 3 and face-mesh 33 → 263) as each source sees it:
  FaceLandmarker pixels, pose pixels, pose-world (x, y), and the SMPL-X corners rotated by R_head
  and by K. A head roll is visible directly in the image, so the pixel sources are the reference.

## Per-clip results

Angles are in degrees: median over frames (p95 in brackets; for sim_all, p95 of |angle|). "Roll
share" is the median of |roll(D)| / angle(D). 09849 has no usable face frames, so its R_head
columns are empty.

| clip | frames (Kabsch) | sim_all | sim_face | ortho pitch | Kabsch vs R_head | D roll | roll share | after roll fix | shoulder tilt img | r(sim_all, shoulder) |
|---|---|---|---|---|---|---|---|---|---|---|
| 05631 | 24/24 | +4.2 (5.8) | +7.6 | −9.0 | 27.3 (33.5) | +10.0 | 0.39 | 26.5 (32.7) | −1.4 | −0.76 |
| 07389 | 72/82 | +0.3 (1.1) | +5.2 | −0.7 | 13.0 (17.3) | +4.4 | 0.35 | 12.8 (17.3) | −1.6 | +0.69 |
| 08917 | 32/32 | +1.0 (2.0) | +7.6 | −10.3 | 20.9 (26.6) | +6.6 | 0.32 | 20.7 (26.3) | −0.7 | −0.49 |
| 08918 | 30/30 | +1.6 (2.7) | +8.3 | −9.5 | 26.3 (29.9) | +7.0 | 0.26 | 26.0 (29.5) | −0.6 | −0.29 |
| 08925 | 41/41 | +1.1 (1.5) | +7.5 | −8.0 | 10.2 (10.8) | +10.1 | 0.99 | 9.2 (9.8) | +2.4 | −0.59 |
| 08938 | 55/56 | +1.6 (3.5) | +9.4 | −7.2 | 26.4 (32.3) | +9.8 | 0.37 | 25.7 (31.9) | +0.4 | −0.34 |
| 09849 | 0/81 | +0.5 (1.6) | +5.1 | −8.4 | – | – | – | – | −1.5 | +0.56 |
| 12328 | 88/88 | +1.6 (2.6) | +8.4 | −17.5 | 8.6 (12.3) | +5.9 | 0.79 | 7.5 (11.8) | +2.1 | −0.72 |
| 13630 | 106/106 | −2.4 (3.2) | +9.0 | −0.2 | 14.5 (15.2) | +10.0 | 0.68 | 14.6 (17.5) | +2.8 | +0.96 |
| 17083 | 62/62 | +1.9 (2.5) | +1.6 | −8.0 | 17.5 (21.4) | +0.3 | 0.09 | 17.8 (21.4) | +1.9 | −0.54 |
| 21871 | 79/79 | +0.5 (1.4) | +3.0 | −4.2 | 15.5 (17.7) | −1.2 | 0.27 | 15.5 (17.7) | +1.0 | +0.10 |
| 21942 | 68/68 | +0.9 (2.7) | +3.2 | −5.2 | 14.5 (18.1) | +1.2 | 0.22 | 14.8 (18.5) | +1.9 | +0.76 |
| 28111 | 35/35 | +3.9 (8.1) | +6.0 | −7.5 | 18.7 (23.9) | +2.2 | 0.13 | 19.4 (24.2) | +2.1 | +0.60 |
| 57641 | 27/29 | +1.0 (1.3) | +7.4 | −9.2 | 13.2 (14.4) | +7.0 | 0.53 | 12.5 (14.1) | +5.8 | −0.33 |
| 57949 | 53/53 | +2.3 (3.1) | +3.6 | −8.3 | 12.1 (14.3) | +2.5 | 0.22 | 11.9 (13.6) | −1.6 | +0.05 |
| 63229 | 47/47 | +3.5 (5.1) | +10.0 | −5.4 | 26.3 (30.7) | +14.1 | 0.56 | 25.0 (29.7) | +6.4 | +0.63 |
| 63665 | 72/72 | +1.3 (2.7) | +4.9 | −5.2 | 16.1 (19.0) | +3.1 | 0.18 | 16.0 (18.7) | −0.0 | +0.36 |
| 63669 | 93/94 | +0.5 (2.3) | +5.1 | −6.6 | 13.5 (18.0) | +3.5 | 0.24 | 13.4 (18.1) | −0.8 | +0.71 |
| 69257 | 56/56 | +0.8 (2.0) | +8.5 | −3.4 | 13.8 (18.3) | +7.4 | 0.59 | 13.1 (18.0) | +0.2 | −0.19 |
| 69364 | 75/75 | −0.2 (1.0) | +10.5 | −10.8 | 12.9 (18.7) | +7.2 | 0.56 | 13.2 (18.8) | +1.4 | −0.38 |
| 69411 | 67/67 | +2.7 (3.6) | +6.8 | −7.9 | 14.1 (21.2) | +6.1 | 0.39 | 13.5 (20.1) | +1.2 | −0.66 |

Pooled over all frames:

| quantity | value |
|---|---|
| median of per-clip \|sim_all\| | 1.3° (largest clip median 4.2°, 05631; largest p95 8.1°, 28111) |
| Kabsch vs R_head | median 14.9°, p95 27.7° |
| after roll fix | median 15.0°, p95 27.2° (no improvement) |
| r(sim_all, image shoulder tilt), across clips / pooled frames | +0.05 / +0.17 |
| r(sim_all, D roll), across clips / pooled frames | +0.07 / −0.11 |
| r(D roll, pose-world eye tilt − pose-pixel eye tilt), pooled | +0.53 |
| eye tilt − FaceLandmarker eye tilt: pose pixels | −0.4° median (p95 \|·\| 5.7°) |
| … pose-world | −5.7° (15.0°) |
| … R_head-rotated SMPL-X corners | −0.3° (1.4°) |
| … K-rotated SMPL-X corners | −5.9° (11.7°) |
| ortho pitch, median of clip medians | −7.9° (range −17.5° to −0.2°); yaw +2.2°, roll +1.2° |
| r(ortho rotvec, D rotvec) across clips, x / y / z | −0.15 / −0.18 / −0.13 |

## Findings

1. **The frames agree in roll.** Fitted over the whole body, the world→image rotation is small on
   every clip: median |sim_all| 1.3°, and per-clip medians range from −2.4° to +4.2°. The
   orthographic fit gives a median roll of +1.2°. At 10 cm from the head joint, 1-2° of roll moves
   a face point by 0.2-0.35 cm, which is negligible for proximity.
2. **The spike's 7.5° came from the face points, not the frame.** sim_face fits only points 0-10
   and is +1.6° to +10.5° on every clip (08925: +7.5°). sim_all on 08925 is +1.1°. So pose-world's
   face cluster is rolled relative to its own image, while the rest of the body is not.
3. **The K vs R_head disagreement is pose-world face error.** The image settles it. R_head's eye
   line matches FaceLandmarker's within 0.3° (p95 1.4°). Pose's own image eye line matches it
   within 0.4°, so the two detectors agree in pixels. The pose-world eye line is 5.7° off (p95
   15°), and K inherits that error (5.9°). The roll of D tracks this pose-world-vs-image eye
   discrepancy (r = +0.53) and does not track the frame roll (r ≈ 0). Rotating R_head back by
   sim_all leaves the disagreement unchanged (14.9° → 15.0°).
4. **The offset is mostly not roll.** Outside 08925 (0.99), 12328 and 13630, the roll share of D
   is 0.1-0.6. Most of D is swing (pitch), which depends on pose-world face depth, the quantity
   O16 already flags as unreliable.
5. **Frame roll does not follow shoulder tilt.** r(sim_all, image shoulder tilt) is +0.05 across
   clips and +0.17 pooled, and within clips the sign varies (−0.76 to +0.96). Pose-world is
   therefore not levelled to the shoulders. Its residual roll looks like fit noise around ~+1°.
6. **Why the earlier ~1° check missed this.** `head_camera_rotation_deg` (median 0.4-1.1° per
   clip) is fitted between head points already posed with R_head and the face pixels. It is ~0
   by construction and cannot detect an O11 offset.
7. **Possible pitch offset, not established.** Over the body, the orthographic fit suggests a
   consistent pitch of about −8° between pose-world and the camera. The 3-D image-z fit agrees on
   08925 (9.4° swing vs 9.2°). The estimate rests on MediaPipe's world depth and on a near-planar
   torso, and it does not correlate with D across clips (r ≈ −0.15). If it is real, R_head's
   pitch is off by that much in pose-world: about 1.4 cm at the face surface, mostly along the
   camera axis. That is small next to O16's ~20 cm depth gap.

## Verdict on O11

- **Roll: aligned, no correction needed before attaching the face.** Pose-world and the
  image/FaceLandmarker frame agree in roll to about 1-2° (median 1.3°, worst clip median 4.2°).
  Applying the measured roll as a correction does not help.
- **R_head is the better head orientation.** The 10° roll on 08925, and the 13-27° K vs R_head
  angle elsewhere, come from distortion of pose-world's face points (eye line 5.7° off the image,
  p95 15°), not from a frame mismatch. Do not re-derive head rotation from pose-world face points.
- **Pitch/yaw: still open.** It is tied to O16. A body-wide orthographic fit suggests ~8° of
  pitch, but it rests on pose-world depth. An independent check would need a known camera pose
  or a depth source.
