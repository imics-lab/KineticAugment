# V1.3 design: the face — head pose, jaw, hand-to-face geometry, and the face stream

**Date:** 2026-10-07. **Branch:** `feat/kinematics-and-ik`.
**Status:** phase A implemented; gate results in [V1.3 findings](V1_3_Findings.md).
**Earlier results:** [V1 findings](V1_Findings.md) (body),
[V1.2 findings](V1_2_Findings.md) (hands), [Week 2 findings](Week_2_Findings.md) (unseen clips).

V1 and V1.2 fit the arms and hands and copy MediaPipe's 468 face landmarks through untouched
(`smplx_to_mp.py:112`). The face matters for two reasons:

1. **Hand-to-face geometry.** Where the hand is relative to the face is a phonological parameter
   of a sign: MOTHER (chin) and FATHER (forehead) differ only in location. Measuring it needs the
   SMPL-X head posed correctly in the same frame as the fitted hands.
2. **Facial cues.** Brow raises, mouthing and head tilts carry grammar and affect. Mouth features
   add 7–8 points of recognition accuracy on DGS ([arXiv 2507.20884](https://arxiv.org/html/2507.20884v1)).

The work comes in two phases. **Phase A** fits the SMPL-X neck, head and jaw and computes signed
hand-to-face distances with location labels. **Phase C** carries MediaPipe's 478 face points and
52 blendshapes alongside, attached rigidly to the fitted head. C starts only after A passes its
gate (§7). Fitting SMPL-X expression (called option B during design) is out of scope.

## 1. Scope

**In scope**

- A FaceLandmarker extraction pass: 478 landmarks, 52 blendshapes and a head-pose matrix per frame.
- A correspondence between MediaPipe points and the SMPL-X surface, transferred from a published
  MediaPipe→FLAME embedding.
- Closed-form fitting of head orientation, neck, and jaw pitch, robust to hands over the face.
- Moving the body + hand stitching from `scripts/view_body_hand_round_trip.py` into `src`.
- Signed distances from hand points to the head and neck surface, with region labels aligned to
  ASL-LEX location codes, and a contact flag.
- Phase C: the face point stream and blendshapes in the SMPL-X frame.
- A verification gate on 20 unseen clips, with every metric split by hand-near-face frames.

**Out of scope**

- SMPL-X `expression` fitting (it stays zero). The 52 blendshapes are ARKit-style coefficients and
  are not converted to SMPL-X expression.
- Temporal smoothing. It follows the plan already agreed in [Week 2 findings](Week_2_Findings.md);
  frames here are fitted independently.
- Spine rotation. The neck absorbs head placement (§4.3); the verified arm fit is unchanged.
- Augmentations that preserve hand-to-face contact. This design provides the measurements they
  will need (§9).
- Any change to the legacy 543-landmark layout in `utils/data_formats.py`.

## 2. Background facts the design relies on

Each fact was checked in source unless marked *inferred*.

| Fact | Source |
| --- | --- |
| Legacy Holistic gives metric world coordinates for the 33 pose points only. Its face points are normalised image coordinates. | `scripts/extract_world_landmarks.py:20-22`; [V1 findings](V1_Findings.md) |
| FaceLandmarker landmark `z` is scaled by image **width**, like `x` (`Array3f(x_scale, y_scale, x_scale)`). | MediaPipe `geometry_pipeline.cc` |
| FaceLandmarker's `facial_transformation_matrixes` map `canonical_face_model.obj` (cm) into a right-handed metric camera space looking down −Z (OpenGL style). They come from a weighted Procrustes fit and **include uniform scale**. Translation depends on an assumed 63° vertical FOV and is unreliable; rotation is more robust (*inferred*). | MediaPipe `face_mesh.md`, `face_geometry_from_landmarks_graph.cc`, `procrustes_solver.h` |
| MPI's metrical-tracker ships `mediapipe_landmark_embedding.npz`: 105 MediaPipe points (eyebrows, eyes, lips and the nose list `[168,6,197,195,5,4,129,98,97,2,326,327,358]`) embedded on the FLAME surface as triangle + barycentric weights. SMIRK bundles the same file. Licence: MPI non-commercial research. | [metrical-tracker](https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe), [SMIRK](https://github.com/georgeretsi/smirk/tree/main/assets/mediapipe_landmark_embedding) |
| `models/mano/SMPL-X__FLAME_vertex_ids.npy` maps FLAME's 5023 vertices to SMPL-X vertices. | [smplx README](https://github.com/vchoutas/smplx#mano-and-flame-correspondences) |
| Community MediaPipe-468 → iBUG-68 tables disagree with each other. One maps the outer lip onto the inner lip, and its jaw order looks mirrored. | Real3DPortrait `face_alignment_utils.py`; Akjava `mp_constants.py` |
| SGNify and SMPLify-X fit the jaw with a prior that makes pitch 10× cheaper than yaw or roll. SGNify takes no face landmarks and reports hand–face contact as its main contact case. | [SGNify config](https://github.com/MPForte/SGNify/blob/master/cfg_files/fit_sgnifyx_sv.yaml), [SMPLify-X config](https://github.com/vchoutas/smplify-x/blob/master/cfg_files/fit_smplx.yaml) |
| No standard splits head rotation evenly between neck and head. PIXIE puts all of it on the head. The atlanto-axial joint provides about 50–70% of cervical yaw; the occiput–C1 joint provides little pitch. | PIXIE `pose_abs2rel`; anatomy references in the research notes |
| ASL-LEX 2.0 head locations: HeadTop, Forehead, Eye, CheekNose, UpperLip, Mouth, Chin, UnderChin; Neck sits under Body. | [ASL-LEX 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685), [WLASL-LEX](https://arxiv.org/pdf/2203.06096) |
| Self-contact work defines contact as Euclidean distance < 2 cm. | [selfcontact](https://github.com/muelea/selfcontact) |

## 3. Data flow and components

```
video ─► scripts/extract_face_landmarks.py ─► <clip>_face.npz
          (MediaPipe Tasks FaceLandmarker,     landmarks (T,478,3) pixels, blendshapes (T,52),
           VIDEO mode, num_faces=1)            head_matrix (T,4,4), valid (T,), width, height

<clip>_world.npz ─► fit_body  ─► BodyFit ─┐
<clip>_hands.npz ─► fit_hand  ─► HandFit ─┤
<clip>_face.npz  ─► fit_head  ─► HeadFit ─┴─► assemble ─► posed SMPL-X (vertices, joints)
                                                             ├─► hand_face_geometry  (phase A)
                                                             └─► align_face_stream   (phase C)
```

New modules under `src/kinetic_augment/body_model/`. Each has one job and can be tested alone.

| Module | Responsibility |
| --- | --- |
| `face_observations.py` | Loads `<clip>_face.npz` into a `FaceClip`. Converts landmarks to pixels as (x·W, y·H, z·W), so all three axes share one scale. |
| `face_correspondence.py` | Loads the MediaPipe→FLAME embedding and transfers it to SMPL-X: FLAME triangle (a, b, c) becomes SMPL-X vertices (ids[a], ids[b], ids[c]) with the same barycentric weights. Exposes the **rigid set** (13 nose points + eye corners 33, 133, 362, 263) and the **jaw pair** (inner lip 13, 14). |
| `head_converter.py` | `fit_head(face_frame, hand_frames, body_fit, skeleton) -> HeadFit` with `neck_pose`, `head_pose`, `jaw_pose`, and the flags `valid`, `reliable`, `jaw_reliable`. |
| `assembly.py` | Poses one SMPL-X model from `BodyFit`, `HandFit` and `HeadFit`. The body + hand part moves here unchanged from `scripts/view_body_hand_round_trip.py` (wrist placement and the palm Kabsch); the script then calls it. |
| `face_regions.py` | Builds the head and neck surface and the per-vertex region labels once per shape (§5). |
| `hand_face_geometry.py` | Signed distance, nearest region and contact flag for each hand point, per frame (§5). |
| `face_stream.py` | Phase C (§6). |

`scripts/extract_face_features.py` writes `<clip>_face_features.npz` with the fitted neck, head and
jaw, all flags, the distances, regions and contact, and (phase C) the face stream and blendshapes.

## 4. Fitting the head

### 4.1 Head orientation

1. **Template.** Compute the rigid-set points on the SMPL-X rest pose with the clip's fitted
   `betas`. At rest every joint rotation is the identity, so a rotation fitted from this template
   is the head's global rotation.
2. **Occlusion weights.** Each rigid point gets weight 1, or 0 if it falls inside either hand's
   2-D outline (the convex hull of the hand's image landmarks, padded by 15% of the distance
   between the outer eye corners). Hand and face image coordinates come from the same video.
3. **Fit.** Weighted Umeyama with scale (`alignment.py:17`, extended to take weights), from the
   template to the observed pixel points. Scale and translation are discarded.
4. **One robust pass.** Points whose residual exceeds 2.5× the weighted median residual are set
   to weight 0, and the fit is repeated once.
5. **Flags.** `reliable=False` if fewer than 8 rigid points keep a non-zero weight, or if the
   weighted RMS residual exceeds 10% of the outer-eye-corner distance. Both thresholds are
   provisional and are revisited once the floor (§7, metric 1) is measured.

The result is `R_head`, the head's rotation in camera axes. Using it as a rotation in the pose
world frame assumes FaceLandmarker's image axes (x right, y down, z away from the camera) match
the pose world axes. That is assumption F2 (§8) and has its own test.

### 4.2 Jaw

The jaw is pitch only, following SGNify and SMPLify-X.

- **Observation.** The gap between inner-lip points 13 and 14, divided by the outer-eye-corner
  distance, so the ratio is scale-free.
- **Model.** The same ratio on SMPL-X, tabulated over jaw pitch 0°–30° in 0.25° steps for the
  clip's `betas`. The curve is monotonic, so the fit is a 1-D interpolation, clamped to the range.
- **Occlusion.** If point 13 or 14 is inside a padded hand outline, the jaw stays at rest and
  `jaw_reliable=False`.
- **Known bias.** With expression fixed at zero, lips that part without the jaw opening are read
  as jaw opening. Metric 3 measures how much this matters.

### 4.3 Splitting the rotation between neck and head

The neck places the head; the head carries the remaining orientation.

1. **Relative rotation.** `R_rel = R_torso⁻¹ · R_head`, where `R_torso` is the global rotation of
   the neck's parent (`global_orient` with the spine at rest).
2. **Neck swing (position).** The observed head centre is the midpoint of the pose-world ear
   points (MediaPipe 7 and 8), which are metric and in the body fit's frame. The neck swings so
   that the direction from the neck joint to the SMPL-X ear midpoint matches the direction to the
   observed ear midpoint. This corrects forward or sideways head lean that a spine at rest cannot
   represent. The neck sits above the shoulders, so the arm fit is unchanged.
3. **Neck twist.** The neck also takes 40% of the yaw of `R_rel`, as a twist about its own axis
   (anatomy: the atlanto-axial joint provides most cervical rotation).
4. **Head.** `head_pose = (R_torso · R_neck)⁻¹ · R_head`. This reproduces the face-derived
   orientation exactly, whatever the neck did.
5. **Fallback.** If either ear point is invalid, the neck takes a per-axis share of `R_rel`
   instead: 40% of yaw, 70% of pitch and 50% of roll, with the head again taking the exact
   remainder.

If no face is detected, neck, head and jaw stay at rest and `valid=False`.

## 5. Hand-to-face geometry (phase A)

**Surface.** SMPL-X vertices in the FLAME map (head and neck), plus every triangle whose three
vertices are all in that set, posed by `assembly.py`.

**Hand points.** Per hand: the wrist joint, the five fingertip vertices (`smplx.vertex_ids`), and
the palm centre (midpoint of the wrist and the middle-finger knuckle joint).

**Distance.** Exact point-to-triangle distance to the nearest triangle. The sign comes from that
triangle's outward normal, so negative means inside the head. The contact flag is
`|d| < 2 cm`. Near concave regions such as the ear the sign can be wrong for deep penetrations;
this is reported as a limitation, not corrected.

**Region labels** are aligned to ASL-LEX, with finer subregions underneath:

| Region | Subregions | Rule on the rest template (head frame) |
| --- | --- | --- |
| Neck | — | Skinning weight dominated by the neck joint |
| UnderChin | — | Jaw-dominated, normal pointing down (n_y < −0.5) |
| Chin | — | Jaw-dominated, below the lower lip, front-facing |
| Mouth | — | Inside the outer-lip contour, in the frontal projection |
| UpperLip | — | Between the subnasale (point 2) and the upper outer lip, between the mouth corners |
| Eye | — | Within a radius of the eye contour, plus the eyeball vertices |
| CheekNose | Nose, Cheek | Remaining front-facing face; Nose inside the nose landmarks' outline |
| Forehead | Brow | Above the brow landmarks and front-facing; Brow in a band around them |
| Ear | — | Near the SMPL-X ear vertices, beyond the outer eye corners laterally |
| HeadTop | — | Remaining scalp, including the back of the head |

The rules use the transferred landmarks and SMPL-X skinning weights, so no further licensed asset
is needed. Thresholds are set by inspecting a rendered image of the segmentation, which goes into
the evidence for the gate. Each hand point takes the label of its nearest surface point.

## 6. The face stream (phase C)

Per frame:

1. Fit a scaled Umeyama transform from the observed rigid-set pixel points to the posed SMPL-X
   rigid-set points, with the occlusion weights from §4.1.
2. Apply it to all 478 MediaPipe points, which puts them in the body's metric frame at the scale
   of the fitted head.
3. Store them relative to the head joint (the inverse of its global transform), so that a later
   re-posing of the head carries them along rigidly.

The 52 blendshapes are stored unchanged as a separate feature stream. FaceLandmarker's head-pose
matrix is kept for verification only (§7, metric 2).

## 7. Verification gate

The gate runs on the 20 clips in `data/round_trip_eval`, which needs a FaceLandmarker pass over
their source videos (`~/Downloads/raw_videos`). Every metric is reported twice: for
**hand-near-face** frames (either hand's padded outline overlaps the bounding box of the 478 face
points) and for the rest. Targets are provisional: as with the body, measure first, then decide.

| # | Metric | Target |
| --- | --- | --- |
| 1 | Weighted RMS residual of the rigid set after the head fit, as % of the outer-eye-corner distance | Report median and p95 against a **floor**: the same residual for MediaPipe's `canonical_face_model.obj` fitted to the SMPL-X neutral template. The floor is the correspondence error that remains even with a perfect fit. |
| 2 | Angle between the fitted head rotation and FaceLandmarker's (scale removed, axes flipped), each taken relative to its clip median | median ≤ 5°, p95 ≤ 10° |
| 3 | Rank correlation per clip between fitted jaw pitch and the `jawOpen` blendshape | ≥ 0.7 |
| 4 | Distance between the observed ear midpoint and the fitted SMPL-X ear midpoint, % shoulder width, with and without the neck swing | Report. This shows whether the swing fixes head placement. |
| 5 | Fraction of frames with any hand point deeper than 1 cm inside the head | Report. A large fraction means the head or hands are misplaced. |
| 6 | Fraction of frames flagged `valid=False`, `reliable=False`, `jaw_reliable=False` | Report |
| C1 | Phase C: residual of the 105 embedded points between the aligned MediaPipe points and their SMPL-X positions, split into rigid and non-rigid points | Report. Non-rigid residual reflects expression, which SMPL-X does not model here. |

If metric 2 fails, the first suspect is assumption F2 (axis conventions), and the second is the
rigid set.

## 8. Assumptions

These rows join the ledger in [V1 problem scope §12](V1_Problem_Scope.md#12-assumptions-ledger)
when implementation starts.

| ID | Assumption | Verified by |
| --- | --- | --- |
| F1 | The MediaPipe→FLAME embedding transfers to SMPL-X correctly. A FLAME version mismatch (FLAME 2023 changed the eye region) would show up as offsets. | Transferred eye corners and nose tip land within a few mm of smplx's own landmarks for the same points; visual check |
| F2 | FaceLandmarker's image axes, after the (x·W, y·H, z·W) conversion, match the pose world frame's axes. | Axis test (§10); metric 2 |
| F3 | The 17-point rigid set is stable enough under expression and partial occlusion to give head rotation. | Metrics 1 and 2, split by hand-near-face |
| F4 | The neck swing toward the observed ear midpoint fixes head placement without touching the arms. | Metric 4; the arm metrics are unchanged |
| F5 | Lip gap is a usable proxy for jaw pitch with expression at zero. | Metric 3 |
| F6 | Region rules on the rest template give anatomically sensible ASL-LEX regions. | Rendered segmentation |

## 9. Constraints recorded for later augmentation work

- **Location and contact are invariants.** An augmentation must not move a hand point across a
  region boundary, or into or out of contact, because that can change the sign.
- **Non-manual signals are not augmented.** Brow raises, headshakes, head tilts and mouthing carry
  grammar. Small static rigid offsets are probably safe (*inferred*).
- **Blendshapes are not mixed into SMPL-X expression.** They remain a separate stream.

## 10. Tests

Tests that need model assets skip without them, using the existing `requires_smplx` pattern in
`tests/hand_assets.py`.

- **Correspondence.** The embedding has 105 points; the rigid set and the jaw pair are among them;
  transferred vertex ids are in range; left and right are consistent with `FACE_LEFT_TO_RIGHT`;
  F1's check on eye corners and nose tip.
- **Synthetic head round trip.** Pose SMPL-X with known neck, head and jaw under a non-identity
  `global_orient`. Read the rigid-set points from the mesh, apply a random scaled transform and
  noise, and fit. Head orientation is recovered within 0.5° and jaw pitch within 1°.
- **Occlusion.** Points inside a synthetic hand outline get weight 0; with fewer than 8 rigid
  points left, the frame is flagged unreliable.
- **Neck swing.** With a synthetic head lean, the swing recovers the ear midpoint, `head_pose`
  still reproduces `R_head` exactly, and the arm joints do not move.
- **Axes.** Pins the image-to-world axis convention and the OpenGL flip of the head-pose matrix.
- **Assembly regression.** `assembly.py` reproduces the current script's output on a stored clip.
- **Distances.** Points at known offsets along surface normals give the exact signed distance;
  labelled anchor points (chin, forehead, mouth) get the right region.
- **Face stream.** Re-posing the head moves the stored 478 points rigidly with it.

## 11. Assets, licences and housekeeping

- `scripts/download_models.py` fetches `face_landmarker.task` (Apache-2.0), MediaPipe's
  `canonical_face_model.obj` (Apache-2.0) and `mediapipe_landmark_embedding.npz` (MPI
  non-commercial). The embedding is git-ignored, like the SMPL-X weights.
- `pyproject.toml` requires `mediapipe` (unpinned) and pins `mediapipe==0.10.21` in the `dev`
  extra. That version includes the Tasks FaceLandmarker.
- Documentation: update the out-of-scope table in [V1 problem scope](V1_Problem_Scope.md) (§3),
  add the ledger rows from §8, and update every affected `index.md`.
