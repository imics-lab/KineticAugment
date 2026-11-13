# V1.3 face integration implementation plan

Date: 2026-10-11. Baseline: `2205f16`. Spec: the user's request to implement the recommendations
following [the training-data audit](V1_3_Training_Data_Audit.md), for landmark/geometry recognition.

## Scope and acceptance

Deliver observed and head-local facial motion alongside the existing body/hands, independent
quality masks and provenance, synchronized sample transformations, a usable WLASL training adapter,
a small masked multi-stream classifier, and empirical integration results. Preserve the default
body/hand fit. Absolute-depth/contact recovery and a learned joint camera optimizer remain separate
research: the recommendation made those conditional on independent geometry evidence.

Success means lossless numerical face attachment inversion, unchanged arm/hand reconstruction in
the default path, unavailable geometry masked, no cross-stream timing drift, finite model training
with missing modalities, and results on the available real clips. Recognition accuracy improvements
require a meaningful labelled held-out dataset and must not be inferred from smoke training.

## Global Constraints

- Work only in this existing feature worktree; preserve pre-existing audit files and the untracked
  `data/round_trip_eval` symlink. Never overwrite source observation/feature caches or licensed assets.
- Python 3.9+, NumPy/SciPy library geometry, optional PyTorch for model code. Black line length 100.
- Keep legacy 543-landmark readers and existing fitting APIs working. New feature archives are
  explicitly versioned; reject incompatible archives rather than silently guessing layouts.
- Use `rtk proxy` for exact/parsed/saved output and `PYTHONPATH=src .venv/bin/python` for this branch.
- No new external service, asset download, package install, push, merge or publication. Local commits
  are appropriate. Stage only the task's files and named index edits, never `git add .`.
- Each agent works itself without spawning subagents. Full reports and test output go in the task
  workspace; return only short status, commit(s), test summary, concerns and report path.
- Keep affected folder indexes current. Meaningful behavior tests required; focused checks per task,
  final full suite once. Extraction checks may need macOS OpenGL outside the sandbox.
- Use independent masks: observations are not lost because body fitting failed. A boolean contact
  without a usable geometry mask must never reach training as a measured negative.

## Assumptions and gate

| ID | Claim | Evidence / status | Verification / failure signal |
| --- | --- | --- | --- |
| FI01 | Similarity + head transforms can preserve the full observed face | NumPy/SciPy alignment and actual skeleton APIs exist; likely | Inverse transform and re-pose tests; failure is lost deformation or wrong axes |
| FI02 | Default head attachment can leave body/hands unchanged | Existing `attach_head` changes neck/head/jaw only; likely | Compare full arm and hand geometry before/after with real assets |
| FI03 | Source observations share identifiers and frame times | FaceClip and HandClip expose metadata, size and timestamps; confirmed interfaces | Reject contradictory hashes, dimensions and time grids |
| FI04 | New streams can reach a classifier with safe missingness | Current reader only supports legacy arrays; implementation required | Dataset -> collate -> classifier -> trainer smoke with unavailable face |
| FI05 | Contact can be used as certified labels | Audit falsified sufficiency of current gates; risky | Keep experimental and masked; no claim of metric truth |
| FI06 | Recognition gains can be measured locally | Labelled split inventory not established; unknown | Inspect available annotations; report limitations if no valid evaluation corpus |

Gate: **go-with-risks** for FI01–FI04; FI05 is excluded from certified training labels. FI06 is
not a prerequisite for validating implementation, but it is required for an accuracy-gain claim.

## Task 1: Reversible full-face attachment primitives

Implement `src/kinetic_augment/body_model/face_stream.py`, its focused tests, nearest indexes.
Relevant existing interfaces: alignment.similarity_transform/apply_similarity, face_correspondence.RIGID,
face_observations.FaceClip, assembly/global_rotations. Do not modify pipeline yet.

Provide a small `FaceAttachment` dataclass with `local` (478,3), `world_from_head` (4,4),
`pixel_to_world` (4,4 similarity), and explicit validity. Public helpers:
`attach_face(face_px, rigid_weights, posed_rigid, head_rotation, head_position)`,
`to_head_frame(points, rotation, position)`, `from_head_frame(points, rotation, position)`,
`transform_points(points, matrix)` (homogeneous affine, preserving missing rows).
Use one weighted similarity from observed RIGID points to posed RIGID points; transform all 478
points and express them in the specified head joint frame. Never replace observed facial
deformation with zero-expression mesh vertices. Head transform is SE(3), attachment similarity
is distinct. Return invalid NaN attachment for insufficient/degenerate anchors or non-finite
head transform, with documented shape validation errors. Finite valid individual rows survive
missing unrelated points; exclude invalid/zero-weight anchors before fitting. Validate rotations
as proper orthonormal matrices, not arbitrary finite arrays. No implicit coordinate conversion.

Test a known nonidentity rotation, translation and scale, inverse mapping to original observations,
nonrigid mouth/brow changes survive, arbitrary rigid re-posing, partial missing landmarks and
zero/degenerate anchor failures. Use realistic tolerances (1e-8 synthetic float64).
Run focused tests, self-review and commit; report exact commands/results.

## Task 2: Integrate attachment, masks and versioned feature exports

Depends on Task 1. Modify face_pipeline.py, small supporting modules as needed,
scripts/extract_face_features.py, tests and indexes. Keep `run_clip` and `save_face_features`
calls compatible. Extract serialization to a focused new `face_features.py` if needed rather
than growing face_pipeline.py excessively; re-export loader from that module.

Add default-on attachment with opt-out `attach_face_stream=False` ONLY for regression comparison.
Default depth correction remains False. For reliable fitted heads attach all 478 points using
actual mesh head joint position and global head rotation. Add to ClipFaceResult and NPZ:
`face_landmarks` (T,478,3 normalized original), `face_pixels` (T,478,3),
`face_stream_local` (T,478,3), `head_transform` (T,4,4 world_from_head),
`face_pixel_to_world` (T,4,4), `face_observed` (T bool), `face_attachment_valid` (T bool),
`face_point_valid` (T,478 bool finite/detected), `face_point_visible` (T,478 bool estimated
not-in-padded-hand-outline, not detector confidence), `head_world_rotation` (T,3,3),
`head_relative_rotation` (T,3,3 torso-relative),
`pose_image` (T,33,3), `pose_world` (T,33,3), `pose_visibility` (T,33),
`hand_image` (T,2,21,3), `hand_world` (T,2,21,3), `hand_present` (T,2),
`geometry_valid` (T,2 bool), `timestamp_ms`, `width`, `height`, `fps`.
Existing feature keys stay. Raw face and blendshapes must survive invalid body fits.
Mask geometry when face/head unreliable, body invalid, hand unposed, or corrected wrist
unreached: distance NaN, region -1, contact False, geometry_valid False. Publish an explicit
`contact_valid` all-False mask until independently calibrated; geometry_valid only denotes
usable fitted geometry, not ground truth contact. Keep unmasked diagnostics in a documented
separate field if needed for the viewer/audit, without presenting them as accepted geometry.

Before fitting verify shapes, positive image dimensions, same width/height, equal finite strictly
increasing timestamps and consistent frame counts. When both source hashes are supplied reject
a mismatch; when absent mark provenance incomplete. Do not require hashes for old test fixtures.
Use feature schema `FACE_FEATURE_SCHEMA = 1` in meta, preserving original face/hand metadata,
source identity, coordinate conventions, depth_corrected and fov_deg, selected frame indices,
extractor/model identifiers, and readable explicit unknowns for unavailable hashes/revision.
Add `load_face_features(path)` returning a plain dictionary including decoded `meta`, validates
schema and essential shapes. Add `--output-dir` to feature extraction, defaulting compatibly.

Fix hand_near_face's polygon edge-crossing false negative. Test absent face/body separately,
occluded region masks, source mismatch, serialization, invalid geometry and corrected-unreached
contact rejection. Asset regression on 01991 sample frames: on/off attachment identical rotations,
translation, arms/hands geometry; face inverse reconstruction max <1e-6 pixels float64 storage.
Run focused existing face/pipeline/depth tests too. Do not overwrite cached features.

## Task 3: Training sample adapter and synchronized temporal operations

Depends on Task 2. Add `evaluation/datasets/face_features.py` plus tests and exports/indexes.
No classifier changes here. Implement `FaceTrainingSample` using dictionaries `streams` and
`masks` of aligned (T,D) arrays, timestamp_ms, source-frame mapping and metadata. Preserve
raw archives as the authoritative geometric representation, not inside every model tensor.

`sample_from_face_features(features, face_space="image", include_blendshapes=True)` creates:
- `body_hands`: image x/y for pose joints (11,12,13,14,15,16) and all 21 points each hand,
  isotropic pixel units normalized by reliable clip median shoulder width and per-frame shoulder
  midpoint. Mask unavailable pose points/hand sides and unavailable normalization anchors.
- `face`: compact 26 unique MediaPipe IDs, exactly
  (61,291,0,17,13,14,78,308,33,133,362,263,70,63,105,66,107,336,296,334,293,300,1,4,168,152).
  This is a documented project subset, NOT a claim of identical CoSign topology. Default image x/y
  centred at eye outer-corner midpoint, normalized by reliable clip median eye distance;
  use face_point_valid AND face_point_visible. Head-local option uses x/y/z with clip median
  local eye distance and centres at local eye midpoint; attachment validity is required.
  Normalize using only valid visible nondegenerate anchors. Do not subtract mean expressions.
- `head`: first two columns of head_relative_rotation flattened in a documented order (6-D),
  reliable-only; include via mask rather than identity for unknown head.
- optional `blendshapes`: original ordered 52 scores; mask invalid/occluded facial observations
  conservatively (mask whole coefficient stream when mouth/eye/brow subset is occluded), keep
  exact names in metadata; standardize missing names for entirely undetected clips or mask safely.
Every value reaching tensors finite, use np.where(mask & finite, value, 0); masks same (T,D).
Do not include metric contact in default streams. Save subset IDs/order, units, source, schema.

Implement `select_frames(indices)` and `resample(timestamp_ms, max_gap_ms=100)` returning new
samples. Apply one time grid to ALL streams and masks. Exact source observations retained;
linear interpolation only where both endpoints observed and span <= max_gap_ms; no extrapolation,
no filling across masked spans. Mark interpolated rows and retain source time mapping. Head
rotations must be interpolated on SO(3) (Slerp then 6-D), not elementwise. Reject unordered or
duplicate input times; output times strictly increasing. Do not smooth or introduce spatial
augmentation by default; re-posing is supported by Task 1 with original local motion preserved.

Implement `WLASLFaceDataset(root_path, feature_dir, annotation_file="WLASL_v0.3.json",
split="train", subset="wlasl100", face_space="image", include_blendshapes=True)` using
official annotation class ordering and splits, `<video_id>_face_features.npz`. Return (sample,label),
preserve source identity; explicit no usable data error instead of silent empty training.
`collate_face_samples(batch)` returns `(x, lengths, labels)` where x is
`{"streams": {name: torch.float32[B,T,D]}, "masks": {name: torch.bool[B,T,D]}}`, padding
masked. All samples must share stream layout, subset and blendshape channel order.
Avoid duplicating the existing WLASL reader wholesale; reuse its annotation logic where clean
or introduce a focused shared helper without changing legacy behavior.

Tests: cross-video/schema/shape mismatches, variable lengths, source subsets, fully missing face
retains body, facial observations survive unavailable body, NaN sanitation, synchronized time
selection/resampling including gap/missing guards and a 180deg rotation case. Tiny annotation
fixture demonstrates split/class consistency and no augmented leakage by source split.

## Task 4: Masked facial temporal branch and existing trainer integration

Depends on Task 3. Add `evaluation/models/face_classifier.py`, tests and model exports/indexes.
Implement `FaceStreamClassifier(num_classes, stream_dims, hidden_dim=32, dropout=0.1)` as a
small BaseSLRClassifier using separate per-stream GRU encoders (batch-first) and fusion/classifier.
Accept x dict from collate and optional lengths. Sanitize with torch.where BEFORE projection;
concatenate feature masks to stream inputs so zero expression differs from unknown. Respect
sequence lengths/padding. Gate fully missing streams so changing masked values never changes
predictions, and outputs/gradients are finite with any/all modalities missing. Provide a way to
instantiate body_hands-only vs body_hands+face/head/blendshapes without architectural rewrites.

Modify SLRTrainer input-to-device handling to recurse through this dictionary while preserving
existing tensor batches. No unrelated trainer redesign. End-to-end DataLoader from Task 3 through
trainer must perform optimization and evaluate. Add usage example `examples/train_face_streams.py`
with CLI feature/annotation roots, train/val official split, epochs, face_space, optional output
checkpoint, seeds, and body-only ablation. Default CPU; no download, no accidental test-set tuning.
Tests: padding invariance, masked NaN/garbage invariance, finite backward/update, legacy LSTM
trainer compatibility, and synthetic face-discriminative sequences learn beyond body-only baseline.
Synthetic experiment is a wiring test, never described as recognition benchmark evidence.

## Task 5: Real-clip validation, final tests and documented results

Depends on Tasks 1–4. Implement `scripts/validate_face_integration.py` to run selected/all available
clips through the pipeline with `--output-dir` outside input dirs and separate JSON report.
Use all 5 pilot and 20 evaluation clips if runtime practical; log coverage if a smaller selection
is needed. Preserve source caches. Validate attachment inverse/re-posing error, default-body
regression via attachment on/off (sample frames sufficient), invalid face -> no accepted geometry,
corrected-unreached -> masked geometry (01986 frames14–21), face-only survival without body on
a controlled copy, finite collated multi-stream tensors and classifier optimization.
Store small JSON results under docs, do not commit video/npz/model files. Inspect available label
annotations under project data / parent project data; if not enough real labelled train/val data,
explicitly state no meaningful real recognition accuracy measured, and run synthetic wiring test.

Run the final full suite with PYTHONPATH=src; rerun only extraction failures with graphics access
if sandbox OpenGL fails. Include exact commands/counts and any limitations in
`docs/V1_3_Face_Integration_Results.md`, along with feature contract, reproduction commands and
training example. Update indexes, AGENTS.md/agent.md only if folder locations or critical setup
conventions changed. Original audit remains historical, link from it to new results if useful.
Self-review, commit and report. Do not modify earlier production logic unless a measured validation
failure demands a focused fix, and disclose it for independent task review.
