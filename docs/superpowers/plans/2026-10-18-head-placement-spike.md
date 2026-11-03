# Head placement spike on clip 08925

**Kind:** throwaway experiment (spike). Minimal tests, no library refactors.
**Branch / worktree:** `exp/head-placement-08925` at `.claude/worktrees/head-placement-exp`, cut from
`feat/face-v1.3` (c818dd9).
**Spec:** none written. The user asked in chat for: (1) a pose-only MediaPipe extraction of clip
08925 and a video showing how MediaPipe's own pose face points (0-10) relate to the pose hand
points, and (2) an experiment placing the SMPL-X head from all 11 pose face points instead of the
ear midpoint, with a video. The depth-from-apparent-size idea already exists as task 14
(`hand_depth.py`, `run_clip(depth_corrected=True)`) and was run on 08925 already; it is not
re-implemented here.

## Global constraints

- Run Python as `PYTHONPATH=src:scripts .venv/bin/python ...` from the worktree root. The shared
  venv's editable install points at another checkout's `src`, which lacks the face modules.
- Inputs (already extracted): `data/test_output/exp_08925/08925_hands.npz`,
  `data/test_output/exp_08925/08925_face.npz`. Video:
  `/Users/bemnetbeshah/Workspace/cv-research/KineticAugment/data/metsis_new_unseen_sign_videos/08925.mp4`
  (41 decoded frames, 288x192 px, 29.97 fps). Write every output under
  `data/test_output/exp_08925/` (git-ignored).
- Python 3.9+, Black-compatible at 100 columns, matplotlib only in scripts (not in `src`).
- Do not modify existing library modules under `src/kinetic_augment`. New experiment code goes in
  new files under `scripts/` (one may add a new module under `src/kinetic_augment/body_model/` if a
  pure-numpy function needs a unit test; tests go under `tests/`).
- Update `scripts/index.md` (and `tests/index.md` if a test is added) for every new file.
- MediaPipe pose indices: 0 nose; 1/2/3 left eye inner/centre/outer; 4/5/6 right eye
  inner/centre/outer; 7/8 left/right ear; 9/10 mouth left/right; 15/16 wrists; 17/18 pinky;
  19/20 index; 21/22 thumb. "Left" is the subject's left.
- Reuse existing rendering helpers (`scripts/view_round_trip.py` `VideoSink`, `read_frames`;
  `scripts/view_proximity_round_trip.py` and `scripts/view_face_round_trip.py` for panel style).

## Task 1: pose-only extraction and pose face/hand proximity video

Create `scripts/extract_pose_only.py`: run the legacy `mediapipe.solutions.pose.Pose` alone (video
mode, `model_complexity=1`, no segmentation) over one video and save
`<output-dir>/<clip>_pose_only.npz` with, per decoded frame: `image` (T, 33, 3) normalised x, y, z;
`visibility` (T, 33); `presence` (T, 33) if available else omit; `world` (T, 33, 3) metres; `valid`
(T,) bool (pose detected); plus `width`, `height`, `fps`. NaN where no pose.

Create `scripts/view_pose_face_proximity.py` that reads that file and the video and writes
`<output-dir>/<clip>_pose_face_proximity.mp4` plus 2-3 PNG stills. It uses only the pose-only
landmarks (no SMPL-X, no FaceLandmarker). Per frame, for the face set {0..10} and the hand set
{15, 17, 19, 21} (left) / {16, 18, 20, 22} (right), find the closest (hand point, face point) pair
per hand and report:
- 3-D distance in pose-world metres;
- its depth component |dz| (world z) and image-plane component (world x, y);
- image distance in pixels and in metres, converted with a per-frame pixels-per-metre scale (a 2-D
  similarity fit from world x, y to image pixels over the visible pose points 0-24);
- a CONTACT flag when the 3-D distance is under 2 cm, and an image NEAR flag when the image
  distance in metres is under 2 cm.
Panels: (1) video frame with the pose skeleton, face points and hand points drawn in distinct
colours and the closest pair joined by a line, with per-hand readouts; (2) front view (x, y) and
(3) side view (z, y, depth to the right) of the world landmarks with the same pair line;
(4) a strip over all frames of the per-frame minimum 3-D distance, image-plane distance and |dz|,
with the 2 cm band and a cursor at the current frame. Print a summary: frames, frames with a
hand, median/min of each distance, CONTACT and NEAR frame counts.

Done when both scripts run on 08925 and the mp4 plays (41 frames).

## Task 2: head placement from the 11 pose face points

Today the neck swings so that the SMPL-X ear midpoint points at the observed pose ear midpoint
(`head_converter.split_neck_head`, `face_pipeline.run_clip`). The head's global orientation comes
from the FaceLandmarker rigid fit and is reproduced exactly whatever the neck does, so the neck
only moves the head joint's position. The experiment replaces the one-point target with a
weighted least-squares target from up to 11 pose face points.

Correspondence (pose index -> SMPL-X point, posed with the clip's betas):
- 0 nose -> `smplx.vertex_ids` "nose" vertex
- 2 / 5 eye centres -> "leye" / "reye" vertices
- 1 / 3 left eye inner / outer -> transferred face-mesh points 362 / 263
  (`face_correspondence`); 4 / 6 right eye inner / outer -> 133 / 33
- 7 / 8 ears -> `head_converter.EAR_VERTICES` (lear, rear)
- 9 / 10 mouth left / right -> transferred face-mesh points 291 / 61
Verify the side conventions on the data: in the image, each pose eye-corner / mouth-corner point
should land on its face-mesh counterpart (FaceLandmarker pixels from `08925_face.npz`) and not on
the mirrored one. Report the median pixel offset per pair; swap a pair and record it if the data
says the mapping is mirrored.

Weights: 1 for a point with pose visibility >= 0.5 that is not inside a padded hand outline
(`head_converter.hand_outlines`, `occluded`, pad as in `fit_head`); else 0.

Neck fit: with the head's world rotation R_head fixed (from the existing fit), each SMPL-X point is
J_head + R_head o_k, where o_k is its rest offset from the head joint. The head-joint target is the
weighted mean of (observed_k - R_head o_k). Swing the neck so the neck-to-head-joint direction
points at that target, keep the existing yaw-share twist, and recompute head_pose so R_head is
still reproduced exactly (same contract as `split_neck_head`). Unit-test this function on a
synthetic case: pose a known neck, generate observed points, and recover the head joint to under
1 mm.

Create `scripts/exp_head_placement.py` that runs the face pipeline's first pass on 08925 (no depth
correction), and for each frame computes three placements: neck at rest, ear-midpoint swing
(current), 11-point swing (new). Per frame and placement record: weighted RMS and per-point
residual of the 11 points (cm and % shoulder width); ear-midpoint residual (cm); the head-joint
shift between the ear and 11-point placements (cm); and the minimum hand-to-head-surface distance
(`hand_face_geometry`) with each placement. Also fit an independent head rotation by weighted
Kabsch from the SMPL-X rest points to the observed pose face points and report its angle to
R_head in degrees. Save per-frame arrays to `<clip>_head_placement.npz` and print medians.

Video `<clip>_head_placement.mp4` plus 2-3 stills: (1) video frame with the observed pose face
points and the three placements' SMPL-X points projected into the image (use a per-frame 2-D
similarity from world x, y to pixels fitted on the pose points, as in Task 1); (2) front and
(3) side 3-D views of observed points vs the ear-swing and 11-point SMPL-X points; (4) a strip of
the 11-point RMS residual for the three placements and the min hand-to-head distance for the
ear and 11-point placements, with a cursor.

Done when the test passes, the script runs on 08925 and the mp4 plays.
