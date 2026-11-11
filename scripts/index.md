# scripts/

Run from the repo root.

| File | Description |
|---|---|
| `download_models.py` | Check SMPL-X model files (`--check`), fetch MediaPipe face assets (`--face`) |
| `convert_mano.py` | Convert MANO `.pkl` to chumpy-free `.npz` |
| `extract_world_landmarks.py` | Cache MediaPipe pose world landmarks for metric fitting |
| `extract_hand_landmarks.py` | Cache MediaPipe hand world/image landmarks plus pose (`*_hands.npz`) |
| `extract_face_landmarks.py` | Cache MediaPipe FaceLandmarker output (`*_face.npz`) |
| `extract_face_features.py` | Write per-clip fitted face features and hand-to-face geometry |
| `export_round_trip_demo.py` | Export a clip's round trip and augmentations for `demo/round_trip` |
| `augment_trials.py` | Apply Tier 1 augmentations to whole clips through the SMPL-X + MANO round trip |
| `demo_smplx_augmentation.py` | Demo of anatomically defined edits through SMPL-X |
| `evaluate_round_trip.py` | Run and score the body + hand round trip on any footage |
| `evaluate_face.py` | V1.3 phase-A face gate metrics over clips |
| `measure_round_trip.py` | Measure the cost of the V1 body round trip |
| `measure_hand_round_trip.py` | V1.2 hand round-trip pilot measurement |
| `view_round_trip.py` | Render the body round trip as a three-panel video (also shared helpers) |
| `view_body_hand_round_trip.py` | Render the body + hand round trip of one clip |
| `legacy/` | **Legacy.** Phase smoke scripts and old renderers |
