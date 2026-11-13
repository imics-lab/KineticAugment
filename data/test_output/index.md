# data/test_output/

Generated artifacts; not source. The `*_landmarks.npy`, `*_analysis.png` and `test_summary.json`
files come from `scripts/legacy/phase1_smoke.py`; only `01986_landmarks.npy` of those is read (by
`examples/run_augmentation.py` and the phase-2 smoke script). The `*_world.npz`, `*_hands.npz`,
`*_face*.npz` caches and the JSON measurements feed the V1 to V1.3 round trip.

| File | Description |
|---|---|
| `01986_landmarks.npy` | MediaPipe landmark cache for clip 01986, shape (frames, 543, 3) |
| `01986_analysis.png` | Analysis plot for clip 01986 |
| `01987_landmarks.npy` | Landmark cache for clip 01987 |
| `01987_analysis.png` | Analysis plot for clip 01987 |
| `01988_landmarks.npy` | Landmark cache for clip 01988 |
| `01988_analysis.png` | Analysis plot for clip 01988 |
| `01991_landmarks.npy` | Landmark cache for clip 01991 |
| `01991_analysis.png` | Analysis plot for clip 01991 |
| `01992_landmarks.npy` | Landmark cache for clip 01992 |
| `01992_analysis.png` | Analysis plot for clip 01992 |
| `test_summary.json` | Per-video detection statistics written by the phase-1 script |
| `hand_round_trip.json` | Per-track and pooled V1.2 hand round-trip errors, observability and environment hashes from `scripts/measure_hand_round_trip.py` |
| `hand_scale_drift.json` | Hand round trip with a per-frame scale about the wrist: pooled and per-track errors, fitted scales and what they follow, from `scripts/measure_hand_scale_drift.py` |
| `*_world.npz` | Metric MediaPipe world landmarks and visibility, used for body-model fitting; extracted one MediaPipe instance per clip |
| `*_hands.npz` | MediaPipe hand world/image landmarks per anatomical side with the paired pose, for the V1.2 hand round trip |
| `*_face.npz` | MediaPipe FaceLandmarker landmarks, blendshapes and head matrix per frame, frame-aligned with `*_hands.npz`, for the V1.3 face work |
| `*_face_features.npz` | Per-frame V1.3 face features (neck/head/jaw fit and flags, signed hand-to-face distances, regions, contact, blendshapes) from `scripts/extract_face_features.py` |
| `*_face_features_dc.npz` | The same features after the depth-corrected second pass (image-derived wrists), plus wrist target, corrected flag, depth ratio, residual and shift, from `extract_face_features.py --depth-corrected` (V1.3 task 14) |
| `videos/` | Rendered YAML augmentation overlays and round-trip previews for the sample clips (see its `index.md`) |
| `index.md` | This index |
