# Test output index

- `01986_analysis.png` — Pose and landmark analysis plot generated from clip 01986.
- `01986_landmarks.npy` — NumPy landmark sequence extracted from clip 01986.
- `01987_analysis.png` — Pose and landmark analysis plot generated from clip 01987.
- `01987_landmarks.npy` — NumPy landmark sequence extracted from clip 01987.
- `01988_analysis.png` — Pose and landmark analysis plot generated from clip 01988.
- `01988_landmarks.npy` — NumPy landmark sequence extracted from clip 01988.
- `01991_analysis.png` — Pose and landmark analysis plot generated from clip 01991.
- `01991_landmarks.npy` — NumPy landmark sequence extracted from clip 01991.
- `01992_analysis.png` — Pose and landmark analysis plot generated from clip 01992.
- `01992_landmarks.npy` — NumPy landmark sequence extracted from clip 01992.
- `hand_round_trip.json` — Per-track and pooled V1.2 hand round-trip errors, observability and environment hashes from `scripts/measure_hand_round_trip.py`.
- `hand_scale_drift.json` — Hand round trip with a per-frame scale about the wrist: pooled and per-track errors, fitted scales and what they follow, from `scripts/measure_hand_scale_drift.py`.
- `test_summary.json` — Per-video extraction metrics plus joint-mapping and conversion status from the reference run.
- `videos/` — Rendered YAML augmentation overlays and a side-by-side preview for clip 01986.
- `index.md` — Index of extracted arrays, analysis plots, and run summary in this folder.
- `*_world.npz` — Metric MediaPipe world landmarks and visibility, used for body-model fitting; extracted one MediaPipe instance per clip.
- `*_hands.npz` — MediaPipe hand world/image landmarks per anatomical side with the paired pose, for the V1.2 hand round trip.
