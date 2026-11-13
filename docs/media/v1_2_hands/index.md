# V1.2 hand round-trip figures index

Written by `scripts/visualize_hand_round_trip.py` from the pilot hand clips and the in-sample-shape fits.

- `01_detection_overview.png` — One video frame per clip with each detected hand's 21 image landmarks, the pose wrists and the detector's raw label.
- `02_detection_timeline.png` — Per clip, the frames with a left hand, a right hand, no hand and an unassigned detection, with the coverage counts.
- `03_round_trip_examples.png` — The best, median and worst frame-sides by MANO error: observed landmarks against the MANO and SMPL-X rebuilds, front and side views.
- `04_round_trip_mesh.png` — The fitted MANO mesh of the median example from two angles, with the observed and rebuilt landmarks.
- `05_overlay_on_video.png` — The same three examples cropped from the video, with observed and MANO landmarks drawn through one fitted scale and shift.
- `06_hand_size_drift.png` — Per track, MediaPipe's wrist-to-middle-knuckle length over time against the fitted MANO hand's fixed length and the per-frame MANO error.
- `index.md` — Index of the V1.2 hand round-trip figures in this folder.
