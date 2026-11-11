# data/

| Item | Description |
|---|---|
| `sign_videos/` | Five sample WLASL sign clips (licence unclear) used by the extraction examples |
| `test_output/` | Landmark caches, plots and summaries: legacy phase-1 smoke output plus the V1 to V1.3 world, hand and face caches and round-trip measurements for the sample clips |
| `augment_trials/` | Output of `scripts/augment_trials.py` / `scripts/augment_face_trials.py`: per-clip augmentation videos and JSON measurements (videos not committed; regenerate with the scripts) |
| `hand_variants/` | Output of `scripts/export_hand_variants.py`: every clip's hands rebuilt by the two hand pipelines (one shape per track, with and without a per-frame scale), for a classifier comparison |
| `round_trip_eval/` | Default output of `scripts/evaluate_round_trip.py`: evaluation set of 20 unseen clips, mostly not committed (derived from footage outside the repo); holds `*_hands.npz`, `*_world.npz`, the face caches `*_face.npz` and `*_face_features.npz`, and the gate report `face_report.json` |
| `wlasl/` | WLASL annotation file (not committed) and the clip → gloss map with mirror safety |
| `index.md` | This index |
