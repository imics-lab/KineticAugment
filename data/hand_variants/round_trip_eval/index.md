# Hand pipeline variants: round_trip_eval

Written by `scripts/export_hand_variants.py --data-dir data/round_trip_eval` (MANO). Layout and meaning: `docs/Hand_Pipeline_Variants.md`.

- `<clip>_hand_variants.npz` — One per clip: MediaPipe's hand landmarks and the hands rebuilt by `track_shape`, `track_shape_scale` and `track_shape_scale_clean`, with their fitted parameters.
- `manifest.json` — Input hashes and per-clip and pooled mean errors against MediaPipe.
- `index.md` — This index.
