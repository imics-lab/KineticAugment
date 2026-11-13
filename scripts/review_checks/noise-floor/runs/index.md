# `noise-floor/runs/`

Captured `measure_noise_floor.py` / `measure_round_trip.py` outputs on landmarks re-extracted under different MediaPipe settings (the `.npz` extractions were not kept — regenerate with `../reextract.py`).

- `default/` — Holistic defaults, one instance per clip (smoothing on).
- `default_sharedgraph/` — Defaults with one Holistic instance shared across clips, which is how `scripts/extract_world_landmarks.py` works.
- `nosmooth/` — `smooth_landmarks=False`.
- `nosmooth_resize75/` — No smoothing, frames resized to 75%.
- `resize75/` — Smoothing on, frames resized to 75%.
- `static/` — `static_image_mode=True` (independent per-frame detection).
- `complexity2/` — `model_complexity=2` (heavy pose model).
- `index.md` — This index.
