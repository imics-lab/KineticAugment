# `noise-floor/`

**Tester** (first-pass measurement) for cluster **noise-floor** — A004: noise-floor estimators, MediaPipe smoothing, frame-rate dependence.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/noise-floor .venv/bin/python scripts/review_checks/noise-floor/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `analyze.py` | A004 / noise-floor cluster: empirical checks. |
| `analyze_out.txt` | Captured output (from the script matching `analyze*`). |
| `check_c2.py` | Sanity check the model_complexity 1 vs 2 disagreement and per-clip structure. |
| `check_c2.txt` | Captured output (from the script matching `check_c2*`). |
| `check_c2b.py` | Is the full-vs-heavy disagreement a left/right swap? Compare swapped correspondences, |
| `check_c2b.txt` | Captured output (from the script matching `check_c2b*`). |
| `check_c2c.py` | Per-axis split of the full-vs-heavy disagreement, and of benign-perturbation disagreement. |
| `check_c2c.txt` | Captured output (from the script matching `check_c2c*`). |
| `check_repro.py` | Check that our default re-extraction reproduces the repo npz, and how variants differ. |
| `check_repro.txt` | Captured output (from the script matching `check_repro*`). |
| `framerate.py` | Frame-rate sensitivity of the script's estimators: decimate existing tracks by 2 and 3 |
| `framerate.txt` | Captured output (from the script matching `framerate*`). |
| `identify_filter.py` | Identify the temporal filter MediaPipe applies to pose_world_landmarks. |
| `identify_filter.txt` | Captured output (from the script matching `identify_filter*`). |
| `noise_floor_repo_current.txt` | Captured output (from the script matching `noise_floor_repo_current*`). |
| `noise_floor_run.txt` | Captured output (from the script matching `noise_floor_run*`). |
| `reextract.py` | Re-extract MediaPipe Holistic world landmarks under several configs/perturbations. |
| `reextract_log.txt` | Captured output (from the script matching `reextract_log*`). |
| `round_trip_current.txt` | Captured output (from the script matching `round_trip_current*`). |
| `runs/` | Subfolder (see its index.md) |
| `index.md` | This index. |
