# `main-check/`

Coordinator spot-checks for cluster **main-check** — Final spot-checks by the coordinating agent: upside-down torso and elbow hinge axis.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/main-check .venv/bin/python scripts/review_checks/main-check/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `elbow.py` | Verification script (no docstring); see the cluster summary above. |
| `upside.py` | Independent check: is the fitted SMPL-X body upside down relative to the MediaPipe data? |
| `index.md` | This index. |
