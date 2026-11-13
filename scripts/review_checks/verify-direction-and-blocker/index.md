# `verify-direction-and-blocker/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **direction-and-blocker** — Step 7 wiring blocker, camera/delta-transfer feasibility, augmentation reachability.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-direction-and-blocker .venv/bin/python scripts/review_checks/verify-direction-and-blocker/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `v1_orientation.py` | Independent check: is the current fit_body output upside-down / backward in MediaPipe world frame? |
| `v1_out.txt` | Captured output (from the script matching `v1*`). |
| `v2_out.txt` | Captured output (from the script matching `v2*`). |
| `v2_step8_flex_clamp.py` | Independent: (a) a shoulders+nose Kabsch root (my own implementation) -> upright? elbow sign? |
| `v3_misc.py` | (1) 180deg about shoulder axis invariance of arm landmarks |
| `v3_out.txt` | Captured output (from the script matching `v3*`). |
| `v3b_flip_debug.py` | Verification script (no docstring); see the cluster summary above. |
| `v3b_out.txt` | Captured output (from the script matching `v3b*`). |
| `v3c_flip_fixed.py` | Verification script (no docstring); see the cluster summary above. |
| `v3c_out.txt` | Captured output (from the script matching `v3c*`). |
| `v4_camera_transfer.py` | Independent re-check of D-BLOCK-2 numbers using the tester's single-pass Holistic extraction (data only). |
| `v4_out.txt` | Captured output (from the script matching `v4*`). |
| `v4b_camera_only_gap.py` | Independent re-check of D-BLOCK-2 numbers using the tester's single-pass Holistic extraction (data only). |
| `v4b_out.txt` | Captured output (from the script matching `v4b*`). |
| `v5_out.txt` | Captured output (from the script matching `v5*`). |
| `v5_reextraction_spread.py` | Side observation: cached world landmarks vs the tester's re-extraction (same videos, stated-identical |
| `v6_elbow_axes.py` | Verification script (no docstring); see the cluster summary above. |
| `v6_out.txt` | Captured output (from the script matching `v6*`). |
| `index.md` | This index. |
