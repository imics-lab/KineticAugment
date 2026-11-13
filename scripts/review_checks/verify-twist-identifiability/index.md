# `verify-twist-identifiability/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **twist-identifiability** — A003: twist/rotation identifiability, joint-angle plausibility on real data.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-twist-identifiability .venv/bin/python scripts/review_checks/verify-twist-identifiability/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `v1_harness_rot.out` | Captured output (from the script matching `v1_harness_rot*`). |
| `v1_harness_rot.py` | Independent replica of measure_round_trip.run_synthetic that also scores rotations. |
| `v2_nullspace.out` | Captured output (from the script matching `v2_nullspace*`). |
| `v2_nullspace.py` | Null-space check on the REAL SMPL-X forward pass (float64). |
| `v3_hinge_range.out` | Captured output (from the script matching `v3_hinge_range*`). |
| `v3_hinge_range.py` | Verification script (no docstring); see the cluster summary above. |
| `v4_real_plaus.out` | Captured output (from the script matching `v4_real_plaus*`). |
| `v4_real_plaus.py` | Plausibility of fit_body output on the real world-landmark clips. |
| `v5_limits.out` | Captured output (from the script matching `v5_limits*`). |
| `v5_limits.py` | zero-noise clamp on real data, using fits after a fixed MP->SMPL-X flip (anatomically oriented root) |
| `v6_aug_twist.out` | Captured output (from the script matching `v6_aug_twist*`). |
| `v6_aug_twist.py` | Same additive rotvec noise on two parameter sets with identical joints (differ only by a compensated humerus twist). |
| `v7_palm_roll.out` | Captured output (from the script matching `v7_palm_roll*`). |
| `v7_palm_roll.py` | Is palm roll about the forearm observable from MediaPipe pose landmarks 17-22? |
| `index.md` | This index. |
