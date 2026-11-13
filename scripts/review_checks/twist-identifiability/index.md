# `twist-identifiability/`

**Tester** (first-pass measurement) for cluster **twist-identifiability** — A003: twist/rotation identifiability, joint-angle plausibility on real data.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/twist-identifiability .venv/bin/python scripts/review_checks/twist-identifiability/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `armlib.py` | Shared helpers for the twist-identifiability cluster (A003). float64 SMPL-X forward. |
| `t0_frame_check.out` | Captured output (from the script matching `t0_frame_check*`). |
| `t0_frame_check.py` | Check the coordinate convention the delivered converter actually produces on real clips. |
| `t1_identifiability.out` | Captured output (from the script matching `t1_identifiability*`). |
| `t1_identifiability.py` | A003: which arm twists are identifiable from shoulder/elbow/wrist joint positions? |
| `t1b_extra.out` | Captured output (from the script matching `t1b_extra*`). |
| `t1b_extra.py` | Supplements to t1: (A') forearm twist compensated by the 3-DOF wrist -> all 55 joints identical; |
| `t2_harness_rotation.out` | Captured output (from the script matching `t2_harness_rotation*`). |
| `t2_harness_rotation.py` | A003 step 1: replicate measure_round_trip.run_synthetic (seed 42, 200 trials, N(0,0.4) per |
| `t3_hinge.out` | Captured output (from the script matching `t3_hinge*`). |
| `t3_hinge.py` | A003 step 2b: which twists are identifiable?  Real SMPL-X forward (float64). |
| `t4_real_plausibility.out` | Captured output (from the script matching `t4_real_plausibility*`). |
| `t4_real_plausibility.py` | A003 step 4: anatomical plausibility of the analytic-IK parameters on the five real clips. |
| `t5_downstream.out` | Captured output (from the script matching `t5_downstream*`). |
| `t5_downstream.py` | A003 step 4b: does twist (non-)identifiability matter downstream? |
| `t6_torso_angle.out` | Captured output (from the script matching `t6_torso_angle*`). |
| `t6_torso_angle.py` | Size of the torso-frame rotation (shoulders+nose vs SMPL-X rest) on real clips, and its |
| `index.md` | This index. |
