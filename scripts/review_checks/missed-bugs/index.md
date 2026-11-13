# `missed-bugs/`

**Tester** (first-pass measurement) for cluster **missed-bugs** — Independent bug hunt: conventions, orphan correspondences, noise per axis, elbow jumps.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/missed-bugs .venv/bin/python scripts/review_checks/missed-bugs/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `bone_lengths_fresh.out` | Captured output (from the script matching `bone_lengths_fresh*`). |
| `bone_lengths_fresh.py` | Finding 10's per-clip bone-length table: stored (shared Holistic instance) vs fresh instance per clip. |
| `collinear.out` | Captured output (from the script matching `collinear*`). |
| `collinear.py` | fit_body accepts any 3 valid correspondences (MIN_CORRESPONDENCES = 3) and has no rank check. |
| `convention_check.out` | Captured output (from the script matching `convention_check*`). |
| `convention_check.py` | Does fit_body produce anatomically meaningful SMPL-X parameters on REAL |
| `convention_min.out` | Captured output (from the script matching `convention_min*`). |
| `convention_min.py` | Minimal synthetic demonstration of the coordinate-convention bug. |
| `direction_error.out` | Captured output (from the script matching `direction_error*`). |
| `direction_error.py` | Bone-direction error of the Step 7 converter's DIRECT output on real clips (doc: 'exact by construction'). |
| `elbow_jumps.out` | Captured output (from the script matching `elbow_jumps*`). |
| `elbow_jumps.py` | Are the ~1.5 rad frame-to-frame elbow jumps caused by wrist-visibility dropouts (bone left at zero)? |
| `extra_checks.out` | Captured output (from the script matching `extra_checks*`). |
| `extra_checks.py` | Follow-ups: exact-collinear Kabsch, real-data swing angles / temporal jumps, head-above-shoulder, |
| `extract_fresh.out` | Captured output (from the script matching `extract_fresh*`). |
| `extract_fresh.py` | Re-extract each clip with a FRESH Holistic instance into work/missed-bugs/freshroot (never touches snap/data). |
| `extract_state.out` | Captured output (from the script matching `extract_state*`). |
| `extract_state.py` | extract_world_landmarks.py reuses ONE Holistic(static_image_mode=False) instance across all videos and |
| `fresh_noise_floor.out` | Captured output (from the script matching `fresh_noise_floor*`). |
| `fresh_round_trip.out` | Captured output (from the script matching `fresh_round_trip*`). |
| `head_check.out` | Captured output (from the script matching `head_check*`). |
| `head_check.py` | Real SMPL-X forward on fit_body output: where does the model's head land vs MediaPipe's nose? |
| `measure_noise_floor.out` | Captured output (from the script matching `measure_noise_floor*`). |
| `measure_round_trip.out` | Captured output (from the script matching `measure_round_trip*`). |
| `noise_axes.out` | Captured output (from the script matching `noise_axes*`). |
| `noise_axes.py` | measure_noise_floor.py compares a 2-axis band ('xy': norm of two per-axis SDs) to a 1-axis band ('z'). |
| `noise_dims.out` | Captured output (from the script matching `noise_dims*`). |
| `noise_dims.py` | Band comparison in measure_noise_floor mixes a 1-axis z band with a 2-axis xy band. |
| `orphan_cases.out` | Captured output (from the script matching `orphan_cases*`). |
| `orphan_cases.py` | fit_body: a landmark whose bone could not be solved (because its PARENT landmark is invalid) is |
| `pytest_now.out` | Captured output (from the script matching `pytest_now*`). |
| `rerun/` | Subfolder (see its index.md) |
| `sanity.out` | Captured output (from the script matching `sanity*`). |
| `sanity.py` | Checks that came back CLEAN (reported so the report can say what is NOT a bug). |
| `synthetic_bugs.out` | Captured output (from the script matching `synthetic_bugs*`). |
| `synthetic_bugs.py` | Synthetic demonstrations of converter / IK / alignment edge-case bugs. |
| `index.md` | This index. |
