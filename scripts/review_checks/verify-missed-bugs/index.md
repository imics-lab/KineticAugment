# `verify-missed-bugs/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **missed-bugs** — Independent bug hunt: conventions, orphan correspondences, noise per axis, elbow jumps.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-missed-bugs .venv/bin/python scripts/review_checks/verify-missed-bugs/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `measure_noise_floor.out` | Captured output (from the script matching `measure_noise_floor*`). |
| `measure_round_trip.out` | Captured output (from the script matching `measure_round_trip*`). |
| `pytest.out` | Captured output (from the script matching `pytest*`). |
| `v_convention.out` | Captured output (from the script matching `v_convention*`). |
| `v_convention.py` | Independent check of N1/A002: is the fitted body upside-down on real MediaPipe world data? |
| `v_direction_a005.out` | Captured output (from the script matching `v_direction_a005*`). |
| `v_direction_a005.py` | R-direction and A005: bone-direction error of the DIRECT output; rigid re-alignment of direct output; scale of aligned fit. |
| `v_extract.out` | Captured output (from the script matching `v_extract*`). |
| `v_extract.py` | Independent check of N2: does extract_world_landmarks.py output depend on clip order? |
| `v_extract_cmp.out` | Captured output (from the script matching `v_extract_cmp*`). |
| `v_extract_cmp.py` | Verification script (no docstring); see the cluster summary above. |
| `v_misc.out` | Captured output (from the script matching `v_misc*`). |
| `v_misc.py` | A001, N5 (betas), N6 (origin), N8 (collinear), N7 (elbow jumps/fabrication), N9 (antipode). |
| `v_misc2.out` | Captured output (from the script matching `v_misc2*`). |
| `v_misc2.py` | A001, N5 (betas), N6 (origin), N8 (collinear), N7 (elbow jumps/fabrication), N9 (antipode). |
| `v_noise.out` | Captured output (from the script matching `v_noise*`). |
| `v_noise.py` | A004-gate, A004-dim, N4: own re-implementation of Methods A/B; validate vs script output; per-axis; synthetic calibration. |
| `v_orphan.out` | Captured output (from the script matching `v_orphan*`). |
| `v_orphan.py` | N3 / R-missing: valid landmarks downstream of an invalid non-leaf landmark. |
| `v_root_axes.out` | Captured output (from the script matching `v_root_axes*`). |
| `v_root_axes.py` | A002 vs N1: root error about a generic axis costs position error; a pitch about the shoulder axis (like C=diag(1,-1,-1)) costs none. |
| `v_roundtrip_fresh.out` | Captured output (from the script matching `v_roundtrip_fresh*`). |
| `v_roundtrip_fresh.py` | Own implementation of the direct/aligned headline (same definitions as measure_round_trip.run_real), on stored vs fresh-per-clip data. |
| `v_swingmax.out` | Captured output (from the script matching `v_swingmax*`). |
| `v_swingmax.py` | Verification script (no docstring); see the cluster summary above. |
| `index.md` | This index. |
