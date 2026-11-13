# `verify-root-counterexample/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **root-counterexample** — A002: root-orientation counterexample, nose vs hip torso frame, Step 8 root design.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-root-counterexample .venv/bin/python scripts/review_checks/verify-root-counterexample/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `measure_round_trip_out.txt` | Captured output (from the script matching `measure_round_trip*`). |
| `noise_floor_out.txt` | Captured output (from the script matching `noise_floor*`). |
| `v1_counterexample.py` | Independent re-check of the root counterexample. |
| `v1_out.txt` | Captured output (from the script matching `v1*`). |
| `v2_out.txt` | Captured output (from the script matching `v2*`). |
| `v2_real.py` | Independent real-data checks for the root-orientation cluster. |
| `v3_optroot.py` | Is the doc's 'root orientation only' row (5.13%) an oracle? Optimise the root. |
| `v3_out.txt` | Captured output (from the script matching `v3*`). |
| `v4_nose_leak.py` | How much head/neck motion leaks into a shoulders+nose torso estimate (true torso fixed). |
| `v4_out.txt` | Captured output (from the script matching `v4*`). |
| `v5_out.txt` | Captured output (from the script matching `v5*`). |
| `v5_partA_and_table.py` | Reproduce Part A's 70 mm with the script's own RNG stream, report mean too; |
| `v6_out.txt` | Captured output (from the script matching `v6*`). |
| `v6_rigid_vs_transl.py` | After a shoulder-line root: translation-only vs an extra rigid post-fit. |
| `v7_hip_inframe.py` | Doc Finding 5 says hips 'Observed in frame' 32%/32%. Reproduce from the image-space caches. |
| `v7_out.txt` | Captured output (from the script matching `v7*`). |
| `v8_noise_hips.py` | Does pooling the hip landmarks (which sit at MediaPipe's world origin by construction) bias the noise floor? |
| `v8_out.txt` | Captured output (from the script matching `v8*`). |
| `index.md` | This index. |
