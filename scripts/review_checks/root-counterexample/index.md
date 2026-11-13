# `root-counterexample/`

**Tester** (first-pass measurement) for cluster **root-counterexample** — A002: root-orientation counterexample, nose vs hip torso frame, Step 8 root design.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/root-counterexample .venv/bin/python scripts/review_checks/root-counterexample/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `counterexample_rerun.txt` | Captured output (from the script matching `counterexample_rerun*`). |
| `counterexample_rerun2.txt` | Captured output (from the script matching `counterexample_rerun2*`). |
| `kinetic_v1_root_counterexample.py` | Reproduce the frozen-root counterexample for the V1 arm swing IK. |
| `measure_round_trip_out.txt` | Captured output (from the script matching `measure_round_trip*`). |
| `proto_tests/` | Subfolder (see its index.md) |
| `t10_nose_vis.txt` | Captured output (from the script matching `t10_nose_vis*`). |
| `t11_fallback_and_torso_relative.py` | (a) Size of the global_orient jump if the nose drops out and Step 8 falls back to the shoulder-only |
| `t11_out.txt` | Captured output (from the script matching `t11*`). |
| `t11_out_camera.txt` | Captured output (from the script matching `t11_out_camera*`). |
| `t1_counterexample_sweep.py` | Stress the reviewer's root counterexample: axis, angle, pose, alignment type, converter path. |
| `t1_out.txt` | Captured output (from the script matching `t1*`). |
| `t1_out_rerun.txt` | Captured output (from the script matching `t1_out_rerun*`). |
| `t2_out.txt` | Captured output (from the script matching `t2*`). |
| `t2_real_root.py` | Real-clip root analysis: what the current converter's root is, how much the nose |
| `t3_full.txt` | Captured output (from the script matching `t3_full*`). |
| `t3_full_rerun.txt` | Captured output (from the script matching `t3_full_rerun*`). |
| `t3_out.txt` | Captured output (from the script matching `t3*`). |
| `t3_upside_down_and_subsample.py` | (a) Is the current converter's SMPL-X body upside down in the MediaPipe world frame? |
| `t4_out.txt` | Captured output (from the script matching `t4*`). |
| `t4_xfail_test.py` | The strict-xfail rotated-root test: its own error today, and whether a shoulder-only root fixes it. |
| `t5_mechanism_and_realism.py` | A. mechanism: which joint rotations can move a shoulder; B. counterexample shoulder-line error; |
| `t5_out.txt` | Captured output (from the script matching `t5*`). |
| `t6_pytest_converter.txt` | Captured output (from the script matching `t6_pytest_converter*`). |
| `t7_out.txt` | Captured output (from the script matching `t7*`). |
| `t7_real_proto.py` | Step 8 prototype on the real clips: error (direct + similarity-aligned) and torso plausibility, |
| `t8_out.txt` | Captured output (from the script matching `t8*`). |
| `t8_perclip_aligned.py` | Per-clip similarity-aligned medians: shipped converter vs prototype shoulder-line root (doc: 01988 5.06 -> 5.11). |
| `t9_doc_table_search.py` | Try to reproduce the doc's retracted-section table (10/20/45 deg: mean 9.87/16.55/47.04 mm, max 81/119/307 mm). |
| `t9_out.txt` | Captured output (from the script matching `t9*`). |
| `index.md` | This index. |
