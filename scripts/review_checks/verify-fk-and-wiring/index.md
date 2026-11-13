# `verify-fk-and-wiring/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **fk-and-wiring** — A001/A007: forward kinematics vs real SMPL-X, legacy converter wiring, Step 7 contract claims.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-fk-and-wiring .venv/bin/python scripts/review_checks/verify-fk-and-wiring/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `measure_round_trip_output.txt` | Captured output (from the script matching `measure_round_trip*`). |
| `v1_fk.py` | Independent check of A001: repo FK vs real SMPL-X forward, all 55 joints, |
| `v1_fk_output.txt` | Captured output (from the script matching `v1_fk*`). |
| `v2_legacy.py` | Independent check of A007: do MediaPipeToSMPLX / SMPLXToMediaPipe still use raw |
| `v2_legacy_output.txt` | Captured output (from the script matching `v2_legacy*`). |
| `v3_contract.py` | Independent checks of the Step 7 contract claims (converter.py). |
| `v3_contract_output.txt` | Captured output (from the script matching `v3_contract*`). |
| `v4_orphan_collinear.py` | collinear triple with a hanging straight left arm |
| `v4_output.txt` | Captured output (from the script matching `v4*`). |
| `v4b_output.txt` | Captured output (from the script matching `v4b*`). |
| `v4b_single_arm.py` | Verification script (no docstring); see the cluster summary above. |
| `v5_output.txt` | Captured output (from the script matching `v5*`). |
| `v5_realdata_validity.py` | Real-data prevalence of (a) defaulted-but-unmarked modelled landmarks, |
| `wtest/` | Subfolder (see its index.md) |
| `index.md` | This index. |
