# `fk-and-wiring/`

**Tester** (first-pass measurement) for cluster **fk-and-wiring** — A001/A007: forward kinematics vs real SMPL-X, legacy converter wiring, Step 7 contract claims.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/fk-and-wiring .venv/bin/python scripts/review_checks/fk-and-wiring/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `_lib_caller.py` | Verification script (no docstring); see the cluster summary above. |
| `t1_fk_vs_smplx.py` | A001: FK (skeleton.py + kinematics.py) vs real SMPL-X forward. |
| `t1_output.txt` | Captured output (from the script matching `t1*`). |
| `t2_legacy_converters.py` | A007: do the public converter classes (mp_to_smplx / smplx_to_mp) still |
| `t2_output.txt` | Captured output (from the script matching `t2*`). |
| `t3_output.txt` | Captured output (from the script matching `t3*`). |
| `t3_step7_contract.py` | Step 7 contract claims about converter.py, tested independently. |
| `t4_extras.py` | Extra checks for the fk-and-wiring cluster (run against the refreshed snapshot). |
| `t4_output.txt` | Captured output (from the script matching `t4*`). |
| `t5_output.txt` | Captured output (from the script matching `t5*`). |
| `t5_residuals.py` | T5: (1) explain the float64 FK residual (smplx batch_rodrigues adds 1e-8 before the norm); |
| `t5b_eps.py` | float64: FK using smplx's exact (slightly non-orthogonal) rodrigues matrices. |
| `t5b_output.txt` | Captured output (from the script matching `t5b*`). |
| `t6_access_audit.py` | Which fields does reconstruct_body actually read? (attribute-access recorder) |
| `t6_output.txt` | Captured output (from the script matching `t6*`). |
| `index.md` | This index. |
