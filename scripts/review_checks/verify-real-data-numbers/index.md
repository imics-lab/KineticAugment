# `verify-real-data-numbers/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **real-data-numbers** — A005: reproduces real-clip headline numbers and the Step 6 oracle decomposition.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-real-data-numbers .venv/bin/python scripts/review_checks/verify-real-data-numbers/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `measure_round_trip.out` | Captured output (from the script matching `measure_round_trip*`). |
| `v_boot.out` | Captured output (from the script matching `v_boot*`). |
| `v_boot.py` | Bootstrap CIs, order-dependence, scale under oracles. Imports own closed form (v_cf). |
| `v_cf.py` | Independent closed-form oracle decomposition (verify-real-data-numbers). |
| `v_contract.out` | Captured output (from the script matching `v_contract*`). |
| `v_contract.py` | Independent checks: real SMPL-X serialization, betas side channel, torso orientation, |
| `v_count.out` | Captured output (from the script matching `v_count*`). |
| `v_count.py` | Verification script (no docstring); see the cluster summary above. |
| `v_floor.out` | Captured output (from the script matching `v_floor*`). |
| `v_floor.py` | is 2.29 (sim) / 2.52 (rigid) a floor for per-clip constant shape + per-frame root? |
| `v_headline.out` | Captured output (from the script matching `v_headline*`). |
| `v_headline.py` | Independent re-check of headline real-data numbers (verify-real-data-numbers). |
| `v_oracle.out` | Captured output (from the script matching `v_oracle*`). |
| `v_oracle.py` | Independent closed-form oracle decomposition (verify-real-data-numbers). |
| `v_step4.out` | Captured output (from the script matching `v_step4*`). |
| `v_step4.py` | Verification script (no docstring); see the cluster summary above. |
| `v_tip.out` | Captured output (from the script matching `v_tip*`). |
| `v_tip.py` | Verification script (no docstring); see the cluster summary above. |
| `index.md` | This index. |
