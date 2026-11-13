# `verify-doc-consistency/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **doc-consistency** — Plan-doc internal consistency checks and number reproduction.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-doc-consistency .venv/bin/python scripts/review_checks/verify-doc-consistency/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `v_contract.py` | Independent verifier checks for the doc-consistency cluster (contract, rotation, betas, |
| `v_contract_out.txt` | Captured output (from the script matching `v_contract*`). |
| `v_counterexample.py` | Sweep the frozen-root counterexample (doc table L525-530; review 0.7 rad numbers). |
| `v_counterexample_out.txt` | Captured output (from the script matching `v_counterexample*`). |
| `v_direct.py` | Verification script (no docstring); see the cluster summary above. |
| `v_direct_out.txt` | Captured output (from the script matching `v_direct*`). |
| `v_oracle.py` | Independent oracle decomposition (doc L651-663): what do an oracle root and oracle per-clip |
| `v_oracle_out.txt` | Captured output (from the script matching `v_oracle*`). |
| `v_presentation.py` | Check presentation_one.md section 4 numbers on the cached image-space data. |
| `v_presentation_out.txt` | Captured output (from the script matching `v_presentation*`). |
| `v_ratio_variants.py` | sum of medians per bone pooled (doc table style) |
| `v_ratio_variants_out.txt` | Captured output (from the script matching `v_ratio_variants*`). |
| `v_similarity.py` | Can a 7-number similarity act as the 'camera' between metric world landmarks and the cached |
| `v_similarity_out.txt` | Captured output (from the script matching `v_similarity*`). |
| `v_world_ratio_out.txt` | Captured output (from the script matching `v_world_ratio*`). |
| `index.md` | This index. |
