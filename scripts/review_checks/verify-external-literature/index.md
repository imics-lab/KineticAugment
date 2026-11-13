# `verify-external-literature/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **external-literature** — External claims: MediaPipe docs, smoothing filter, HybrIK adaptive IK, re-extraction configs.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-external-literature .venv/bin/python scripts/review_checks/verify-external-literature/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `nf_fresh_s.txt` | Captured output (from the script matching `nf_fresh_s*`). |
| `nf_fresh_u.txt` | Captured output (from the script matching `nf_fresh_u*`). |
| `nf_rev_s.txt` | Captured output (from the script matching `nf_rev_s*`). |
| `nf_shared_s.txt` | Captured output (from the script matching `nf_shared_s*`). |
| `nf_shared_u.txt` | Captured output (from the script matching `nf_shared_u*`). |
| `nf_static.txt` | Captured output (from the script matching `nf_static*`). |
| `nf_stored.txt` | Captured output (from the script matching `nf_stored*`). |
| `rt_fresh_s.txt` | Captured output (from the script matching `rt_fresh_s*`). |
| `rt_fresh_u.txt` | Captured output (from the script matching `rt_fresh_u*`). |
| `rt_rev_s.txt` | Captured output (from the script matching `rt_rev_s*`). |
| `rt_shared_s.txt` | Captured output (from the script matching `rt_shared_s*`). |
| `rt_shared_u.txt` | Captured output (from the script matching `rt_shared_u*`). |
| `rt_static.txt` | Captured output (from the script matching `rt_static*`). |
| `rt_stored.txt` | Captured output (from the script matching `rt_stored*`). |
| `vx_adaptive.py` | Verifier's independent implementation of naive vs adaptive (HybrIK-style) arm IK, |
| `vx_adaptive.txt` | Captured output (from the script matching `vx_adaptive*`). |
| `vx_compare.py` | Verification script (no docstring); see the cluster summary above. |
| `vx_compare.txt` | Captured output (from the script matching `vx_compare*`). |
| `vx_extract.py` | Independent re-extraction (verifier). Writes only into this dir. |
| `vx_extract.txt` | Captured output (from the script matching `vx_extract*`). |
| `vx_finding10.py` | Recompute Finding 10 per-clip median lengths (vis>=0.5 gated, as the doc's table) per extraction config. |
| `vx_finding10.txt` | Captured output (from the script matching `vx_finding10*`). |
| `vx_oneeuro_cutoff.py` | Implied One-Euro cutoff (MediaPipe world-landmark filter: min_cutoff 0.1, beta 40, value scaling disabled) |
| `vx_oneeuro_cutoff.txt` | Captured output (from the script matching `vx_oneeuro_cutoff*`). |
| `vx_visibility.py` | Does MediaPipe visibility track positional uncertainty?  Two error proxies: |
| `vx_visibility.txt` | Captured output (from the script matching `vx_visibility*`). |
| `index.md` | This index. |
