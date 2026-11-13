# `verify-shape-identifiability/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **shape-identifiability** — A006: length Jacobian w.r.t. betas, shape non-uniqueness, held-out shape fits.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-shape-identifiability .venv/bin/python scripts/review_checks/verify-shape-identifiability/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `pairs.npy` | Cached array produced by a script in this folder and loaded by another. |
| `v1_jacobian.py` | 0) pose invariance of the 5 lengths (collars zero) vs collar motion |
| `v1_jacobian.txt` | Captured output (from the script matching `v1_jacobian*`). |
| `v2_nonunique.py` | Verification script (no docstring); see the cluster summary above. |
| `v2_nonunique.txt` | Captured output (from the script matching `v2_nonunique*`). |
| `v3_realdata.py` | Independent real-data checks: per-clip medians, L-R asymmetry, half-split |
| `v3_realdata.txt` | Captured output (from the script matching `v3_realdata*`). |
| `v4_shapefit.py` | Independent re-run of the 'what does a per-clip shape fit buy' numbers. |
| `v4_shapefit.txt` | Captured output (from the script matching `v4_shapefit*`). |
| `v5_misc.py` | Verification script (no docstring); see the cluster summary above. |
| `v5_misc.txt` | Captured output (from the script matching `v5_misc*`). |
| `v6_root_confound.py` | Is 'root pose' a real confounder of a shape fit, as the smplx audit says and the |
| `v6_root_confound.txt` | Captured output (from the script matching `v6_root_confound*`). |
| `v7_acf.py` | Verification script (no docstring); see the cluster summary above. |
| `v7_acf.txt` | Captured output (from the script matching `v7_acf*`). |
| `vcommon.py` | Independent helpers: use the real smplx torch forward (not a hand-rolled linear model). |
| `index.md` | This index. |
