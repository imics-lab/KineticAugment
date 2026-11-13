# `verify-noise-floor/`

Adversarial **verifier** (independently re-measured the tester's results) for cluster **noise-floor** — A004: noise-floor estimators, MediaPipe smoothing, frame-rate dependence.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/verify-noise-floor .venv/bin/python scripts/review_checks/verify-noise-floor/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `acf_sim.py` | ACF of second difference under iid noise + real arm motion (no temporal correlation in the noise). |
| `acf_sim_out.txt` | Captured output (from the script matching `acf_sim*`). |
| `heavy.py` | Full vs heavy model: floors, per-clip disagreement, per-axis, and L/R-swap check. |
| `heavy_out.txt` | Captured output (from the script matching `heavy*`). |
| `hips.py` | Verification script (no docstring); see the cluster summary above. |
| `my_extract.py` | Independent re-extraction (verifier). Writes only into this dir's ext/ folder. |
| `my_extract_log.txt` | Captured output (from the script matching `my_extract_log*`). |
| `nf_now.txt` | Captured output (from the script matching `nf_now*`). |
| `nf_repo.txt` | Captured output (from the script matching `nf_repo*`). |
| `ratio.py` | A/B ratios per band: observed (repo, raw) vs iid+real-motion simulation (pooled medians, same as script). |
| `ratio_out.txt` | Captured output (from the script matching `ratio*`). |
| `rt_now.txt` | Captured output (from the script matching `rt_now*`). |
| `rt_repo.txt` | Captured output (from the script matching `rt_repo*`). |
| `sim.py` | Verifier simulation: what do Method A / B read when the TRUE per-axis noise sigma is known? |
| `sim_out.txt` | Captured output (from the script matching `sim*`). |
| `verify.py` | Verifier's independent checks of the noise-floor (A004) cluster. |
| `verify_out.txt` | Captured output (from the script matching `verify*`). |
| `index.md` | This index. |
