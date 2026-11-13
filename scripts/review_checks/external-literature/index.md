# `external-literature/`

**Tester** (first-pass measurement) for cluster **external-literature** — External claims: MediaPipe docs, smoothing filter, HybrIK adaptive IK, re-extraction configs.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/external-literature .venv/bin/python scripts/review_checks/external-literature/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `compare_configs.py` | Compare stored world landmarks vs re-extracted configs; quantify smoothing + cross-clip state leakage. |
| `compare_configs.txt` | Captured output (from the script matching `compare_configs*`). |
| `finding10_by_config.py` | Finding 10 per-clip proportions recomputed for each extraction config (median over frames where |
| `finding10_by_config.txt` | Captured output (from the script matching `finding10_by_config*`). |
| `hybrik_adaptive.py` | Prior-art test: HybrIK (Li et al., CVPR 2021) distinguishes 'naive' IK (bone direction = |
| `hybrik_adaptive.txt` | Captured output (from the script matching `hybrik_adaptive*`). |
| `leak_decay.py` | Verification script (no docstring); see the cluster summary above. |
| `leak_decay.txt` | Captured output (from the script matching `leak_decay*`). |
| `metrics_per_config.py` | Run the project's own noise-floor estimators and round-trip harness on each re-extracted config. |
| `metrics_per_config.txt` | Captured output (from the script matching `metrics_per_config*`). |
| `modelcard.txt` | Captured output (from the script matching `modelcard*`). |
| `pdftext.py` | collect text in TJ/Tj |
| `pdftext2.py` | map object number -> raw object bytes |
| `reextract.py` | Re-extract MediaPipe Holistic world landmarks under several configurations. |
| `sanity_leak.py` | Independent sanity check of the cross-clip leakage claim (re-run 2026-10-05). |
| `sanity_leak.txt` | Captured output (from the script matching `sanity_leak*`). |
| `sgnify_main.py` | Do it inside a loop due to ncg memory issue: |
| `shared_nosmooth.py` | minimal inline copy of extract() to avoid executing reextract's CLI |
| `shared_nosmooth.txt` | Captured output (from the script matching `shared_nosmooth*`). |
| `visibility_mask_lag.py` | Visibility is low-pass filtered (alpha=0.1) when smoothing is on. How often does the |
| `visibility_mask_lag.txt` | Captured output (from the script matching `visibility_mask_lag*`). |
| `visibility_vs_jitter.py` | Is MediaPipe `visibility` usable as a 3-D positional uncertainty? Test on the unsmoothed (U) and |
| `visibility_vs_jitter.txt` | Captured output (from the script matching `visibility_vs_jitter*`). |
| `index.md` | This index. |
