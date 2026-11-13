# Week 2 exploratory experiments

Scratch experiments behind the error-metric and smoothing discussion in `docs/Week_2_Findings.md`. Not project code.

- `decomp.py` — Per-frame error breakdown for 25 clips: clip-wide body vs per-frame plausible body vs floor; per-track vs per-frame hand shape.
- `deviation.py` — Normal-range analysis of the breakdown: own-clip vs all-clip bands, excursion counts, and the two figures.
- `deviation_body.png` — Per-clip body error over time: shape offset and non-human-arm error, with both bands.
- `deviation_hands.png` — Per-clip hand error over time: per-frame-shape error and hand shape offset, with both bands.
- `smooth_exp.py` — Smoothing comparison: running average and centred smoothing of output points vs centred smoothing of fitted parameters.
- `smooth_results.txt` — Output of `smooth_exp.py` on 01988, 28111, 69257, 05631 and 09849.
- `render_smooth.py` — Renders source | raw SMPL-X figure | parameter-smoothed figure, with wrist and fingertip traces.
- `01988_smoothing_comparison.mp4`, `28111_smoothing_comparison.mp4`, `69257_smoothing_comparison.mp4` — Those renders (sigma 1.5 frames).
- `index.md` — This index.
