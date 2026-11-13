# V1.2 feasibility probes

**Date:** 2026-10-05. **Plan:** [V1.2 implementation plan](V1_2_Implementation_Plan.md). **Ledger:** [V1.2 assumptions](V1_2_Assumptions.md).

Measured before the implementation plan was written, with one-off scripts run against the local model assets, the installed packages and the five pilot clips. The probes changed no repository file. These are pre-implementation measurements; the implementation's tests and `scripts/measure_hand_round_trip.py` supersede them where they overlap.

## Model assets

| Check | Result |
| --- | --- |
| MANO loading | `MANO_{LEFT,RIGHT}.npz` load with `smplx.create(..., model_type="mano", ext="npz", use_pca=False, flat_hand_mean=True)`: 10 betas, `v_template` (778, 3), `posedirs` 135 rows, `J_regressor` (16, 778). The forward pass returns 16 joints and no fingertips. |
| MANO provenance | The npz files equal the official pkl files bit for bit, except the left model's shapedirs x-sign fix. |
| Left-hand shape fix | Works: with the same pose and shape, left and right hands mirror each other to within 0.006 mm (the unfixed pkl: 32.4 mm). The left template is an exact x-mirror of the right. |
| Joint order and tips | Verified from geometry on both sides: thumb 13–15, index 1–3, middle 4–6, pinky 7–9, ring 10–12. Tip vertices 744, 320, 443, 554, 671 sit 20–35 mm past the distal joint, within 13° of the distal bone. |
| Tip skinning | Each MANO tip vertex is skinned 100% to its own distal joint. Pose correctives still move the tips: median 0.6–1.5 mm, maximum 11.8 mm (pinky) over MANO's 1,554 training poses. |
| Units and mean pose | Metres (wrist to middle MCP 94.7 mm). `flat_hand_mean=True` is nearly flat (bends up to 26°, at the thumb); `False` curls the hand by up to 87 mm. |
| Gradients | Finite everywhere, matching central differences to a relative 1.5e-7. |
| SMPL-X version | v1.1 by content: `shapedirs` has 400 components (300 shape + 100 expression). `models/smplx/version.txt` says 1.0 and is stale. Loads with `num_betas=300`. |
| SMPL-X hands | Wrists 20/21, finger joints 25–39 (left) and 40–54 (right), and tip vertices 5361/4933/5058/5169/5286 (left), 8079/7669/7794/7905/8022 (right) verified; the tips equal SMPL-X's own output joints 66–75. Hand-pose mean and PCA components are identical to MANO's. |
| SMPL-X vs MANO template | Wrist to middle MCP 107.9 vs 94.7 mm (×1.14). Finger bones agree within ×0.93–1.08; palm vectors are ×1.13–1.32 longer in SMPL-X. SMPL-X's left and right hands are not exact mirrors (5.9 mm). |

## MediaPipe hands (0.10.21) on the pilot clips

| Check | Result |
| --- | --- |
| Which API | Only legacy `mp.solutions.hands.Hands` returns hand world landmarks. Legacy `Holistic` has pose world landmarks only. The Tasks API classes exist, but no `.task` model files are installed. |
| Coverage | 410 frames. `Hands` finds 0/1/2 hands in 95/83/232 frames (547 detections; 23% of frames have none). `Holistic` sees the left hand in 265 frames and the right in 304. All 21 world points are finite whenever a hand is detected. |
| Resolution | 320×240, 1280×720, 640×480, 288×192, 288×192; palms span 20–55 px. |
| Units and origin | Metres: wrist to middle MCP has a median of 88.7 mm (80–100 mm per clip). The origin is the knuckle-line centre, 79–98 mm from the wrist. |
| Left and right | The detector's label names the opposite hand in 540/547 detections (98.7%): it assumes mirrored (selfie) input. The 7 exceptions are all in 01987. `Holistic`'s `left_hand` sits at pose wrist 15 (the signer's left) in 261/265 frames. World-landmark chirality matches MANO's side in 96–98% of frames. |
| Axes | Hand-world axes equal pose-world axes: x right, y down, z away from the camera. No basis change is needed. |
| Duplicates | 9 frames hold two detections assigned to the same side: duplicate detections of one hand (01991), and overlapping hands at the frame edge (01988). |
| Determinism | Extraction is bit-identical across runs and clip orders when each clip gets fresh detector instances. |

## How rigid are MediaPipe's hands?

A model-free estimate built from each clip-side's own frames: a rigid palm template, each finger bone at its median length along the observed direction. This is a rough estimate, not a lower bound.

| Measure | Result |
| --- | --- |
| Error over the 20 non-wrist points, wrist-relative | Mean 14.9 mm, tips 17.1 mm, p95 36.1 mm (per clip-side 5.2–25.1 mm). |
| The same with free translation | 7.6 mm mean, 8.6 mm tips. |
| The same with free translation and per-frame scale | 4.9 mm mean, 6.7 mm tips. |
| Palm rigidity | 6.6 mm mean residual after a rigid fit. |
| Bone lengths | Within-clip coefficient of variation: median 15.9%; worst ring MCP→PIP 43%. |
| Hand size | Per-frame palm scale p5–p95: 0.69–1.31. |

## What this means for the plan

- **Feasible:** the assets, metric hand observations, left/right assignment and the hand/pose axes all hold up.
- **At risk:** the provisional real-data target (mean ≤ 5 mm, tips ≤ 8 mm, p95 ≤ 15 mm against raw MediaPipe) for any hand model with one shape per track. MediaPipe's hands change size and proportions from frame to frame, and a rigid skeleton cannot follow that.
- **Response:** the implementation measures the plan's primary metric as specified, without loosening it. It adds a per-frame-shape diagnostic and a per-frame-scale reference, to show how much of the error is the detector's size changes and how much is the model.
