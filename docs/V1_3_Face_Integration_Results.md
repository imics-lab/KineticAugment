# V1.3 Face Integration Results

Date: 2026-10-11. This report measures the Tasks 1–5 face stream and classifier integration on
the cached pilot and evaluation clips. Machine-readable counts, per-clip coverage, and cache hashes
are in [the evidence JSON](V1_3_Face_Integration_Evidence.json). The earlier training-data audit
remains the source for broader contact-accuracy limitations.

## Real-cache coverage

The validator processed all 5 pilot clips and all 20 evaluation clips: **25 clips and 1,646 of
1,646 frames**, at stride 1. It took 245.2 seconds of pipeline time. The evaluation cache is the
existing `data/round_trip_eval` symlink; the validator read its target without modifying it. It
wrote per-clip feature archives and its detailed report to `/tmp/face-v1.3-validation-full`,
outside both input trees. SHA-256 hashes for all 50 face and hand input archives matched before
and after processing.

There were 1,550 observed face frames and 1,549 successful head attachments (99.94% of observed
face frames). Body fitting succeeded in 1,643 frames. The independent per-hand geometry mask
accepted 1,472 hand-frame rows. **No invalid-face frame had accepted fitted geometry.** Clip
coverage is listed below; geometry counts are hand-frame rows and can be greater than frame counts
because a frame can have both hands accepted.

| Pilot clip | Frames | Face observed | Attached | Body valid | Geometry-valid hands |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01986 | 83 | 83 | 83 | 83 | 100 |
| 01987 | 75 | 75 | 75 | 75 | 76 |
| 01988 | 144 | 143 | 143 | 143 | 193 |
| 01991 | 49 | 49 | 49 | 49 | 49 |
| 01992 | 59 | 59 | 59 | 59 | 117 |

| Evaluation clip | Frames | Face observed | Attached | Body valid | Geometry-valid hands |
| --- | ---: | ---: | ---: | ---: | ---: |
| 05631 | 24 | 24 | 24 | 24 | 17 |
| 07389 | 82 | 72 | 72 | 82 | 52 |
| 08917 | 32 | 32 | 32 | 32 | 23 |
| 08918 | 30 | 30 | 30 | 30 | 24 |
| 08938 | 56 | 55 | 55 | 55 | 69 |
| 09849 | 81 | 0 | 0 | 81 | 0 |
| 12328 | 88 | 88 | 88 | 88 | 176 |
| 13630 | 106 | 106 | 106 | 106 | 47 |
| 17083 | 62 | 62 | 62 | 62 | 35 |
| 21871 | 79 | 79 | 79 | 79 | 41 |
| 21942 | 68 | 68 | 68 | 68 | 59 |
| 28111 | 35 | 35 | 35 | 35 | 19 |
| 57641 | 29 | 27 | 26 | 29 | 26 |
| 57949 | 53 | 53 | 53 | 53 | 53 |
| 63229 | 47 | 47 | 47 | 47 | 19 |
| 63665 | 72 | 72 | 72 | 72 | 43 |
| 63669 | 94 | 93 | 93 | 93 | 55 |
| 69257 | 56 | 56 | 56 | 56 | 74 |
| 69364 | 75 | 75 | 75 | 75 | 60 |
| 69411 | 67 | 67 | 67 | 67 | 45 |

## Geometry and mask checks

Across 740,422 finite facial points from attached frames, mapping the stored head-local points back
to world coordinates reproduced the pixel-to-world transform with median error **9.7×10⁻¹⁷ m**,
95th percentile **1.8×10⁻¹⁶ m**, and maximum **4.1×10⁻¹⁶ m**. Applying the inverse pixel transform
recovered the original face coordinates with median error **6.4×10⁻¹⁴ px**, 95th percentile
**2.6×10⁻¹³ px**, and maximum **7.3×10⁻¹³ px**. An independent arbitrary rigid re-pose/inverse
check had maximum head-local error **9.3×10⁻¹⁷ m**. These test reversibility and coordinate
handling; they do not validate absolute metric depth or contact labels.

With attachment enabled and disabled over the same full-clip fit context, the first three
geometry-valid frames matched exactly on two pilot clips. For 01986 (frames 10–12) and 01991
(frames 0–2), body-valid and geometry-valid masks and all per-value finite masks matched. Both
probes compared 495 finite rotation entries and 9 translation entries; hand-to-face distance
compared 35 finite entries on 01986 and 21 on 01991. Every maximum absolute difference was zero.
The initial comparison that refit only three frames was discarded because the hand-shape fit
depends on the selected frame set; both accepted comparisons used every clip frame in both runs.

The depth-corrected regression on 01986 frames 14–21 produced 16 corrected-but-unreached hand
rows, and all 16 had invalid geometry masks. A separate controlled copy removed pose-world
observations on 01986 frame 0. The producer reported body-fit failure while retaining observed
face points; the image-space adapter produced 52 valid face values and zero body/hand values for
that row.

The original integration check used four synthetic sequences, synthetic labels, and body/hand,
face, and head streams. The final review fix adds the real-archive probe below. A bounded filename
inventory of the current worktree `data` and parent project `data` directories (following scoped
symlinks) found zero matching WLASL/annotation/label/gloss JSON, CSV, TSV, or Parquet files. The
search command was:

```sh
rtk proxy rg --files --hidden --no-ignore --follow data \
  /Users/bemnetbeshah/Workspace/cv-research/KineticAugment/data \
  | rtk proxy rg -i '(wlasl|annot|label|gloss).*(\.json|\.csv|\.tsv|\.parquet)$|(^|/)(annotations?|labels?)(/|\.)' || true
```

Therefore **no meaningful real recognition accuracy was measured**.

## Mixed-archive classifier integration

Archives without detector blendshape names receive the canonical MediaPipe 52-channel layout for
collation, while every score remains invalid when its channel names are unknown. The collator checks
channel order only when the blendshape stream is present. Real named channel orders remain strict,
and incompatible orders are rejected. This lets no-face clips join batches with the default
blendshape stream and body-only batches where that metadata is absent.

The classifier probe loaded the already generated archives in `/tmp/face-v1.3-validation-full`
without rerunning fitting: all 25 clips and 1,646 frames, including entirely no-face clip 09849,
were collated and passed through both modes. The multimodal batch used `body_hands`, `face`, `head`,
and `blendshapes`; all 72,852 named blendshape values were valid, logits, gradients, and parameter
updates were finite, and the optimizer changed 20 parameters (loss before update 0.76181). The
body-only batch used only `body_hands`; logits, gradients, and updates were finite, 8 parameters
changed (loss before update 0.99812). Its padded sequence length was 144. Labels were synthetic in
both probes, so these measurements establish finite training wiring only and do not measure
recognition accuracy.

Reproduce the archive-only probe without fitting:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python scripts/validate_face_integration.py \
  --probe-feature-dir /tmp/face-v1.3-validation-full \
  --output-dir /tmp/face-v1.3-final-fix-probe
```

The focused adapter, classifier, and validator tests reported **19 passed**. They cover default
mixed collation with an unnamed no-face clip, blendshapes-disabled and body-only batches, and
rejection of incompatible named channel orders. Reproduce them with:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_face_training_dataset.py tests/test_face_classifier.py \
  tests/test_face_integration_validation.py -q --tb=long
```

Machine-readable probe metrics and counts are recorded in the evidence JSON.

## Feature contract and training

The serialized per-clip archive uses schema 1 and keeps all 478 observed face points, timestamps,
source frame indices, source metadata, and separate observation, visibility, attachment, and
geometry masks. It is not the legacy 543-landmark layout. The adapter emits synchronized,
finite-filled streams with same-shaped boolean masks: `body_hands` (96 values), `face` (52 image
values or 78 head-local values), `head` (6-D rotation encoding), and optional ordered 52-channel
blendshapes. Missing values are filled only at the tensor boundary and remain false in their
masks. Metric contact diagnostics are not classifier inputs.

Train against the official split only when both official annotation JSON and matching feature
archives are available:

```sh
PYTHONPATH=src .venv/bin/python examples/train_face_streams.py \
  --annotation-root /path/to/WLASL \
  --feature-root /path/to/face_features \
  --subset wlasl100 --face-space image --epochs 50
```

The validation split controls early stopping; the example does not load the test split. `--body-only`
selects the body/hand ablation. This example was not run on real labels in this validation because
the annotation data was unavailable.

## Reproduction

Run all paired cached clips and write outputs/report outside the input directories:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python scripts/validate_face_integration.py \
  --output-dir /tmp/face-v1.3-validation
```

The default input directories are `data/test_output` and `data/round_trip_eval`. Repeat
`--input-dir` to select others, or repeat `--clip` to select exact clip IDs. `--frame-stride` and
`--max-clips` explicitly reduce work and are recorded in the JSON report.

The full project suite previously ran with:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest -q --tb=long
```

That sandbox run reported **279 passed, 1 xfailed, 3 failed, and 3 errors**. All six failures and
errors were the MediaPipe face/hand/world extraction tests; the traceback showed NSOpenGL pixel
format creation failing in the sandbox. The only rerun was those six extraction tests with graphics
access, and all six passed in 14.63 seconds. The graphics-enabled rerun command was:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_extract_face_landmarks.py tests/test_extract_hand_landmarks.py \
  tests/test_extract_world_landmarks.py -q --tb=long
```

The full-suite counts above are retained historical evidence; this final fix was verified with the
focused adapter, classifier, and validator command documented above.

## Final review and remaining decision

Every implementation task received an independent review. The final whole-branch review covered
`347b059..19454c1`; its mixed no-face batch finding was fixed in `a340878`. The scoped final
re-review found all findings addressed and no new breakage. The final focused run passed 19 tests,
and all 25 saved real archives passed multimodal and body-only batching and finite optimization.
Temporary task reports were removed from the worktree after review; committed historical test logs
remain in Git history. The commands and results needed to reproduce the delivered checks are above.

The implementation preserves observations and adds the trainable facial branch. Joint camera/depth
optimization remains deferred because no independent metric reference was available. The cost of
that decision is that absolute hand-to-face depth stays uncertain and would need separate work;
contact diagnostics remain excluded from the default classifier and are not certified labels.
