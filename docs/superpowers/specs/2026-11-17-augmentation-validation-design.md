# Augmentation validation: manifest, replay and per-augmentation checks

Status: design for approval (2026-11-17). First slice of
[Augmentation_Validation_Proposal.md](../../Augmentation_Validation_Proposal.md), made concrete.

## 1. Intent

Every generated training sample must be traceable to exactly what was done to it, and every
augmentation that was done must be checked against what it claims to do. The user's framing:
**map each generated output's name to the augmentations applied to it, then validate the sample
against those augmentations.**

This is the "is each augmentation implemented and applied correctly" layer. It is the foundation
for the later keep/drop decision, but it does not make that decision on usefulness or meaning.

The classifier reads raw MediaPipe landmarks (normalised image x/y and unnormalised coordinates,
no shoulder centring), so image-space checks are judged in that format
([README § Classifier input](../../../README.md#classifier-input)).

## 2. Goals, non-goals, anti-goals

Goals
- A stable name for each generated sample and a manifest record mapping it to source clip,
  profile, seed, applied entries, draws and fallbacks.
- A validator that, from the manifest record alone, replays the chain, captures the state before
  and after every entry, and runs that entry's own check.
- One check per week-4 chain entry (all 15 in `chain.ORDER`), plus universal checks every step gets.
- A per-sample JSON report with statuses PASS / FAIL / REVIEW / NOT EVALUATED.
- Tests proving each check passes on correct augmentations **and fails on a deliberately broken one**.
- Fix the two known `report.py` defects (all-missing reads as 0; body parts pooled for in-frame).

Non-goals
- Semantic validity (does the sign still mean the same thing) and recognition benefit.
- Corpus-calibrated thresholds or a keep/drop policy on realism; that is the next slice.
- The final classifier exporter (landmark order, z convention, missing-value encoding are still
  open, proposal A005).
- The older Tier 1 scripts (`scripts/augment_trials.py`, `tier1.py` paths outside the chain)
  and the legacy `evaluation/metrics` package. Neither is changed.

Anti-goals (would make this a failure even if it runs)
- A check that recomputes the expected result by calling the same implementation it is checking.
  Expected values must come from an independent calculation (closed-form geometry, the draw's
  numbers, or the input snapshot).
- An unavailable measurement reported as 0 or PASS. Unavailable is NOT EVALUATED with a reason.
- Reporting "all checks pass" as "the sample is good for training".

## 3. Constraints

- Python 3.9+, pytest + NumPy only; no new dependencies. Black, 100 columns, on new files only.
- Replay needs the SMPL-X assets and the source round-trip cache. Tests that need assets use the
  existing `requires_smplx` skip; a skipped check is NOT EVALUATED, never a pass.
- Must not change any existing draw: the RNG streams in `AugmentationChain` stay as they are.
- Licensed model weights are never written into samples or reports.

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **Replay with step snapshots**: validator re-runs the chain from the manifest with recorded draws, capturing before/after of every entry via a new step hook | Exact, isolated per-entry checks; needs only the manifest; costs one chain replay per validated sample | **Chosen by the user** |
| Snapshots saved at generation time | Faster validation, but much larger files and old samples cannot be re-validated | Rejected |
| Compare only original vs final output | No chain changes, but composed entries blur, so most checks become NOT EVALUATED | Rejected |

## 5. Design

### 5.1 Sample naming and manifest

- Sample id: `<clip_id>__<profile_name>__b<base_seed>__v<variant>` (variant = 0..N-1 when
  several samples are drawn per clip). The chain seed of a variant is
  `int(np.random.SeedSequence([base_seed, variant]).generate_state(1)[0])`; the manifest stores
  `base_seed`, `variant` and the derived `seed`, and `profile.resolved.seed` is overridden to that
  derived `seed` (the chain reads `profile.seed`). Ids are unique within a generation run; the
  generator refuses to overwrite an existing id.
- Source clips are snapshotted once. `run_clip` + `clip_from_result` + `fit_clip_camera` are slow
  and need the models, so the generator writes `sources/<clip_id>.npz`: every `ClipData` field,
  camera included. Replay loads this file and never re-fits, so the camera that `scene.uncentre`
  depends on is reproduced exactly. The synthetic test clip is saved the same way.
- Generation writes, per run directory:
  - `sources/<clip_id>.npz` (above).
  - `samples/<sample_id>.npz`: every `ClipOutputs` array (world and image arrays, `head_world`,
    `hand_valid`, `face_valid`, `body_ok`, `times_s`, `blendshapes`, `blendshape_names`; no
    vertices).
  - `manifest.jsonl`: one JSON object per sample:
    `schema_version, sample_id, source{clip_id, landmarks_dir, clip_path, clip_sha256}`,
    `profile{name, path, sha256, resolved}, base_seed, variant, seed, pivot`,
    `sampled, draws, applied, fallbacks, code_revision, created`.
    `sampled` holds each entry's draw as returned by `sample` (deep-copied by the generator's
    `draw_hook` before `apply` adds diagnostics), or `{"gated": true}`; `draws` is the
    post-apply `ChainResult.draws`. `resolved` is the YAML-shaped dict of the profile after
    enable/disable overrides (a new `profile_to_dict` serializer, round-tripping through
    `profile_from_dict`), so replay never depends on a YAML file that may have changed. JSON has
    no NaN/Infinity (`allow_nan=False`); NumPy values are converted to lists/scalars.
- A generator script (`scripts/generate_samples.py`) produces these from a profile and a clip list.

### 5.2 Replay

- `AugmentationChain` gains an optional `step_hook(name, before, after, draw)` called after each
  entry that ran (gated-out and fallback entries included, with `after is before`). With no hook,
  behaviour and draws are unchanged (a test pins this).
- The validator loads `sources/<clip_id>.npz` (checking `clip_sha256`), builds the chain from
  `profile.resolved`, and runs it with:
  - a `draw_hook` that compares each freshly sampled draw with the manifest's `sampled` entry and
    then returns a **deep copy** of the manifest's draw (`apply` writes diagnostics into the dict
    it is given), so replay is exact even if a numeric drift appeared;
  - the `step_hook`, which records snapshots and, because `draw_hook` never sees gated-out
    entries, also compares gating with `sampled`.
- `replay.draws` compares sampled keys only, after the same JSON normalisation as the manifest:
  strings, bools and ints exactly; floats with `rtol=1e-9, atol=1e-12`. A mismatch, a gating
  difference, or a pivot that differs from the manifest's `pivot` is a FAIL.
- Checks run on `ClipOutputs`: for clip-stage entries the validator calls `build_outputs` on the
  before and after `ClipData`, caching so each snapshot is built once (after of step k is before
  of step k+1). Rotation-space checks (fingers, arm joints) may also read the `ClipData` itself.
- The replayed final outputs are compared with the stored `samples/<id>.npz` (`replay.final`):
  masks and frame counts exactly, coordinates with `atol=1e-6` (m or normalised units), so float
  drift across machines does not cause false failures.
- `StepSnapshot` holds `name, stage, before, after` (a `ClipData` for clip-stage entries, a
  `ClipOutputs` for `temporal.hand_dropout`), the cached `before_outputs` / `after_outputs`, the
  post-apply `draw`, and `gated` / `fallback`. `CheckContext` holds the entry's resolved ranges,
  the scene pivot and Tier 1 draw from `ChainContext`, the skeleton parents and the camera. The
  pivot and draw are inputs like any other datum; using them does not break independence, as long
  as the expected geometry is computed by the check's own code.
- Checks key off `draws` (every entry that ran), not `applied` (which omits identity scene draws).

### 5.3 Check registry and statuses

- `CHECKS: Dict[str, Callable[[StepSnapshot, CheckContext], List[CheckResult]]]`, keyed by
  chain entry name. A test fails if any name in `chain.ORDER` has no registered check.
- `CheckResult`: `name, status, value, units, tolerance, numerator, denominator, reason,
  frames` (worst frame indices). Unavailable numbers are `null` with a reason.
- Status meaning: FAIL = the augmentation did not do what it claims (an engineering defect);
  REVIEW = valid but worth a look (e.g. a hand newly leaves the frame); NOT EVALUATED = could not
  be measured (no camera, no valid points, assets missing); PASS otherwise.
- Sample status = worst of its checks (FAIL > REVIEW > NOT EVALUATED > PASS).

### 5.4 Universal checks (every step)

| Check | Rule |
|---|---|
| `no_mutation` | the `before` object is byte-identical before and after the step |
| `draw_in_range` | every sampled value lies inside the resolved profile's range / truncation |
| `missingness` | no frame or point gains validity (`hand_valid`, `face_valid`, `body_ok`) except as the entry allows; valid points are finite; documented-missing slots excluded explicitly |
| `coverage` | per part (left hand, right hand, face, arms): expected, finite, in-frame, newly out-of-frame vs before, worst frame and longest run. Newly out-of-frame = REVIEW. No camera = NOT EVALUATED |
| `untouched_parts` | parts the entry must not move are unchanged (per entry, listed below) |

### 5.5 Per-entry checks

Tolerances: exact-geometry checks use 1e-6 m / 1e-6 normalised units; checks through IK or
re-skinning use tolerances the builder measures on the pilot clip and records in the code with
the measured value.

| Entry | Expected relationship (independent of the implementation) | Untouched |
|---|---|---|
| `fingers.joint_coupled_noise` | each finger's MCP/PIP local flexion changes by the drawn `mcp_deg`/`pip_deg`, DIP by `pip × dip_ratio`; hand bone lengths unchanged | arms, face |
| `fingers.handshape_jitter` | `features_changed(before, after)` is false; offsets within `truncate_sigmas × σ`; `fallback = no_valid_redraw` ⇒ output equals input; hand bone lengths unchanged | arms, face |
| `body.tier1_arm_joints`, `body.joint_coupled_noise` | arm bone lengths unchanged; each hand rigid (internal distances unchanged); recorded `wrist_moved_mm_max` matches a recomputation from the snapshots; each joint angle ≤ its truncation | face, fingers' local shape |
| `body.limb_length_scaling` | upper-arm / forearm length ratio after/before equals the drawn factor per bone; with `retarget`, wrists stay within the measured tolerance or a fallback is recorded | face |
| `body.movement_size` | in the torso frame, each wrist path about its fitted-frame centroid scales by `factor`; bone lengths unchanged | face |
| `body.location_jitter` | in the torso frame, every fitted wrist shifts by the constant `offset_mm`; bone lengths unchanged | face |
| `face.expression_jitter` | blendshape change equals the drawn constant offsets (plus recorded noise); disabled groups (mouth/jaw by default) change by exactly 0; rigid skull points unchanged | arms, hands |
| `scene.mirror` | after = reflection of before about the scene pivot with left/right swapped (pose via `POSE_MIRROR`, hands L↔R, face via `FACE_MIRROR`, blendshapes via `blendshape_mirror`), checked against an independently written reflection (mirror-twice stays a unit test in `test_week4_scene.py`, not a validator check, since it would rerun the implementation) | — |
| `scene.view` | all 3-D pairwise distances within a frame unchanged; the rotation recovered by Kabsch matches the drawn yaw/pitch. No 2-D distance requirement | — |
| `scene.scale` | 3-D same-frame pairwise distances and image-space same-frame differences multiply by `factor` (catches scaling body but not hands) | — |
| `scene.uncentre` | 3-D distances unchanged; image points shift by one constant equal to the recorded shift (catches pixel offsets used as normalised); hand points that were in frame stay in frame (the promise) else FAIL | — |
| `temporal.time_warp` | times strictly increasing; first and last source frames kept; mode `none`: output duration / input duration = `duration` within one frame; `varied`/`segment`: output length matches the drawn rates | — |
| `temporal.frame_drop` | `keep` sorted, unique, a subset of input frames; no gap on `times_s` longer than `max_gap_ms`; effective rate ≥ `min_fps`; fill `none` drops the frames, `hold` repeats the previous kept frame, `interpolate` lies between neighbours | — |
| `temporal.hand_dropout` | hand rows are NaN / invalid exactly on the recorded runs; runs lie only on frames where `before.hand_valid` is true (fitted and present, what `sample` uses); run length ≤ `max_run_frames`; everything else unchanged | pose, face |

Temporal entries change the frame count, so `untouched_parts` and frame-by-frame comparisons are
replaced by the timing rules above.

### 5.6 Reports and scripts

- `scripts/validate_samples.py <run_dir>` writes `validation/<sample_id>.json` and
  `validation/summary.json` (counts per status, per entry, per check) plus a short `summary.md`.
- `report.py` fixes: `_nanmax` on all-missing returns NaN and callers report NOT EVALUATED;
  `in_frame_pct` is reported per part (left hand, right hand, face, arms). Existing keys stay for
  current consumers (`augment_week4_trials.py`, `demo_week4.py`, the lab), new per-part keys are
  added. The hosted lab copy in `docs/augmentation-lab/` is not rebuilt.

### 5.7 Tests

- Fault injection needs no production changes (`chain.ORDER` stays fixed). Most faults call the
  check functions directly on a snapshot made by the real entry whose `after` is then corrupted;
  at least one end-to-end case monkeypatches a real entry's `apply` inside a chain run and expects
  the validator to FAIL it. Faults, one per check at least: (scale on body but not hands; mirror without side swap; uncentre with pixel offsets; frame drop
  leaving a 300 ms gap; expression jitter moving the mouth; arm noise stretching a bone;
  hand dropout on an undetected frame; a mutated input). Each must produce FAIL on its check.
- Every correct entry on the synthetic clip (`tests/clip_fixtures.synthetic_clip`) passes.
- Negative report tests from the proposal: all values missing; one hand absent; no camera;
  already out-of-frame source; single-frame clip.
- Manifest round trip: generate → validate on the synthetic clip gives `replay.draws` and
  `replay.final` PASS; editing a draw in the manifest gives FAIL.
- Hook pinning: chain results with and without hooks are identical.

### 5.8 Code layout (builder's call within this)

New package `src/kinetic_augment/validation/` (`manifest.py`, `replay.py`, `checks/` per region,
`report.py`), scripts above, tests `tests/test_validation_*.py`, and `index.md` updates for each
new folder. `docs/Augmentation_Validation_Proposal.md` gets a line pointing here.

## 6. Calls made by the designer (not discussed with the user)

These came out of the builder check. Each is a proposal that can be overturned:
source `ClipData` snapshots instead of re-fitting on replay; the derived-seed rule; storing
pre-apply `sampled` draws beside post-apply `draws`; the comparison tolerances for
`replay.draws` and `replay.final`; fault injection without production changes; dropout runs
checked against `hand_valid`; a new `profile_to_dict` serializer; mirror-twice kept out of the
validator; the hosted lab copy not rebuilt.

## 7. Left to the builder

- Exact tolerances for IK / re-skinning checks, measured on the pilot clip and recorded.
- How the torso frame is computed in checks (reuse `arm_ik` torso-frame math only if the check
  stays independent of the entry under test; otherwise reimplement).
- Internal data structures, caching, CLI flags.
