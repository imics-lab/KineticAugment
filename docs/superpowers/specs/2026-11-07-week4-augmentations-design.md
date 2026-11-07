# Week 4: more augmentations, region tags and YAML profiles

Status: design, awaiting approval. Branch: `feat/week4-augmentations` (from
`fix/restore-tier1-export-face-trials`, which is `main` plus the Tier 1 export fix).

## 1. Intent

Week 4 widens the augmentation library well beyond Tier 1. The professor wants the project to
be ambitious with augmentations, so this week builds augmentations that the project's own
evidence ([Augmentation_Grounding.md](../../Augmentation_Grounding.md)) calls risky, as well
as the safe ones. The risky ones are not dropped. They are built, but stay **off by default**,
so a later classifier ablation can measure whether they help or hurt (Grounding §7 step 5).

Three things come out of the week:

1. New augmentations, each drawing its values **once per clip** (V1 Problem Scope §11), as
   Tier 1 does.
2. Each augmentation is tagged with the **region** it acts on: `face`, `body`, `fingers`,
   `temporal` or `scene`.
3. YAML profiles in `configs/` that list, per region, which augmentations run, whether each is
   on by default, its probability, and **its sampling ranges** (scale range, jitter size, and
   so on). The ranges come from the YAML, not from constants buried in code.

## 2. Goals, non-goals, anti-goals

**Goals**
- The five augmentations the user asked for: limb length scaling, joint coupled noise, facial
  expression jitter, partial time warping (constant and varied), and motion slicing (frame
  dropping).
- Four more the user accepted: movement-size scaling, location jitter, within-category
  handshape jitter, and hand-dropout runs.
- Region tags in the registry, and profiles grouped by region.
- Every new augmentation callable from a YAML profile, with its ranges read from that profile.
- Ranges set about 25% beyond the proven literature ranges (§5.0), with the proven range shown
  beside each used range in the YAML, the README and `docs/Augmentation_Ranges.md`.
- Tests and a visual trial script, matching how Tier 1 was checked.

**Non-goals**
- Running the classifier ablation itself. This week only makes the arms switchable.
- Weak-hand drop (Tier 2 item 9). It needs ASL-LEX Sign Type labels, which the project does not
  have yet.
- The invariant checks of Grounding §4 (location region, contact, palm orientation) as reject
  filters. Only the handshape-feature check below is in scope, because handshape jitter cannot
  be built safely without it.
- Rewriting or deleting the older `intrinsic.py`, `temporal.py` and `extrinsic.py` classes, or
  the older profiles (`slr_profile*.yaml`, `har_running_profile.yaml`). The phase scripts
  `test_phase*.py` still import them.

**Anti-goals** (what would make this a failure even if it runs)
- A risky augmentation switched on by default. Limb length scaling, varied/segment time warping
  and facial expression jitter must be opt-in.
- Ranges hard-coded in Python with the YAML values ignored or only partly used.
- Per-frame random draws where the design says per clip. The old classes did this, and it adds
  jitter that the round trip then has to fight.
- Zero strength not reproducing the input exactly.
- Augmentations that move a hand out of its sign location without saying so. Each augmentation
  must report how far the wrists moved (§6).

## 3. Decisions made with the user

- All nine augmentations above are built this week.
- **On by default:** joint coupled noise and frame dropping, at the conservative ranges in §5.
- **Opt-in experiment arms:** limb length scaling, varied and segment time warping, and facial
  expression jitter (the user's call). The four suggested augmentations are also opt-in (see §8,
  item 1).
- Every profile entry carries its ranges.
- The face is the observed MediaPipe face stream (478 points and 52 blendshape scores) carried
  on the SMPL-X head, as in `scripts/augment_face_trials.py`. Expression jitter acts on that
  stream. SMPL-X's own expression stays zero.

## 4. Data the augmentations act on

Everything plugs into the V1.3 body + hands + face round trip
(`kinetic_augment.body_model.face_pipeline.run_clip`, used by `scripts/augment_face_trials.py`),
so one clip has all three regions:

| Part | Shape | Notes |
|---|---|---|
| SMPL-X rotations | (T, 55, 3) axis-angle | body 0–21, jaw/eyes 22–24, MANO hands 25–54; NaN where the body was not fitted |
| translation | (T, 3) | |
| betas | (300,) | one body shape per clip |
| face points in the head frame | (T, 478, 3) | `face_stream_local`; put back with `face_stream.from_head_frame` |
| blendshapes | (T, 52) | scores in [0, 1]; NaN where there is no face |
| frame rate | scalar | source fps of the clip |
| hand / body validity masks | (T, 2), (T,) | |
| weak-perspective camera | `tier1.WeakCamera` | fitted once per clip with `tier1.fit_weak_camera` from the round trip's arm landmarks to MediaPipe's 2-D pose, as `augment_trials.py` does; needed by uncentre and the in-frame checks |

**What the chain outputs.** The augmented SMPL-X parameters, plus a landmark sequence in
MediaPipe's layout (33 pose points, 21 points per hand, 478 face points), in world space and in
image space through the weak camera. This makes the output usable as classifier input for the
later ablation, not only for the visual trial. Tier 1's scene augmentations are re-implemented on
this container by calling the existing `tier1` functions, so the maths stays in one place.
`export_round_trip_demo.py` keeps its own chain and is not switched over this week.

Augmentations either edit the SMPL-X parameters (the body, fingers and face ones), re-time the
clip (temporal), or edit the output landmarks and vertices (scene, hand dropout). The builder
defines one small clip container holding these arrays so every augmentation has the same
signature. Its name and layout are left to the builder.

**Order of the chain.** Fingers → body → face → mirror → view → scale → uncentre → temporal →
hand dropout. This extends Tier 1's chain (arm joints, mirror, view, scale, uncentre, duration).
Temporal comes after everything spatial, and hand dropout comes last because it imitates the
detector seeing the final clip.

**Randomness.** Each augmentation gets its own `numpy.random.Generator` per clip, from a
`SeedSequence` keyed on the profile seed, the clip id and the augmentation's name. Adding,
removing or reordering augmentations then never changes another augmentation's draw. Tier 1
keeps its existing stream, `sample_tier1(default_rng([seed, clip]))`, so its draws stay
bit-identical to the existing exports and review evidence for the same seed and clip. Every draw
is recorded (as `Tier1Draw.as_dict` does) so a sample can be reproduced and reported. No use of
the global `np.random` state.

## 5. The augmentations

### 5.0 Range policy: slightly beyond what is proven

The user wants the ranges to be ambitious. Every range in the week-4 profiles is therefore set
**about 25% beyond the widest range the evidence supports** ("proven"), in the direction of more
variation. Log-uniform ranges are widened by 25% in log space, so that the low and high ends
stay symmetric. Probabilities (mirror p) are not widened. The proven range is recorded next to
every used range in the YAML comments, the README and a docs page (§7), so a reader can always
see how far past the evidence each one goes.

"Proven" means the widest range with sign-language or human-perception evidence in
[Augmentation_Grounding.md](../../Augmentation_Grounding.md) that was not shown to hurt. Where
the literature has no range, the row says so, and the used value is the designer's estimate
widened by 25%. **[M]**, **[L]** and **[T]** refer to Grounding's evidence keys.

| Augmentation | Proven range (source) | Used in week-4 profiles |
|---|---|---|
| View yaw | ±20° (Spoken2Sign **[M7]**) | ±25° |
| View pitch | ±5° ("a few degrees" **[T]**) | ±6° |
| Mirror | p = 0.5 **[M6]**; cost 1.3 pp on WLASL once **[T]** | p = 0.5 (a probability, not widened) |
| Uniform duration | 0.67–1.5×; 0.5× hurt once **[L6.2, M6, M3, T]** | 0.6–1.65×, log-uniform |
| Arm joints: shoulder, wrist | ±4° at p = 0.3 (SPOTER **[M1]**) | ±5° |
| Arm joints: elbow | ±7.5°, no harm in pose-based translation **[T]** | ±9.5° |
| Whole-scene scale | 0.8–1.2 (GISLR **[M10]**) | 0.75–1.25 |
| Uncentre | ±0.1 of the frame (GISLR **[M10]**) | ±0.125, hands still kept in the frame |
| Limb length | no literature; between-signer spread 5.7% in this project's MediaPipe data (Grounding §6.1) | ±7% |
| Movement size | no measured range; 0.85–1.15 is Grounding's linguistic example **[L2.10, L2.11]** | 0.8–1.2, log-uniform |
| Location jitter | σ = 10 mm, truncated at 20 mm (production spread **[T]**) | σ = 12.5 mm, truncated at 25 mm |
| Joint coupled noise, arms | as arm joints above **[M1, T]** | shoulder/wrist σ = 2.5°, truncated at 5°; elbow σ = 4.75°, truncated at 9.5° |
| Joint coupled noise, fingers | no published range | σ = 3.75°, truncated at 7.5° (estimate 3° / 6°, widened) |
| Handshape jitter | no published range; GISLR "finger-tree rotate" gives no angle **[T]** | non-selected and thumb σ = 7.5°; selected σ = 3.75°; selected flexion σ = 2°; all truncated at 2σ |
| Expression jitter | no published range; lexical mouth patterns excluded **[L7.3, L1.5]** | brows and eyes σ = 0.0625, truncated at 0.125; cheeks/nose σ = 0.0375, truncated at 0.075; mouth/jaw off |
| Varied time warp | none: non-uniform warps are excluded **[L6.3, L6.4]** | knot rates 0.7–1.43×, log-uniform |
| Segment time warp | none: excluded as above | window 20–60% of the clip, factor 0.6–1.65× |
| Frame dropping | ≥ 10 fps, where human intelligibility stops improving **[T]**; masked runs ≤ 8 frames **[M4, M10]** | ≥ 8 fps; max gap 125 ms; p = 0–0.2; target 12–25 fps |
| Hand-dropout runs | runs ≤ 8 frames **[M4, M10]** | runs of 1–10 frames |

Excluded items with no range to widen stay excluded: camera roll, shear, temporal reversal,
cropping. The mouth/jaw blendshape group stays off by default even in the ambitious profile,
because lexical mouth patterns can change the sign; it has a range in the YAML so an
experiment can switch it on.

`tier1.TIER1` keeps its current constants, so `augment_trials.py`, the round-trip demo and
`test_tier1.py` behave as before. The week-4 profiles pass their own ranges in.

The per-augmentation descriptions below give the mechanics. The numbers in this table are
authoritative where they differ.

### Body

**5.1 Limb length scaling** — `body`, **opt-in**
- Scale bone lengths directly in the skeleton (upper arm, forearm, and optionally the hand as a
  whole), not through `betas`. The removed `LimbLengthScaling` shifted `betas`, which mostly
  encodes a constant shoulder offset in this project's fits (Grounding §6.1).
- One factor per bone, shared by the left and right sides, so the signer stays symmetric.
- Range: uniform in ±7% (the measured between-signer spread is 2.9–5.7%).
- `retarget: true` (the default): re-solve the shoulder and elbow with the existing swing IK
  (`body_model/swing_ik.py`) so each wrist keeps its world position. With `retarget: false`
  the hands move, and the wrist displacement is reported. The trial moved wrists 68–83 mm
  without retargeting (Grounding §6).

**5.2 Joint coupled noise** — `body` and `fingers`, **on by default**
- Joints: the shoulder, elbow and wrist of each arm only. Spine, neck and head are not touched,
  because moving the head moves the carried face, which is the landmark for hand-to-face
  proximity.
- It **replaces** Tier 1's arm-joint draw in the week-4 profiles rather than stacking on it, so
  a default sample gets one arm perturbation, not two. Tier 1's arm-joint draw stays available
  as `body.tier1_arm_joints` with `enabled: false`; the loader rejects a profile that enables
  both.
- A constant rotation offset per joint, drawn once per clip, with correlations between joints.
  Unlike Tier 1's independent arm joints, these offsets are coupled:
  - **along the chain:** shoulder, elbow and wrist offsets share a common component with
    correlation `chain_coupling` (default 0.5);
  - **between sides:** the left and right offsets are tied with `side_coupling` (default 0.7).
    Mode `mirror` gives mirror-symmetric offsets (the default for signing), `synchronous` gives
    identical ones, and `independent` gives none.
- Body range: σ = 2.5° per joint, truncated at ±5°. Elbow σ = 4.75°, truncated at ±9.5°.
  Proven: ±4°, and ±7.5° for the elbow (Grounding §3 item 4).
- Finger mode: the same idea inside each MANO finger. Offsets are drawn on the MCP and PIP
  flexion, and DIP follows PIP at a ratio of about 2/3 (the usual PIP–DIP coupling).
  Neighbouring fingers share `finger_coupling` (default 0.5). Range: σ = 3.75°, truncated at
  ±7.5° (no published range).
- Optional `drift_deg` (default 0, so off): a smooth low-frequency wobble added on top of the
  constant offset, cut off at about 1 Hz. This is the only time-varying spatial noise in the
  design, and it is off unless a profile turns it on.

**5.3 Movement-size scaling** — `body`, opt-in (Tier 2 item 5)
- Scale each wrist's path about its own centroid by one factor, then re-solve the arm with IK
  so the wrist follows the new path. The fingers ride along with the wrist.
- Range: 0.8–1.2, log-uniform. Both hands use the same factor.
- The start and end locations should stay in their region (Grounding item 5). Until the
  location-region invariant exists, report the start and end wrist displacement.

**5.4 Location jitter** — `body`, opt-in (Tier 2 item 6)
- One constant offset per clip added to both wrist paths in a torso-anchored frame (shoulder
  centre, shoulder-width axis, torso up), then IK. Both hands share the offset, so the distance
  between them is unchanged.
- Range: σ = 12.5 mm per axis, truncated at 25 mm. Proven: σ = 10 mm, truncated at 20 mm.
- Face-anchored location shifts, including forehead lowering, wait for the location-region
  invariant and are out of scope.

### Fingers

**5.5 Within-category handshape jitter** — `fingers`, opt-in (Tier 2 item 7)
- Per-clip offsets on MANO finger joints. They are larger on non-selected fingers and the
  thumb, smaller on selected fingers, and smallest on the flexion of selected fingers
  (Grounding *Recommended*).
- Range: non-selected fingers and thumb σ = 7.5°; selected fingers σ = 3.75°; selected-finger
  flexion σ = 2°. All truncated at 2σ. There is no published range.
- Symmetric signs: when both hands start in the same handshape, the same offsets are applied to
  both hands, as SGNify does. "Same handshape" is decided by the feature check below.
- **Handshape-feature check (required):** a geometric classifier over the MANO pose that labels,
  per finger, extended / flexed / curved, plus spread, plus thumb position. Selected fingers
  are those that differ from the hand's resting state. Each threshold has a tolerance margin
  (`feature_margin_deg`, default 5°). Frames where the original sits within that margin of a
  threshold are ambiguous and are not compared, and a flip counts only when the jittered value
  crosses the threshold by more than the margin. A sample whose features differ from the
  original at any compared frame is redrawn, up to `max_redraws` (default 5). After that the clip gets
  no handshape jitter, and that is recorded.

### Face

**5.6 Facial expression jitter** — `face`, **opt-in**
- One additive offset per blendshape score, drawn once per clip, then clipped to [0, 1].
- Blendshapes are grouped, and each group has its own range in the YAML:
  - `brows` (brow*): σ = 0.0625, truncated at 0.125;
  - `eyes` (eyeBlink*, eyeSquint*, eyeWide*, eyeLook*): σ = 0.0625, truncated at 0.125;
  - `cheeks_nose` (cheekSquint*, noseSneer*): σ = 0.0375, truncated at 0.075;
  - `mouth_jaw` (jaw*, mouth*, cheekPuff, tongueOut): **off by default** (`enabled: false`,
    with σ = 0.0375, truncated at 0.075 ready for an experiment), because mouth patterns can be
    lexical (Grounding, Excluded).
- The 478 face points have to change to match. For each clip, fit a ridge regression from the
  clip's own blendshape scores to its face-point offsets in the head frame (rest = the clip's
  median face). The edit is Δpoints = W · Δblendshapes. Rigid skull points
  (`face_correspondence.py`) are held fixed.
- If a clip has too few face frames for the fit (fewer than `min_face_frames`, default 30),
  only the scores are jittered and the points are left alone. This is recorded.
- Optional `frame_noise` (default 0): small smoothed per-frame noise on the scores, off unless a
  profile turns it on.

### Temporal

**5.7 Time warping** — `temporal`. One augmentation, three modes:
- `uniform` (on, as Tier 1): one duration factor for the whole clip, 0.6–1.65× log-uniform
  (proven: 0.67–1.5×). This is Tier 1's `duration`, moved under the profile so its range is in
  the YAML.
- `varied` (**opt-in**): a smooth monotonic re-timing whose local rate varies across the clip.
  Built from `num_knots` (default 4) rate values, each drawn log-uniform in 0.7–1.43,
  interpolated and integrated. The total duration is then rescaled to the uniform factor (or
  kept). The last `protect_end` fraction of the clip (default 0.2) is re-timed only uniformly,
  so the end deceleration (telicity) keeps its shape.
- `segment` (**opt-in**): the "partial" warp. One window covering 20–60% of the clip, starting
  anywhere, is re-timed by a factor 0.6–1.65×. The rest keeps its speed. The rate is blended
  over 3 frames at each edge so speed does not jump.
- **How the modes combine.** `uniform` always runs when time warping is enabled. On top of
  it, at most one of `varied` and `segment` runs per clip, chosen with weights from the YAML
  (`mode_weights: {none: 1, varied: 0, segment: 0}` in the default profile; `{none: 1, varied:
  1, segment: 1}` in `ambitious`). `varied` keeps the uniform factor's total
  duration; `segment` multiplies the uniform factor inside its window.
- Resampling reuses Tier 1's `resample` / `resample_points`, which interpolate SMPL-X rotations
  properly and handle missing frames.

**5.8 Frame dropping (motion slicing)** — `temporal`, **on by default** (Tier 2 item 10)
- Mode `rate`: lower the frame rate by keeping frames on a coarser grid, for example 30 → 20 or
  12 fps. The target fps is drawn from `target_fps` (default 12–25), never below `min_fps`
  (default 8; proven: 10, where human intelligibility stops improving [T]).
- Mode `random`: drop individual frames with probability p (default 0–0.2 drawn per clip),
  never more than `max_gap_ms` in a row (default 125 ms, which keeps ≥8 fps everywhere).
- `fill: none | hold | interpolate`. The default `none` removes the dropped frames, so the clip
  gets shorter, as a lower-fps video would. Timestamps are kept so the clip's real duration is
  unchanged.
- One mode per clip, chosen with `mode_weights` like time warping. The default profile uses
  `random` only; `ambitious` weights `rate` and `random` equally.

**5.9 Hand-dropout runs** — `temporal`, opt-in (Tier 2 item 10)
- Imitates MediaPipe losing a hand. For each hand, mark runs of 1–10 frames (proven: ≤ 8) as
  missing: NaN in
  that hand's 21 output landmarks (world and image space), and its validity mask false. The
  SMPL-X parameters are left alone. Whole axes are never masked.
- The run rate and length distribution come from the project's cached MediaPipe output
  (`*_hands.npz`, 25 clips in `data/round_trip_eval`), which misses about half of WLASL's hands. The builder measures it once,
  writes the fitted values into the YAML with a comment saying where they came from, and keeps
  the measurement script in `scripts/`.
- Only frames where the hand was actually detected can be dropped.

### Scene

The Tier 1 scene augmentations (mirror, view, scale, uncentre) and Tier 1's arm-joint draw get
region tags (`scene`, and `body` for arm joints) and their ranges move into the profile. Their
behaviour does not change, and `test_tier1.py` must still pass. In the week-4 profiles they use
the widened ranges of §5.0 (yaw ±25°, pitch ±6°, scale 0.75–1.25, uncentre ±0.125); the
constants in `tier1.TIER1` itself are unchanged.

## 6. Region tags and registry

- `AugmentationRegistry` gains a `region` per augmentation (`face`, `body`, `fingers`,
  `temporal`, `scene`), and `list_by_region(region)`. One augmentation can carry two regions
  (joint coupled noise is `body` and `fingers`). Each region has its own YAML block, so the
  same augmentation can be configured separately per region.
- The existing `category` field (extrinsic / intrinsic / temporal) stays for the old classes.
- Every augmentation reports, per clip: its draw, the maximum wrist displacement in mm, the
  maximum change in bone length, whether frame-to-frame jitter increased, and any fallback
  taken (no redraw found, too few face frames). Face augmentations also report face rigidity on
  the head, as `augment_face_trials.py` measures it.

## 7. YAML profiles

A new profile format, read by a new loader. The older profiles stay as they are.

```yaml
profile_name: slr_week4_default
seed: 20261107
schema_version: 1

regions:
  body:
    joint_coupled_noise:
      enabled: true
      probability: 1.0
      ranges:
        sigma_deg: 2.5            # proven: ±4° shoulder/wrist (SPOTER [M1])
        truncate_deg: 5.0
        elbow_sigma_deg: 4.75     # proven: ±7.5° elbow, no harm [T]
        elbow_truncate_deg: 9.5
        chain_coupling: 0.5
        side_coupling: 0.7
        side_mode: mirror
        drift_deg: 0.0
    limb_length_scaling:
      enabled: false        # opt-in experiment arm
      probability: 1.0
      ranges:
        factor: [0.93, 1.07]  # proven: none in literature; between-signer spread 5.7% (§6.1)
        bones: [upper_arm, forearm]
        retarget: true
  fingers: { ... }
  face: { ... }
  temporal: { ... }
  scene: { ... }
```

- Every key under `ranges` is read. A key the augmentation does not know is an error, not
  silently ignored. A missing key falls back to the code default, and the loader prints the
  default it used.
- Ranges are given as `[low, high]` with an optional `distribution: uniform | log_uniform`.
  Scalar σ values are used for truncated normals.
- Every range key carries a YAML comment with its proven range and source, or "proven: none"
  with the reason, as in the example above.
- Profiles to ship:
  - `slr_week4_default.yaml`: Tier 1 scene augmentations + uniform duration + joint coupled
    noise + frame dropping, at the §5.0 "used" ranges. Everything else is present but has
    `enabled: false`, so every range is visible in one file.
  - `slr_week4_ambitious.yaml`: everything enabled at the §5.0 "used" ranges (mouth/jaw still
    off).
  - `slr_week4_literature.yaml`: the same augmentations as the default profile, at the proven
    ranges. It is the conservative baseline for the ablation; augmentations with no proven range
    are disabled in it.

**Documenting the ranges.** The proven-versus-used table of §5.0 also goes in two more places:
- `README.md`: a new section "Augmentation ranges" after "Seeing the augmentations", with the
  table and a sentence explaining that the used ranges are deliberately about 25% beyond the
  literature. It links to the docs page.
- `docs/Augmentation_Ranges.md`: the full table with sources, the 25% rule, and per-row notes
  where Grounding gives more context (for example 0.5× duration hurting once). It is linked from
  `docs/index.md`, and from Grounding §3 as the place where the week-4 ranges live.
The table lives in these files, the YAML comments and this design. When a range changes, all
four are updated together.
- A single augmentation arm (for the later ablation) is made by overriding `enabled` on the
  default profile. The loader accepts an override list such as
  `--enable face.expression_jitter`, so arms do not each need a file.
- `configs/index.md` lists the new profiles.

## 8. Calls made by the designer (not discussed with the user)

1. The four suggested augmentations (movement size, location jitter, handshape jitter, hand
   dropout) are **opt-in**, because Grounding puts them in Tier 2, which needs per-sign gates
   or invariant checks that do not exist yet. The `ambitious` profile turns them on.
2. "Partial time warping" is read as two opt-in modes: `varied` (rate varies smoothly over the
   whole clip) and `segment` (one window re-timed). The constant version is Tier 1's `uniform`.
3. "Motion slicing" is read as dropping frames, not cropping the clip, since cropping is
   excluded by Grounding.
4. Limb length scaling scales bones directly and retargets the wrists by default.
5. Joint coupled noise also covers fingers, as finger-synergy coupling.
6. "Slightly above the proven range" (the user's request) is set at about 25% beyond the widest
   proven range, widened in log space for log-uniform ranges. Probabilities are not widened.
   Where no range is published (finger σ, blendshape σ, warp knot rates) the used value is the
   designer's estimate widened by 25%, marked "proven: none" in the YAML.
6a. The ablation baseline at the proven ranges is a third profile, `slr_week4_literature.yaml`,
   replacing the earlier "wide" profile, which the widened defaults made redundant.
7. The temporal and scene augmentations get regions of their own beside face, body and fingers.
8. Joint coupled noise touches the arms only and replaces Tier 1's arm-joint draw in the week-4
   profiles instead of stacking on it.
9. The chain outputs classifier-ready landmarks (world and image space) as well as SMPL-X
   parameters. It fits the weak camera itself, and `export_round_trip_demo.py` is left alone.
10. Time warping always applies `uniform`, plus at most one of `varied` / `segment` per clip by
    weight. Frame dropping picks one mode per clip by weight.
11. Tier 1 keeps its existing random stream so its draws stay bit-identical. Every new
    augmentation gets its own seeded stream.
12. The handshape check has a 5° tolerance margin, so frames near a threshold do not force
    every sample to be rejected.

## 9. Constraints

- Python 3.9+, Black at 100 characters. New code under `src/kinetic_augment/augmentations/`,
  tests under `tests/`, profiles under `configs/`.
- No licensed model weights committed. Tests that need SMPL-X or MANO assets skip when they are
  missing, as `tests/hand_assets.py` and `tests/face_assets.py` do.
- Update every `index.md` touched, and `AGENTS.md`'s folder list if a folder is added.
- Do not break `test_tier1.py`, the phase scripts, `augment_trials.py`, or
  `augment_face_trials.py`.

## 10. How it is checked

- **Unit tests per augmentation:** zero strength reproduces the input exactly; the same seed
  gives the same draw; draws stay inside the YAML ranges; re-timings are monotonic; bone
  lengths are unchanged (except limb scaling, which changes only the bones named); frame
  dropping never goes below `min_fps`; face rigid points do not move under expression jitter;
  the handshape features are unchanged after handshape jitter.
- **Profile tests:** each shipped profile loads; unknown keys fail; `enabled: false` really
  skips; overrides work.
- **Visual trial:** a script in the style of `augment_face_trials.py` that renders one panel per
  new augmentation for a few cached clips, and writes the per-augmentation report of §6 to
  JSON. Its videos go under `data/augment_trials/`.

## 11. Left to the builder

- Module layout and file names (for example one module per region).
- The clip container's name and fields beyond those in §4.
- How the old `TimeWarping`, `FrameDropping` and `JointCoupledNoise` relate to the new ones
  (new names, or a `v2` suffix), provided the old ones keep working.
- The IK details for retargeting, reusing `swing_ik.py` where it fits.
- The exact thresholds of the handshape-feature classifier, as long as they are documented and
  tested on the cached clips.
