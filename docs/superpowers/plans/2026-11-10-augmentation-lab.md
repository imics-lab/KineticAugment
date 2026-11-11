# Augmentation Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser workbench to view and tune the week-4 augmentations: live knobs on a local
server with the models (full mode), or a preloaded zip that opens by double-click (share mode).

**Architecture:** One frontend (plain HTML + classic `<script>` files, three.js 0.147.0 vendored)
talks to a `DataSource`: `LiveSource` calls a local FastAPI server that runs one chain entry per
pane on a cached fit; `ShareSource` reads precomputed pane bundles packed into `.js` files. The
Python side lives in a new package `kinetic_augment.lab`; the chain gets one optional hook so
extreme mode can pin draws to range edges without changing default behaviour.

**Tech Stack:** Python ≥ 3.9, numpy, FastAPI + uvicorn + python-multipart (new `lab` extra),
three.js 0.147.0 (classic build), Node ≥ 18 for JS unit tests (`node --test`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-augmentation-lab-design.md`

## Global Constraints

- Python 3.9+; Black at 100 characters, **run only on files this plan creates or edits** (pass
  explicit paths; never reformat whole folders).
- Implementation under `src/kinetic_augment/lab/`; tests under `tests/`; scripts in `scripts/`.
- Every new folder gets an `index.md`; update the nearest `index.md` whenever a file is added,
  removed or renamed; add new index paths to the folder list in `AGENTS.md`.
- No licensed weights (SMPL-X, MANO, FLAME, MediaPipe task files) in git or in the share zip.
- With extreme mode off and no hook, every existing chain output and seed is unchanged.
- Frontend: no build step, no ES modules (they fail from `file://` in Chrome), no CDN at runtime.
  Every frontend file is a classic script that adds to `window.Lab`; files that Node tests also
  load end with `if (typeof module !== 'undefined') module.exports = …`.
- Plain English first, technical name second, for every augmentation and every knob.
- Server binds 127.0.0.1 only.
- Tests that need the models use `tests.hand_assets.requires_smplx` (they skip without assets).
- Commits end with the attribution line configured for this session; never name the agent as a
  contributor anywhere else.

## Review Focus

1. **A knob dragged to a value the augmentation rejects** (e.g. coupling 1.2, truncation 0 with
   σ > 0): the pane shows "this value isn't allowed: <reason>" and keeps its last good frames; the
   server returns 422, never 500. Test in Task 7 (`test_pane_rejects_bad_value`).
2. **Share zip opened in Safari as well as Chrome, offline**: no `fetch`, no modules, gzip via
   `DecompressionStream` (Safari ≥ 16.4). Fallback text if `DecompressionStream` is missing.
   Test in Task 10 (`test_share_index_has_no_module_or_fetch`) plus the manual Safari check.
3. **A clip whose hands or face are missing for frames** (NaN points, `body_ok` 0): panes draw
   what exists and skip the rest without throwing. Test in Task 8 (`bundle.test.js` NaN case).
4. **Uploading a second video while the first is still fitting**: jobs queue one at a time; the
   status shows "waiting". Test in Task 6 (`test_jobs_run_one_at_a_time`).
5. **Saving with a name that has spaces, slashes or an existing file's name**: the name becomes
   a safe slug; an existing name asks to overwrite (409 from the server unless `overwrite`).
   Test in Task 7 (`test_save_name_slug_and_conflict`).

---

## File structure

```
src/kinetic_augment/augmentations/chain.py      modify: optional draw_hook
src/kinetic_augment/augmentations/profile.py    modify: profile_from_dict (load_profile uses it)
src/kinetic_augment/lab/
  __init__.py      package marker, LAB_VERSION
  index.md
  runs.py          Run, RUNS, merge, run_profile (moved from scripts/demo_week4.py)
  bundle.py        write_data, bundle_arrays, jsonable, source_frames (moved), encode/decode pane blobs
  extreme.py       EXTREME_ENTRIES, extreme_draw, make_hook
  knobs.yaml       hand-written knob + plain-English data for every lab entry
  schema.py        load_knobs, build_schema(profile) -> dict sent to the frontend
  pane.py          PaneRequest, pane_profile, run_pane
  clipstore.py     PreparedClip, prepare_clip, ClipStore (disk cache of fits + frames)
  models.py        REQUIRED_MODELS, check_models, LabConfig (model dir persisted)
  ingest.py        ingest_video (MediaPipe + fit), JobQueue
  saving.py        slugify, validate_profile_text, save_setting
  server.py        create_app(...) FastAPI app
  cli.py           main() for the `kinetic-lab` command
  share.py         plan_grid, choose_grid, decimation, build_share
  static/
    index.html, setup.html, style.css
    vendor/three.min.js, vendor/OrbitControls.js   (three@0.147.0, classic)
    bundle.js      decode a pane blob (gzip, int16 mesh deltas)
    sources.js     LiveSource, ShareSource
    panes.js       two three.js panes, shared camera, view toggles
    timeline.js    play/scrub, playback vs source-time clock
    knobs.js       two-marker sliders, enum dropdowns, mode switch, dice, copy A→B
    explain.js     plain-English box and readout
    yaml_edit.js   build a saved profile's YAML text from the base text + edits
    app.js         state and wiring
scripts/demo_week4.py              modify: import moved pieces from kinetic_augment.lab
scripts/build_lab_share.py         new: CLI around lab.share.build_share
tests/test_lab_*.py, tests/js/*.test.js
```

---

### Task 1: Chain hook and in-memory profiles

**Files:**
- Modify: `src/kinetic_augment/augmentations/chain.py` (`AugmentationChain.__init__`, `run.step`)
- Modify: `src/kinetic_augment/augmentations/profile.py` (`load_profile`)
- Test: `tests/test_lab_chain_hook.py`

**Interfaces:**
- Produces: `AugmentationChain(profile, skeleton, draw_hook=None)`. `draw_hook(name: str,
  draw: dict, aug) -> dict` is called after `aug.sample(...)` and before `aug.apply(...)`; the
  returned dict is applied and recorded.
- Produces: `profile_from_dict(data: dict, source: str = "<dict>", enable=(), disable=()) ->
  Profile` with exactly `load_profile`'s validation; `load_profile` reads the YAML and calls it.

- [ ] **Step 1: Write the failing tests**

```python
def test_profile_from_dict_matches_load_profile():
    path = CONFIGS / "slr_week4_default.yaml"
    data = yaml.safe_load(path.read_text())
    a, b = load_profile(path), profile_from_dict(data, source=str(path))
    assert a == b

def test_profile_from_dict_rejects_like_load_profile():
    data = yaml.safe_load((CONFIGS / "slr_week4_default.yaml").read_text())
    data.pop("seed")
    with pytest.raises(ValueError, match="seed"):
        profile_from_dict(data)

@requires_smplx
def test_hook_none_is_unchanged():
    clip, skeleton = synthetic_clip(frames=24), get_skeleton()
    profile = load_profile(CONFIGS / "slr_week4_ambitious.yaml")
    a = AugmentationChain(profile, skeleton).run(clip)
    b = AugmentationChain(profile, skeleton, draw_hook=None).run(clip)
    assert a.draws == b.draws and np.array_equal(a.outputs.pose_world, b.outputs.pose_world, equal_nan=True)

@requires_smplx
def test_hook_replaces_draw_before_apply():
    clip, skeleton = synthetic_clip(frames=12), get_skeleton()
    profile = load_profile(CONFIGS / "slr_week4_default.yaml", enable=["body.location_jitter"])
    def hook(name, draw, aug):
        return {"offset_mm": [25.0, 0.0, 0.0]} if name == "body.location_jitter" else draw
    res = AugmentationChain(profile, skeleton, draw_hook=hook).run(clip)
    assert res.draws["body.location_jitter"]["offset_mm"] == [25.0, 0.0, 0.0]
```

- [ ] **Step 2: Run** `pytest tests/test_lab_chain_hook.py -v`. Expected: FAIL (`profile_from_dict` / `draw_hook` missing).
- [ ] **Step 3: Implement** both interfaces. In `step`, the hook runs only for entries that pass the gate; the `wrist_moved_mm_max` measure and `_non_identity` use the hooked draw.
- [ ] **Step 4: Run** the new tests and `pytest tests/test_week4_profile.py tests/test_week4_*.py -q`. Expected: all pass (or skip without assets).
- [ ] **Step 5: Commit** `feat: add a draw hook to the chain and profiles from dicts`.

### Task 2: Move shared run and bundle code into `kinetic_augment.lab`

**Files:**
- Create: `src/kinetic_augment/lab/{__init__.py,index.md,runs.py,bundle.py}`
- Modify: `scripts/demo_week4.py` (delete the moved definitions; import them)
- Modify: `src/kinetic_augment/index.md`, `AGENTS.md` (folder list)
- Test: `tests/test_lab_bundle.py`

**Interfaces:**
- Produces (moved unchanged): `runs.Run`, `runs.RUNS`, `runs.merge` (was `_merge`),
  `runs.run_profile(run, profiles)`, `bundle.write_data`, `bundle.bundle_arrays`,
  `bundle.jsonable`, `bundle.source_frames`, `bundle.SCALE_I16 = 10000.0`.
- Produces (new): `bundle.encode_pane(meta: dict, arrays: dict[str, tuple[np.ndarray, str]],
  mesh_delta: bool = True) -> bytes` and `bundle.decode_pane(blob: bytes) -> tuple[dict,
  dict[str, np.ndarray]]`.
  Blob layout: `b"KLAB"` + uint32 LE header length + UTF-8 JSON header (`meta` plus
  `arrays: {name: {dtype, shape, offset, delta}}`) + 4-byte-aligned array bytes. When
  `mesh_delta`, `mesh_verts` (int16, `(T, V, 3)`) is stored as frame 0 then frame-to-frame
  differences (int16), `delta: true`. Gzip is applied by callers that need it, not here.
- Pane arrays (the only ones the Lab sends): `times` f32 `(T,)`, `mesh_verts` i16 `(T, V, 3)`,
  `aug_arm` f32 `(T, 6, 3)`, `aug_hands` f32 `(T, 2, 21, 3)`, `aug_face` f32 `(T, F, 3)`.
  Image-space arrays are not sent.

- [ ] **Step 1: Write the failing tests**: `test_encode_decode_round_trip` (random arrays incl.
  NaN, mesh with large frame jumps within int16; decoded arrays equal the inputs exactly),
  `test_blob_header` (starts with `b"KLAB"`, offsets 4-byte aligned), and
  `test_demo_week4_still_imports` (`import demo_week4` with `scripts/` on `sys.path`;
  `demo_week4.RUNS is kinetic_augment.lab.runs.RUNS`).
- [ ] **Step 2: Run** `pytest tests/test_lab_bundle.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement**: move the code verbatim; add encode/decode (delta via `np.diff` on
  int32 then cast to int16; decode with `np.cumsum(..., dtype=np.int32)`).
- [ ] **Step 4: Run** `pytest tests/test_lab_bundle.py tests/test_week4_*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `refactor: move week-4 run and bundle helpers into kinetic_augment.lab`.

### Task 3: Extreme draws

**Files:**
- Create: `src/kinetic_augment/lab/extreme.py`
- Test: `tests/test_lab_extreme.py`

**Interfaces:**
- Produces: `EXTREME_ENTRIES: frozenset[str]` = `body.joint_coupled_noise`,
  `body.limb_length_scaling`, `body.movement_size`, `body.location_jitter`,
  `fingers.joint_coupled_noise`, `face.expression_jitter`, `scene.view`, `scene.scale`,
  `temporal.time_warp` (uniform mode only), `temporal.frame_drop`.
- Produces: `extreme_draw(name: str, draw: dict, ranges: dict, tier1: Tier1Ranges, edge: str) ->
  dict` (`edge` is `"low"` or `"high"`; returns a new dict; other keys copied).
- Produces: `make_hook(edge: str, profile: Profile) -> Callable[[str, dict, Any], dict]` that
  applies `extreme_draw` to entries in `EXTREME_ENTRIES` and passes others through.

Edge rules (the spec's "pinned to the edge"; sign: high = +, low = −):

| Entry | Draw keys changed |
|---|---|
| body.joint_coupled_noise | each `joints[i]["angle"]`: shoulders/wrists ±`truncate_deg`, elbows ±`elbow_truncate_deg`; right side keeps the left side's mirrored axis when `side_mode` is `mirror`/`synchronous`; `drift` → None |
| body.limb_length_scaling | every `factors[bone]` → range lo (low) / hi (high) |
| body.movement_size | `factor` → lo / hi |
| body.location_jitter | `offset_mm` → the drawn direction scaled to length `truncate_mm` (high) or its negative (low); zero direction → +x |
| fingers.joint_coupled_noise | every `mcp_deg`, `pip_deg` → ±`truncate_deg` |
| face.expression_jitter | each enabled group's offsets → ±its `truncate`; `noise` → None |
| scene.view | `yaw_deg` → ±`tier1.yaw_deg`, `pitch_deg` → ±`tier1.pitch_deg` |
| scene.scale | `factor` → `tier1.scale` lo / hi |
| temporal.time_warp | `duration` → `tier1.duration` lo / hi, only when `mode == "none"` (uniform); otherwise unchanged |
| temporal.frame_drop | `rate` mode: target fps → range lo (low, more dropped) / hi; `random` mode: p → range hi (low) / lo (high); `keep` recomputed with the augmentation's own `_rate_keep` / `_close_gaps` |

The implementer reads each entry's `sample` to get exact key names (e.g. the `ArmJointDraw`
field for the angle) before writing the rule.

- [ ] **Step 1: Write the failing tests**, one per row, building the draw with the real
  augmentation's `sample` on `synthetic_clip` then checking the pinned values, e.g.

```python
@requires_smplx
def test_joint_noise_high_pins_to_truncation():
    profile = load_profile(CONFIGS / "slr_week4_default.yaml")
    aug = AugmentationRegistry.get("body.joint_coupled_noise")(ranges=resolved_ranges(profile, "body.joint_coupled_noise"))
    draw = aug.sample(np.random.default_rng(0), synthetic_clip(frames=12), None)
    out = extreme_draw("body.joint_coupled_noise", draw, aug.ranges, tier1_ranges(profile), "high")
    angles = {(j["side"], j["joint"]): j["angle"] for j in out["joints"]}
    assert angles[("L", "elbow")] == 9.5 and angles[("L", "shoulder")] == 5.0
```

  plus `test_hook_ignores_other_entries` and `test_low_is_negative_of_high` for the noise entries.
- [ ] **Step 2: Run** `pytest tests/test_lab_extreme.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `extreme.py` per the table.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: pin augmentation draws to their range edges for the lab`.

### Task 4: Knob schema and plain-English text

**Files:**
- Create: `src/kinetic_augment/lab/knobs.yaml`, `src/kinetic_augment/lab/schema.py`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]` adds `lab/knobs.yaml`, `lab/static/**/*`)
- Test: `tests/test_lab_schema.py`

**Interfaces:**
- `knobs.yaml` is keyed by lab entry id = the `RUNS` keys with an `entry` (so
  `temporal.time_warp.uniform` etc.), each with: `title`, `does`, `changes` (list), `keeps`
  (list), `proven`, `used`, `note` (start from `W4_EXPLAIN` in `demo/round_trip/week4.js`;
  copy the text), `extreme: bool`, and `knobs`: a list of
  `{key, kind: scalar|range|enum|bool, plain, technical, unit, proven: [lo, hi] | null, limits:
  [min, max], step, options (enum), share: true|false}`. Group knobs (face blendshape groups)
  use dotted keys (`brows.sigma`, `brows.enabled`). `plain` is one sentence a non-specialist
  understands; `technical` is the symbol and YAML key, e.g. `"elbow σ (elbow_sigma_deg)"`.
  `share: true` marks the 1–2 knobs per entry that get share-mode stops (σ / factor / main
  range); the rest are full-mode only and show "live only" in share mode.
- Slider limits: about 2× the larger of the proven and used values, inside what the
  augmentation's constructor accepts.
- Produces: `load_knobs() -> dict` and `build_schema(profile: Profile) -> dict` returning
  `{"profile": name, "seed": int, "entries": [{id, entry, group, title, does, changes, keeps,
  proven, used, note, extreme, overrides, knobs: [{…, default}]}], "profiles": ["default",
  "ambitious", "literature"]}` where `default` is the value from `resolved_ranges(profile, entry)`
  merged with the run's `overrides`.

- [ ] **Step 1: Write the failing tests**:
  `test_every_lab_entry_has_knobs` (ids == RUNS keys with an entry),
  `test_every_range_key_is_a_knob` (for each entry, every key of the augmentation's
  `DEFAULT_RANGES` except settings fixed by `overrides` (`mode_weights`) appears, dotted for nested),
  `test_defaults_within_limits`, `test_plain_text_present` (non-empty `plain`, `does`),
  `test_extreme_flag_matches_extreme_entries`,
  `test_limits_accepted_by_augmentation` (constructing the augmentation with each knob at its
  min and max, others default, raises nothing).
- [ ] **Step 2: Run** `pytest tests/test_lab_schema.py -v`. Expected: FAIL.
- [ ] **Step 3: Write `knobs.yaml` and `schema.py`.** Use the `# proven:` / `# setting:` comments
  in `configs/slr_week4_*.yaml` and `docs/Augmentation_Ranges.md` for proven ranges.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: describe every lab knob in plain English with its ranges`.

### Task 5: Pane runner and clip store

**Files:**
- Create: `src/kinetic_augment/lab/pane.py`, `src/kinetic_augment/lab/clipstore.py`
- Test: `tests/test_lab_pane.py`

**Interfaces:**
- `PaneRequest` (frozen dataclass): `clip_id: str, entry_id: str, values: dict, mode: str
  ("random"|"extreme"), reroll: int = 0, edge: str = "high", profile: str = "default"`.
  `entry_id` may also be `combined.<profile>` (a shipped profile whole) or `saved:<name>` (a
  file in the saved folder, whole; live mode only); `values` is ignored for both.
  `PaneRequest.key() -> str`: canonical JSON (sorted keys) used for caching and share manifests.
- `pane_profile(base: Profile, req: PaneRequest) -> Profile`: deep copy of `base`, only the run's
  entry enabled at probability 1, ranges = merge(resolved ranges, run overrides, `req.values`
  un-dotted), `seed = base.seed + req.reroll`; validated by re-running
  `profile_from_dict` on the result (raises `ValueError` for rejected values).
- `PreparedClip`: `clip_id, clip (ClipData with camera), frames (np.ndarray), base_meta,
  base_arrays, n_images: int`. `base_arrays` holds the pane arrays plus `mesh_faces` (uint16
  `(F, 3)`, `skeleton.model.faces`), so the frontend gets the topology from the base blob.
- `prepare_clip(clip_id, result, hands, face, models) -> PreparedClip`: the first half of
  `demo_week4.export_week4` (clip_from_result, fit_clip_camera, base outputs with vertices).
- `run_pane(prep: PreparedClip, req: PaneRequest, base: Profile, models) -> tuple[dict, dict]`:
  meta (`draws`, `applied`, `fallbacks`, `metrics` from `report.measure`, `ranges`, `n`, `fps`,
  `mirrored`, `reroll`, `mode`, `edge`) and pane arrays; extreme mode passes
  `make_hook(edge, profile)` when the entry is in `EXTREME_ENTRIES`, else raises
  `ValueError("no extreme for this one")`.
- `ClipStore(root: Path, models)`: `list() -> list[dict]` (id, frames, source), `get(clip_id) ->
  PreparedClip` (memory LRU of 4), `frame_path(clip_id, i) -> Path`, `add(clip_id, result, hands,
  face, frames_dir)`. On disk per clip: `root/<id>/fit.pkl` (pickle of `(result, hands, face)`),
  `root/<id>/frames/NNNN.jpg`. Default root `demo/round_trip/clips` so clips exported earlier
  appear: when `fit.pkl` is missing but `hands.npz`/`face.npz` exist, `get` refits with
  `run_clip` once and writes `fit.pkl`.

- [ ] **Step 1: Write the failing tests**: `test_pane_profile_only_entry_enabled`,
  `test_pane_profile_reroll_changes_seed`, `test_pane_profile_rejects_bad_value` (coupling 1.2 →
  ValueError), `test_request_key_is_canonical` (dict order does not matter), and with
  `@requires_smplx` on `real_clip()`: `test_run_pane_random_matches_week4_export`
  (`run_pane` for `body.location_jitter` at defaults with the ambitious profile gives the same
  `aug_hands` as running `run_profile(RUNS[...], profiles)` through the chain),
  `test_run_pane_extreme_location_offset_is_truncation`, `test_run_pane_under_two_seconds`.
- [ ] **Step 2: Run** `pytest tests/test_lab_pane.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `pane.py` and `clipstore.py`, reusing `bundle.bundle_arrays`
  (dropping image arrays) and `bundle.source_frames` for `times`.
- [ ] **Step 4: Run** the tests. Expected: PASS (model tests skip without assets).
- [ ] **Step 5: Commit** `feat: run one lab pane on a cached clip fit`.

### Task 6: Models check, config and video ingest

**Files:**
- Create: `src/kinetic_augment/lab/models.py`, `src/kinetic_augment/lab/ingest.py`
- Test: `tests/test_lab_models.py`, `tests/test_lab_ingest.py`

**Interfaces:**
- `REQUIRED_MODELS: tuple[str, ...]` = `smplx/SMPLX_NEUTRAL.npz`, `mano/MANO_LEFT.npz`,
  `mano/MANO_RIGHT.npz`, `mano/SMPL-X__FLAME_vertex_ids.npy`, `flame/generic_model.pkl`,
  `flame/mediapipe_landmark_embedding.npz`, `mediapipe/face_landmarker.task` (relative to the
  model dir). `MODEL_LINKS: dict[str, str]` with the official pages (SMPL-X
  `https://smpl-x.is.tue.mpg.de/`, FLAME `https://flame.is.tue.mpg.de/`, MediaPipe face
  landmarker via `scripts/download_models.py --face`, metrical-tracker embedding
  `https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe`).
- `check_models(model_dir: Path) -> list[str]` (missing relative paths).
- `LabConfig.load() / .save()`, field `model_dir: str | None`, stored at
  `~/.kinetic_lab/config.json` (`KINETIC_LAB_HOME` env var overrides the folder, for tests).
- `ingest.ingest_video(video: Path, clip_id: str, store: ClipStore, models, progress:
  Callable[[str, float], None]) -> str`: runs `extract` and `run_with_mano` and `read_frames`
  from `scripts/export_round_trip_demo.py` (imported by putting the repo's `scripts/` on
  `sys.path`; raise a clear error if the checkout has no `scripts/`), then `store.add`.
- `JobQueue`: `submit(video: Path, clip_id: str) -> job_id`, `status(job_id) -> {state:
  waiting|running|done|failed, step, progress, error}`; one worker thread, FIFO.

- [ ] **Step 1: Write the failing tests**: `test_check_models_lists_missing(tmp_path)`,
  `test_config_round_trip(tmp_path, monkeypatch)`, `test_jobs_run_one_at_a_time` (monkeypatch
  `ingest_video` with a slow fake; second job is `waiting` while the first runs, both end `done`),
  `test_failed_job_reports_error`.
- [ ] **Step 2: Run** `pytest tests/test_lab_models.py tests/test_lab_ingest.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: check lab model files and ingest new videos in a queue`.

### Task 7: Saving, server and `kinetic-lab` command

**Files:**
- Create: `src/kinetic_augment/lab/saving.py`, `src/kinetic_augment/lab/server.py`, `src/kinetic_augment/lab/cli.py`
- Modify: `pyproject.toml` (extra `lab = ["fastapi>=0.110", "uvicorn>=0.27", "python-multipart>=0.0.9"]`; script `kinetic-lab = "kinetic_augment.lab.cli:main"`)
- Modify: `.gitignore` (add `.superpowers/`, `configs/viewer_saved/*.yaml` is **not** ignored)
- Test: `tests/test_lab_server.py`, `tests/test_lab_saving.py`

**Interfaces:**
- `saving.slugify(name: str) -> str` (lowercase, `[a-z0-9_-]`, max 60 chars, non-empty or
  ValueError); `validate_profile_text(text: str) -> Profile` (yaml.safe_load +
  `profile_from_dict`); `save_setting(text: str, name: str, folder: Path, overwrite: bool) ->
  Path` (writes `folder/<slug>.yaml`, creates `folder/index.md` and adds a line per file;
  `FileExistsError` when present and not `overwrite`). Default folder `configs/viewer_saved/`.
- `server.create_app(store: ClipStore | None, config: LabConfig, models_loader=load_face_models,
  saved_dir: Path, static_dir: Path) -> FastAPI` with routes:

| Route | Returns |
|---|---|
| `GET /api/status` | `{mode: "live"|"setup", missing: [...], model_dir, links: MODEL_LINKS}` |
| `POST /api/models` `{model_dir}` | re-checks, loads models when complete, returns status |
| `GET /api/schema?profile=default` | `build_schema(...)` |
| `GET /api/profile-text?profile=default` | the base profile YAML text (for `yaml_edit.js`) |
| `GET /api/clips` | `store.list()` |
| `GET /api/clips/{id}/base` | pane blob (`application/octet-stream`) of the unaugmented clip |
| `GET /api/clips/{id}/frames/{i}.jpg` | source frame |
| `POST /api/pane` (PaneRequest JSON) | pane blob; 422 `{detail}` on ValueError |
| `POST /api/videos` (multipart file) | `{job_id, clip_id}` (clip id = file stem, slugified, suffixed if taken) |
| `GET /api/jobs/{job_id}` | job status |
| `GET /api/saved` / `GET /api/saved/{name}` | list / YAML text |
| `POST /api/saved` `{name, text, overwrite}` | 200 path, 409 exists, 422 invalid profile |
| `GET /` and static files | `static_dir` |

  In setup mode (models missing) every model-needing route returns 503 `{detail: "models
  missing"}`; `/api/status`, `/api/models`, static files and `/api/saved` still work.
- `cli.main(argv=None)`: `--port 8770`, `--host 127.0.0.1` (refuse anything else),
  `--clips-dir demo/round_trip/clips`, `--model-dir` (saved into LabConfig), `--no-browser`;
  starts uvicorn and opens `http://127.0.0.1:<port>/`.

- [ ] **Step 1: Write the failing tests** with `fastapi.testclient.TestClient` and a fake store /
  fake `run_pane` (monkeypatched) so no models are needed:
  `test_status_setup_when_models_missing`, `test_model_routes_503_in_setup`,
  `test_pane_returns_blob` (decodes with `decode_pane`), `test_pane_rejects_bad_value` (422 with
  the ValueError text), `test_save_name_slug_and_conflict` (`"My run / v2"` → `my-run-v2.yaml`;
  second POST → 409; with `overwrite` → 200), `test_save_rejects_invalid_profile` (422),
  `test_cli_refuses_public_host`.
- [ ] **Step 2: Run** `pytest tests/test_lab_server.py tests/test_lab_saving.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules and the pyproject/.gitignore edits; `pip install -e '.[lab]'`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: serve the lab locally with live panes and saved settings`.

### Task 8: Frontend core: data, panes, timeline, view toggles

**Files:**
- Create: `src/kinetic_augment/lab/static/{index.html,style.css,bundle.js,sources.js,panes.js,timeline.js,app.js}`, `static/vendor/{three.min.js,OrbitControls.js}`, `static/index.md`, `static/vendor/index.md`
- Test: `tests/js/bundle.test.js`, `tests/js/sources.test.js`

**Interfaces:**
- Vendor: three@0.147.0 `build/three.min.js` and `examples/js/controls/OrbitControls.js` from
  the npm tarball (`npm pack three@0.147.0`), MIT licence file alongside; note the version in
  `vendor/index.md`.
- `Lab.decodePane(bytes: Uint8Array) -> {meta, arrays: {name: {data, shape, stride}}}` (sync,
  mirrors `bundle.decode_pane`, cumsum for `delta`); `Lab.gunzip(bytes) -> Promise<Uint8Array>`
  via `DecompressionStream('gzip')`, throwing `Error("browser too old")` if missing.
- `DataSource` interface, implemented by `Lab.LiveSource(baseUrl)` and `Lab.ShareSource(data =
  window.LAB_DATA)`:
  `status()`, `schema(profile)`, `profileText(profile)`, `clips()`, `base(clipId)`,
  `pane(request) -> Promise<{meta, arrays}>`, `frameUrl(clipId, i)`, `saved()`,
  `savedText(name)`, `save(name, text, overwrite)`, `stops(entryId, knobKey) -> number[] | null`
  (null in live mode), `addVideo(file) -> {job_id, clip_id}`, `job(jobId)`, `isShare: boolean`.
  Share data scripts call `Lab.addBlob(sha1, base64)`, which stores into `Lab.blobs`;
  `ShareSource.pane` maps `PaneRequest` key → sha1 via `LAB_DATA.panes`, then base64 →
  `Lab.gunzip` → `Lab.decodePane`. App start: try `fetch('/api/status')` only when
  `location.protocol` starts with `http`; otherwise use `ShareSource`.
- `Lab.Panes(container, {onCameraChange})`: two side-by-side viewports, one shared
  `PerspectiveCamera` + `OrbitControls`; `setBase(pane)`, `setPane('A'|'B', pane)`,
  `showFrame(fA, fB, fBase)`, `setToggles({video, original, augmented, mesh, points, skeleton,
  face, hands})`, `cameraPreset('front'|'side'|'hands'|'face')`. Colours: original `#4a8fe7`,
  A `#e8913a`, B `#b56cf0` (CSS variables in `style.css`). Mesh faces come from
  `meta.mesh_faces` in the base blob (array `mesh_faces` uint16 for both modes).
- `Lab.Timeline(el, {fps, onFrame})`: play/pause, scrub, frame label, clock
  `'playback'|'source'` (behaviour as `demo/round_trip/all.js`: shorter pane holds last frame
  and shows "ended").
- Layout as the spec §5.1: left sidebar (clips, augmentations by group, whole profiles,
  saved settings), middle (source frame, pane A, pane B, timeline, toggles row), right knob
  panel container (filled in Task 9), explanation box above the panes.

- [ ] **Step 1: Write the failing Node tests**: `bundle.test.js` decodes a fixture blob written by
  a small Python helper `tests/js/make_fixture.py` (NaN points, delta mesh) and checks values;
  `sources.test.js` checks `ShareSource.pane` finds a blob by `PaneRequest` key and
  `stops()` returns the manifest stops.
- [ ] **Step 2: Run** `python tests/js/make_fixture.py && node --test tests/js/`. Expected: FAIL.
- [ ] **Step 3: Implement** the files. Also add `tests/test_lab_js.py` that runs `node --test
  tests/js/` via subprocess and skips when `node` is not on PATH, so `pytest` covers the JS.
- [ ] **Step 4: Run** `node --test tests/js/` and `pytest tests/test_lab_js.py`. Expected: PASS.
- [ ] **Step 5: Verify in the browser**: `kinetic-lab --no-browser`, open it in the Browser pane,
  load clip 57949, toggle every view, orbit, play; screenshot. No console errors.
- [ ] **Step 6: Commit** `feat: draw the lab panes, timeline and view toggles`.

### Task 9: Knobs, explanations, save and load

**Files:**
- Create: `src/kinetic_augment/lab/static/{knobs.js,explain.js,yaml_edit.js}`
- Modify: `src/kinetic_augment/lab/static/{app.js,index.html,style.css}`
- Test: `tests/js/yaml_edit.test.js`, `tests/js/knobs.test.js`, `tests/test_lab_yaml_edit.py`

**Interfaces:**
- `Lab.KnobPanel(el, schemaEntry, state, {onChange, isShare, stops})`: per knob one bar with
  green (proven), amber (past proven up to the used/default value), grey (rest up to limit),
  two markers A/B; enum → two dropdowns; bool → two checkboxes. Above the bars: per pane a
  random/extreme switch (extreme disabled with "no extreme for this one" when
  `schemaEntry.extreme` is false), a low/high toggle in extreme, a dice button, the reroll
  number; a "copy A → B" button. Each knob row shows `plain` first, then `technical` and unit
  in smaller text. No proven range → amber band labelled "no proven range".
- Share mode: markers snap to `stops`; knobs with `share: false` are greyed with "live only";
  moving a second knob in a pane resets the first to default and shows "one knob at a time in
  the shared version". Dice cycles reroll 0..R where R comes from the share manifest.
- Live mode: `onChange` fires debounced (150 ms while dragging, immediately on release); only
  the changed pane recomputes; an in-flight request for the same pane is superseded (ignore
  stale responses by a per-pane counter). A 422 shows its detail under the pane and keeps the
  last good frames.
- State: `state.panes.A|B = {perEntry: {entryId: {values, mode, edge, reroll}}}`, kept across
  clip switches and entry switches (spec §9).
- `Lab.explainHtml(schemaEntry)` and `Lab.readoutHtml(meta)`: does / changes / keeps /
  proven vs used / note; readout of draws, metrics (with units, in words) and fallbacks,
  collapsed by default.
- `Lab.buildSavedYaml(baseText: string, edits: {entryKey: {rangeKey: value}}, entries: string[])
  -> string`: line-based edit of the base profile text: for each edited entry sets `enabled:
  true` (probability left as the base has it); for each edited range leaf replaces the value, keeps the existing comment and appends
  `  # viewer: was <old>`; renames `profile_name: <name>`. Ranges are written in flow style
  (`[0.9, 1.1]`). Entry keys are YAML paths `region.key`; nested range keys dotted
  (`brows.sigma`). Run-level overrides that are not range keys (`mode_weights`) are written as
  range keys when the edited entry is a time-warp/frame-drop variant (they are range keys:
  `run_profile` merges overrides into `ranges`).
- "＋ add video" (live only): file input → `source.addVideo` → poll `source.job` every 2 s,
  showing the step and progress ("waiting", "extracting landmarks", "fitting", "done") → the
  new clip appears and opens; a failure shows the job's error.
- Save button: asks for a name and which pane; live → `source.save` (409 asks "replace?");
  share → download `<slug>.yaml`. Load menu: live → `source.saved()`; share → file picker;
  loading sets the chosen pane's values for every entry the file enables.

- [ ] **Step 1: Write the failing tests**: `yaml_edit.test.js` (edits keep comments, add
  `# viewer: was`, set `enabled: true`, handle a dotted face group key and a `[lo, hi]` range);
  `knobs.test.js` (snapping to nearest stop; band widths from proven/default/limits as fractions;
  second-knob reset rule in share mode) with knob math in pure functions exported for Node;
  `tests/test_lab_yaml_edit.py` runs a Node script that builds YAML from
  `configs/slr_week4_default.yaml` with edits to every lab entry and checks each output passes
  `validate_profile_text` and carries the edited values (skips without `node`).
- [ ] **Step 2: Run** `node --test tests/js/ && pytest tests/test_lab_yaml_edit.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the files and wire them in `app.js`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Verify in the browser**: live mode, `body.joint_coupled_noise` on 57949, drag
  elbow σ A to 4.75 and B to 10, see the panes differ; extreme high; dice; save "elbow-test",
  confirm `configs/viewer_saved/elbow-test.yaml` loads with `load_profile`; reload and load it
  back. Screenshot.
- [ ] **Step 6: Commit** `feat: tune lab knobs with two markers and save settings as profiles`.

### Task 10: Share build

**Files:**
- Create: `src/kinetic_augment/lab/share.py`, `scripts/build_lab_share.py`
- Modify: `scripts/index.md`
- Test: `tests/test_lab_share.py`

**Interfaces:**
- `plan_grid(schema: dict, stops: int, rerolls: int) -> list[PaneRequest]` (without clip id):
  per entry, for each knob with `share: true`, `stops` values evenly spread over its limits,
  always including its default and the proven-range edge(s) inside the limits (dedupe, sort);
  each in random reroll 0 and, when the entry has `extreme`, extreme low and high; plus random
  rerolls 1..`rerolls` at defaults; plus each enum option at defaults; plus `combined.default`,
  `combined.ambitious`, `combined.literature`.
- `choose_grid(schema, n_clips, bytes_per_pane, budget_bytes) -> (stops, rerolls)`: first of
  `(7, 3), (5, 3), (5, 2), (3, 2), (3, 0)` whose estimated size fits; else `(3, 0)` with a warning.
- `decimate_mesh(vertices: np.ndarray, faces: np.ndarray, target: int = 3000) -> (index:
  np.ndarray, faces: np.ndarray)`: quadric decimation of the rest mesh (`trimesh` +
  `fast-simplification`, added to the `lab` extra), then each kept vertex mapped to its nearest
  original vertex so per-frame meshes are `verts[:, index]`.
- `build_share(clip_ids, store, models, out_zip: Path, budget_mb: int = 200, profile="default")
  -> dict` (summary: panes, unique blobs, size). Steps: prepare each clip; measure one pane's
  gzipped size; `choose_grid`; run every request (dedupe identical blobs by sha1); gzip each blob;
  write `data/<clip>/<sha1>.js` files of the form `Lab.addBlob("<sha1>", "<base64>")`, a
  `data/manifest.js` (`window.LAB_DATA = {schema, profileText, clips: [{id, n, fps}], base:
  {clip: sha1}, panes: {requestKey: sha1}, stops: {entry: {knob: [...]}}, rerolls}`),
  frames as `data/<clip>/frames/NNNN.jpg`; copy `static/` unchanged; `index.html` lists the data scripts via a generated
  `data/load.js` that writes `<script>` tags in order; zip it all under one folder
  `augmentation-lab/` with a `README.txt` ("double-click index.html; works offline").
- `scripts/build_lab_share.py --clips 57949 01986 <third> --out dist/augmentation-lab.zip
  --budget-mb 200 --profile default`; `dist/` is git-ignored already (`dist/` in .gitignore).
- The third clip: the first clip in `demo/round_trip/clips.json` `order` that is not 57949 or
  01986 and has a video in `--video-dir`; export it with the existing exporter first if needed.

- [ ] **Step 1: Write the failing tests**: `test_plan_grid_includes_defaults_and_proven_edges`,
  `test_plan_grid_extreme_only_where_allowed`, `test_choose_grid_shrinks_to_budget`,
  `test_share_index_has_no_module_or_fetch` (static `index.html` and every `static/*.js` contain
  no `type="module"`, no `import `, and no `fetch(` outside `LiveSource`),
  `@requires_smplx test_decimate_mesh_target_and_valid_faces`, and
  `@requires_smplx test_build_share_small` (one clip, `stops=3, rerolls=0` forced through a
  parameter, budget large; zip contains `augmentation-lab/index.html`, `data/manifest.js`, no
  `.npz`/`.pkl`/`.task` files).
- [ ] **Step 2: Run** `pytest tests/test_lab_share.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `share.py` and the script.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Build the real zip** with the three clips; record size and pane count.
  Expected: under 200 MB.
- [ ] **Step 6: Verify offline in the browser**: unzip to the scratchpad, open
  `index.html` via `file://` in the Browser pane: clips load, knobs snap, extreme and dice work,
  save downloads a YAML, mesh/points toggles work, no console errors. Screenshot. Note for the
  user to also open it once in Safari.
- [ ] **Step 7: Commit** `feat: build the shareable offline lab zip`.

### Task 11: Setup page, docs and indexes

**Files:**
- Create: `src/kinetic_augment/lab/static/setup.html`, `docs/augmentation-lab.md`
- Modify: `src/kinetic_augment/lab/index.md`, `src/kinetic_augment/lab/static/index.md`,
  `docs/index.md`, `scripts/index.md`, `tests/index.md`, `src/kinetic_augment/index.md`,
  `AGENTS.md` (folder index list; a line on `kinetic-lab`), `agent.md` (sync), `README.md`
  (one short section linking `docs/augmentation-lab.md`)
- Test: `tests/test_lab_docs.py`

**Interfaces:**
- `setup.html`: plain steps (clone, `pip install -e '.[smplx,lab]'`, model files with
  `MODEL_LINKS` and what each is for, `kinetic-lab`, enter the model folder). In live mode it
  shows the missing files from `/api/status` and a folder field posting to `/api/models`; in
  share mode it is reached from the "Want live knobs? Set up full mode" link and shows the
  steps only.
- `docs/augmentation-lab.md`: what the Lab is, both modes, how to build the zip, where saved
  settings go and how to use one (`load_profile("configs/viewer_saved/<name>.yaml")`).

- [ ] **Step 1: Write the failing test** `test_every_lab_file_is_indexed`: every file in
  `src/kinetic_augment/lab/`, `static/` and `static/vendor/` appears in that folder's `index.md`;
  `AGENTS.md` lists those index paths.
- [ ] **Step 2: Run** `pytest tests/test_lab_docs.py -v`. Expected: FAIL.
- [ ] **Step 3: Write** the page, docs and index updates.
- [ ] **Step 4: Run** `pytest -q` (whole suite) and `node --test tests/js/`. Expected: all pass
  or skip; no new failures compared with `main`.
- [ ] **Step 5: Commit** `docs: document the augmentation lab and index its files`.
