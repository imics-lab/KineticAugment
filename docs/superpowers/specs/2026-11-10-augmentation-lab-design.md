# Augmentation Lab — design

Date: 2026-11-10. Status: ready for approval.

Calls marked **(builder's call)** were made while writing this design, not agreed with the user;
everything else was decided in conversation.

## 1. Intent

A visual tool for looking at and tuning the week-4 augmentations without regenerating exports.
Today every new setting (say elbow σ = 10° instead of 4.75°) means asking for the export
(`scripts/export_round_trip_demo.py` / `scripts/demo_week4.py`) to be rerun. The Lab lets the user
pick a clip, pick an augmentation, move its knobs and see the result straight away.

It serves two people:

- **The user (researcher)**: tunes ranges, compares settings, checks augmentations across clips,
  and saves settings to bring back into the pipeline.
- **Their professor**: follows progress remotely. He prefers looking at augmentations to reading
  numbers. He must be able to open it with nothing to install. If he wants live knobs, he can take
  a further step himself (install the package and the models) without hand-holding.

## 2. Goals

1. One screen (the **workbench**) to see an augmentation on a clip, comparing two settings side by
   side, each against the original.
2. Every augmentation and every knob explained in plain English first, with the technical name
   (σ, truncation, coupling …) second.
3. Knob ranges shown visually: the proven range, the part past it, and the rest.
4. Two ways to run it:
   - **Share mode**: a zip the professor unzips and double-clicks. A few preloaded clips and
     settings. No install, no models, no videos beyond the preloaded frames, no server.
   - **Full mode**: on a machine with the package and the models, a local server makes knobs live
     and lets the user add a new sign video, which runs through MediaPipe → SMPL-X fit → the chain.
5. **Save setting**: writes a profile YAML the pipeline's `load_profile` reads as is.

## 3. Non-goals

- Not a desktop app (no Electron/Tauri, no installer, no code signing, no auto-updater).
- Not a hosted website (no GitHub Pages). The share build is a local zip.
- Not a replacement for `demo/round_trip/` (round-trip error checking stays there).
- No training, metrics dashboards or batch processing of datasets.
- No editing of the YAML profiles in `configs/` in place; saves go to a separate folder.
- No multi-clip grid view in v1 (comparing across clips is done by switching clips with the knobs
  kept; see §5.4).

## 4. Anti-goals (would make it a failure even if it works)

- **Share mode that is not simple.** It must open on double-click and be obvious to a
  non-programmer. No setup steps, no errors about missing models, no server, no jargon-first text.
- **Jargon before meaning.** A knob labelled only "elbow_sigma_deg" fails. Plain words first.
- **Distributing licensed model files.** SMPL-X, MANO and FLAME weights must never be in the zip
  or in git (project rule: do not commit licensed model weights).
- **A save that the pipeline can't load.** Saved settings must pass `load_profile` validation.

## 5. What the user sees

### 5.1 Workbench layout (agreed from mockups)

- **Left sidebar**
  - **Clips**: the loaded clips. In full mode, a "＋ add video" entry.
  - **Augmentations**, grouped by region: body, fingers, face, scene, timing. These are the
    week-4 chain entries (`chain.ORDER`), including the off-by-default ones (limb length, movement
    size, location jitter, handshape jitter, face expression, varied/segment warps, hand dropout).
- **Middle**
  - The source video frame, then **pane A** and **pane B** side by side.
  - Each pane overlays the **original** (blue) with **its own augmented version** (A orange,
    B purple).
  - Both panes share one camera (drag to orbit either; front/side/hands/face camera buttons, as
    the existing pages have) and one timeline (play/pause, scrub, frame number).
- **View toggles** (both modes, apply to both panes): source video, original, augmented, body
  mesh, landmark points, skeleton lines, face points, hand points. Each one switches on and off.
  Defaults: everything on except what clutters (builder's call).
- **Right: knob panel** for the selected augmentation (§5.2).
- **Top of the knob panel / above the panes**: the plain-English explanation box (§5.3).

### 5.2 Knobs

- Each knob is **one slider with two markers**: an orange A marker and a purple B marker on the
  same range bar. Dragging a marker sets that pane's value.
- The bar is coloured: **green** = proven range (from the literature, as in
  `docs/Augmentation_Ranges.md`), **amber** = past proven up to the profile's used value, **grey**
  = beyond that up to the slider limit. Where there is no proven range ("proven: none"), the bar
  shows the used range in amber and says "no proven range" in words.
- Knob kinds:
  - scalar (σ, truncation, coupling, probability …): one value per marker;
  - range-valued (e.g. `factor: [0.93, 1.07]`): each marker sets the range's half-width/ends
    (builder's call on the exact control, e.g. a symmetric "spread" around the centre, as long as
    the saved YAML keeps the `[lo, hi]` form);
  - enum settings (`side_mode: mirror|independent|synchronous`, `bones`, fill mode): a dropdown
    per pane;
  - on/off for sub-groups (face blendshape groups).
- Slider limits: builder's call, defaulting to about 2× the larger of the proven and used values,
  never into values `load_profile` rejects (it already refuses strengths that hang or turn NaN).
- **Per pane, a draw-mode switch: random / extreme.**
  - **Random**: one seeded draw at that setting. A dice button rerolls (new seed for that pane).
    The seed is shown.
  - **Extreme**: every sampled value is pinned to the edge of its range: the truncation bound for a
    σ knob, the range end for a range knob, with a **low/high** toggle for which edge (builder's
    call on this definition; signs for multi-dimensional draws: all at the chosen edge, mirrored
    sides kept mirrored). Probabilities are treated as "it fires". The existing export already
    forces visible draws for mirror and hand dropout; extreme mode generalises that.
- A **copy A → B** button (builder's call, convenience).

### 5.3 Plain-English explanations

- For each augmentation, before any knob: what it does, what it changes, what it keeps, proven vs
  used range. Start from `demo/round_trip/week4.js` (`W4_EXPLAIN`: does / changes / keeps /
  proven / used / note), which already covers every week-4 run.
- **New**: a plain-English line for every knob, then its technical name. Example:
  "How far the elbow is bent or straightened, typical size — *elbow σ (elbow_sigma_deg), degrees*."
  The source of these lines is a single data file shared by both modes (builder's call; one
  place to update when an augmentation is added). The YAML comments (`# proven: …`,
  `# setting: …`) are the factual basis.
- Readout under the panes: the draws for each pane, the `report.measure` numbers (wrist movement,
  bone-length change, …) in words with units, and any fallback ("the handshape check failed 5
  times; clip left unchanged"). Secondary to the visuals: collapsed or small.

### 5.4 Switching clips

Switching clips keeps both panes' knob settings and draw modes, so the same two settings can be
checked on clip after clip.

### 5.5 Timing augmentations

Time warps and frame dropping change a clip's length. Reuse the existing clock choice from
`all.js`: **playback** (each pane plays its own frames; shorter clip holds its last frame, marked
"ended") and **source time** (each pane shows its frame nearest the current source frame).

### 5.6 Whole profiles

The sidebar also lists the whole profiles (default, ambitious, literature) and any saved settings,
shown as the full chain. Knob editing is per augmentation only; a whole profile is view-only in
v1 (builder's call).

### 5.7 Save and load settings

- **Save setting** (with a name typed by the user) writes a complete profile YAML: the base profile
  (the one the session started from, default `configs/slr_week4_default.yaml`) with that pane's
  changed values applied. Each changed line keeps its comment and adds
  `# viewer: was <old value>`. The entry is enabled if it was edited. Save asks which pane (A or B).
- Full mode: written to `configs/viewer_saved/<name>.yaml` (folder created with an `index.md`).
- Share mode: the file downloads in the browser instead.
- **Load**: a menu listing saved settings (full mode: the folder; share mode: a file picker for
  a YAML file) that sets a pane's knobs from it.
- Every saved file must load with `load_profile` without error (test this).

## 6. The two modes

### 6.1 Share mode (the zip)

- Built by a script (e.g. `scripts/build_lab_share.py --clips 57949 01986 … --out <zip>`) on the
  user's machine, which has the models.
- **Opens by double-clicking `index.html`** from the unzipped folder in Chrome or Safari, offline.
  Chrome blocks `fetch()` of local files from `file://`, so: all clip data and settings are packed
  into `.js` files loaded with `<script>` tags (e.g. base64 or typed-array literals assigned to a
  global), and three.js is vendored into the zip (no CDN).
- **Knobs snap** to values computed ahead of time: about 5–7 stops per knob across the bar,
  including the profile default and the proven-range edge. Only one knob differs from the profile
  default at a time; moving a second knob resets the first to default, with a short note saying
  so. Draw mode random uses the profile seed plus a small fixed set of rerolls (e.g. 3); extreme
  low/high are precomputed. (Builder's call on exact counts, within the size budget below.)
- Enum knobs: every option precomputed.
- Content (builder's call, user to confirm in the build command): 3 clips; every week-4
  augmentation; the three whole profiles.
- **Size budget**: the zip should stay small enough to send by a link (target under ~200 MB).
  Today one augmentation bundle with meshes is ~3.3 MB per clip, so a full stop grid with meshes
  will not fit uncompressed. Meshes are included for every stop (§10) but quantised and/or
  decimated; if the budget still can't be met, cut stops or rerolls before dropping meshes. Licensed model files are never included (derived per-frame meshes:
  see §10, Q1).
- Share mode shows **no** full-mode controls (no "add video", no model setup). One small link at
  the bottom, "Want live knobs? Set up full mode", opens a page with the steps and model links
  (§6.3).

### 6.2 Full mode (local server)

- Installed from the repo: `pip install -e '.[smplx,viewer]'` (a new `viewer` extra adds the web
  server, builder's choice: FastAPI + uvicorn). A console command (e.g. `kinetic-lab`) starts the
  server on localhost and opens the browser.
- The same frontend as share mode. It detects the server (e.g. a `/api/status` call) and turns on
  live knobs, "＋ add video", save-to-folder and load-from-folder. Opened from `file://` with no
  server, it behaves as share mode.
- **Models**: on start the server checks the model files (the list in
  `demo/round_trip/README.md` step 1: SMPL-X neutral, MANO left/right, FLAME + MediaPipe embedding,
  MediaPipe face landmarker). If any are missing it shows the **setup page** (§6.3) and a field to
  give the model folder path. The path is remembered locally (e.g. a small config file in the user's
  home or the repo's git-ignored area; builder's call).
- **Add video**: the user picks an `.mp4`; the server runs extraction (MediaPipe) and the SMPL-X
  fit once (about 1–2 minutes per clip on a laptop CPU), with a progress indicator, then caches the
  result so reopening the clip is instant. Reuse the export script's steps (`extract`,
  `run_with_mano`, the week-4 `ClipData` build) rather than reimplementing them; refactor shared
  pieces into `src/kinetic_augment` if needed. Clips already exported under
  `demo/round_trip/clips/` should be loadable without refitting where their caches allow.
- **Live knobs**: on marker release (and debounced while dragging), the server runs the chain for
  that one augmentation with the pane's values and returns the pane's frames. Target: under about
  1 s per update on a laptop CPU (today the 20 week-4 runs take about 10 s together after the fit).
  Only the changed pane recomputes.
- Server runs locally only (bind 127.0.0.1).

### 6.3 Setup page (both modes)

Plain steps, not spoon-feeding: clone the repo, `pip install -e '.[smplx,viewer]'`, get the model
files from their official sites (SMPL-X https://smpl-x.is.tue.mpg.de/ , MANO via the SMPL-X page and
`scripts/convert_mano.py`, FLAME https://flame.is.tue.mpg.de/ plus the MediaPipe embedding,
`python scripts/download_models.py --face` for the face landmarker), run `kinetic-lab`, enter the
model folder. Reuse wording from `demo/round_trip/README.md`.

## 7. Constraints

- Python ≥ 3.9; Black, 100-character lines, on new code only.
- Implementation under `src/kinetic_augment` (server, shared pipeline pieces) and tests under
  `tests`; scripts in `scripts`; every new folder gets an `index.md`, and existing `index.md` files
  are updated; `AGENTS.md` folder index list updated.
- Frontend: plain HTML/JS modules + three.js, no build step, as `demo/round_trip/` does. The
  frontend files live where the installed package can serve them (e.g.
  `src/kinetic_augment/lab/static/`; builder's call) and the share build copies them.
- The augmentations themselves are not changed except to add what extreme mode needs (a way to
  force draws to range edges, likely in `augmentations/ranges.py` and the per-entry draw code) and
  any refactor needed to call one entry with given values. Existing behaviour and seeds must be
  unchanged when extreme mode is off.
- No licensed weights in git or in the zip. `.superpowers/` added to `.gitignore`.

## 8. Left to the builder

- Exact data format of pane frames over the API and in the share bundle (reusing the
  `clip.json` + `data.bin` layout of the existing exporter is the default).
- Visual styling (dark theme as in `demo/round_trip/style.css` is a fine start).
- How the knob schema (kinds, limits, plain-English text, proven/used values) is produced:
  hand-written data file vs generated from the YAML comments plus a hand-written text file.
- Server framework details, port choice, caching layout.
- Tests: knob schema covers every chain entry and range key; saved YAML round-trips through
  `load_profile`; extreme mode puts draws at edges; extreme off reproduces current seeded output;
  share build opens from `file://` (a headless browser smoke check if practical).

## 9. Decisions from the builder check

- **A pane runs the selected entry alone** (every other entry off), as the existing `aug/<name>/`
  exports do, so the pane shows what that one augmentation does. Whole profiles (§5.6) are the
  place to see the chain. (Builder's call.)
- **Pane state**: each pane keeps its own knob values per augmentation, so switching from elbow
  noise to scale and back keeps both. Save writes every entry that pane has edited into one
  profile (each enabled). In share mode the "one knob off default at a time" rule applies per pane
  within the selected augmentation. (Builder's call.)
- **Seeds**: pane draws use the profile seed plus a reroll index k (k = 0 is the profile seed).
  The dice increments k in both modes, so share and full mode show the same draw for the same k.
  A saved YAML stores ranges and the base profile's seed, not the reroll index: the pipeline draws
  per clip anyway. The readout shows k. (Builder's call.)
- **Extreme mode** is available only for knobs with a natural edge (σ/truncation, range ends,
  probabilities as "fires"). Elsewhere (enums, mirror, curve shapes, drop patterns) the switch is
  disabled with a short "no extreme for this one" note. (Builder's call.)
- **Share clips**: 57949, 01986 and a third clip exported from the `clips.json` review order;
  the build command's `--clips` changes them. The zip includes the source frames: the user wants
  a few videos preloaded for the professor. (Third clip: builder's call.)
- **Meshes in the zip**: included, with view toggles (§10).

## 10. Answered by the user

Q1. Should the share zip include the fitted body mesh, or only points and skeleton?
**Show everything, with toggles to turn each view on and off** (user, 2026-11-10). The share zip
includes meshes for the original and every augmented stop, and the view toggles (§5.1) apply in
both modes. To stay within the size budget the meshes are compressed (quantised and/or decimated,
builder's call on how). Licensed model files themselves are still never included.
