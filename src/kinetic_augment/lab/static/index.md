# Lab static index

The Augmentation Lab page. Classic scripts only (no ES modules, no CDN, no build step), each adding to `window.Lab`, so the same files work from the local server and from `file://` in the share zip.

- `index.html` — The page: sidebar (clips, augmentations by group, whole profiles, saved settings), explanation box, source frame, panes A and B, timeline, view toggles, readout, knob panel container; script order.
- `setup.html` — The setup page (steps, model files and links); in full mode it lists the missing files from `/api/status` and posts the model folder to `/api/models`.
- `style.css` — Dark workbench theme and layout; stream colours `--orig`, `--pane-a`, `--pane-b`.
- `bundle.js` — `Lab.decodePane` (pane blob from `bundle.encode_pane`, delta mesh undone), `Lab.gunzip` (DecompressionStream), `Lab.at`.
- `sources.js` — `Lab.LiveSource` (local server API) and `Lab.ShareSource` (`window.LAB_DATA`; blob scripts added on first use by `Lab.loadScript`, `Lab.addBlob`), `Lab.requestKey`, `Lab.chooseSource`, `Lab.watchJob` (polls an add-video job).
- `timeline.js` — `Lab.Timeline` (play/pause, scrub, speed, playback / source-time clock) and the clock maths `Lab.frameMap`, `Lab.nearestFrames`, `Lab.clockLength`.
- `panes.js` — `Lab.Panes`: two three.js viewports with one shared camera and orbit, original vs augmented overlay, view toggles, camera presets.
- `app.js` — `Lab.app`: start-up, sidebar, clip and entry selection, pane requests (newest wins), toggles; the state/event API for the knob panel; wires the knob panel, the Save / Load forms and the full-mode "＋ add video" flow.
- `knobs.js` — `Lab.KnobPanel`: one bar per knob with A/B markers and proven / used / beyond bands, enum dropdowns and on/off boxes per pane, random / extreme, low / high, new draw, copy A → B; share-mode stops and one knob at a time; `Lab.knobMath` (the pure maths).
- `explain.js` — `Lab.explainHtml` (does / changes / keeps / proven vs used / note), `Lab.readoutHtml` (draws, measures in words with units, fallbacks), `Lab.unchangedReason`.
- `yaml_edit.js` — Saved settings without a YAML library: `Lab.buildSavedYaml` (line edit of the base profile, `# viewer: was`), `Lab.savedEdits`, `Lab.parseYaml`, `Lab.valuesFromProfile`.
- `data/` — Share-mode data scripts; only a placeholder `load.js` in the repo (see `data/index.md`).
- `vendor/` — three.js r147 and its OrbitControls, vendored (see `vendor/index.md`).
- `index.md` — Index of this folder.
