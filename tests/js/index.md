# JS tests index

Node tests (`node --test tests/js/*.test.js`) for the Lab page's classic scripts; `tests/test_lab_js.py` runs them under pytest on freshly written fixtures (skips without `node`).

- `make_fixture.py` — Writes the fixtures with `bundle.encode_pane` and `PaneRequest.key()` (`python tests/js/make_fixture.py [out_dir]`).
- `bundle.test.js` — `Lab.decodePane` against `decode_pane` (dtypes, NaN points, wrapping int16 delta, unaligned bytes) and `Lab.gunzip`.
- `sources.test.js` — `Lab.ShareSource` (manifest keyed by Python's `PaneRequest.share_key`, found from UI requests with or without default values and edges; misses; stops, base, frames) and `Lab.LiveSource` (pane POST, 422 detail), `Lab.watchJob`.
- `timeline.test.js` — The clock maths: nearest frames, playback vs source-time lengths, "ended", wall-clock playback.
- `knobs.test.js` — `Lab.knobMath` (bands as fractions, snapping to stops, range markers, the share one-knob rule, the 150 ms scheduler) and `Lab.KnobPanel` rendered into a stub element (plain then technical text, locked knobs, live and share behaviour).
- `explain.test.js` — `Lab.explainHtml`, `Lab.readoutHtml` and `Lab.unchangedReason`.
- `yaml_edit.test.js` — `Lab.buildSavedYaml` on `configs/slr_week4_default.yaml` (comments kept, `# viewer: was`, `enabled: true`, dotted face keys, flow ranges), `Lab.savedEdits`, `Lab.parseYaml`, `Lab.valuesFromProfile`.
- `build_saved.js` — Helper for `tests/test_lab_yaml_edit.py` (not a test): writes the saved YAML of every lab entry.
- `panes.test.js` — The panes' per-frame buffers with the vendored three.js (axis flip, missing points, no-body frames draw points but no mesh, mirrored winding, ghost vs solid meshes).
- `fixtures/` — Generated fixtures, committed so `node --test` runs alone (see `fixtures/index.md`).
- `index.md` — Index of this folder.
