# Augmentation Lab (share build)

Static, offline build of the Augmentation Lab, unpacked from `dist/augmentation-lab.zip` so it can be
hosted as-is. Open `index.html` (or serve this folder). Built by `scripts/build_lab_share.py`; see
[`../augmentation-lab.md`](../augmentation-lab.md).

- `index.html` — Entry page of the Lab.
- `setup.html` — Steps for setting up full mode (live knobs).
- `README.txt` — Short how-to-open note shipped with the zip.
- `app.js` — Main app wiring.
- `bundle.js` — Share-mode data access (precomputed stops).
- `explain.js` — Plain-English explanations of augmentations and knobs.
- `knobs.js` — Knob bars with proven and used ranges.
- `panes.js` — The A/B 3D panes.
- `sources.js` — Clip sources and loading.
- `timeline.js` — Playback timeline.
- `yaml_edit.js` — Writes saved profiles as YAML.
- `style.css` — Page styles.
- `data/` — Baked clips and the manifest.
- `vendor/` — Third-party libraries (three.js).
- `index.md` — This index.
