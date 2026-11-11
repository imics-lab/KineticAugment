# Lab static index

The Augmentation Lab page. Classic scripts only (no ES modules, no CDN, no build step), each adding to `window.Lab`, so the same files work from the local server and from `file://` in the share zip.

- `index.html` — The page: sidebar (clips, augmentations by group, whole profiles, saved settings), explanation box, source frame, panes A and B, timeline, view toggles, readout, knob panel container; script order.
- `style.css` — Dark workbench theme and layout; stream colours `--orig`, `--pane-a`, `--pane-b`.
- `bundle.js` — `Lab.decodePane` (pane blob from `bundle.encode_pane`, delta mesh undone), `Lab.gunzip` (DecompressionStream), `Lab.at`.
- `sources.js` — `Lab.LiveSource` (local server API) and `Lab.ShareSource` (`window.LAB_DATA` + `Lab.addBlob`), `Lab.requestKey`, `Lab.chooseSource`.
- `timeline.js` — `Lab.Timeline` (play/pause, scrub, speed, playback / source-time clock) and the clock maths `Lab.frameMap`, `Lab.nearestFrames`, `Lab.clockLength`.
- `panes.js` — `Lab.Panes`: two three.js viewports with one shared camera and orbit, original vs augmented overlay, view toggles, camera presets.
- `app.js` — `Lab.app`: start-up, sidebar, clip and entry selection, pane requests (newest wins), toggles; the state/event API for the knob panel.
- `data/` — Share-mode data scripts; only a placeholder `load.js` in the repo (see `data/index.md`).
- `vendor/` — three.js r147 and its OrbitControls, vendored (see `vendor/index.md`).
- `index.md` — Index of this folder.
