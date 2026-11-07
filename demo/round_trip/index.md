# demo/round_trip/

Plain HTML/JS viewer (three.js from a CDN, no build step). Data is exported by `scripts/export_round_trip_demo.py`.

| Item | Description |
|---|---|
| `README.md` | What the viewer shows, plus step-by-step quick start for one clip |
| `index.html` | Round-trip viewer page (video, 2-D extraction, 3-D comparison, SMPL-X body, error strips) |
| `viewer.js` | Logic for `index.html` (loads exported clip data, draws panels and strips) |
| `augment.html` | Augmentation viewer page (original vs augmented bodies for each Tier 1 augmentation) |
| `augment.js` | Logic for `augment.html` |
| `style.css` | Shared styles for both pages |
| `clips/`, `clips.json` | Not in git: written by the export script (per-clip frames and data) |
