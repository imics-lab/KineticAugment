# demo/round_trip/

Plain HTML/JS viewer (three.js from a CDN, no build step). Serve this folder
(`python3 -m http.server 8765 --directory demo/round_trip`) after exporting clips with
`scripts/export_round_trip_demo.py`.

| Item | Description |
|---|---|
| `README.md` | What the viewer shows, how to export clips and serve the pages, plus a step-by-step quick start for one clip |
| `index.html` | Round-trip page: sign video, MediaPipe 2-D extraction, 3-D extraction vs round trip, SMPL-X body with round-trip points, error strips and readout |
| `viewer.js` | Loads a clip's export, draws the 2-D overlays and the two three.js views (shared camera), the error / stream-agreement strips and the per-frame readout |
| `style.css` | Dark theme and stream colours shared by the pages |
| `augment.html` | Augmentations page: the Tier 1 and week-4 augmentations of a clip, one at a time, with what each does and its proven and used ranges |
| `augment.js` | Loads one `aug/<name>/` bundle and its original (`v/off` or `aug/w4.base`) and draws the three shared-camera views, the explanation box and the readout |
| `all.html` | All-augmentations page: every exported augmentation of a clip in a grid, playing together |
| `all.js` | Draws the grid with one WebGL renderer into each panel's rectangle, one shared camera and a playback or source-time clock |
| `week4.js` | Plain-words explanations, proven and used ranges, and draw formatting for the week-4 runs, shared by both augmentation pages |
| `clips/` | Generated, git-ignored: per clip, `frames/`, the raw MediaPipe caches (`hands.npz`, `face.npz`) and `v/<smoothing>/{clip.json, data.bin}` for each smoothing setting and `aug/<augmentation>/{clip.json, data.bin}` for each augmentation |
| `clips.json` | Generated, git-ignored: the seeded 50-clip review order, the smoothing settings, and each clip's summary per setting |
| `index.md` | This index |
