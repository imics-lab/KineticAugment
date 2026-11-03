# Round-trip viewer index

Static page; serve this folder (`python3 -m http.server 8765 --directory demo/round_trip`) after
exporting clips with `scripts/export_round_trip_demo.py`.

- `index.html` — Page layout: sign video, MediaPipe 2-D extraction, 3-D extraction vs round trip, SMPL-X body with round-trip points, error strips and readout.
- `viewer.js` — Loads a clip's export, draws the 2-D overlays and the two three.js views (shared camera), the error / stream-agreement strips and the per-frame readout.
- `style.css` — Dark theme and stream colours shared by the panels.
- `clips/` — Generated, git-ignored: per clip, `frames/`, the raw MediaPipe caches (`hands.npz`, `face.npz`) and `v/<smoothing>/{clip.json, data.bin}` for each smoothing setting.
- `clips.json` — Generated, git-ignored: the seeded 50-clip review order, the smoothing settings, and each clip's summary per setting.
- `index.md` — Index of this folder.
