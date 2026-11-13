# Lab static data index

Share-mode data. The share build writes the real files here in its zip (`manifest.js` with `window.LAB_DATA`, one `Lab.addBlob` script per pane blob in `<clip>/`, frames; blobs load on first use); the repo only holds the placeholder.

- `load.js` — Placeholder loaded by `index.html` after `sources.js`; the share build replaces it with a script that writes the manifest's `<script>` tag.
- `index.md` — Index of this folder.
