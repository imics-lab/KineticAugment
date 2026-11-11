# Lab static data index

Share-mode data. The share build writes the real files here in its zip (`manifest.js` with `window.LAB_DATA`, one `Lab.addBlob` script per pane blob, frames); the repo only holds the placeholder.

- `load.js` — Placeholder loaded by `index.html` after `sources.js`; the share build replaces it with a script that writes a `<script>` tag per data file.
- `index.md` — Index of this folder.
