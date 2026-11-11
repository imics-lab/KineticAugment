# JS test fixtures index

Written by `tests/js/make_fixture.py`; regenerate rather than edit.

- `pane.bin` — A small pane blob from `bundle.encode_pane` (NaN points, a wrapping int16 delta mesh, uint16 faces above 32767).
- `pane.bin.gz` — The same blob gzipped (mtime 0).
- `pane.expected.json` — `bundle.decode_pane` of the blob: meta and every array's dtype, shape and values (NaN as null).
- `share.json` — A share manifest keyed by `PaneRequest.share_key()` with its schema, the UI requests each key must answer, requests that must miss, raw `key()`s, stops and rerolls.
- `index.md` — Index of this folder.
