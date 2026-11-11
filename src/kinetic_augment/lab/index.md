# Lab index

- `__init__.py` — Package marker; defines `LAB_VERSION` and keeps imports light.
- `bundle.py` — Round-trip bundle helpers (`write_data`, `bundle_arrays`, `jsonable`, `source_frames`) and the pane blob `encode_pane` / `decode_pane`.
- `extreme.py` — Pins a sampled draw to its range edge (`EXTREME_ENTRIES`, `extreme_draw`, `make_hook`) for the lab's extreme switch.
- `runs.py` — The week-4 run table (`Run`, `RUNS`) and `merge` / `run_profile`.
- `index.md` — Index of this folder.
- `knobs.yaml` — Hand-written plain-English text for every lab augmentation and knob (title, does/changes/keeps, proven and used ranges, per-knob plain sentence, technical name, limits, step, share).
- `schema.py` — `load_knobs` reads `knobs.yaml`; `build_schema(profile)` merges it with a profile's values (and each run's overrides) into the schema the browser UI renders.
