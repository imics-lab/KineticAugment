# Lab index

- `__init__.py` — Package marker; defines `LAB_VERSION` and keeps imports light.
- `bundle.py` — Round-trip bundle helpers (`write_data`, `bundle_arrays`, `jsonable`, `source_frames`) and the pane blob `encode_pane` / `decode_pane`.
- `clipstore.py` — `ClipStore`: cached clip fits (`fit.pkl`), video frames, refit of earlier exports, LRU of prepared clips.
- `cli.py` — The `kinetic-lab` command: `--port`, local-only `--host`, `--clips-dir`, `--model-dir`, `--no-browser`; falls back to the repo's complete `models/` for the run (`startup_config`); starts uvicorn.
- `extreme.py` — Pins a sampled draw to its range edge (`EXTREME_ENTRIES`, `extreme_draw`, `make_hook`) for the lab's extreme switch.
- `ingest.py` — `ingest_video` (MediaPipe + SMPL-X fit + frames into a `ClipStore`) and `JobQueue`, a one-worker FIFO job queue with per-job status.
- `models.py` — `REQUIRED_MODELS`, `MODEL_LINKS`, `check_models`, `repo_model_dir`, and the persisted `LabConfig` (`~/.kinetic_lab/config.json`).
- `pane.py` — `PaneRequest` (with `key` and the share-manifest `share_key`), `schema_defaults`, `pane_profile`, `prepare_clip`, `run_pane`: one augmentation for one lab pane as a blob payload.
- `runs.py` — The week-4 run table (`Run`, `RUNS`) and `merge` / `run_profile`.
- `index.md` — Index of this folder.
- `knobs.yaml` — Hand-written plain-English text for every lab augmentation and knob (title, does/changes/keeps, proven and used ranges, per-knob plain sentence, technical name, limits, step, share).
- `saving.py` — `slugify`, `validate_profile_text`, `save_setting`: saved settings as YAML in `configs/viewer_saved/` with an index line each.
- `schema.py` — `load_knobs` reads `knobs.yaml`; `build_schema(profile)` merges it with a profile's values (and each run's overrides) into the schema the browser UI renders.
- `server.py` — `create_app`: the FastAPI routes (status, models, schema, clips, panes, uploads, jobs, saved settings) and static files; setup mode while models are missing.
- `static/` — The Lab web page (HTML, CSS, classic scripts, vendored three.js), served by the server and copied into the share zip (see `static/index.md`).
