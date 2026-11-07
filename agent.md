# agent.md

Portable entry point for coding agents (Cursor, Copilot Workspace, Aider, ...).
**The authoritative instructions are in [`AGENTS.md`](./AGENTS.md)** (`CLAUDE.md` is a symlink to it). Read it first.

## Project

KineticAugment is a geometry-aware augmentation framework for human motion data (sign language first).
MediaPipe landmarks are fitted to one SMPL-X body (body, MANO hands, face), edited anatomically, and
read back as landmarks. The code is in `src/kinetic_augment`. `legacy/` folders (src, scripts, configs, docs)
are deprecated: do not build on them.

## Key commands (run from the repo root)

```bash
pip install -e '.[smplx,dev]'                # install
pytest tests/                                # tests (PYTHONPATH=src if not installed)
python scripts/convert_mano.py               # after placing licensed models in models/ (see models/README.md)
# demo: see demo/round_trip/README.md
```

## Conventions

Every folder has an `index.md` (keep it current). Scripts run from the repo root.
