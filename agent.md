# agent.md

Portable entry point for coding agents (Cursor, Copilot Workspace, Aider, ...).
**The authoritative instructions are in [`AGENTS.md`](./AGENTS.md)** (`CLAUDE.md` is a symlink to it). Read it first.

## Project

KineticAugment is a geometry-aware augmentation framework for human motion data (sign language first).
MediaPipe landmarks are fitted to one SMPL-X body (body, MANO hands, face), edited anatomically, and
read back as landmarks. The code is in `src/kinetic_augment` (Python 3.9+). `legacy/` folders (src,
scripts, configs, docs) are deprecated: do not build on them.

The downstream sign classifier reads **raw MediaPipe landmarks as a time series**, in both MediaPipe's
normalised image coordinates and unnormalised coordinates. It does not centre on the shoulders or
divide by shoulder width, so augmentations are judged in that output format.

## Key commands (run from the repo root)

```bash
pip install -e '.[smplx,dev]'                # install
pytest tests/                                # tests (PYTHONPATH=src if not installed)
node --test tests/js/*.test.js               # the Lab page's JavaScript tests
python scripts/convert_mano.py               # after placing licensed models in models/ (see models/README.md)
# demo: see demo/round_trip/README.md
```

The Augmentation Lab is a local web page for comparing augmentations: install with
`pip install -e '.[smplx,lab]'`, put the model files in `models/` (or enter the folder on the setup
page), and run `kinetic-lab` (`python -m kinetic_augment.lab.cli`). See `docs/augmentation-lab.md`.

## Conventions

Every folder has an `index.md` (keep it current). Scripts run from the repo root. Black, 100-character
lines. Package code in `src/kinetic_augment`, tests in `tests`, profiles in `configs`.
