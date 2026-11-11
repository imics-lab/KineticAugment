# KineticAugment: root

| Item | Description |
|---|---|
| `AGENTS.md` | Canonical instructions for coding agents |
| `CLAUDE.md` | Symlink to `AGENTS.md` |
| `agent.md` | Portable agent entry point pointing at `AGENTS.md` |
| `README.md` | Project overview, install, API, demo, known limitations |
| `CONTRIBUTING.md` | How to report bugs, propose features, and submit PRs (incl. running tests) |
| `CODE_OF_CONDUCT.md` | Community code of conduct |
| `LICENSE` | MIT licence |
| `pyproject.toml` | Packaging, optional extras (`smplx`, `physics`, `dev`, ...), pytest/black/mypy config |
| `.gitignore` | Git ignore rules |
| `.github/` | Issue templates (`ISSUE_TEMPLATE/`); deliberately has no `index.md` |
| `configs/` | YAML pipeline configs (current format; `legacy/` for the old format) |
| `data/` | Sample sign videos and generated test output |
| `demo/` | Round-trip and augmentation web viewer |
| `docs/` | Concepts, research logs, evidence notes; `legacy/` for outdated docs |
| `examples/` | Usage examples |
| `models/` | Model weights location (gitignored weights) and setup notes |
| `scripts/` | Extraction, evaluation, rendering and model-setup scripts; `legacy/` for deprecated ones |
| `src/` | The `kinetic_augment` Python package |
| `tests/` | pytest suite |
