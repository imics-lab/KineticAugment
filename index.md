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
| `pyproject.toml` | Packaging, optional extras (`smplx`, `physics`, `lab`, `dev`, ...), `kinetic-lab` script, pytest/black/mypy config |
| `.gitignore` | Git ignore rules |
| `.github/` | Issue templates (`ISSUE_TEMPLATE/`); its `index.md` sits at `.github/` level only |
| `configs/` | YAML pipeline configs and week-4 profiles (current format; `legacy/` for the old format) |
| `data/` | Sample sign videos, hand-variant and augmentation trials, and generated test output |
| `demo/` | Round-trip and augmentation web viewer |
| `docs/` | Concepts, research logs, evidence notes, Augmentation Lab guide; `legacy/` for outdated docs |
| `examples/` | Usage examples |
| `models/` | Model weights location (gitignored weights) and setup notes |
| `scripts/` | Extraction, evaluation, rendering, lab share and model-setup scripts; `legacy/` for deprecated ones |
| `src/` | The `kinetic_augment` Python package |
| `tests/` | pytest suite and the Lab's JavaScript tests (`js/`) |
| `index.md` | This index |
