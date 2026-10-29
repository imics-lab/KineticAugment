# KineticAugment Agent Guide

KineticAugment is a Python framework for geometry-aware augmentation of human skeletal motion data. Source code lives under `src/kinetic_augment`, YAML profiles live in `configs`, and the phase scripts in `scripts` provide the main project checks.

## Getting started

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the test suite with `pytest`. The documented phase checks can also be run with `CUDA_VISIBLE_DEVICES="" python scripts/test_phase1.py` through `test_phase5.py`; some phases require optional model assets or dependencies.

## Conventions

- Support Python 3.9 and newer.
- Keep formatting compatible with Black's 100-character line length.
- Add implementation code beneath `src/kinetic_augment` and tests beneath `tests`.
- Keep YAML task profiles in `configs` and do not commit licensed model weights.
- Update the nearest `index.md` whenever files or folders are added, removed, or renamed.
- Keep `CLAUDE.md` as a relative symlink to this file, and keep `agent.md` synchronized with significant changes here.

## Folder indexes

- [`index.md`](index.md)
- [`.github/index.md`](.github/index.md)
- [`.github/ISSUE_TEMPLATE/index.md`](.github/ISSUE_TEMPLATE/index.md)
- [`configs/index.md`](configs/index.md)
- [`data/index.md`](data/index.md)
- [`data/hand_variants/index.md`](data/hand_variants/index.md)
- [`data/hand_variants/round_trip_eval/index.md`](data/hand_variants/round_trip_eval/index.md)
- [`data/hand_variants/test_output/index.md`](data/hand_variants/test_output/index.md)
- [`data/sign_videos/index.md`](data/sign_videos/index.md)
- [`data/test_output/index.md`](data/test_output/index.md)
- [`data/test_output/videos/index.md`](data/test_output/videos/index.md)
- [`docs/index.md`](docs/index.md)
- [`docs/augmentation-evidence/index.md`](docs/augmentation-evidence/index.md)
- [`docs/media/index.md`](docs/media/index.md)
- [`docs/research-guide/index.md`](docs/research-guide/index.md)
- [`examples/index.md`](examples/index.md)
- [`models/index.md`](models/index.md)
- [`models/mano/index.md`](models/mano/index.md)
- [`models/smplx/index.md`](models/smplx/index.md)
- [`models/vposer/index.md`](models/vposer/index.md)
- [`presentation 2/index.md`](<presentation 2/index.md>)
- [`scripts/index.md`](scripts/index.md)
- [`scripts/review_checks/index.md`](scripts/review_checks/index.md) — frozen V1 review evidence; per-cluster subfolders each have their own `index.md`
- [`src/index.md`](src/index.md)
- [`src/kinetic_augment/index.md`](src/kinetic_augment/index.md)
- [`src/kinetic_augment/augmentations/index.md`](src/kinetic_augment/augmentations/index.md)
- [`src/kinetic_augment/body_model/index.md`](src/kinetic_augment/body_model/index.md)
- [`src/kinetic_augment/constraints/index.md`](src/kinetic_augment/constraints/index.md)
- [`src/kinetic_augment/constraints/collision/index.md`](src/kinetic_augment/constraints/collision/index.md)
- [`src/kinetic_augment/evaluation/index.md`](src/kinetic_augment/evaluation/index.md)
- [`src/kinetic_augment/evaluation/benchmark/index.md`](src/kinetic_augment/evaluation/benchmark/index.md)
- [`src/kinetic_augment/evaluation/datasets/index.md`](src/kinetic_augment/evaluation/datasets/index.md)
- [`src/kinetic_augment/evaluation/metrics/index.md`](src/kinetic_augment/evaluation/metrics/index.md)
- [`src/kinetic_augment/evaluation/models/index.md`](src/kinetic_augment/evaluation/models/index.md)
- [`src/kinetic_augment/pipeline/index.md`](src/kinetic_augment/pipeline/index.md)
- [`src/kinetic_augment/utils/index.md`](src/kinetic_augment/utils/index.md)
- [`tests/index.md`](tests/index.md)
