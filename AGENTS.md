# AGENTS.md

Canonical instructions for coding agents working in this repo. `CLAUDE.md` is a symlink to this file;
`agent.md` is a short portable entry point for other harnesses. Edit this file only.

## What this is

KineticAugment is a geometry-aware augmentation framework for human motion data (sign language
recognition first, also HAR). Landmarks are fitted to one SMPL-X body (body, MANO hands, head and
jaw), edited anatomically, and read back as MediaPipe-format landmarks. Package: `src/kinetic_augment`
(v0.4.0, Python >= 3.9, MIT).

## Three code areas: know which one you are in

1. **Current stack (V1 to V1.3), the active research line.**
   - `body_model/`: MediaPipe world landmarks <-> SMPL-X round trip (`converter`, `skeleton`, `shape`,
     `swing_ik`, `kinematics`; `hand_*` for MANO; `head_converter`, `face_*`, `assembly` for the face).
   - `augmentations/tier1.py`: Tier 1 augmentations (mirror, view, duration, arm joints, scale, uncentre).
     Not in the registry or `Pipeline`. Design: `docs/Augmentation_Grounding.md`.
   - Scripts: `scripts/` (not `scripts/legacy/`); viewer: `demo/round_trip/`; research log: `docs/V1_*`.
2. **Registry library API.** `pipeline/` (`Pipeline.from_preset` / `from_yaml`, `AugmentationChain`,
   datasets), `augmentations/{base,extrinsic,intrinsic,temporal}.py`, `constraints/`, `evaluation/`,
   `body_model/smplx_wrapper.py`. YAML configs for it: `configs/*.yaml`.
3. **Deprecated, do not build on it.** Everything under `src/kinetic_augment/legacy`,
   `scripts/legacy`, `configs/legacy`, `docs/legacy`: the old `AugmentationPipeline`, image-space
   `MediaPipeToSMPLX` / `SMPLXToMediaPipe` converters, canonicalization, phase smoke scripts. Kept for
   history and reachable via `kinetic_augment.AugmentationPipeline`. New work must not import from or
   extend `legacy/`.

## Setup

```bash
pip install -e '.[smplx,dev]'      # torch, smplx, trimesh, pytest, mediapipe
```

Model weights are licensed and live in `models/`, which is gitignored (only `models/.gitignore`,
`README.md`, the FLAME vertex map and placeholder files are tracked). Fetch SMPL-X and MANO manually,
then:

```bash
python scripts/download_models.py --check   # verify SMPL-X files
python scripts/convert_mano.py              # MANO .pkl -> chumpy-free .npz
python scripts/download_models.py --face    # MediaPipe face assets (face pipeline only)
```

Details and the file list: `models/README.md`. Landmark-mode augmentations need no model files.

## Tests

```bash
pytest tests/                       # from the repo root
PYTHONPATH=src pytest tests/        # if the package is not installed
```

No models or GPU needed; SMPLXWrapper tests skip without torch and model files. `scripts/legacy/phase*_smoke.py`
are print scripts, not pytest tests.

## Demo

Round-trip and augmentation viewer: follow `demo/round_trip/README.md` (export a clip with
`scripts/export_round_trip_demo.py`, then `python3 -m http.server --directory demo/round_trip`).
Needs all model files and an internet connection (three.js from a CDN).

## Known limitations

- Constraint enforcement is a no-op in `mode='landmark'` (TODO in `pipeline/pipeline.py`); it only acts in
  `mode='smplx'`. The constraint ablation in `evaluation/benchmark/experiments.py` therefore measures nothing.
- `BenchmarkRunner` quality metrics run on synthetic random-walk data, not real clips.
- `PipelineConfig.canonicalize` / `output_space` are accepted and ignored.
- `configs/legacy/slr_profile.yaml` has every probability at 0.0.
- `models/vposer/` is vendored under a proprietary MPG licence and unused; `data/test_output/` holds
  generated artifacts.
- The V1-V1.3 `body_model` stack has little pytest coverage; docs cite some scripts/tests that are not in the repo.

## Conventions

- Run scripts from the repo root (default paths are root-relative).
- Every folder has an `index.md` listing its files and subfolders. Update it when you add, remove or
  rename a file.
- `AGENTS.md` is canonical; keep the `CLAUDE.md` symlink intact and keep `agent.md` in sync after big changes.
- Keep it simple: reuse existing patterns, add no dependencies without approval, add tests for new code.
- Docs reference code by path; fix doc links when you move files.

## Folder indexes

| Folder | Index |
|---|---|
| root | `index.md` |
| `configs/` | `configs/index.md` |
| `configs/legacy/` | `configs/legacy/index.md` |
| `data/` | `data/index.md` |
| `data/sign_videos/` | `data/sign_videos/index.md` |
| `data/test_output/` | `data/test_output/index.md` |
| `demo/` | `demo/index.md` |
| `demo/round_trip/` | `demo/round_trip/index.md` |
| `docs/` | `docs/index.md` |
| `docs/augmentation-evidence/` | `docs/augmentation-evidence/index.md` |
| `docs/legacy/` | `docs/legacy/index.md` |
| `docs/media/` | `docs/media/index.md` |
| `examples/` | `examples/index.md` |
| `models/` | `models/index.md` |
| `models/mano/` | `models/mano/index.md` |
| `models/smplx/` | `models/smplx/index.md` |
| `models/vposer/` | `models/vposer/index.md` |
| `scripts/` | `scripts/index.md` |
| `scripts/legacy/` | `scripts/legacy/index.md` |
| `src/` | `src/index.md` |
| `src/kinetic_augment/` | `src/kinetic_augment/index.md` |
| `src/kinetic_augment/augmentations/` | `src/kinetic_augment/augmentations/index.md` |
| `src/kinetic_augment/body_model/` | `src/kinetic_augment/body_model/index.md` |
| `src/kinetic_augment/constraints/` | `src/kinetic_augment/constraints/index.md` |
| `src/kinetic_augment/constraints/collision/` | `src/kinetic_augment/constraints/collision/index.md` |
| `src/kinetic_augment/evaluation/` | `src/kinetic_augment/evaluation/index.md` |
| `src/kinetic_augment/evaluation/benchmark/` | `src/kinetic_augment/evaluation/benchmark/index.md` |
| `src/kinetic_augment/evaluation/datasets/` | `src/kinetic_augment/evaluation/datasets/index.md` |
| `src/kinetic_augment/evaluation/metrics/` | `src/kinetic_augment/evaluation/metrics/index.md` |
| `src/kinetic_augment/evaluation/models/` | `src/kinetic_augment/evaluation/models/index.md` |
| `src/kinetic_augment/legacy/` | `src/kinetic_augment/legacy/index.md` |
| `src/kinetic_augment/pipeline/` | `src/kinetic_augment/pipeline/index.md` |
| `src/kinetic_augment/utils/` | `src/kinetic_augment/utils/index.md` |
| `tests/` | `tests/index.md` |

`.github/` is intentionally without an `index.md` (an `.md` under `ISSUE_TEMPLATE/` would become an issue template).
