# AGENTS.md

Canonical instructions for coding agents working in this repo. `CLAUDE.md` is a symlink to this file;
`agent.md` is a short portable entry point for other harnesses. Edit this file only.

## What this is

KineticAugment is a geometry-aware augmentation framework for human motion data (sign language
recognition first, also HAR). Landmarks are fitted to one SMPL-X body (body, MANO hands, head and
jaw), edited anatomically, and read back as MediaPipe-format landmarks. Package: `src/kinetic_augment`
(v0.4.0, Python >= 3.9, MIT).

The downstream sign classifier reads **raw MediaPipe landmarks as a time series**, in both MediaPipe's
normalised image coordinates and unnormalised coordinates. It does not centre on the shoulders or
divide by shoulder width, so augmentations are judged in that output format. See
[README § Classifier input](README.md#classifier-input).

## Three code areas: know which one you are in

1. **Current stack (V1 to V1.3, week 4), the active research line.**
   - `body_model/`: MediaPipe world landmarks <-> SMPL-X round trip (`converter`, `skeleton`, `shape`,
     `swing_ik`, `kinematics`; `hand_*` for MANO; `head_converter`, `face_*`, `assembly` for the face).
   - `augmentations/tier1.py`: Tier 1 augmentations (mirror, view, duration, arm joints, scale, uncentre).
     Not in the registry or `Pipeline`. Design: `docs/Augmentation_Grounding.md`.
   - Week-4 augmentations: `augmentations/{clip,chain,profile,outputs,body,fingers,face,scene,timing,...}.py`
     driven by the `configs/slr_week4_*.yaml` profiles (`load_profile` + `AugmentationChain`). Ranges:
     `docs/Augmentation_Ranges.md`.
   - Augmentation Lab: `lab/` (`kinetic-lab` server and web page, `scripts/build_lab_share.py` for the
     offline zip). See `docs/augmentation-lab.md`.
   - Scripts: `scripts/` (not `scripts/legacy/`); viewer: `demo/round_trip/`; research log: `docs/V1_*`.
2. **Registry library API.** `pipeline/` (`Pipeline.from_preset` / `from_yaml`, `AugmentationChain`,
   datasets), `augmentations/{base,extrinsic,intrinsic,temporal}.py`, `constraints/`, `evaluation/`,
   `body_model/smplx_wrapper.py`. YAML configs for it: `configs/slr_profile.yaml`.
3. **Deprecated, do not build on it.** Everything under `src/kinetic_augment/legacy`,
   `scripts/legacy`, `configs/legacy`, `docs/legacy`: the old `AugmentationPipeline`, image-space
   `MediaPipeToSMPLX` / `SMPLXToMediaPipe` converters, canonicalization, phase smoke scripts. Kept for
   history and reachable via `kinetic_augment.AugmentationPipeline`. New work must not import from or
   extend `legacy/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[smplx,dev]'      # torch, smplx, trimesh, pytest, mediapipe
pip install -e '.[smplx,lab]'      # add the Augmentation Lab server
```

Model weights are licensed and live in `models/`, which is gitignored (only `models/.gitignore`,
`README.md`, the FLAME vertex map, the `index.md` files and placeholder files are tracked). Fetch
SMPL-X and MANO manually, then:

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
node --test tests/js/*.test.js      # the Lab page's JavaScript tests
```

SMPLXWrapper tests skip without torch and model files; several V1 to week-4 tests skip without the
licensed models or cached clips. `scripts/legacy/phase*_smoke.py` are print scripts, not pytest tests;
run them with `CUDA_VISIBLE_DEVICES="" python scripts/legacy/phase1_smoke.py` (some need optional
model assets or dependencies).

## Demo

Round-trip and augmentation viewer: follow `demo/round_trip/README.md` (export a clip with
`scripts/export_round_trip_demo.py`, then `python3 -m http.server --directory demo/round_trip`).
Needs all model files and an internet connection (three.js from a CDN).

Augmentation Lab: `kinetic-lab` starts a local web page for comparing augmentations (needs
`pip install -e '.[smplx,lab]'` and the model files). See `docs/augmentation-lab.md`.

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

- Support Python 3.9 and newer. Keep formatting compatible with Black's 100-character line length.
- Add implementation code beneath `src/kinetic_augment` and tests beneath `tests`.
- Keep YAML task profiles in `configs` and do not commit licensed model weights.
- Run scripts from the repo root (default paths are root-relative).
- Every folder has an `index.md` listing its files and subfolders (exception: `.github/ISSUE_TEMPLATE/`,
  where an `.md` would become an issue template; `.github/index.md` lists it). Update the nearest
  `index.md` when you add, remove or rename a file.
- `AGENTS.md` is canonical; keep `CLAUDE.md` as a relative symlink to it and keep `agent.md` in sync
  after big changes.
- Keep it simple: reuse existing patterns, add no dependencies without approval, add tests for new code.
- Docs reference code by path; fix doc links when you move files.

## Folder indexes

- [`index.md`](index.md)
- [`.github/index.md`](.github/index.md)
- [`configs/index.md`](configs/index.md)
- [`configs/legacy/index.md`](configs/legacy/index.md)
- [`configs/viewer_saved/index.md`](configs/viewer_saved/index.md)
- [`data/index.md`](data/index.md)
- [`data/augment_trials/index.md`](data/augment_trials/index.md)
- [`data/hand_variants/index.md`](data/hand_variants/index.md)
- [`data/hand_variants/round_trip_eval/index.md`](data/hand_variants/round_trip_eval/index.md)
- [`data/hand_variants/test_output/index.md`](data/hand_variants/test_output/index.md)
- [`data/round_trip_eval/index.md`](data/round_trip_eval/index.md)
- [`data/round_trip_eval/videos/index.md`](data/round_trip_eval/videos/index.md)
- [`data/round_trip_eval/week2_experiments/index.md`](data/round_trip_eval/week2_experiments/index.md)
- [`data/sign_videos/index.md`](data/sign_videos/index.md)
- [`data/test_output/index.md`](data/test_output/index.md)
- [`data/test_output/videos/index.md`](data/test_output/videos/index.md)
- [`data/wlasl/index.md`](data/wlasl/index.md)
- [`demo/index.md`](demo/index.md)
- [`demo/round_trip/index.md`](demo/round_trip/index.md)
- [`docs/index.md`](docs/index.md)
- [`docs/augmentation-evidence/index.md`](docs/augmentation-evidence/index.md)
- [`docs/augmentation-evidence/reports/index.md`](docs/augmentation-evidence/reports/index.md)
- [`docs/augmentation-evidence/research_notes/index.md`](docs/augmentation-evidence/research_notes/index.md) — each report's notes subfolder has its own `index.md`
- [`docs/legacy/index.md`](docs/legacy/index.md)
- [`docs/media/index.md`](docs/media/index.md)
- [`docs/media/v1_2_hands/index.md`](docs/media/v1_2_hands/index.md)
- [`docs/research-guide/index.md`](docs/research-guide/index.md)
- [`docs/superpowers/index.md`](docs/superpowers/index.md)
- [`docs/superpowers/plans/index.md`](docs/superpowers/plans/index.md)
- [`docs/superpowers/specs/index.md`](docs/superpowers/specs/index.md)
- [`examples/index.md`](examples/index.md)
- [`models/index.md`](models/index.md)
- [`models/flame/index.md`](models/flame/index.md)
- [`models/mano/index.md`](models/mano/index.md)
- [`models/mediapipe/index.md`](models/mediapipe/index.md)
- [`models/smplx/index.md`](models/smplx/index.md)
- [`models/vposer/index.md`](models/vposer/index.md)
- [`presentation 2/index.md`](<presentation 2/index.md>)
- [`scripts/index.md`](scripts/index.md)
- [`scripts/legacy/index.md`](scripts/legacy/index.md)
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
- [`src/kinetic_augment/lab/index.md`](src/kinetic_augment/lab/index.md)
- [`src/kinetic_augment/lab/static/index.md`](src/kinetic_augment/lab/static/index.md)
- [`src/kinetic_augment/lab/static/data/index.md`](src/kinetic_augment/lab/static/data/index.md)
- [`src/kinetic_augment/lab/static/vendor/index.md`](src/kinetic_augment/lab/static/vendor/index.md)
- [`src/kinetic_augment/legacy/index.md`](src/kinetic_augment/legacy/index.md)
- [`src/kinetic_augment/pipeline/index.md`](src/kinetic_augment/pipeline/index.md)
- [`src/kinetic_augment/utils/index.md`](src/kinetic_augment/utils/index.md)
- [`tests/index.md`](tests/index.md)
- [`tests/js/index.md`](tests/js/index.md)
- [`tests/js/fixtures/index.md`](tests/js/fixtures/index.md)
