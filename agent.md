# KineticAugment Agent Entry Point

The authoritative project instructions are in [`AGENTS.md`](AGENTS.md).

KineticAugment is a Python package for anatomically and temporally plausible augmentation of skeletal motion data. Start by creating a Python 3.9+ environment and running `pip install -e '.[dev]'`.

The downstream sign classifier reads **raw MediaPipe landmarks as a time series**, in both MediaPipe's normalised image coordinates and unnormalised coordinates. It does not centre on the shoulders or divide by shoulder width, so augmentations are judged in that output format.

Use `pytest` for the standard test suite. The project also provides `scripts/test_phase1.py` through `scripts/test_phase5.py`; run them with `CUDA_VISIBLE_DEVICES="" python <script>`. Keep package code in `src/kinetic_augment`, tests in `tests`, configuration profiles in `configs`, and update the nearest `index.md` after structural changes.

The Augmentation Lab is a local web page for comparing augmentations: install with `pip install -e '.[smplx,lab]'`, put the model files in `models/` (or enter the folder on the setup page), and run `kinetic-lab` (`python -m kinetic_augment.lab.cli`). See `docs/augmentation-lab.md`.
