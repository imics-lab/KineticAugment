#!/usr/bin/env python3
"""
SMPL-X Model Download Script.

This script helps set up the required SMPL-X model files for KineticAugment.

SMPL-X models must be downloaded manually from the official website due to
licensing requirements. This script provides instructions and validates
the installation.

Official download: https://smpl-x.is.tue.mpg.de/

Usage:
    python scripts/download_models.py --check
    python scripts/download_models.py --setup-dir models/smplx
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MEDIAPIPE_FACE_ASSETS = {
    # Apache-2.0, published by Google.
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/1/face_landmarker.task",
    "canonical_face_model.obj": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/"
                                "master/mediapipe/modules/face_geometry/data/"
                                "canonical_face_model.obj",
}
FLAME_FILES = ["generic_model.pkl", "mediapipe_landmark_embedding.npz"]
FLAME_INSTRUCTIONS = """
FLAME assets (licensed, manual):
  1. Register at https://flame.is.tue.mpg.de/ and download FLAME 2020;
     put generic_model.pkl in models/flame/.
  2. Download mediapipe_landmark_embedding.npz from
     https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe
     (MPI non-commercial licence) into models/flame/.
"""


def fetch_mediapipe_face_assets(models_dir: Path) -> None:
    """Download the Apache-2.0 MediaPipe face assets into models/mediapipe/."""
    target = models_dir / "mediapipe"
    target.mkdir(parents=True, exist_ok=True)
    for name, url in MEDIAPIPE_FACE_ASSETS.items():
        path = target / name
        if path.exists():
            print(f"  present: {path}")
            continue
        print(f"  fetching {name} ...")
        urllib.request.urlretrieve(url, path)
    missing = [f for f in FLAME_FILES if not (models_dir / "flame" / f).exists()]
    if missing:
        print(FLAME_INSTRUCTIONS)


# Required model files for SMPL-X
REQUIRED_FILES = {
    'smplx': [
        'SMPLX_NEUTRAL.npz',
        'SMPLX_MALE.npz',
        'SMPLX_FEMALE.npz',
    ],
    'mano': [
        'MANO_LEFT.pkl',
        'MANO_RIGHT.pkl',
    ],
}

# Optional files
OPTIONAL_FILES = {
    'vposer': [
        'vposer_v1_0/snapshots/TR00_E096.pt',
    ],
}

DOWNLOAD_INSTRUCTIONS = """
================================================================================
                        SMPL-X MODEL DOWNLOAD INSTRUCTIONS
================================================================================

KineticAugment requires SMPL-X model files for the body model integration.
These files must be downloaded manually due to licensing requirements.

STEP 1: Register and Download
------------------------------
1. Go to https://smpl-x.is.tue.mpg.de/
2. Create an account and accept the license agreement
3. Download the following packages:
   - SMPL-X (main model): ~1.5 GB
   - MANO (hand model): ~50 MB (optional, for detailed hands)

STEP 2: Extract Files
---------------------
Extract the downloaded files to the models directory:

    {model_dir}/
    ├── smplx/
    │   ├── SMPLX_NEUTRAL.npz
    │   ├── SMPLX_MALE.npz
    │   └── SMPLX_FEMALE.npz
    └── mano/  (optional)
        ├── MANO_LEFT.pkl
        └── MANO_RIGHT.pkl

STEP 3: Verify Installation
---------------------------
Run this script with --check to verify:

    python scripts/download_models.py --check --model-dir {model_dir}

================================================================================
"""

VPOSER_INSTRUCTIONS = """
================================================================================
                        VPOSER (Optional) DOWNLOAD INSTRUCTIONS
================================================================================

VPoser is a learned pose prior that helps keep augmented poses natural.
It's optional but recommended for high-quality augmentations.

1. Go to https://smpl-x.is.tue.mpg.de/
2. Download VPoser v1.0
3. Extract to: {model_dir}/vposer/

================================================================================
"""


def check_models(model_dir: Path, verbose: bool = True) -> dict:
    """
    Check if required model files exist.

    Args:
        model_dir: Path to models directory
        verbose: Print status messages

    Returns:
        Dictionary with check results
    """
    results = {
        'smplx': {'found': [], 'missing': []},
        'mano': {'found': [], 'missing': []},
        'vposer': {'found': [], 'missing': []},
    }

    # Check SMPL-X files
    smplx_dir = model_dir / 'smplx'
    for filename in REQUIRED_FILES['smplx']:
        filepath = smplx_dir / filename
        if filepath.exists():
            results['smplx']['found'].append(filename)
        else:
            results['smplx']['missing'].append(filename)

    # Check MANO files
    mano_dir = model_dir / 'mano'
    for filename in REQUIRED_FILES['mano']:
        filepath = mano_dir / filename
        if filepath.exists():
            results['mano']['found'].append(filename)
        else:
            results['mano']['missing'].append(filename)

    # Check VPoser files (optional)
    vposer_dir = model_dir / 'vposer'
    for filename in OPTIONAL_FILES['vposer']:
        filepath = vposer_dir / filename
        if filepath.exists():
            results['vposer']['found'].append(filename)
        else:
            results['vposer']['missing'].append(filename)

    if verbose:
        print_check_results(results)

    return results


def print_check_results(results: dict):
    """Print formatted check results."""
    print("\n" + "=" * 60)
    print("MODEL CHECK RESULTS")
    print("=" * 60)

    # SMPL-X (required)
    print("\nSMPL-X (Required):")
    if results['smplx']['found']:
        for f in results['smplx']['found']:
            print(f"  ✓ {f}")
    if results['smplx']['missing']:
        for f in results['smplx']['missing']:
            print(f"  ✗ {f} (MISSING)")

    # MANO (optional but recommended)
    print("\nMANO Hands (Recommended):")
    if results['mano']['found']:
        for f in results['mano']['found']:
            print(f"  ✓ {f}")
    if results['mano']['missing']:
        for f in results['mano']['missing']:
            print(f"  ○ {f} (not found)")

    # VPoser (optional)
    print("\nVPoser (Optional):")
    if results['vposer']['found']:
        for f in results['vposer']['found']:
            print(f"  ✓ {f}")
    if results['vposer']['missing']:
        for f in results['vposer']['missing']:
            print(f"  ○ {f} (not found)")

    # Summary
    print("\n" + "-" * 60)
    smplx_ok = len(results['smplx']['missing']) == 0

    if smplx_ok:
        print("✓ SMPL-X models are properly installed!")
        print("  KineticAugment body model features are ready to use.")
    else:
        print("✗ SMPL-X models are missing!")
        print("  Run with --help for download instructions.")

    print("=" * 60 + "\n")


def setup_directory(model_dir: Path):
    """Create the models directory structure."""
    print(f"Setting up models directory: {model_dir}")

    # Create directories
    (model_dir / 'smplx').mkdir(parents=True, exist_ok=True)
    (model_dir / 'mano').mkdir(parents=True, exist_ok=True)
    (model_dir / 'vposer').mkdir(parents=True, exist_ok=True)

    # Create .gitignore to prevent accidentally committing large files
    gitignore_path = model_dir / '.gitignore'
    gitignore_content = """# Model files (too large for git, must be downloaded separately)
*.npz
*.pkl
*.pt
*.pth
*.onnx
*.h5

# Keep the directory structure
!.gitignore
"""
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)

    # Create README with instructions
    readme_path = model_dir / 'README.md'
    readme_content = """# Model Files

This directory contains model files for KineticAugment.

## Required Files

Download SMPL-X models from: https://smpl-x.is.tue.mpg.de/

Place files as follows:
```
models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
├── mano/
│   ├── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
└── vposer/ (optional)
    └── vposer_v1_0/
```

## Verification

Run the download script to check installation:
```bash
python scripts/download_models.py --check
```
"""
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created directory structure at {model_dir}")
    print(f"✓ Created .gitignore to prevent committing model files")
    print(f"✓ Created README.md with instructions")


def main():
    parser = argparse.ArgumentParser(
        description="SMPL-X model download helper for KineticAugment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DOWNLOAD_INSTRUCTIONS.format(model_dir="models")
    )

    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('models'),
        help='Path to models directory (default: models/)'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if required models are installed'
    )

    parser.add_argument(
        '--setup-dir',
        action='store_true',
        help='Create the models directory structure'
    )

    parser.add_argument(
        '--vposer-info',
        action='store_true',
        help='Show VPoser download instructions'
    )

    parser.add_argument(
        '--face',
        action='store_true',
        help='Fetch MediaPipe face assets and check the FLAME files'
    )

    args = parser.parse_args()

    if args.face:
        fetch_mediapipe_face_assets(args.model_dir)
        return

    # Default action: show instructions
    if not any([args.check, args.setup_dir, args.vposer_info]):
        print(DOWNLOAD_INSTRUCTIONS.format(model_dir=args.model_dir))
        return

    if args.setup_dir:
        setup_directory(args.model_dir)

    if args.check:
        results = check_models(args.model_dir)

        # Return non-zero exit code if required models are missing
        if results['smplx']['missing']:
            sys.exit(1)

    if args.vposer_info:
        print(VPOSER_INSTRUCTIONS.format(model_dir=args.model_dir))


if __name__ == '__main__':
    main()
