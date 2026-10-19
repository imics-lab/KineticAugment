#!/usr/bin/env python3
"""
Write per-clip face features (V1.3): fitted neck, head and jaw with their flags, signed
hand-to-face distances with ASL-LEX regions and contact, image-space proximity to the projected
head, and the blendshapes.

Needs <clip>_hands.npz and <clip>_face.npz in --landmark-dir; writes <clip>_face_features.npz
next to them.

Usage:
    python scripts/extract_face_features.py [--landmark-dir data/test_output] [--clip 01991]
        [--depth-corrected]

--depth-corrected re-places each wrist from the image (task 14) and writes
<clip>_face_features_dc.npz instead; the default output is unchanged.
"""

import argparse
import time
from pathlib import Path

from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip, save_face_features
from kinetic_augment.body_model.hand_observations import load_hand_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-dir", type=Path, default=Path("data/test_output"))
    parser.add_argument(
        "--output-dir", type=Path, help="feature output directory (default: --landmark-dir)"
    )
    parser.add_argument("--clip", action="append", help="clip id; repeatable (default: all)")
    parser.add_argument(
        "--depth-corrected",
        action="store_true",
        help="image-derived wrists; writes *_face_features_dc.npz",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.landmark_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = args.clip or sorted(
        p.name[: -len("_face.npz")] for p in args.landmark_dir.glob("*_face.npz")
    )
    models = load_face_models()
    for clip in clips:
        start = time.time()
        hands = load_hand_clip(args.landmark_dir / f"{clip}_hands.npz")
        face = load_face_clip(args.landmark_dir / f"{clip}_face.npz")
        result = run_clip(hands, face, models, depth_corrected=args.depth_corrected)
        suffix = "_dc" if args.depth_corrected else ""
        path = output_dir / f"{clip}_face_features{suffix}.npz"
        save_face_features(path, result, face)
        print(f"{clip}: {len(result.frames)} frames -> {path} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
