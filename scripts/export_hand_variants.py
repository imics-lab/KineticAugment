#!/usr/bin/env python3
"""
Save the hand round trip of every clip under two pipelines, for a classifier comparison.

Both pipelines fit one shape per hand track and one pose per frame; they differ in how
MediaPipe's drift of hand size is handled (docs/Hand_Pipeline_Variants.md):

    track_shape         one shape per track, fitted once on every frame of the track, and
                        nothing else: the hand keeps one size for the whole clip, so the
                        drift is left as error (the V1.2 pipeline as it ran)
    track_shape_scale   one shape per track, fitted with a per-frame scale about the wrist
                        held to a geometric mean of 1, then each frame's pose and scale; the
                        drift is kept in the scale (docs/V1_2_Findings.md section 10)

From track_shape_scale two outputs are saved: the hand with its scale replayed (closest to
MediaPipe) and the clean hand at scale 1 (one size per track, with the scale fit's shape).

One file per clip, <out-dir>/<clip>_hand_variants.npz, laid out like `HandClip.world`:
(T, 2, 21, 3) metres per side (0 = left, 1 = right), NaN where a side has no hand.

    world, image, present, width, height, fps, timestamp_ms     MediaPipe's observations
    <backend>__<output>__world        rebuilt landmarks; <output> is track_shape,
                                      track_shape_scale or track_shape_scale_clean
    <backend>__<variant>__<param>     wrist_orient (T,2,3), hand_pose (T,2,45),
                                      betas (T,2,K), transl (T,2,3), scale (T,2),
                                      rms_mm (T,2), status (T,2): "absent" where no hand

<out-dir>/manifest.json records the settings, input hashes and per-clip errors.

Usage (from the repo root):
    python scripts/export_hand_variants.py                       # the random 20 clips
    python scripts/export_hand_variants.py --data-dir data/test_output --backends mano smplx
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from measure_hand_round_trip import BACKENDS, MODEL_DIR, sha256

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    landmark_errors_mm,
    reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import NUM_HAND_LANDMARKS, HandModel
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip

DATA_DIR = Path("data/round_trip_eval")
OUT_ROOT = Path("data/hand_variants")
VARIANTS = ("track_shape", "track_shape_scale")
OUTPUTS = ("track_shape", "track_shape_scale", "track_shape_scale_clean")
PARAMS = ("wrist_orient", "hand_pose", "betas", "transl")


def fit_variant(variant, observed, valid, model):
    """One track's fits under `variant`."""
    scaled = variant == "track_shape_scale"
    betas = fit_hand_shape(observed, valid, model, per_frame_scale=scaled)
    return fit_hands(observed, valid, model, betas, per_frame_scale=scaled)


def rebuilt(fits, model, clean=False):
    if clean:
        fits = [dataclasses.replace(f, scale=1.0) if f.status == "ok" else f for f in fits]
    return np.stack([reconstruct_hand(f, model) for f in fits])


def export_clip(path, models, backends):
    """Arrays for one clip's file, and its per-output errors."""
    clip = load_hand_clip(path)
    count = len(clip.present)
    arrays = {"world": clip.world, "image": clip.image, "present": clip.present,
              "width": np.array(clip.width), "height": np.array(clip.height),
              "fps": np.array(clip.fps), "timestamp_ms": clip.timestamp_ms}
    errors = {}
    for backend in backends:
        k = models[(backend, SIDES[0])].num_betas
        for output in OUTPUTS:
            arrays[f"{backend}__{output}__world"] = np.full(
                (count, 2, NUM_HAND_LANDMARKS, 3), np.nan)
        for variant in VARIANTS:
            v = f"{backend}__{variant}__"
            arrays.update({v + "wrist_orient": np.full((count, 2, 3), np.nan),
                           v + "hand_pose": np.full((count, 2, 45), np.nan),
                           v + "betas": np.full((count, 2, k), np.nan),
                           v + "transl": np.full((count, 2, 3), np.nan),
                           v + "scale": np.full((count, 2), np.nan),
                           v + "rms_mm": np.full((count, 2), np.nan),
                           v + "status": np.full((count, 2), "absent", dtype="<U8")})
        for s, side in enumerate(SIDES):
            frames = np.flatnonzero(clip.present[:, s])
            if len(frames) < 2:
                continue
            model = models[(backend, side)]
            observed = clip.world[frames, s].astype(np.float64)
            valid = np.isfinite(observed).all(axis=-1)
            for variant in VARIANTS:
                fits = fit_variant(variant, observed, valid, model)
                v = f"{backend}__{variant}__"
                for name in PARAMS:
                    arrays[v + name][frames, s] = np.stack([getattr(f, name) for f in fits])
                arrays[v + "scale"][frames, s] = [f.scale for f in fits]
                arrays[v + "rms_mm"][frames, s] = [f.rms_mm for f in fits]
                arrays[v + "status"][frames, s] = [f.status for f in fits]
                outs = [variant] + (["track_shape_scale_clean"] if variant != "track_shape" else [])
                for output in outs:
                    points = rebuilt(fits, model, clean=output.endswith("_clean"))
                    arrays[f"{backend}__{output}__world"][frames, s] = points
                    errors.setdefault(f"{backend}: {output}", []).append(
                        landmark_errors_mm(points, observed))
    return arrays, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder of <clip>_hands.npz (HandClip) files")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: data/hand_variants/<data-dir name>")
    parser.add_argument("--backends", nargs="+", default=["mano"], choices=BACKENDS)
    args = parser.parse_args()
    out_dir = args.out_dir or OUT_ROOT / args.data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.data_dir.glob("*_hands.npz"))
    if not paths:
        raise SystemExit(f"No {args.data_dir}/*_hands.npz")
    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR) for b in args.backends for s in SIDES}
    manifest = {"data_dir": str(args.data_dir), "backends": args.backends,
                "outputs": list(OUTPUTS), "inputs": {}, "clips": {}, "pooled_mean_mm": {}}
    pooled = {}
    began = time.time()
    for path in paths:
        clip = path.stem.replace("_hands", "")
        arrays, errors = export_clip(path, models, args.backends)
        np.savez_compressed(out_dir / f"{clip}_hand_variants.npz", **arrays)
        manifest["inputs"][str(path)] = sha256(path)
        manifest["clips"][clip] = {row: float(np.nanmean(np.concatenate(e)))
                                   for row, e in errors.items()}
        for row, e in errors.items():
            pooled.setdefault(row, []).extend(e)
        means = manifest["clips"][clip]
        print(f"{clip}: " + "  ".join(f"{row} {m:.2f}" for row, m in means.items()))
    manifest["pooled_mean_mm"] = {row: float(np.nanmean(np.concatenate(e)))
                                  for row, e in pooled.items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nPooled mean, mm (wrist-relative, against MediaPipe):")
    for row, m in manifest["pooled_mean_mm"].items():
        print(f"  {row:<40}{m:>7.2f}")
    print(f"\nWrote {len(paths)} clips to {out_dir} in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
