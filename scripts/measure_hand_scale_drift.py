#!/usr/bin/env python3
"""
V1.2 follow-up: how much of the hand round-trip error is MediaPipe's drift of hand size?

The hand round trip (scripts/measure_hand_round_trip.py) has no scale-only row for the
models: a per-frame shape closes most of the gap to the targets, but it has 10 values per
frame and absorbs more than size. This script adds one value per frame -- a scale about
the wrist, kept beside the fit as a residual (`per_frame_scale` in hand_converter) -- and
scores the same metric on the same pilot clips:

    in-sample shape                       the existing row, recomputed as a check
    neutral shape + per-frame scale       1 value per frame on top of the neutral fit
    in-sample shape + per-frame scale     the shape fitted with scales held to a geometric
                                          mean of 1, then each frame's scale fitted freely
    held-out shape + per-frame scale      the shape fitted on one part of the track and
                                          scored on the rest, both ways, for the time
                                          halves and the interleaved frames

Then, from the in-sample fit with scale, per track: how far the shape sits from the
plausibility bound, how the scale moves (spread, drift between halves, frame-to-frame
jitter), and whether it follows how squarely the palm faces the camera or how large the
hand appears in the image. The per-frame-shape diagnostic is read from
data/test_output/hand_round_trip.json, not refitted.

Usage:
    python scripts/measure_hand_scale_drift.py                  # the five pilot clips
    python scripts/measure_hand_scale_drift.py --data-dir data/round_trip_eval \
        --earlier data/round_trip_eval/round_trip_report.json   # the random 20 (ledger P22)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from measure_hand_round_trip import (
    BACKENDS,
    MIDDLE_MCP,
    MODEL_DIR,
    TARGETS_MM,
    environment,
    interleaved,
    sha256,
    print_table,
    rebuild,
    summary,
    time_halves,
)
from scipy.stats import spearmanr

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    landmark_errors_mm,
    plausible_radius,
)
from kinetic_augment.body_model.hand_model import LANDMARK_NAMES, WRIST, HandModel
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip

DATA_DIR = Path("data/test_output")
EARLIER = DATA_DIR / "hand_round_trip.json"
INDEX_MCP, PINKY_MCP = 5, 17
SPLITS = {"time halves": time_halves, "interleaved frames": interleaved}


def tracks(data_dir):
    """Yield (clip, side, world (n, 21, 3), image_px (n, 21, 2)) per clip-side, in time order."""
    for path in sorted(Path(data_dir).glob("*_hands.npz")):
        clip = load_hand_clip(path)
        for s, side in enumerate(SIDES):
            frames = np.flatnonzero(clip.present[:, s])
            if len(frames) >= 2:
                image = clip.image[frames, s, :, :2] * np.array([clip.width, clip.height])
                yield (path.stem.replace("_hands", ""), side,
                       clip.world[frames, s].astype(np.float64), image.astype(np.float64))


def errors_of(fits, model, landmarks):
    return landmark_errors_mm(rebuild(fits, model), landmarks)


def held_out_with_scale(model, landmarks, valid, parts):
    """Every frame scored once by a shape fitted (with scales) on the other part."""
    errors = np.full((len(landmarks), len(LANDMARK_NAMES) - 1), np.nan)
    norms = []
    for fit_on, score_on in (parts, parts[::-1]):
        betas = fit_hand_shape(landmarks[fit_on], valid[fit_on], model, per_frame_scale=True)
        norms.append(float(np.linalg.norm(betas)))
        fits = fit_hands(landmarks[score_on], valid[score_on], model, betas, per_frame_scale=True)
        errors[score_on] = errors_of(fits, model, landmarks[score_on])
    return errors, norms


def scale_behaviour(scales, world, image):
    """How one track's fitted scales move, and what they follow."""
    log_s = np.log(scales)
    first, second = time_halves(len(scales))
    relative = world - world[:, :1]
    normal = np.cross(relative[:, INDEX_MCP], relative[:, PINKY_MCP])
    facing = np.abs(normal[:, 2]) / np.linalg.norm(normal, axis=1)  # z is the camera axis
    image_span = np.linalg.norm(image[:, MIDDLE_MCP] - image[:, WRIST], axis=1)
    world_span = np.linalg.norm(relative[:, MIDDLE_MCP], axis=1)
    step = np.diff(log_s)
    return {
        "p5": float(np.percentile(scales, 5)),
        "median": float(np.median(scales)),
        "p95": float(np.percentile(scales, 95)),
        "log_std": float(np.std(log_s)),
        "half_ratio": float(np.median(scales[second]) / np.median(scales[first])),
        "step_log_std": float(np.std(step)) if len(step) > 1 else float("nan"),
        "lag1_autocorrelation": float(np.corrcoef(log_s[:-1], log_s[1:])[0, 1])
        if len(log_s) > 2 else float("nan"),
        "spearman_vs_observed_span": float(spearmanr(log_s, world_span)[0]),
        "spearman_vs_palm_facing_camera": float(spearmanr(log_s, facing)[0]),
        "spearman_vs_image_span_px": float(spearmanr(log_s, image_span)[0]),
    }


def ladder(model, landmarks):
    valid = np.ones(landmarks.shape[:2], dtype=bool)
    rows = {}
    plain = fit_hand_shape(landmarks, valid, model)
    rows["in-sample shape"] = errors_of(fit_hands(landmarks, valid, model, plain), model, landmarks)
    rows["neutral shape + per-frame scale"] = errors_of(
        fit_hands(landmarks, valid, model, per_frame_scale=True), model, landmarks)
    betas = fit_hand_shape(landmarks, valid, model, per_frame_scale=True)
    fits = fit_hands(landmarks, valid, model, betas, per_frame_scale=True)
    rows["in-sample shape + per-frame scale"] = errors_of(fits, model, landmarks)
    norms = {}
    for name, split in SPLITS.items():
        rows[f"held-out shape + per-frame scale ({name})"], norms[name] = held_out_with_scale(
            model, landmarks, valid, split(len(landmarks)))
    scales = np.array([f.scale for f in fits])
    radius = plausible_radius(model.num_betas)
    shape = {
        "radius": radius,
        "in_sample_norm_without_scale": float(np.linalg.norm(plain)),
        "in_sample_norm_with_scale": float(np.linalg.norm(betas)),
        "held_out_norms_with_scale": norms,
        "failed_fits": int(sum(f.status != "ok" for f in fits)),
    }
    return rows, scales, shape


def earlier_rows(path):
    """Pooled rows of an earlier report: the pilot's round-trip JSON or evaluate_round_trip's."""
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    if "hands" in report.get("pooled", {}):  # evaluate_round_trip.py: pooled.hands.<backend>
        return report["pooled"]["hands"]
    return report["pooled"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--earlier", type=Path, default=EARLIER,
                        help="report whose pooled rows are printed for comparison")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <data-dir>/hand_scale_drift.json")
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    args = parser.parse_args()
    output = args.output or args.data_dir / "hand_scale_drift.json"
    backends = tuple(args.backends)

    began = time.time()
    found = list(tracks(args.data_dir))
    if not found:
        raise SystemExit(f"No {args.data_dir}/*_hands.npz")
    earlier = earlier_rows(args.earlier)

    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR) for b in backends for s in SIDES}
    pooled_errors, pooled_scale = {}, {}
    env = environment()
    env["data"] = {str(p): sha256(p) for p in sorted(args.data_dir.glob("*_hands.npz"))}
    report = {"environment": env, "data_dir": str(args.data_dir),
              "targets_mm": TARGETS_MM, "tracks": [],
              "pooled": {}, "earlier_rows": {}}

    for clip, side, landmarks, image in found:
        span = np.linalg.norm(landmarks[:, MIDDLE_MCP] - landmarks[:, WRIST], axis=-1)
        scale_mm = np.full(len(landmarks), 1000.0 * np.median(span))
        entry = {"clip": clip, "side": side, "frames_with_hand": len(landmarks),
                 "hand_scale_mm": float(scale_mm[0]), "rows": {}}
        for backend in backends:
            rows, scales, shape = ladder(models[(backend, side)], landmarks)
            entry[backend] = {"shape": shape,
                              "scale": scale_behaviour(scales, landmarks, image),
                              "scales": scales.tolist()}
            for row, value in rows.items():
                name = f"{backend}: {row}"
                entry["rows"][name] = summary(value, scale_mm)
                pooled_errors.setdefault(name, []).append(value)
                pooled_scale.setdefault(name, []).append(scale_mm)
        report["tracks"].append(entry)
        print_table(f"{clip} {side}  ({len(landmarks)} frames, hand {scale_mm[0]:.0f} mm)",
                    entry["rows"])

    for row in pooled_errors:
        report["pooled"][row] = summary(
            np.concatenate(pooled_errors[row]), np.concatenate(pooled_scale[row]))
        report["pooled"][row]["track_mean_mm"] = float(np.mean(
            [t["rows"][row]["mean_mm"] for t in report["tracks"]]))
    for backend in backends:
        for row in ("refined", "held-out shape (time halves)", "in-sample shape",
                    "per-frame shape (diagnostic)"):
            if f"{backend}: {row}" in earlier:
                report["earlier_rows"][f"{backend}: {row}"] = earlier[f"{backend}: {row}"]
    print_table("POOLED (mm; % of wrist-to-middle-MCP length)", report["pooled"])
    print_table(f"For comparison, from {args.earlier}", report["earlier_rows"])

    print("\nScale per track (in-sample shape + per-frame scale)")
    print(f"  {'track':<14}{'backend':<8}{'p5':>6}{'p95':>6}{'halves':>8}{'lag1':>7}"
          f"{'|b|/r':>7}{'was':>7}{'facing':>8}{'img px':>8}")
    for t in report["tracks"]:
        for backend in backends:
            sc, sh = t[backend]["scale"], t[backend]["shape"]
            print(f"  {t['clip'] + ' ' + t['side']:<14}{backend:<8}{sc['p5']:>6.2f}"
                  f"{sc['p95']:>6.2f}{sc['half_ratio']:>8.2f}{sc['lag1_autocorrelation']:>7.2f}"
                  f"{sh['in_sample_norm_with_scale'] / sh['radius']:>7.3f}"
                  f"{sh['in_sample_norm_without_scale'] / sh['radius']:>7.3f}"
                  f"{sc['spearman_vs_palm_facing_camera']:>8.2f}"
                  f"{sc['spearman_vs_image_span_px']:>8.2f}")

    output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output} in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
