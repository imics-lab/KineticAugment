#!/usr/bin/env python3
"""
Run the current body + hand round trip on any footage and score it.

One script for new, unseen clips. Per video it:

  1. Extracts MediaPipe landmarks in one pass -- a fresh Hands and Holistic instance
     per clip (`extract_hand_landmarks.extract_hands`). Holistic's pose world
     landmarks are the same ones `extract_world_landmarks.py` caches, so the body and
     hands come from one decode.
  2. Body: MediaPipe pose -> SMPL-X -> MediaPipe, scored as `measure_round_trip.py`
     scores it, in % of the clip's median shoulder width over the six arm landmarks:
        direct    fitted shape (`fit_shape`), the real number
        template  SMPL-X's template body (betas = 0)
        floor     rigid skeleton with the clip's median arm geometry: the V1 gate
        held out  shape fitted on one half of the clip in time, scored on the other
     plus the upright-body check (with the nose, and with it masked).
  3. Hands: MediaPipe hand -> MANO (and/or SMPL-X hands) -> MediaPipe, scored as
     `measure_hand_round_trip.py` scores it: mm between predicted and observed
     wrist-relative landmarks 1-20, no registration. Rows: in-sample shape (the
     pipeline as it runs), held-out shape (time halves; interleaved frames), and the
     model-free rigid reference.

Landmarks are cached in the output directory and re-used unless --reextract is given.
Five pilot clips set the provisional hand targets; numbers on new footage are the
first real test of whether they hold.

Usage (from the repo root):
    .venv/bin/python scripts/evaluate_round_trip.py --videos path/to/new_clips/
    .venv/bin/python scripts/evaluate_round_trip.py --videos a.mp4 b.mp4 --render
    .venv/bin/python scripts/evaluate_round_trip.py --videos dir/ --hand-backends mano smplx

Output (default data/round_trip_eval/):
    <clip>_hands.npz, <clip>_world.npz   landmark caches
    round_trip_report.json               every number printed, per clip and pooled
    videos/<clip>_body_hand_round_trip.mp4   with --render
"""

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np  # noqa: E402

from extract_hand_landmarks import (  # noqa: E402
    HANDS_SETTINGS, HOLISTIC_SETTINGS, extract_hands,
)
from measure_hand_round_trip import (  # noqa: E402
    MIDDLE_MCP, TARGETS_MM, held_out, interleaved, print_table, rebuild, rigid_reference,
    summary, time_halves,
)
from measure_round_trip import (  # noqa: E402
    L_SHOULDER, NAMES, NOSE, R_SHOULDER, VISIBILITY_THRESHOLD, errors, floor_skeleton,
    is_upright, pct, pooled,
)
from kinetic_augment.body_model.converter import fit_body  # noqa: E402
from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape, fit_hands, landmark_errors_mm,
)
from kinetic_augment.body_model.hand_model import WRIST, HandModel  # noqa: E402
from kinetic_augment.body_model.hand_observations import (  # noqa: E402
    SIDES, load_hand_clip, save_hand_clip,
)
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

OUTPUT_DIR = Path("data/round_trip_eval")
MODEL_DIR = "models"
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
BODY_KEYS = ("direct", "template", "floor", "held_out_direct", "held_out_floor")


def find_videos(paths):
    videos = []
    for path in map(Path, paths):
        if path.is_dir():
            videos += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            videos.append(path)
        else:
            raise SystemExit(f"not a video or folder: {path}")
    stems = [v.stem for v in videos]
    if len(set(stems)) != len(stems):
        raise SystemExit("two videos share a file name; clip ids must be unique")
    return videos


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(video: Path, out_dir: Path, force: bool):
    """The clip's hand observations (with its pose), cached as <clip>_hands / _world.npz."""
    hands_path = out_dir / f"{video.stem}_hands.npz"
    world_path = out_dir / f"{video.stem}_world.npz"
    if hands_path.exists() and world_path.exists() and not force:
        return load_hand_clip(hands_path)

    import mediapipe as mp

    with mp.solutions.hands.Hands(**HANDS_SETTINGS) as hands, \
            mp.solutions.holistic.Holistic(**HOLISTIC_SETTINGS) as holistic:
        clip = extract_hands(video, hands, holistic)
    clip.meta = {
        "source": str(video), "source_sha256": _sha256(video), "mediapipe": mp.__version__,
        "hands_settings": HANDS_SETTINGS, "holistic_settings": HOLISTIC_SETTINGS,
        "detector_state": "fresh Hands and Holistic instances per clip",
    }
    save_hand_clip(hands_path, clip)
    # The body cache format of extract_world_landmarks.py, from the same Holistic pass.
    np.savez_compressed(world_path, landmarks=clip.pose_world, visibility=clip.pose_visibility)
    return clip


# --------------------------------------------------------------------------- body


def body_round_trip(clip, skeleton):
    """Per-landmark errors (fractions of shoulder width) for each BODY_KEYS row, and checks."""
    pose = clip.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (clip.pose_visibility >= VISIBILITY_THRESHOLD)
    both = ok[:, L_SHOULDER] & ok[:, R_SHOULDER]
    info = {"frames": len(pose), "frames_with_shoulders": int(both.sum()),
            "coverage_pct": {NAMES[m]: float(100 * ok[:, m].mean()) for m in NAMES}}
    rows = {key: {m: [] for m in NAMES} for key in BODY_KEYS}
    if both.sum() < 2:
        info["note"] = "fewer than two frames with both shoulders visible: body not scored"
        return rows, info

    width = float(np.median(np.linalg.norm(pose[both, L_SHOULDER] - pose[both, R_SHOULDER],
                                           axis=-1)))
    betas = fit_shape(pose, ok, skeleton)
    body = skeleton.shaped(betas)
    floor = floor_skeleton(skeleton, pose, ok)
    for key, fit_with, reconstruct_with in (("direct", body, skeleton),
                                            ("template", skeleton, skeleton),
                                            ("floor", floor, floor)):
        rows[key], info["skipped_frames"] = errors(pose, ok, fit_with, reconstruct_with, width)

    half = len(pose) // 2
    for train, test in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        train_both = both[train]
        if train_both.sum() < 2 or not both[test].any():
            continue
        trained = skeleton.shaped(fit_shape(pose[train], ok[train], skeleton))
        found, _ = errors(pose[test], ok[test], trained, skeleton, width)
        trained_floor = floor_skeleton(skeleton, pose[train], ok[train])
        found_floor, _ = errors(pose[test], ok[test], trained_floor, trained_floor, width)
        for m in NAMES:
            rows["held_out_direct"][m] += found[m]
            rows["held_out_floor"][m] += found_floor[m]

    upright = {"fitted": 0, "with_nose": 0, "nose_masked": 0}
    for frame, valid in zip(pose[both], ok[both]):
        masked = valid.copy()
        masked[NOSE] = False
        upright["fitted"] += 1
        upright["with_nose"] += int(is_upright(fit_body(frame, valid, body), body))
        upright["nose_masked"] += int(is_upright(fit_body(frame, masked, body), body))
    info.update(shoulder_width_m=width, betas_norm=float(np.linalg.norm(betas)), upright=upright)
    return rows, info


def body_summary(rows):
    out = {}
    for key, per_landmark in rows.items():
        values = pooled(per_landmark)
        out[key] = {"median_pct": pct(values), "p95_pct": pct(values, 95), "n": len(values)}
    out["gap_to_floor_pts"] = out["direct"]["median_pct"] - out["floor"]["median_pct"]
    out["per_landmark_direct_median_pct"] = {NAMES[m]: pct(np.asarray(v, dtype=float))
                                             for m, v in rows["direct"].items()}
    return out


def print_body(title, s):
    print(f"\n{title}")
    print(f"  {'direct':>9}{'p95':>8}{'template':>10}{'floor':>8}{'gap':>7}"
          f"{'held-out':>10}{'h-o floor':>11}   (% shoulder width, medians)")
    print(f"  {s['direct']['median_pct']:>8.2f}%{s['direct']['p95_pct']:>7.2f}%"
          f"{s['template']['median_pct']:>9.2f}%{s['floor']['median_pct']:>7.2f}%"
          f"{s['gap_to_floor_pts']:>6.2f}%{s['held_out_direct']['median_pct']:>9.2f}%"
          f"{s['held_out_floor']['median_pct']:>10.2f}%")
    print("  per landmark (direct): " + ", ".join(
        f"{name} {v:.1f}%" for name, v in s["per_landmark_direct_median_pct"].items()))


# -------------------------------------------------------------------------- hands


def hand_round_trip(clip, models, backends):
    """Per side: summary rows, (n, 20) error arrays per row, and track info."""
    tracks = {}
    for s, side in enumerate(SIDES):
        frames = np.flatnonzero(clip.present[:, s])
        if len(frames) < 2:
            tracks[side] = {"frames_with_hand": int(len(frames)),
                            "note": "fewer than two frames with this hand: not scored"}
            continue
        landmarks = clip.world[frames, s].astype(np.float64)
        valid = np.ones(landmarks.shape[:2], dtype=bool)
        span = np.linalg.norm(landmarks[:, MIDDLE_MCP] - landmarks[:, WRIST], axis=-1)
        scale = np.full(len(landmarks), 1000.0 * np.median(span))
        first, second = time_halves(len(landmarks))
        errors_mm = {"rigid reference": landmark_errors_mm(rigid_reference(landmarks), landmarks)}
        statuses = {}
        for backend in backends:
            model = models[(backend, side)]
            betas = fit_hand_shape(landmarks, valid, model)
            fits = fit_hands(landmarks, valid, model, betas)
            statuses[backend] = {status: sum(f.status == status for f in fits)
                                 for status in {f.status for f in fits}}
            errors_mm[f"{backend}: in-sample shape"] = landmark_errors_mm(
                rebuild(fits, model), landmarks)
            errors_mm[f"{backend}: held-out shape (time halves)"], _ = held_out(
                model, landmarks, valid, time_halves(len(landmarks)))
            errors_mm[f"{backend}: held-out shape (interleaved frames)"], _ = held_out(
                model, landmarks, valid, interleaved(len(landmarks)))
        tracks[side] = {
            "frames_with_hand": int(len(frames)),
            "hand_scale_mm": float(scale[0]),
            "half_span_ratio": float(np.median(span[second]) / np.median(span[first])),
            "fit_status": statuses,
            "rows": {row: summary(e, scale) for row, e in errors_mm.items()},
            "_errors": errors_mm,
            "_scale": scale,
        }
    return tracks


def target_check(row):
    return {key: row[f"{key}_mm"] <= limit for key, limit in TARGETS_MM.items()}


# --------------------------------------------------------------------------- main


def write_index(out_dir: Path, render: bool):
    """The project keeps an index.md in every folder; this one describes generated files."""
    lines = ["# Round-trip evaluation output", "",
             "Generated by `scripts/evaluate_round_trip.py`.", "",
             "- `*_hands.npz` — MediaPipe hand world/image landmarks with the paired pose, per "
             "clip (`HandClip`).",
             "- `*_world.npz` — Pose world landmarks and visibility from the same pass, in the "
             "`extract_world_landmarks.py` format.",
             "- `round_trip_report.json` — Body and hand round-trip errors per clip and pooled.",
             "- `index.md` — This index."]
    if render:
        lines.insert(-1, "- `videos/` — Stitched body + hand round-trip videos per clip.")
        (out_dir / "videos").mkdir(exist_ok=True)
        (out_dir / "videos" / "index.md").write_text(
            "# Round-trip videos\n\n- `*_body_hand_round_trip.mp4` — MediaPipe 2-D, input vs "
            "round trip (front, side), and the SMPL-X figure, per clip.\n- `index.md` — This "
            "index.\n")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")



def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", nargs="+", required=True,
                        help="video files and/or folders of videos")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hand-backends", nargs="+", default=["mano"],
                        choices=["mano", "smplx"])
    parser.add_argument("--reextract", action="store_true",
                        help="run MediaPipe again even if landmarks are cached")
    parser.add_argument("--render", action="store_true",
                        help="also render the stitched body + hand round-trip video per clip")
    args = parser.parse_args()

    began = time.time()
    videos = find_videos(args.videos)
    if not videos:
        raise SystemExit("no videos found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skeleton = SMPLXSkeleton(model_path=MODEL_DIR)
    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR)
              for b in args.hand_backends for s in SIDES}

    report = {"settings": {"hand_backends": args.hand_backends, "targets_mm": TARGETS_MM,
                           "visibility_threshold": VISIBILITY_THRESHOLD,
                           "body_units": "% of the clip's median shoulder width",
                           "hand_units": "mm, wrist-relative landmarks 1-20, no registration"},
              "clips": {}, "pooled": {}}
    body_pool = {key: {m: [] for m in NAMES} for key in BODY_KEYS}
    hand_pool, hand_scale = {}, {}

    print(f"Evaluating {len(videos)} clip(s) -> {args.output_dir}")
    for video in videos:
        clip_id = video.stem
        clip = extract(video, args.output_dir, args.reextract)
        print(f"\n{'=' * 78}\n{clip_id}: {len(clip.present)} frames {clip.width}x{clip.height} "
              f"@ {clip.fps:.1f} fps; left hand {int(clip.present[:, 0].sum())}, "
              f"right hand {int(clip.present[:, 1].sum())} frames\n{'=' * 78}")

        rows, body_info = body_round_trip(clip, skeleton)
        entry = {"source": str(video), "body": body_info, "hands": {}}
        if "note" in body_info:
            print(f"\nBODY: {body_info['note']}")
        else:
            entry["body"]["summary"] = body_summary(rows)
            print_body("BODY  (MediaPipe -> SMPL-X -> MediaPipe)", entry["body"]["summary"])
            up = body_info["upright"]
            print(f"  upright: {up['with_nose']}/{up['fitted']} frames with the nose, "
                  f"{up['nose_masked']}/{up['fitted']} with it masked; "
                  f"skipped (a shoulder invalid): {body_info['skipped_frames']}")
            for key in BODY_KEYS:
                for m in NAMES:
                    body_pool[key][m] += rows[key][m]

        for side, track in hand_round_trip(clip, models, args.hand_backends).items():
            if "note" in track:
                print(f"\n{side.upper()} HAND: {track['note']}")
            else:
                print_table(f"{side.upper()} HAND  ({track['frames_with_hand']} frames, hand "
                            f"{track['hand_scale_mm']:.0f} mm, second / first half span "
                            f"{track['half_span_ratio']:.2f})  mm", track["rows"])
                for row, e in track.pop("_errors").items():
                    hand_pool.setdefault(row, []).append(e)
                    hand_scale.setdefault(row, []).append(track["_scale"])
                track.pop("_scale")
            entry["hands"][side] = track
        report["clips"][clip_id] = entry

        if args.render:
            from view_body_hand_round_trip import render
            render(clip_id, args.output_dir, args.output_dir / "videos", video_path=video)

    print(f"\n{'=' * 78}\nPOOLED over {len(videos)} clip(s)\n{'=' * 78}")
    if any(body_pool["direct"].values()):
        report["pooled"]["body"] = body_summary(body_pool)
        print_body("BODY", report["pooled"]["body"])
        print("  gate: direct should reach the rigid-skeleton floor; 'gap' is what is left")
    if hand_pool:
        report["pooled"]["hands"] = {
            row: summary(np.concatenate(hand_pool[row]), np.concatenate(hand_scale[row]))
            for row in hand_pool}
        print_table("HANDS (both sides)  mm", report["pooled"]["hands"])
        print(f"\n  provisional targets: mean <= {TARGETS_MM['mean']} mm, "
              f"tips <= {TARGETS_MM['tips']} mm, p95 <= {TARGETS_MM['p95']} mm")
        for row, s in report["pooled"]["hands"].items():
            if row == "rigid reference":
                continue
            s["meets_target"] = target_check(s)
            verdict = ", ".join(f"{k} {'ok' if v else 'MISS'}"
                                for k, v in s["meets_target"].items())
            print(f"  {row:<44}{verdict}")

    write_index(args.output_dir, args.render or (args.output_dir / "videos").exists())
    out = args.output_dir / "round_trip_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out} in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
