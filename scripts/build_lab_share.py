#!/usr/bin/env python3
"""
Build the shareable offline Augmentation Lab: `dist/augmentation-lab.zip`.

The zip opens by double-clicking `index.html` (Chrome or Safari, offline, no models). It holds the
Lab page and a precomputed grid of panes for each clip, sized to `--budget-mb`.

    python scripts/build_lab_share.py --clips 57949 01986 01988 --out dist/augmentation-lab.zip

A clip that is not in the clip store yet is added from `--video-dir/<id>.mp4` (extraction and fit,
about a minute). `--clips` defaults to 57949, 01986 and the first clip of
`demo/round_trip/clips.json` `order` that has a video and is not one of those two.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinetic_augment.lab.clipstore import DEFAULT_ROOT, ClipStore  # noqa: E402
from kinetic_augment.lab.models import check_models, repo_model_dir  # noqa: E402
from kinetic_augment.lab.share import MESH_MM, MESH_VERTICES, build_share  # noqa: E402

FIXED = ["57949", "01986"]
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".avi")


def find_video(video_dir: Path, clip_id: str):
    for suffix in VIDEO_SUFFIXES:
        path = video_dir / f"{clip_id}{suffix}"
        if path.is_file():
            return path
    return None


def default_clips(video_dir: Path, order_file: Path):
    """The two fixed clips and the first clip in `order` that has a video and is not one of them."""
    order = json.loads(order_file.read_text()).get("order", [])
    third = next((c for c in order if c not in FIXED and find_video(video_dir, c)), None)
    return FIXED + ([third] if third else [])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clips", nargs="+", default=None, help="clip ids (default: see above)")
    p.add_argument("--out", type=Path, default=REPO / "dist" / "augmentation-lab.zip")
    p.add_argument("--budget-mb", type=int, default=200)
    p.add_argument("--profile", default="default", choices=["default", "ambitious", "literature"])
    p.add_argument("--clips-dir", type=Path, default=REPO / DEFAULT_ROOT)
    p.add_argument("--video-dir", type=Path, default=REPO / "data" / "sign_videos")
    p.add_argument("--clips-json", type=Path, default=REPO / "demo" / "round_trip" / "clips.json")
    p.add_argument("--model-dir", type=Path, default=None, help="default: <repo>/models")
    p.add_argument("--mesh-vertices", type=int, default=MESH_VERTICES, help="decimated mesh size")
    p.add_argument("--mesh-mm", type=float, default=MESH_MM, help="mesh precision in mm")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_dir = args.model_dir or repo_model_dir()
    missing = check_models(model_dir)
    if missing:
        print("missing model files in %s: %s" % (model_dir, ", ".join(missing)), file=sys.stderr)
        return 2
    from kinetic_augment.body_model.face_pipeline import load_face_models

    started = time.time()
    models = load_face_models(str(model_dir))
    store = ClipStore(args.clips_dir, models)
    clips = args.clips or default_clips(args.video_dir, args.clips_json)
    have = {c["id"] for c in store.list()}
    for clip_id in clips:
        if clip_id in have:
            continue
        video = find_video(args.video_dir, clip_id)
        if video is None:
            print(
                "clip %s is not in %s and has no video in %s"
                % (clip_id, args.clips_dir, args.video_dir),
                file=sys.stderr,
            )
            return 2
        from kinetic_augment.lab.ingest import ingest_video

        print("adding clip %s from %s" % (clip_id, video), flush=True)
        ingest_video(
            video,
            clip_id,
            store,
            models,
            lambda step, frac: print("  %s %.0f%%" % (step, 100 * frac), flush=True),
        )
    print("clips: %s" % " ".join(clips), flush=True)
    summary = build_share(
        clips,
        store,
        models,
        args.out,
        budget_mb=args.budget_mb,
        profile=args.profile,
        mesh_target=args.mesh_vertices,
        mesh_mm=args.mesh_mm,
        progress=lambda text: print(text, flush=True),
    )
    summary["seconds"] = round(time.time() - started, 1)
    summary["size_mb"] = round(summary["size_bytes"] / 1e6, 1)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
