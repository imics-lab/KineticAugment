#!/usr/bin/env python3
"""
Map every local WLASL clip to its gloss and mark whether it may be mirrored.

Reads the WLASL annotation file (data/wlasl/WLASL_v0.3.json, downloaded from
github.com/dxli94/WLASL and not committed) and the mirror blacklist
(configs/mirror_blacklist.yaml, docs/Mirror_Unsafe_Signs.md), and writes one row per clip:

    clip_id, gloss, wlasl_rank, signer_id, split, variation_id, mirror_safe, mirror_rule

wlasl_rank is the gloss's position in WLASL (0-99 is WLASL100). mirror_safe is "no" for a
blacklisted gloss, with the blacklist category in mirror_rule.

It also checks the blacklist against all 2000 WLASL glosses and prints the ones present.

Usage (from the repo root):
    python scripts/map_clip_glosses.py
    python scripts/map_clip_glosses.py --clips-dir data/sign_videos --out clip_glosses.csv
"""

import argparse
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_blacklist(path):
    """Return {gloss: category} for every gloss on the blacklist."""
    config = yaml.safe_load(path.read_text())
    return {
        gloss.lower(): category
        for category, entry in config["categories"].items()
        for gloss in entry["glosses"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/wlasl/WLASL_v0.3.json"
    )
    parser.add_argument(
        "--clips-dir", type=Path, default=ROOT / "data/metsis_new_unseen_sign_videos"
    )
    parser.add_argument("--blacklist", type=Path, default=ROOT / "configs/mirror_blacklist.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data/wlasl/clip_glosses.csv")
    args = parser.parse_args()

    entries = json.loads(args.annotations.read_text())
    blacklist = load_blacklist(args.blacklist)

    by_video = {}
    for rank, entry in enumerate(entries):
        for instance in entry["instances"]:
            by_video[instance["video_id"]] = (entry["gloss"], rank, instance)

    rows, missing = [], []
    for clip in sorted(p.stem for p in args.clips_dir.glob("*.mp4")):
        if clip not in by_video:
            missing.append(clip)
            continue
        gloss, rank, instance = by_video[clip]
        category = blacklist.get(gloss.lower())
        rows.append(
            {
                "clip_id": clip,
                "gloss": gloss,
                "wlasl_rank": rank,
                "signer_id": instance["signer_id"],
                "split": instance["split"],
                "variation_id": instance["variation_id"],
                "mirror_safe": "no" if category else "yes",
                "mirror_rule": category or "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    glosses = sorted({row["gloss"] for row in rows})
    blocked = sorted({row["gloss"] for row in rows if row["mirror_safe"] == "no"})
    print(f"{len(rows)} clips, {len(glosses)} glosses -> {args.out}")
    if missing:
        print(f"not in WLASL: {', '.join(missing)}")
    print(f"blacklisted glosses among the clips: {', '.join(blocked) or 'none'}")

    in_wlasl = {entry["gloss"].lower(): rank for rank, entry in enumerate(entries)}
    present = sorted((in_wlasl[g], g, c) for g, c in blacklist.items() if g in in_wlasl)
    print(f"blacklisted glosses in WLASL2000 ({len(present)} of {len(blacklist)}):")
    for rank, gloss, category in present:
        print(f"  {gloss:<16} rank {rank:<5} {category}")


if __name__ == "__main__":
    main()
