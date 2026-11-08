#!/usr/bin/env python3
"""
Week-4 visual trial: one video panel per augmentation, plus a JSON report of what each did.

For each clip the V1.3 round trip runs on the signing span (`face_pipeline.run_clip`), the result
becomes a `ClipData` with its weak camera fitted (`scene.fit_clip_camera`), and the profile is run
through `AugmentationChain` once per enabled entry (only that entry on, probability forced to 1)
and once as a whole ("combined"). Every run is measured against the unaugmented clip with
`report.measure`.

Video: <output-dir>/<clip>_week4_augmentations.mp4 is a grid with one panel per run: a front view
(the image projection of the clip's camera, y down) of the arm bones (pose 11-16), both hands
(MediaPipe topology) and the 478 face points, over the unaugmented clip in grey. The title is the
run's name and its first draw values. Temporal runs have a different frame count: each panel is
shown at its own frame index modulo its length (it loops), while the grey base plays once.

Usage (from the repo root):
    .venv/bin/python scripts/augment_week4_trials.py --clips 57949
    .venv/bin/python scripts/augment_week4_trials.py --disable face.expression_jitter

Reads <landmark-dir>/<clip>_hands.npz and <clip>_face.npz. Writes
<output-dir>/<clip>_week4_augmentations.mp4 and <output-dir>/augment_week4_trials.json.
"""

import argparse
import copy
import dataclasses
import json
import math
import os
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

from augment_face_trials import signing_span  # noqa: E402
from view_body_hand_round_trip import HAND_BONES  # noqa: E402
from view_round_trip import VideoSink  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kinetic_augment.augmentations.chain import AugmentationChain  # noqa: E402
from kinetic_augment.augmentations.clip import clip_from_result  # noqa: E402
from kinetic_augment.augmentations.outputs import build_outputs  # noqa: E402
from kinetic_augment.augmentations.profile import load_profile  # noqa: E402
from kinetic_augment.augmentations.report import measure  # noqa: E402
from kinetic_augment.augmentations.scene import fit_clip_camera  # noqa: E402
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip  # noqa: E402
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402

PROFILE = Path("configs/slr_week4_ambitious.yaml")
LANDMARK_DIR = Path("data/round_trip_eval")
OUTPUT_DIR = Path("data/augment_trials")
MODEL_DIR = "models"
CLIPS = ("57949", "12328", "63669", "63229")
ARM_BONES = ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12))
COLUMNS = 4
GREY, BLUE, ORANGE, GREEN = "#b8b8b8", "#1f5fbf", "#d9730d", "#2a7f3f"


def only_entry(profile, name):
    """A copy of `profile` with just `name` enabled, at probability 1."""
    copied = copy.deepcopy(profile)
    for entry in copied.entries:
        entry.enabled = entry.name == name
        if entry.enabled:
            entry.probability = 1.0
    return copied


def jsonable(value):
    """Plain-JSON copy of nested draws and measures (numpy to lists, NaN / inf to null)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def draw_summary(draws, name, limit=3):
    """The first few scalar (or short list) values of a draw, for a panel title."""
    parts = []
    for key, value in (draws.get(name) or {}).items():
        if isinstance(value, bool) or isinstance(value, (int, float, np.floating, np.integer)):
            parts.append(f"{key}={value:.3g}" if not isinstance(value, bool) else f"{key}={value}")
        elif isinstance(value, (list, tuple)) and len(value) <= 3 and value:
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                parts.append(f"{key}=[{', '.join(f'{v:.3g}' for v in value)}]")
        if len(parts) == limit:
            break
    return ", ".join(parts)


def scene_bone_change_mm(base, aug, draws):
    """
    Largest bone-length change of a scene run after accounting for what the scene entry does.

    Mirror relabels the sides (left bones become the right ones) and scale multiplies every
    length by its factor; neither changes a bone's shape, so the base lengths are swapped /
    scaled the same way before comparing. Zero up to rounding means the scene entry is rigid.
    """
    worst = 0.0
    swap = bool(draws.get("scene.mirror", {}).get("mirror"))
    factor = float(draws.get("scene.scale", {}).get("factor", 1.0))
    pairs = ((11, 13), (13, 15), (12, 14), (14, 16))
    other = {pairs[0]: pairs[2], pairs[1]: pairs[3], pairs[2]: pairs[0], pairs[3]: pairs[1]}
    for bone in pairs:
        a, b = other[bone] if swap else bone
        ref = np.linalg.norm(base.pose_world[:, a] - base.pose_world[:, b], axis=-1) * factor
        got = np.linalg.norm(aug.pose_world[:, bone[0]] - aug.pose_world[:, bone[1]], axis=-1)
        diff = np.abs(got - ref)
        worst = max(worst, float(diff[np.isfinite(diff)].max()))
    return 1000.0 * worst


def points_of(outputs):
    """(arms (T,6,2), hands (T,2,21,2), face (T,478,2)) in the front view, NaN where unusable."""
    if outputs.pose_image is not None:
        arms, hands, face = outputs.pose_image[:, 11:17], outputs.hands_image, outputs.face_image
    else:
        arms = outputs.pose_world[:, 11:17, :2]
        hands, face = outputs.hands_world[..., :2], outputs.face_world[..., :2]
    hands = np.where(outputs.hand_valid[:, :, None, None], hands, np.nan)
    face = np.where(outputs.face_valid[:, None, None], face, np.nan)
    return arms, hands, face


class Panel:
    """One axes: the grey base and the coloured augmentation, updated frame by frame."""

    def __init__(self, ax, title, base_pts, aug_pts, limits):
        self.ax, self.base, self.aug = ax, base_pts, aug_pts
        ax.set_xlim(*limits[0])
        ax.set_ylim(limits[1][1], limits[1][0])  # y down
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7)
        self.artists = {}
        for tag, color, width in (("base", GREY, 1.0), ("aug", BLUE, 1.4)):
            self.artists[tag] = dict(
                arms=[ax.plot([], [], "-", color=color, lw=width * 1.5)[0] for _ in ARM_BONES],
                hands=[
                    ax.plot([], [], "-", color=(color if tag == "base" else ORANGE), lw=width)[0]
                    for _ in range(2 * len(HAND_BONES))
                ],
                face=ax.plot(
                    [],
                    [],
                    ".",
                    color=(color if tag == "base" else GREEN),
                    ms=0.8 if tag == "aug" else 0.6,
                )[0],
            )

    @staticmethod
    def _update(art, pts, t):
        arms, hands, face = pts
        t = t % len(arms)
        for line, (a, b) in zip(art["arms"], ARM_BONES):
            line.set_data(
                [arms[t, a - 11, 0], arms[t, b - 11, 0]], [arms[t, a - 11, 1], arms[t, b - 11, 1]]
            )
        k = 0
        for side in range(2):
            for a, b in HAND_BONES:
                p, q = hands[t, side, a], hands[t, side, b]
                art["hands"][k].set_data([p[0], q[0]], [p[1], q[1]])
                k += 1
        art["face"].set_data(face[t, :, 0], face[t, :, 1])

    def show(self, t):
        self._update(self.artists["base"], self.base, t)
        self._update(self.artists["aug"], self.aug, t)


def view_limits(point_sets):
    """Shared (x, y) limits that hold every finite arm / hand / face point of the base."""
    pts = np.concatenate([p.reshape(-1, 2) for p in point_sets])
    pts = pts[np.isfinite(pts).all(axis=1)]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.08 * (hi - lo).max()
    return (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)


def render(path, fps, base_out, runs, base_frames):
    """Write the panel grid video: `runs` is a list of (title, outputs)."""
    base_pts = points_of(base_out)
    limits = view_limits(base_pts)
    rows = math.ceil(len(runs) / COLUMNS)
    fig = plt.figure(figsize=(3.2 * COLUMNS, 3.3 * rows), dpi=70)
    panels = []
    for i, (title, outputs) in enumerate(runs):
        ax = fig.add_subplot(rows, COLUMNS, i + 1)
        panels.append(Panel(ax, title, base_pts, points_of(outputs), limits))
    fig.tight_layout(pad=0.6)
    sink = VideoSink(path, fps)
    for t in range(base_frames):
        for panel in panels:
            panel.show(t)
        sink.add(fig)
    sink.close()
    plt.close(fig)


def run_clip_trial(clip_id, args, profile, models, output_dir):
    started = time.time()
    hands = load_hand_clip(args.landmark_dir / f"{clip_id}_hands.npz")
    face = load_face_clip(args.landmark_dir / f"{clip_id}_face.npz")
    span = signing_span(hands)
    if span is None:
        return {"error": "no frame with both arms observed"}
    result = run_clip(hands, face, models, frames=span)
    skeleton = models.skeleton
    clip = clip_from_result(clip_id, result, hands, face, skeleton)
    camera = fit_clip_camera(clip, hands, skeleton, frames=result.frames)
    clip = clip.replace(camera=camera)
    base_out = build_outputs(clip, skeleton)

    names = [e.name for e in profile.entries if e.enabled]
    plan = [(name, only_entry(profile, name)) for name in names] + [("combined", profile)]
    report, panels = {}, []
    for name, prof in plan:
        entry = {}
        try:
            t0 = time.time()
            res = AugmentationChain(prof, skeleton).run(clip)
            temporal = name.startswith("temporal.") or any(
                a.startswith("temporal.") for a in res.applied
            )
            aligned = res.clip.frames == clip.frames and not temporal
            entry = {
                "applied": list(res.applied),
                "fallbacks": dict(res.fallbacks),
                "frames": int(res.clip.frames),
                "aligned": bool(aligned),
                "draws": res.draws,
                "measure": measure(base_out, res.outputs, aligned),
                "seconds": round(time.time() - t0, 2),
            }
            if name.startswith("scene."):
                entry["measure"]["bone_length_change_scene_adjusted_mm"] = scene_bone_change_mm(
                    base_out, res.outputs, res.draws
                )
            title = name if name != "combined" else f"combined ({len(res.applied)} applied)"
            summary = draw_summary(res.draws, name) if name != "combined" else ""
            panels.append((f"{title}\n{summary}" if summary else title, res.outputs))
        except Exception:  # one failing run must not stop the others
            entry = {"error": traceback.format_exc()}
            print(f"[{clip_id}] {name} FAILED\n{entry['error']}")
        report[name] = entry
        print(f"[{clip_id}] {name}: {entry.get('applied', 'error')}")
    video = output_dir / f"{clip_id}_week4_augmentations.mp4"
    render(video, float(clip.fps), base_out, panels, clip.frames)
    return {
        "frames": int(clip.frames),
        "fps": float(clip.fps),
        "video": str(video),
        "runs": report,
        "seconds": round(time.time() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--clips", nargs="+", default=list(CLIPS))
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--enable", nargs="*", default=[], metavar="region.key")
    parser.add_argument("--disable", nargs="*", default=[], metavar="region.key")
    args = parser.parse_args()

    profile = load_profile(args.profile, enable=args.enable, disable=args.disable)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_face_models(args.model_dir)
    out = {"profile": profile.name, "seed": profile.seed, "clips": {}}
    for clip_id in args.clips:
        out["clips"][clip_id] = run_clip_trial(clip_id, args, profile, models, args.output_dir)
        json_path = args.output_dir / "augment_week4_trials.json"
        json_path.write_text(json.dumps(jsonable(out), indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
