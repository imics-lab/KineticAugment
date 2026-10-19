#!/usr/bin/env python3
"""
Visualise the V1 MediaPipe -> SMPL-X -> MediaPipe round trip, frame by frame.

For each clip it fits one body shape (`fit_shape`), then per frame:

    fit_body(world landmarks, validity, body)  ->  SMPL-X parameters
    reconstruct_body(parameters, skeleton)     ->  reconstructed landmarks

and renders a video with three panels:

    source video  |  front view  |  side view
    BLUE  = MediaPipe world landmarks (input)
    GREEN = reconstruction from SMPL-X parameters alone (joints coloured by error)

plus a strip of the per-frame error over time. Errors are in % of the observed
shoulder width, the same statistic `measure_round_trip.py` reports.

World landmarks are metric and hip-centred, so they are drawn in their own 3-D
space, not over the image: there is no camera model yet to project them back.

Usage (from the repo root, which must hold the V1 converter, data/ and models/):
    .venv/bin/python scripts/view_round_trip.py                 # all clips
    .venv/bin/python scripts/view_round_trip.py --clips 01986   # one clip

Output: data/test_output/videos/<clip>_round_trip.mp4
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kinetic_augment.body_model.converter import (  # noqa: E402
    ARM_CORRESPONDENCE, fit_body, reconstruct_body,
)
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

VIDEO_DIR = Path("data/sign_videos")
LANDMARK_DIR = Path("data/test_output")
OUTPUT_DIR = Path("data/test_output/videos")
VISIBILITY_THRESHOLD = 0.5  # matches measure_round_trip.py

ARM_MP = [mp for _, mp in ARM_CORRESPONDENCE]          # 11..16
ARM_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
CONTEXT_BONES = [(11, 23), (12, 24), (23, 24)]         # torso, drawn faintly
JOINT_NAMES = {11: "L sho", 12: "R sho", 13: "L elb", 14: "R elb", 15: "L wri", 16: "R wri"}
ERROR_CMAP_MAX = 15.0                                  # % shoulder width at full red


def load_clip(clip: str):
    data = np.load(LANDMARK_DIR / f"{clip}_world.npz")
    pose = data["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (data["visibility"] >= VISIBILITY_THRESHOLD)
    return pose, ok


def read_frames(clip: str, count: int, video_path=None):
    """RGB frames of the clip's video (VIDEO_DIR/<clip>.mp4 unless `video_path` is given)."""
    capture = cv2.VideoCapture(str(video_path or VIDEO_DIR / f"{clip}.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    while len(frames) < count:  # some clips have one more landmark frame than video frame
        frames.append(frames[-1] if frames else np.zeros((240, 320, 3), np.uint8))
    return frames, fps


def round_trip_clip(pose, ok, skeleton):
    """Fit one shape for the clip, then fit and reconstruct every frame."""
    body = skeleton.shaped(fit_shape(pose, ok, skeleton))
    fits, recon = [], np.full_like(pose, np.nan)
    for t in range(len(pose)):
        try:
            fit = fit_body(pose[t], ok[t], body)
        except ValueError:  # a shoulder is invalid: the torso frame is undefined
            fits.append(None)
            continue
        fits.append(fit)
        recon[t] = reconstruct_body(fit, skeleton)
    return body, fits, recon


def frame_errors(pose_t, ok_t, recon_t):
    """Per-joint error in % shoulder width (absent where unobserved or unfitted)."""
    errors = {}
    if not (ok_t[11] and ok_t[12]):
        return errors
    width = np.linalg.norm(pose_t[11] - pose_t[12])
    for mp in ARM_MP:
        if ok_t[mp] and np.isfinite(recon_t[mp]).all():
            errors[mp] = 100.0 * np.linalg.norm(recon_t[mp] - pose_t[mp]) / width
    return errors


def _project(points, view):
    """MediaPipe world is y-down; plot with up = up. Front: (x, -y). Side: (z, -y)."""
    if view == "front":
        return points[..., 0], -points[..., 1]
    return points[..., 2], -points[..., 1]


def draw_skeleton(ax, points, valid, view, color, bones=ARM_BONES, lw=2.5, alpha=1.0,
                  joint_colors=None, label=None, joints=True):
    first = True
    for a, b in bones:
        if valid[a] and valid[b]:
            xs, ys = _project(points[[a, b]], view)
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label if first else None)
            first = False
    if not joints:
        return
    for mp in ARM_MP:
        if valid[mp]:
            x, y = _project(points[mp], view)
            c = joint_colors.get(mp, color) if joint_colors else color
            ax.scatter([x], [y], s=36, color=c, edgecolors="k", linewidths=0.4, zorder=3,
                       alpha=alpha)


def view_limits(*clouds):
    """Fixed axis limits for the whole clip so the skeleton does not jump around."""
    pts = np.concatenate([c[:, ARM_MP + [23, 24]].reshape(-1, 3) for c in clouds])
    pts = pts[np.isfinite(pts).all(axis=1)]
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.25 + 0.05
    return centre, half


def set_limits(ax, centre, half, view, title):
    cx = centre[0] if view == "front" else centre[2]
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(-centre[1] - half, -centre[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def error_color(err):
    return plt.cm.RdYlGn_r(min(err / ERROR_CMAP_MAX, 1.0))


class VideoSink:
    """Collects matplotlib figures into an mp4 at the clip's frame rate."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.writer = path, fps, None

    def add(self, fig):
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if self.writer is None:
            h, w = image.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (w, h))
        self.writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()


def render_clip(clip, skeleton, out_dir: Path):
    pose, ok = load_clip(clip)
    frames, fps = read_frames(clip, len(pose))
    _, fits, recon = round_trip_clip(pose, ok, skeleton)

    per_frame = [frame_errors(pose[t], ok[t], recon[t]) for t in range(len(pose))]
    mean_err = np.array([np.mean(list(e.values())) if e else np.nan for e in per_frame])
    centre, half = view_limits(pose, recon)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_round_trip.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(14, 6.4), dpi=100)

    for t in range(len(pose)):
        fig.clf()
        grid = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.25, wspace=0.08)
        ax_vid, ax_front, ax_side = (fig.add_subplot(grid[0, i]) for i in range(3))
        ax_err = fig.add_subplot(grid[1, :])

        ax_vid.imshow(frames[t])
        ax_vid.set_title(f"clip {clip}  frame {t + 1}/{len(pose)}", fontsize=10)
        ax_vid.axis("off")

        recon_valid = np.isfinite(recon[t]).all(axis=1)
        colors = {mp: error_color(e) for mp, e in per_frame[t].items()}
        for ax, view in ((ax_front, "front"), (ax_side, "side")):
            draw_skeleton(ax, pose[t], ok[t], view, "#7f9cc4", bones=CONTEXT_BONES, lw=1.2,
                          alpha=0.6, joints=False)
            draw_skeleton(ax, pose[t], ok[t], view, "#1f5fbf", label="MediaPipe input")
            if fits[t] is not None:
                draw_skeleton(ax, recon[t], recon_valid, view, "#2ca02c", lw=2.0,
                              joint_colors=colors, label="SMPL-X reconstruction")
            set_limits(ax, centre, half, view,
                       "front view" if view == "front" else "side view (depth →)")
        ax_front.legend(loc="lower left", fontsize=8, framealpha=0.8)

        if fits[t] is None:
            text = "no fit: a shoulder is not visible"
        else:
            text = "\n".join(f"{JOINT_NAMES[mp]}: {e:4.1f}%" for mp, e in per_frame[t].items())
            text += f"\nmean : {mean_err[t]:4.1f}%"
        ax_side.text(0.98, 0.02, text, transform=ax_side.transAxes, ha="right", va="bottom",
                     fontsize=8, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        median = np.nanmedian(mean_err)
        ax_err.plot(np.arange(1, len(pose) + 1), mean_err, color="#555", lw=1.2)
        ax_err.axvline(t + 1, color="#d62728", lw=1)
        ax_err.axhline(median, color="#2ca02c", lw=0.8, ls="--")
        ax_err.set_xlim(1, len(pose))
        ax_err.set_ylim(0, max(np.nanmax(mean_err) * 1.1, 1.0))
        ax_err.set_ylabel("mean err\n% shoulder w.", fontsize=8)
        ax_err.set_xlabel(f"frame   (dashed: clip median {median:.2f}%; joint colour: "
                          f"green 0% → red ≥{ERROR_CMAP_MAX:.0f}%)", fontsize=8)
        ax_err.tick_params(labelsize=7)
        sink.add(fig)

    sink.close()
    plt.close(fig)
    print(f"  {clip}: median frame error {np.nanmedian(mean_err):.2f}% of shoulder width, "
          f"{sum(f is not None for f in fits)}/{len(pose)} frames fitted -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="*", help="clip ids, e.g. 01986 (default: all cached)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    clips = args.clips or sorted(p.name.split("_")[0] for p in LANDMARK_DIR.glob("*_world.npz"))
    skeleton = SMPLXSkeleton(model_path="models")
    print(f"Rendering round trip for {len(clips)} clip(s):")
    for clip in clips:
        render_clip(clip, skeleton, args.output_dir)


if __name__ == "__main__":
    main()
