#!/usr/bin/env python3
"""
Hand-to-face proximity from MediaPipe pose landmarks alone (08925 head-placement spike).

Reads <landmark-dir>/<clip>_pose_only.npz (see extract_pose_only.py) and the video
<video-dir>/<clip>.mp4. No SMPL-X, no FaceLandmarker. Per frame and per hand, the closest
(hand point, face point) pair is found by 3-D pose-world distance, over the face set {0..10} and the
hand set {15, 17, 19, 21} (left) / {16, 18, 20, 22} (right). For that pair it reports

    3-D       distance in pose-world metres; CONTACT below 2 cm
    |dz|      its world-z component (depth), and `plane` its world (x, y) component
    image     distance in pixels between the two image landmarks, and in metres via a per-frame
              pixels-per-metre scale (2-D similarity fit of world x, y to image pixels over the
              visible pose points 0-24); NEAR below 2 cm
    img-min   the smallest image distance over all hand/face pairs (not tied to the 3-D pair)

A point is usable when its visibility is >= 0.5 and its normalised image x, y lie inside the frame
(0..1). Only usable points enter the closest-pair search, and a hand counts as present in a frame
when its wrist (15 / 16) is usable and at least one face point is. Image "metres" use the body-plane
pixels-per-metre scale, so they are approximate for a hand nearer the camera than the body.

Panels: 1. video frame with the pose skeleton, face points (orange), left hand points (blue), right
hand points (green) and the closest pair joined by a line (red at CONTACT), per-hand readouts below;
2. front view (x, y) and 3. side view (z, y; depth to the right) of the world landmarks with the
same pair line; 4. strip over all frames: per-frame minimum 3-D distance (over hands), and the image
distance and |dz| of that same pair, the 2 cm band and a cursor at the current frame.

Usage (from the repo root):
    PYTHONPATH=src:scripts .venv/bin/python scripts/view_pose_face_proximity.py 08925 \
        --video-dir /path/to/videos

Output: <output-dir>/<clip>_pose_face_proximity.mp4 and 2-3 stills
<clip>_pose_face_proximity_frame<N>.png (smallest 3-D distance, smallest image distance, and the
largest 3-D distance for contrast).
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from view_round_trip import VideoSink  # noqa: E402

VIDEO_DIR = Path("data/sign_videos")
LANDMARK_DIR = Path("data/test_output/exp_08925")
OUTPUT_DIR = Path("data/test_output/exp_08925")

FACE = list(range(11))
HANDS = {"L": [15, 17, 19, 21], "R": [16, 18, 20, 22]}
WRIST = {"L": 15, "R": 16}
CONTACT_M = 0.02
VISIBILITY = 0.5
SCALE_POINTS = 25  # pose points 0-24 feed the world -> pixel fit
COLOURS = {"face": "#ff8c00", "L": "#1f77b4", "R": "#2ca02c", "line": "#d62728",
           "grey": "#9a9a9a", "pair": "#c2185b"}
# Body bones among points 0-24 (face points are drawn as points only).
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (15, 17), (15, 19), (17, 19), (15, 21), (16, 18), (16, 20), (18, 20), (16, 22)]


def similarity_scale(world_xy: np.ndarray, image_px: np.ndarray) -> float:
    """Pixels per metre of the 2-D similarity (rotation, scale, shift) taking world to pixels."""
    ok = np.isfinite(world_xy).all(axis=1) & np.isfinite(image_px).all(axis=1)
    if ok.sum() < 2:
        return float("nan")
    w = world_xy[ok, 0] + 1j * world_xy[ok, 1]
    p = image_px[ok, 0] + 1j * image_px[ok, 1]
    w, p = w - w.mean(), p - p.mean()
    denom = float(np.sum(np.abs(w) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(abs(np.sum(np.conj(w) * p)) / denom)  # |a| of the least-squares p = a * w + b


def closest_pair(hand_world: np.ndarray, face_world: np.ndarray):
    """(hand index, face index, distance) of the closest pair of two 3-D point sets."""
    gap = np.linalg.norm(hand_world[:, None, :] - face_world[None, :, :], axis=-1)
    gap = np.where(np.isfinite(gap), gap, np.inf)
    h, f = np.unravel_index(np.argmin(gap), gap.shape)
    return int(h), int(f), float(gap[h, f])


def measure(data) -> dict:
    """Per-frame, per-hand closest-pair measures; arrays are (T, 2) over hands (L, R)."""
    world, image, vis = data["world"], data["image"], data["visibility"]
    w, h = float(data["width"]), float(data["height"])
    n = len(world)
    keys = ("present", "hand_pt", "face_pt", "d3", "dz", "plane", "img_px", "img_m", "img_min_m",
            "contact", "near", "ppm")
    out = {k: np.full((n, 2), np.nan) for k in keys}
    for k in ("present", "contact", "near"):
        out[k] = np.zeros((n, 2), dtype=bool)
    out["hand_pt"][:], out["face_pt"][:] = -1, -1
    px = image[..., :2] * np.array([w, h])
    for t in range(n):
        if not data["valid"][t]:
            continue
        usable = (vis[t] >= VISIBILITY) & ((image[t, :, :2] >= 0) & (image[t, :, :2] <= 1)).all(1)
        seen = np.where(vis[t, :SCALE_POINTS] >= VISIBILITY)[0]
        if len(seen) < 3:
            seen = np.arange(SCALE_POINTS)
        ppm = similarity_scale(world[t, seen, :2], px[t, seen])
        out["ppm"][t] = ppm
        for i, side in enumerate("LR"):
            idx = [j for j in HANDS[side] if usable[j]]
            face = [j for j in FACE if usable[j]]
            if not usable[WRIST[side]] or not face:
                continue
            hp, fp, d3 = closest_pair(world[t, idx], world[t, face])
            a, b = idx[hp], face[fp]
            delta = world[t, a] - world[t, b]
            img_px = float(np.linalg.norm(px[t, a] - px[t, b]))
            img_all = np.linalg.norm(px[t, idx][:, None] - px[t, face][None], axis=-1)
            out["present"][t, i] = True
            out["hand_pt"][t, i], out["face_pt"][t, i] = a, b
            out["d3"][t, i], out["dz"][t, i] = d3, abs(delta[2])
            out["plane"][t, i] = float(np.hypot(delta[0], delta[1]))
            out["img_px"][t, i], out["img_m"][t, i] = img_px, img_px / ppm
            out["img_min_m"][t, i] = float(img_all.min()) / ppm
            out["contact"][t, i] = d3 < CONTACT_M
            out["near"][t, i] = img_px / ppm < CONTACT_M
    return out


def read_frames(path: Path, count: int):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < count:
        raise ValueError(f"{path}: {len(frames)} frames decoded, expected {count}")
    return frames


def view_limits(world: np.ndarray, valid: np.ndarray):
    pts = world[valid][:, :SCALE_POINTS].reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, float((hi - lo).max() / 2) * 1.08


def draw_world(ax, pts, cols, m, t, centre, half, label, depth):
    """Front (x, y) or side (z, y) view of the world landmarks with the closest-pair lines."""
    u = 2 if depth else 0
    for a, b in BONES:
        ax.plot(pts[[a, b], u], pts[[a, b], 1], color=COLOURS["grey"], lw=1)
    ax.scatter(pts[:SCALE_POINTS, u], pts[:SCALE_POINTS, 1], s=6, c=COLOURS["grey"], zorder=2)
    ax.scatter(pts[FACE, u], pts[FACE, 1], s=22, c=COLOURS["face"], zorder=3)
    for i, side in enumerate("LR"):
        idx = HANDS[side]
        ax.scatter(pts[idx, u], pts[idx, 1], s=22, c=COLOURS[side], zorder=3)
        if m["present"][t, i]:
            a, b = int(m["hand_pt"][t, i]), int(m["face_pt"][t, i])
            colour = COLOURS["line"] if m["contact"][t, i] else COLOURS["pair"]
            ax.plot(pts[[a, b], u], pts[[a, b], 1], color=colour, lw=1.8, zorder=4)
    ax.set_xlim(centre[u] - half, centre[u] + half)
    ax.set_ylim(centre[1] + half, centre[1] - half)  # world y points down
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("z, away from camera (m) →" if depth else "x (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def readout(m, t) -> str:
    lines = []
    for i, side in enumerate("LR"):
        if not m["present"][t, i]:
            lines.append(f"{side}: no in-frame visible pair")
            continue
        flags = ("  CONTACT" if m["contact"][t, i] else "") + ("  NEAR" if m["near"][t, i] else "")
        lines.append(
            f"{side}: 3D {m['d3'][t, i] * 100:5.1f} cm  |dz| {m['dz'][t, i] * 100:5.1f}  "
            f"plane {m['plane'][t, i] * 100:4.1f}\n    img {m['img_px'][t, i]:4.1f} px = "
            f"{m['img_m'][t, i] * 100:4.1f} cm (min {m['img_min_m'][t, i] * 100:4.1f}){flags}")
    return "\n".join(lines)


def best_hand(m):
    """Per frame, the hand with the smallest 3-D distance (NaN where no hand is present)."""
    d3 = np.where(m["present"], m["d3"], np.inf)
    hand = np.argmin(d3, axis=1)
    rows = np.arange(len(d3))
    ok = np.isfinite(d3[rows, hand])
    return hand, rows, ok


def pick_stills(m, count=3):
    """0-based frames of the smallest 3-D, smallest image and largest 3-D distance (distinct)."""
    hand, rows, ok = best_hand(m)
    if not ok.any():
        return [0]
    d3 = np.where(ok, m["d3"][rows, hand], np.nan)
    img = np.where(ok, m["img_m"][rows, hand], np.nan)
    picks = [int(np.nanargmin(d3)), int(np.nanargmin(img)), int(np.nanargmax(d3))]
    chosen = []
    for t in picks:
        if t not in chosen:
            chosen.append(t)
    return sorted(chosen[:count])


def render(clip, landmark_dir: Path, video_dir: Path, out_dir: Path):
    data = np.load(landmark_dir / f"{clip}_pose_only.npz")
    world, image = data["world"], data["image"]
    n = len(world)
    frames = read_frames(video_dir / f"{clip}.mp4", n)
    fps = float(data["fps"])
    m = measure(data)
    hand, rows, ok = best_hand(m)
    d3 = np.where(ok, m["d3"][rows, hand], np.nan) * 100
    img = np.where(ok, m["img_m"][rows, hand], np.nan) * 100
    dz = np.where(ok, m["dz"][rows, hand], np.nan) * 100
    centre, half = view_limits(world, data["valid"])
    still_set = set(pick_stills(m))
    px = image[..., :2] * np.array([float(data["width"]), float(data["height"])])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_pose_face_proximity.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    for t in range(n):
        fig.clf()
        grid = fig.add_gridspec(2, 3, height_ratios=[3.6, 1.3], hspace=0.38, wspace=0.12,
                                left=0.05, right=0.98, top=0.94, bottom=0.07)
        ax_vid, ax_front, ax_side = (fig.add_subplot(grid[0, i]) for i in range(3))
        ax_strip = fig.add_subplot(grid[1, :])

        ax_vid.imshow(frames[t])
        if data["valid"][t]:
            for a, b in BONES:
                ax_vid.plot(px[t, [a, b], 0], px[t, [a, b], 1], color=COLOURS["grey"], lw=1)
            ax_vid.scatter(px[t, FACE, 0], px[t, FACE, 1], s=16, c=COLOURS["face"], zorder=3)
            for i, side in enumerate("LR"):
                idx = HANDS[side]
                ax_vid.scatter(px[t, idx, 0], px[t, idx, 1], s=16, c=COLOURS[side], zorder=3)
                if m["present"][t, i]:
                    a, b = int(m["hand_pt"][t, i]), int(m["face_pt"][t, i])
                    colour = COLOURS["line"] if m["contact"][t, i] else COLOURS["pair"]
                    ax_vid.plot(px[t, [a, b], 0], px[t, [a, b], 1], color=colour, lw=1.6, zorder=4)
        ax_vid.set_xlim(0, data["width"])
        ax_vid.set_ylim(data["height"], 0)
        ax_vid.set_title(f"{clip}  frame {t + 1}/{n}  (face orange, L blue, R green)", fontsize=9)
        ax_vid.set_xlabel(readout(m, t), fontsize=7, family="monospace", loc="left")
        ax_vid.tick_params(labelsize=7)

        if data["valid"][t]:
            draw_world(ax_front, world[t], COLOURS, m, t, centre, half, "front view (x, y)", False)
            draw_world(ax_side, world[t], COLOURS, m, t, centre, half,
                       "side view (z, y), depth to the right", True)

        xs = np.arange(1, n + 1)
        ax_strip.plot(xs, d3, color="#222222", lw=1.6, label="3-D min")
        ax_strip.plot(xs, img, color="#1f77b4", lw=1.3, label="image (same pair)")
        ax_strip.plot(xs, dz, color="#d62728", lw=1.3, ls="--", label="|dz|")
        ax_strip.axhspan(0, CONTACT_M * 100, color="#2ca02c", alpha=0.2, label="2 cm")
        ax_strip.axvline(t + 1, color="k", lw=1)
        ax_strip.set_xlim(1, n)
        ax_strip.set_ylim(0, max(5.0, float(np.nanmax(np.r_[d3, img, dz])) * 1.35))
        ax_strip.set_ylabel("cm", fontsize=8)
        ax_strip.set_xlabel("frame (closest hand per frame)", fontsize=8)
        ax_strip.tick_params(labelsize=7)
        ax_strip.legend(fontsize=7, ncol=4, loc="upper right")
        sink.add(fig)
        if t in still_set:
            fig.savefig(out_dir / f"{clip}_pose_face_proximity_frame{t + 1}.png", dpi=100)
    sink.close()
    plt.close(fig)

    capture = cv2.VideoCapture(str(out_path))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return m, ok, hand, rows, written, sorted(t + 1 for t in still_set), out_path


def summarise(clip, m, ok, hand, rows, written, stills, out_path):
    d3, dz = m["d3"][rows, hand], m["dz"][rows, hand]
    plane, img = m["plane"][rows, hand], m["img_m"][rows, hand]
    img_min = np.where(m["present"], m["img_min_m"], np.inf).min(axis=1)
    print(f"{clip}: {len(ok)} frames, {int(ok.sum())} with a hand "
          f"(L {int(m['present'][:, 0].sum())}, R {int(m['present'][:, 1].sum())}); "
          f"mp4 {written} frames -> {out_path}")
    print(f"{'closest pair, cm':<28}{'median':>8}{'min':>8}")
    for name, v in (("3-D", d3), ("|dz| (depth)", dz), ("plane (x, y)", plane),
                    ("image (same pair)", img), ("image min over pairs", img_min)):
        v = v[ok] * 100
        print(f"{name:<28}{np.median(v):8.2f}{v.min():8.2f}")
    print(f"CONTACT frames (3-D < 2 cm): {int(m['contact'].any(axis=1).sum())}   "
          f"NEAR frames (image < 2 cm): {int(m['near'].any(axis=1).sum())}")
    print(f"median pixels per metre: {np.nanmedian(m['ppm'][:, 0]):.0f}   stills: {stills}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clip", help="clip id, e.g. 08925")
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = render(args.clip, args.landmark_dir, args.video_dir, args.output_dir)
    summarise(args.clip, *result)


if __name__ == "__main__":
    main()
