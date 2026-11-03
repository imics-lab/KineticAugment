#!/usr/bin/env python3
"""
V1.2 figures: how MediaPipe finds the hands, and what the hand round trip looks like.

Fits the "in-sample shape" row of docs/V1_2_Findings.md section 2 for all nine hand tracks on
both backends, exactly as `ladder()` in scripts/measure_hand_round_trip.py does: one plausible
shape per track, fitted jointly with every frame of the whole track, then every frame refitted
at that shape. (Frames in a batch share an optimiser run, so fitting a subset of a track would
not reproduce the measured numbers.) It rebuilds the 21 landmarks from the fitted parameters
alone, checks the per-track means against data/test_output/hand_round_trip.json, and writes
six PNG figures:

    01_detection_overview.png   one frame per clip with the detected hands
    02_detection_timeline.png   which frames have a left hand, a right hand, none, or a stray
    03_round_trip_examples.png  the best, median and worst frame-side, observed vs rebuilt
    04_round_trip_mesh.png      the median example's fitted MANO mesh
    05_overlay_on_video.png     the same three examples drawn on the video frame
    06_hand_size_drift.png      MediaPipe's hand size against the fitted hand's, per track

Usage (from the repository root, after scripts/extract_hand_landmarks.py):
    python scripts/visualize_hand_round_trip.py
    python scripts/visualize_hand_round_trip.py --cache-dir /tmp/hand_fits   # fit once, re-render

With --cache-dir, each fitted track is saved with `save_hand_fits` (named after a hash of its
landmarks) and reused on later runs. The cache does not notice a change of fitting code: clear
the directory after changing `hand_converter.py` or `hand_model.py`.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib as mpl
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    landmark_errors_mm,
    load_hand_fits,
    reconstruct_hand,
    save_hand_fits,
)
from kinetic_augment.body_model.hand_model import (
    HAND_PARENTS,
    MANO_TIP_VERTICES,
    POSE_SLOT_LANDMARKS,
    TIPS,
    WRIST,
    HandModel,
)
from kinetic_augment.body_model.hand_observations import LEFT, RIGHT, SIDES, load_hand_clip

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "test_output"
VIDEO_DIR = ROOT / "data" / "sign_videos"
MODEL_DIR = str(ROOT / "models")
RESULTS_JSON = DATA_DIR / "hand_round_trip.json"
OUT_DIR = ROOT / "docs" / "media" / "v1_2_hands"
BACKENDS = ("mano", "smplx")
BACKEND_NAMES = {"mano": "MANO", "smplx": "SMPL-X"}
MIDDLE_MCP = 9
DPI = 150
JSON_TOLERANCE_MM = 0.01
MESH_TOLERANCE_M = 1e-6

# Colour-blind safe, the same in every figure.
SIDE_COLOURS = ("#0072B2", "#E69F00")  # left, right
SMPLX_COLOUR = "#009E73"
OBSERVED = "#222222"
ERROR = "#D55E00"
UNASSIGNED = "#CC79A7"
NO_HAND = "#BBBBBB"

# Every (landmark, parent) pair of the 21-landmark hand skeleton.
CHILDREN = np.array([c for c, p in enumerate(HAND_PARENTS) if p >= 0])
PARENTS = HAND_PARENTS[CHILDREN]


@dataclass
class Track:
    """One hand of one clip, the frames that have it, and its fits."""

    clip: str
    side: int  # LEFT or RIGHT
    frames: np.ndarray  # (n,) video frame index of each track frame
    world: np.ndarray  # (n, 21, 3) float64 MediaPipe world landmarks, metres
    fits: Dict[str, list] = field(default_factory=dict)
    rebuilt: Dict[str, np.ndarray] = field(default_factory=dict)  # (n, 21, 3) metres
    errors: Dict[str, np.ndarray] = field(default_factory=dict)  # (n, 20) millimetres

    @property
    def side_name(self) -> str:
        return SIDES[self.side]

    @property
    def name(self) -> str:
        return f"{self.clip} {self.side_name}"

    def mean_error(self, backend: str) -> np.ndarray:
        """(n,) mean landmark error of each frame, in millimetres."""
        return np.nanmean(self.errors[backend], axis=1)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def lighten(colour: str, amount: float) -> np.ndarray:
    """The colour mixed with white; amount 0 keeps it, 1 gives white."""
    rgb = np.array(mpl.colors.to_rgb(colour))
    return rgb + (1.0 - rgb) * amount


# ---------------------------------------------------------------------------------------------
# Data: tracks, fits, the check against the measurement
# ---------------------------------------------------------------------------------------------


def load_clips() -> Dict[str, object]:
    clips = {
        p.stem.replace("_hands", ""): load_hand_clip(p)
        for p in sorted(DATA_DIR.glob("*_hands.npz"))
    }
    if not clips:
        raise SystemExit("No data/test_output/*_hands.npz; run scripts/extract_hand_landmarks.py")
    return clips


def collect_tracks(clips) -> List[Track]:
    """The tracks of `tracks()` in scripts/measure_hand_round_trip.py, in the same order."""
    found = []
    for name, clip in clips.items():
        for side in range(2):
            frames = np.flatnonzero(clip.present[:, side])
            if len(frames) >= 2:
                world = clip.world[frames, side].astype(np.float64)
                found.append(Track(name, side, frames, world))
    return found


def fit_track(track: Track, backend: str, model, cache_dir: Optional[Path]):
    """The in-sample-shape fits of `ladder()`; read from the cache when it holds them."""
    path = None
    if cache_dir is not None:
        digest = hashlib.sha1(track.world.tobytes()).hexdigest()[:10]
        path = cache_dir / f"{track.clip}_{track.side_name}_{backend}_{digest}.npz"
        if path.exists():
            fits = load_hand_fits(path)
            if len(fits) == len(track.world):
                return fits, False
    valid = np.ones(track.world.shape[:2], dtype=bool)
    betas = fit_hand_shape(track.world, valid, model)
    fits = fit_hands(track.world, valid, model, betas)
    if path is not None:
        save_hand_fits(path, fits)
    return fits, True


def fit_all(tracks: List[Track], models, cache_dir: Optional[Path]) -> None:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        for backend in BACKENDS:
            began = time.time()
            model = models[(backend, track.side_name)]
            fits, fitted = fit_track(track, backend, model, cache_dir)
            track.fits[backend] = fits
            track.rebuilt[backend] = np.stack([reconstruct_hand(f, model) for f in fits])
            track.errors[backend] = landmark_errors_mm(track.rebuilt[backend], track.world)
            how = f"fitted in {time.time() - began:.0f} s" if fitted else "from the cache"
            print(f"  {track.name:<12}{BACKEND_NAMES[backend]:<8}{how}", flush=True)


def check_against_json(tracks: List[Track]) -> float:
    """Print each track's in-sample mean against the JSON's; return the largest difference."""
    report = json.loads(RESULTS_JSON.read_text())
    measured = {(t["clip"], t["side"]): t for t in report["tracks"]}
    print(f"\n  {'track':<14}{'backend':<8}{'frames':>7}{'mean here':>11}{'JSON':>9}{'diff':>10}")
    worst = 0.0
    for track in tracks:
        entry = measured[(track.clip, track.side_name)]
        for backend in BACKENDS:
            row = entry["rows"][f"{backend}: in-sample shape"]
            errors = track.errors[backend]
            mean = float(np.mean(errors[np.isfinite(errors)]))
            difference = abs(mean - row["mean_mm"])
            worst = max(worst, difference)
            assert int(np.isfinite(errors).any(axis=1).sum()) == row["frames"]
            print(
                f"  {track.name:<14}{backend:<8}{row['frames']:>7}{mean:>11.4f}"
                f"{row['mean_mm']:>9.4f}{difference:>10.6f}"
            )
    for backend in BACKENDS:
        pooled = np.concatenate([t.errors[backend].ravel() for t in tracks])
        reference = report["pooled"][f"{backend}: in-sample shape"]["mean_mm"]
        print(f"  pooled {backend}: {np.mean(pooled):.4f} mm, JSON {reference:.4f} mm")
        worst = max(worst, abs(float(np.mean(pooled)) - reference))
    print(f"  largest absolute difference from the JSON: {worst:.2e} mm")
    if worst > JSON_TOLERANCE_MM:
        print(f"  WARNING: more than {JSON_TOLERANCE_MM} mm from the measured values")
    return worst


def read_frames(path: Path, wanted, expected: int):
    """Decode a video; return {frame index: RGB image} for `wanted`, and check the count."""
    capture = cv2.VideoCapture(str(path))
    frames, count = {}, 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if count in wanted:
            frames[count] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        count += 1
    capture.release()
    if count != expected:
        raise SystemExit(f"{path.name}: decoded {count} frames, the hand clip has {expected}")
    missing = set(wanted) - set(frames)
    if missing:
        raise SystemExit(f"{path.name}: frames {sorted(missing)} were not decoded")
    return frames


# ---------------------------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------------------------


def segments(xy: np.ndarray) -> np.ndarray:
    """(20, 2, 2) bone segments of a (21, 2) skeleton."""
    return np.stack([xy[CHILDREN], xy[PARENTS]], axis=1)


def draw_skeleton(ax, xy, colour, width, size, marker="o", zorder=3, **marker_style):
    ax.add_collection(LineCollection(segments(xy), colors=colour, linewidths=width, zorder=zorder))
    ax.plot(
        xy[:, 0], xy[:, 1], marker, ms=size, color=colour, ls="none", zorder=zorder, **marker_style
    )


def runs(mask: np.ndarray):
    """(start, length) of each run of True in a boolean vector."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def raw_label(side: int, agrees: bool) -> str:
    """The detector's own label for an assigned hand: it names the opposite hand when it agrees."""
    return SIDES[1 - side if agrees else side].capitalize()


def save(fig, name: str, out_dir: Path) -> None:
    fig.savefig(out_dir / name, dpi=DPI)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------------------------
# Figure 1: one frame per clip
# ---------------------------------------------------------------------------------------------


def pick_overview_frame(clip) -> int:
    """The frame nearest the clip's middle with both hands assigned (else with one)."""
    middle = (len(clip.present) - 1) / 2.0
    pool = np.flatnonzero(clip.present.all(axis=1))
    if len(pool) == 0:
        pool = np.flatnonzero(clip.present.any(axis=1))
    return int(pool[np.argmin(np.abs(pool - middle))])


def place_tags(ax, hands, width: int, height: int) -> None:
    """
    Write each hand's tag in its side colour, clear of every hand: the higher hand's above the
    hands, the lower hand's below them (a lone hand's goes above unless there is no room).
    """
    every = np.concatenate([xy for _, xy, _ in hands])
    top, bottom = every[:, 1].min(), every[:, 1].max()
    order = np.argsort([xy[:, 1].mean() for _, xy, _ in hands])
    for rank, index in enumerate(order):
        side, xy, text = hands[index]
        centre = float(np.clip(xy[:, 0].mean(), 0.27 * width, 0.73 * width))
        above = rank == 0 and (len(hands) > 1 or top > 0.16 * height)
        y = top - 0.025 * height if above else bottom + 0.025 * height
        ax.text(
            centre,
            y,
            text,
            ha="center",
            va="bottom" if above else "top",
            fontsize=10,
            color="black",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": SIDE_COLOURS[side], "lw": 1.8},
        )


def figure_detection_overview(clips, picks, images, out_dir: Path) -> None:
    fig = Figure(figsize=(16, 10.6), layout="constrained")
    axes = fig.subplots(2, 3).ravel()
    for ax, (name, clip) in zip(axes, clips.items()):
        t = picks[name]
        width, height = clip.width, clip.height
        ax.imshow(images[name][t])
        hands = []
        for side in range(2):
            wrist = clip.pose_image[t, (15, 16)[side], :2] * (width, height)
            ax.scatter(
                *wrist, s=170, facecolors="none", edgecolors="white", linewidths=4.5, zorder=4
            )
            ax.scatter(
                *wrist,
                s=170,
                facecolors="none",
                edgecolors=SIDE_COLOURS[side],
                linewidths=2.2,
                zorder=5,
            )
            if clip.present[t, side]:
                xy = clip.image[t, side, :, :2] * (width, height)
                draw_skeleton(ax, xy, SIDE_COLOURS[side], 2.0, 3.5, mec="white", mew=0.5)
                label = raw_label(side, bool(clip.label_agrees[t, side]))
                hands.append((side, xy, f'{SIDES[side]} · detector said "{label}"'))
        place_tags(ax, hands, width, height)
        ax.set_xlim(-0.5, width - 0.5)  # a pose wrist can lie outside the frame
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_title(f"Clip {name} · {width}×{height} px · frame {t}")
        ax.set_xlabel("Image x (pixels)")
        ax.set_ylabel("Image y (pixels)")
    legend_ax = axes[len(clips)]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [], [], color=SIDE_COLOURS[0], lw=2.5, marker="o", ms=5, label="Left hand, 21 landmarks"
        ),
        Line2D(
            [],
            [],
            color=SIDE_COLOURS[1],
            lw=2.5,
            marker="o",
            ms=5,
            label="Right hand, 21 landmarks",
        ),
        Line2D(
            [],
            [],
            color="0.35",
            marker="o",
            ms=12,
            mfc="none",
            mew=2,
            ls="none",
            label="Wrist found by the body-pose model\n(blue = left wrist, orange = right wrist)",
        ),
    ]
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11.5)
    legend_ax.text(
        0.0,
        0.42,
        "How to read this\n\n"
        "Each detected hand is given a side by which pose wrist it\n"
        "sits nearest, never by the detector's own label. The tag\n"
        "shows the label the detector printed. The detector assumes\n"
        "a mirror image, so on these videos it names the opposite\n"
        'hand: a left hand tagged "Right" is the expected case.',
        transform=legend_ax.transAxes,
        va="top",
        fontsize=11.5,
    )
    fig.suptitle(
        "How MediaPipe finds the hands: one frame from each of the five clips", fontsize=15
    )
    save(fig, "01_detection_overview.png", out_dir)


# ---------------------------------------------------------------------------------------------
# Figure 2: detection timeline
# ---------------------------------------------------------------------------------------------


def clip_counts(clip) -> Dict[str, int]:
    """The per-clip row of the coverage table in docs/V1_2_Findings.md section 5."""
    return {
        "frames": len(clip.present),
        "no hand": int((~clip.present.any(axis=1) & (clip.unassigned == 0)).sum()),
        "left": int(clip.present[:, LEFT].sum()),
        "right": int(clip.present[:, RIGHT].sum()),
        "unassigned": int(clip.unassigned.sum()),
    }


def figure_detection_timeline(clips, out_dir: Path) -> None:
    fig = Figure(figsize=(16, 12.2), layout="constrained")
    axes = fig.subplots(len(clips), 1)
    for ax, (name, clip) in zip(axes, clips.items()):
        count = clip_counts(clip)
        frames = np.arange(count["frames"])
        no_hand = ~clip.present.any(axis=1) & (clip.unassigned == 0)
        for start, length in runs(no_hand):
            ax.axvspan(start, start + length, color=NO_HAND, alpha=0.35, lw=0, zorder=0)
            ax.axvspan(start, start + length, fc="none", ec="0.55", hatch="///", lw=0, zorder=1)
        for row, side in ((1, 0), (0, 1)):
            ax.broken_barh(
                runs(clip.present[:, side]),
                (row - 0.32, 0.64),
                facecolors=SIDE_COLOURS[side],
                zorder=3,
            )
        stray = frames[clip.unassigned > 0]
        ax.plot(
            stray + 0.5,
            np.full(len(stray), 1.65),
            "v",
            ms=9,
            color=UNASSIGNED,
            mec="black",
            mew=0.6,
            zorder=4,
        )
        ax.set_xlim(0, count["frames"])
        ax.set_ylim(-0.55, 1.95)
        ax.set_yticks([1, 0], ["left hand", "right hand"])
        ax.set_xlabel("Frame number")
        ax.set_title(
            f"Clip {name} · {clip.width}×{clip.height} px · {clip.fps:.2f} frames per second: "
            f"{count['frames']} frames, no hand in {count['no hand']}, "
            f"left hand in {count['left']}, right hand in {count['right']}, "
            f"unassigned detections {count['unassigned']}",
            loc="left",
            fontsize=11.5,
        )
    handles = [
        Patch(fc=SIDE_COLOURS[0], label="Left hand found"),
        Patch(fc=SIDE_COLOURS[1], label="Right hand found"),
        Patch(fc=NO_HAND, alpha=0.5, hatch="///", ec="0.55", label="No hand detected in the frame"),
        Line2D(
            [],
            [],
            marker="v",
            ls="none",
            ms=9,
            color=UNASSIGNED,
            mec="black",
            mew=0.6,
            label="A detection given no side (a duplicate, or too far from either pose wrist)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(
        "When MediaPipe sees each hand: frames with a left hand, a right hand, or neither",
        fontsize=15,
    )
    save(fig, "02_detection_timeline.png", out_dir)


# ---------------------------------------------------------------------------------------------
# Figures 3-5: the best, median and worst frame-side
# ---------------------------------------------------------------------------------------------


@dataclass
class Example:
    label: str
    rank: int  # 1 = lowest error among all frame-sides
    track: Track
    k: int  # index into the track

    @property
    def frame(self) -> int:
        return int(self.track.frames[self.k])

    def error(self, backend: str) -> float:
        return float(self.track.mean_error(backend)[self.k])

    @property
    def heading(self) -> str:
        t = self.track
        return f"clip {t.clip}, {t.side_name} hand, frame {self.frame}"

    def hand_size_mm(self):
        """MediaPipe's wrist-to-middle-knuckle length in this frame, and its median in the track."""
        t = self.track
        lengths = 1000.0 * np.linalg.norm(t.world[:, MIDDLE_MCP] - t.world[:, WRIST], axis=1)
        return float(lengths[self.k]), float(np.median(lengths))


def pick_examples(tracks: List[Track]) -> List[Example]:
    """Best, median and worst frame-side by MANO's in-sample per-frame mean error."""
    owner = np.concatenate([[(i, k) for k in range(len(t.frames))] for i, t in enumerate(tracks)])
    mean = np.concatenate([t.mean_error("mano") for t in tracks])
    order = np.argsort(mean, kind="stable")
    total = len(order)
    chosen = (("Best case", 0), ("Median case", (total - 1) // 2), ("Worst case", total - 1))
    return [
        Example(label, rank + 1, tracks[owner[order[rank]][0]], int(owner[order[rank]][1]))
        for label, rank in chosen
    ]


def wrist_relative_mm(points: np.ndarray) -> np.ndarray:
    return 1000.0 * (points - points[WRIST])


def example_points(example: Example) -> Dict[str, np.ndarray]:
    t, k = example.track, example.k
    return {
        "observed": wrist_relative_mm(t.world[k]),
        "mano": wrist_relative_mm(t.rebuilt["mano"][k]),
        "smplx": wrist_relative_mm(t.rebuilt["smplx"][k]),
    }


def draw_view(ax, points: Dict[str, np.ndarray], side: int, horizontal: int, half: float) -> None:
    """
    One view of an example: `horizontal` picks x (0) or z (2); y (1) runs down the page. The
    window is `half` millimetres either side of the drawing's centre, on both axes.
    """
    pick = [horizontal, 1]
    observed, mano, smplx = (points[k][:, pick] for k in ("observed", "mano", "smplx"))
    ax.add_collection(
        LineCollection(np.stack([observed, mano], axis=1), colors=ERROR, linewidths=0.9, zorder=2)
    )
    draw_skeleton(ax, smplx, SMPLX_COLOUR, 1.6, 6.5, marker="^", zorder=3, mfc="white", mew=1.4)
    draw_skeleton(
        ax, mano, SIDE_COLOURS[side], 1.6, 6.0, marker="s", zorder=4, mfc="white", mew=1.4
    )
    draw_skeleton(ax, observed, OBSERVED, 2.2, 4.5, marker="o", zorder=5)
    ax.plot([0], [0], "*", ms=15, color="black", mfc="gold", zorder=6)
    everything = np.concatenate([observed, mano, smplx])
    centre = (everything.min(axis=0) + everything.max(axis=0)) / 2.0
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] + half, centre[1] - half)  # y runs down the image
    ax.set_box_aspect(1)  # equal spans on a square box: millimetres are equal on both axes
    ax.grid(alpha=0.25)


def example_legend_handles() -> list:
    both_sides = (
        Line2D([], [], color=SIDE_COLOURS[0], lw=1.6, marker="s", mfc="white", mew=1.4, ms=6),
        Line2D([], [], color=SIDE_COLOURS[1], lw=1.6, marker="s", mfc="white", mew=1.4, ms=6),
    )
    return [
        Line2D([], [], color=OBSERVED, lw=2.2, marker="o", ms=4.5, label="Observed by MediaPipe"),
        (
            both_sides,
            "MANO rebuilt from its fitted parameters (blue = left hand, orange = right hand)",
        ),
        Line2D(
            [],
            [],
            color=SMPLX_COLOUR,
            lw=1.6,
            marker="^",
            mfc="white",
            mew=1.4,
            ms=6.5,
            label="SMPL-X rebuilt from its fitted parameters",
        ),
        Line2D([], [], color=ERROR, lw=1.2, label="Error: observed landmark to its MANO landmark"),
        Line2D(
            [], [], marker="*", ms=14, color="black", mfc="gold", ls="none", label="Wrist (origin)"
        ),
    ]


def add_example_legend(fig) -> None:
    handles, labels = [], []
    for item in example_legend_handles():
        if isinstance(item, tuple):
            handles.append(item[0])
            labels.append(item[1])
        else:
            handles.append(item)
            labels.append(item.get_label())
    fig.legend(
        handles,
        labels,
        loc="center",
        ncols=2,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    )


def figure_round_trip_examples(examples: List[Example], total: int, out_dir: Path) -> None:
    fig = Figure(figsize=(13, 18), layout="constrained")
    rows = fig.subfigures(4, 1, height_ratios=[1, 1, 1, 0.14])
    for sub, example in zip(rows[:3], examples):
        points = example_points(example)
        axes = sub.subplots(1, 2)
        side = example.track.side
        half = 0.54 * np.ptp(np.concatenate(list(points.values())), axis=0).max()
        draw_view(axes[0], points, side, 0, half)
        draw_view(axes[1], points, side, 2, half)
        axes[0].set_title("Front: as the camera sees it", fontsize=12)
        axes[1].set_title("Side: depth away from the camera against height", fontsize=12)
        axes[0].set_xlabel("Sideways, to the image right (mm from the wrist)")
        axes[1].set_xlabel("Depth, away from the camera (mm from the wrist)")
        for ax in axes:
            ax.set_ylabel("Down the image (mm from the wrist)")
        size, typical = example.hand_size_mm()
        sub.suptitle(
            f"{example.label} (rank {example.rank} of {total}): {example.heading}\n"
            f"mean error over its 20 landmarks: MANO {example.error('mano'):.1f} mm, "
            f"SMPL-X {example.error('smplx'):.1f} mm\n"
            f"MediaPipe's wrist-to-middle-knuckle length: {size:.0f} mm here, "
            f"{typical:.0f} mm typical for this track",
            fontsize=12.5,
        )
    add_example_legend(rows[3])
    fig.suptitle(
        "The round trip on three hands: what MediaPipe saw and what the fitted models give back",
        fontsize=15,
    )
    save(fig, "03_round_trip_examples.png", out_dir)


# ---------------------------------------------------------------------------------------------
# Figure 4: the MANO mesh of the median example
# ---------------------------------------------------------------------------------------------


def mano_mesh(fit):
    """Vertices, joints and faces of MANO at a fit's parameters, set up like `HandModel`'s."""
    import smplx

    model = smplx.create(
        MODEL_DIR,
        model_type="mano",
        is_rhand=fit.side == "right",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=len(fit.betas),
        ext="npz",
        dtype=torch.float64,
    )

    def as_batch(values):
        return torch.as_tensor(values, dtype=torch.float64).reshape(1, -1)

    with torch.no_grad():
        out = model(
            global_orient=as_batch(fit.wrist_orient),
            hand_pose=as_batch(fit.hand_pose),
            betas=as_batch(fit.betas),
            transl=as_batch(fit.transl),
        )
    return out.vertices[0].numpy(), out.joints[0].numpy(), np.asarray(model.faces, dtype=int)


def mesh_landmarks(vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """The 21 MediaPipe points read off the mesh: wrist and finger joints, tip vertices."""
    points = np.zeros((21, 3))
    points[WRIST] = joints[0]
    for slot, landmark in enumerate(POSE_SLOT_LANDMARKS):
        points[landmark] = joints[1 + slot]
    points[list(TIPS)] = vertices[list(MANO_TIP_VERTICES)]
    return points


def draw_mesh(ax, vertices, faces, points, observed, side, view) -> None:
    """Mesh (light shading), observed landmarks and reconstructed points, in mm from the wrist."""

    # Plot axes: x as in the image, y = depth (away from the camera), z = up (minus MediaPipe's y).
    def to_plot(xyz):
        return np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1)

    triangles = to_plot(vertices)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.array([-0.4, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(normals @ light)
    colours = np.clip(lighten(SIDE_COLOURS[side], 0.35)[None, :] * shade[:, None], 0, 1)
    ax.add_collection3d(
        Poly3DCollection(
            triangles, facecolors=colours, edgecolors="none", linewidths=0, alpha=0.55, zorder=1
        )
    )
    rebuilt, seen = to_plot(points), to_plot(observed)
    for child, parent in zip(CHILDREN, PARENTS):
        ax.plot(*zip(rebuilt[child], rebuilt[parent]), color=SIDE_COLOURS[side], lw=1.4, zorder=4)
    ax.scatter(
        *rebuilt.T,
        s=42,
        marker="D",
        c=SIDE_COLOURS[side],
        edgecolors="black",
        linewidths=0.7,
        depthshade=False,
        zorder=5,
    )
    ax.scatter(*seen.T, s=22, marker="o", c=OBSERVED, depthshade=False, zorder=6)
    every = np.concatenate([to_plot(vertices), rebuilt, seen])
    low, high = every.min(axis=0), every.max(axis=0)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(high - low, zoom=1.0 if view[0] == 0 else 0.85)
    ax.set_proj_type("ortho")  # no perspective, so the camera view is the view the video had
    ax.computed_zorder = False
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("Sideways, to the image right (mm)", labelpad=4)
    ax.set_zlabel("Up the image (mm)", labelpad=10)
    if view[0] == 0:  # looking straight along the depth axis: it is end-on, so drop it
        ax.set_yticks([])
    else:
        ax.set_ylabel("Depth, away from the camera (mm)", labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(4))


def figure_mesh(example: Example, check_m: float, vertices, faces, points, out_dir: Path) -> None:
    track = example.track
    observed = wrist_relative_mm(track.world[example.k])
    reconstructed = 1000.0 * (points - points[WRIST])
    shifted = 1000.0 * (vertices - points[WRIST])
    fig = Figure(figsize=(15, 9.4), layout="constrained")
    body, footer = fig.subfigures(2, 1, height_ratios=[1, 0.1])
    views = (
        ("From the camera (depth points into the page)", (0, -90)),
        ("From above and to one side", (28, -40)),
    )
    for index, (title, view) in enumerate(views):
        ax = body.add_subplot(1, 2, index + 1, projection="3d")
        draw_mesh(ax, shifted, faces, reconstructed, observed, track.side, view)
        ax.set_title(title, fontsize=12.5)
    handles = [
        Patch(
            fc=lighten(SIDE_COLOURS[track.side], 0.35),
            alpha=0.7,
            label="MANO mesh (778 vertices, shaded)",
        ),
        Line2D(
            [],
            [],
            color=SIDE_COLOURS[track.side],
            marker="D",
            mec="black",
            mew=0.7,
            ms=7,
            label="The 21 landmarks rebuilt from the fitted parameters",
        ),
        Line2D(
            [],
            [],
            color=OBSERVED,
            marker="o",
            ms=5,
            ls="none",
            label="The 21 landmarks MediaPipe observed",
        ),
    ]
    footer.legend(handles=handles, loc="center", ncols=3, frameon=False)
    fig.suptitle(
        f"The fitted MANO hand behind the median example: {example.heading}, "
        f"mean error {example.error('mano'):.1f} mm\n"
        f"The mesh's joints and fingertip vertices equal the rebuilt landmarks "
        f"(largest difference {check_m:.1e} m)",
        fontsize=13.5,
    )
    save(fig, "04_round_trip_mesh.png", out_dir)


# ---------------------------------------------------------------------------------------------
# Figure 5: the examples drawn over the video
# ---------------------------------------------------------------------------------------------


def fit_similarity(world_xy: np.ndarray, image_xy: np.ndarray):
    """Scale s and shift t (no rotation) minimising |s * world + t - image|^2 by least squares."""
    world_mean, image_mean = world_xy.mean(axis=0), image_xy.mean(axis=0)
    centred_world, centred_image = world_xy - world_mean, image_xy - image_mean
    scale = float((centred_world * centred_image).sum() / (centred_world**2).sum())
    return scale, image_mean - scale * world_mean


def crop_limits(points: np.ndarray, width: int, height: int):
    """A square window around the points, 1.4 times their extent, kept inside the frame."""
    low, high = points.min(axis=0), points.max(axis=0)
    half = min(0.7 * (high - low).max() + 8.0, 0.5 * min(width, height))
    centre = np.clip((low + high) / 2.0, [half, half], [width - half, height - half])
    return centre - half, centre + half


def overlay_geometry(example: Example, clips):
    """The three sets of image points of one example, and the mapping that makes them."""
    track, k = example.track, example.k
    clip = clips[track.clip]
    pixels = np.array([clip.width, clip.height], dtype=np.float64)
    image = clip.image[example.frame, track.side, :, :2].astype(np.float64) * pixels
    scale, shift = fit_similarity(track.world[k][:, :2], image)
    observed = scale * track.world[k][:, :2] + shift
    rebuilt = scale * track.rebuilt["mano"][k][:, :2] + shift
    rms = float(np.sqrt(((observed - image) ** 2).sum(axis=1).mean()))
    return image, observed, rebuilt, scale, rms


def figure_overlay(examples: List[Example], clips, images, out_dir: Path) -> None:
    fig = Figure(figsize=(16, 8.8), layout="constrained")
    body, footer = fig.subfigures(2, 1, height_ratios=[1, 0.24])
    axes = body.subplots(1, 3)
    for ax, example in zip(axes, examples):
        clip = clips[example.track.clip]
        side = example.track.side
        image, observed, rebuilt, scale, rms = overlay_geometry(example, clips)
        ax.imshow(images[example.track.clip][example.frame])
        low, high = crop_limits(np.concatenate([image, observed, rebuilt]), clip.width, clip.height)
        draw_skeleton(ax, rebuilt, "black", 4.2, 0.0, zorder=3)
        draw_skeleton(ax, observed, "white", 3.4, 0.0, zorder=4)
        draw_skeleton(
            ax, rebuilt, SIDE_COLOURS[side], 2.0, 6.5, marker="o", zorder=5, mec="black", mew=0.8
        )
        draw_skeleton(ax, observed, "white", 1.4, 5.0, marker="o", zorder=6, mec="black", mew=0.8)
        ax.plot(*image.T, "D", ms=4.5, color=OBSERVED, mec="white", mew=0.8, ls="none", zorder=7)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(high[1], low[1])
        ax.set_title(
            f"{example.label}: {example.heading}\nMANO mean error {example.error('mano'):.1f} mm\n"
            f"mapping: {scale / 1000.0:.2f} px per mm, misfit {rms:.1f} px rms",
            fontsize=11,
        )
        ax.set_xlabel("Image x (pixels)")
        ax.set_ylabel("Image y (pixels)")
    side_pair = (
        Line2D([], [], color=SIDE_COLOURS[0], lw=2, marker="o", mec="black", ms=6.5),
        Line2D([], [], color=SIDE_COLOURS[1], lw=2, marker="o", mec="black", ms=6.5),
    )
    handles = [
        Line2D([], [], color="white", lw=3, marker="o", mec="black", ms=5),
        side_pair,
        Line2D([], [], color=OBSERVED, marker="D", mec="white", mew=0.8, ls="none", ms=5),
    ]
    labels = [
        "Observed world landmarks (MediaPipe), through the mapping",
        "MANO rebuilt from its parameters, through the same mapping (blue = left, orange = right)",
        "MediaPipe's own image landmarks",
    ]
    footer.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    )
    footer.text(
        0.5,
        0.02,
        "The gap between white and coloured is the round-trip error in x and y. The mapping, a "
        "scale and a shift fitted from MediaPipe's own\nlandmarks and applied unchanged to the "
        "reconstruction, is a visual aid, not a camera model; depth error is not shown. "
        "Misfit: how far the\nmapped observed landmarks fall from MediaPipe's own image "
        "landmarks (diamonds), in pixels, root mean square over the 21.",
        ha="center",
        va="bottom",
        fontsize=11,
        style="italic",
    )
    fig.suptitle(
        "The round trip drawn on the video: how far the rebuilt hand lands from the observed one",
        fontsize=15,
    )
    save(fig, "05_overlay_on_video.png", out_dir)


# ---------------------------------------------------------------------------------------------
# Figure 6: hand size drift
# ---------------------------------------------------------------------------------------------


def with_gaps(x: np.ndarray, y: np.ndarray, frames: np.ndarray):
    """Insert NaN where frames are missing so a line breaks instead of bridging the gap."""
    gaps = np.flatnonzero(np.diff(frames) > 1) + 1
    return np.insert(x.astype(float), gaps, np.nan), np.insert(y.astype(float), gaps, np.nan)


def figure_size_drift(tracks: List[Track], clips, ratios, out_dir: Path) -> Dict[str, float]:
    spans = {
        t.name: 1000.0 * np.linalg.norm(t.world[:, MIDDLE_MCP] - t.world[:, WRIST], axis=1)
        for t in tracks
    }
    fitted = {}
    for t in tracks:
        length = 1000.0 * np.linalg.norm(
            t.rebuilt["mano"][:, MIDDLE_MCP] - t.rebuilt["mano"][:, WRIST], axis=1
        )
        fitted[t.name] = length
    length_axis = (
        min(min(v.min() for v in spans.values()), min(v.min() for v in fitted.values())) - 6,
        max(max(v.max() for v in spans.values()), max(v.max() for v in fitted.values())) + 6,
    )
    error_top = 1.08 * max(t.mean_error("mano").max() for t in tracks)
    fig = Figure(figsize=(16, 12.6), layout="constrained")
    for ax, track in zip(fig.subplots(3, 3).ravel(), tracks):
        seconds = track.frames / clips[track.clip].fps
        x, span = with_gaps(seconds, spans[track.name], track.frames)
        ax.plot(x, span, "-o", color=OBSERVED, lw=1.8, ms=3, zorder=3)
        ax.axhline(
            float(np.median(fitted[track.name])), color=SIDE_COLOURS[track.side], lw=3.0, zorder=2
        )
        ax.set_ylim(*length_axis)
        ax.set_xlim(0, len(clips[track.clip].present) / clips[track.clip].fps)
        ax.set_xlabel("Time in the clip (seconds)")
        ax.set_ylabel("Wrist to middle knuckle (mm)")
        ax.set_title(f"{track.name} · half_span_ratio {ratios[track.name]:.2f}", fontsize=12.5)
        twin = ax.twinx()
        twin.spines["right"].set_visible(True)
        x, error = with_gaps(seconds, track.mean_error("mano"), track.frames)
        twin.plot(x, error, "--o", color=ERROR, lw=1.8, ms=3, zorder=1)
        twin.set_ylim(0, error_top)
        twin.set_ylabel("MANO error per frame (mm)", color=ERROR)
        twin.tick_params(axis="y", colors=ERROR)
        twin.spines["right"].set_color(ERROR)
    side_pair = (
        Line2D([], [], color=SIDE_COLOURS[0], lw=3),
        Line2D([], [], color=SIDE_COLOURS[1], lw=3),
    )
    handles = [
        Line2D([], [], color=OBSERVED, lw=1.8, marker="o", ms=3),
        side_pair,
        Line2D([], [], color=ERROR, lw=1.8, ls="--", marker="o", ms=3),
    ]
    labels = [
        "MediaPipe's wrist-to-middle-knuckle length in each frame (left axis)",
        "The fitted MANO hand's length, one per track (left axis; blue = left, orange = right)",
        "MANO round-trip error in each frame, mean over 20 landmarks (right axis)",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=1,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    )
    fig.suptitle(
        "Why the target is missed: MediaPipe's hand size drifts, but a fitted hand keeps one size",
        fontsize=15,
    )
    save(fig, "06_hand_size_drift.png", out_dir)
    return {name: float(np.ptp(v)) for name, v in fitted.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, help="save the fits here and reuse them later")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where the PNGs go")
    args = parser.parse_args()
    began = time.time()
    set_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips()
    tracks = collect_tracks(clips)
    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR) for b in BACKENDS for s in SIDES}
    print(f"Fitting {len(tracks)} tracks on {len(BACKENDS)} backends (in-sample shape)")
    fit_all(tracks, models, args.cache_dir)
    print("\nCheck against data/test_output/hand_round_trip.json (in-sample shape rows)")
    check_against_json(tracks)

    print("\nChoosing frames and decoding the videos")
    picks = {name: pick_overview_frame(clip) for name, clip in clips.items()}
    examples = pick_examples(tracks)
    wanted = {name: {picks[name]} for name in clips}
    for example in examples:
        wanted[example.track.clip].add(example.frame)
    images = {
        name: read_frames(VIDEO_DIR / f"{name}.mp4", wanted[name], len(clip.present))
        for name, clip in clips.items()
    }
    for name, clip in clips.items():
        print(f"  clip {name}: {len(clip.present)} frames decoded; overview frame {picks[name]}")
    total = sum(len(t.frames) for t in tracks)
    for example in examples:
        print(
            f"  {example.label:<12}rank {example.rank:>3} of {total}: {example.heading}; error "
            f"MANO {example.error('mano'):.3f} mm, SMPL-X {example.error('smplx'):.3f} mm"
        )

    print("\nDrawing")
    figure_detection_overview(clips, picks, images, args.out_dir)
    figure_detection_timeline(clips, args.out_dir)
    figure_round_trip_examples(examples, total, args.out_dir)

    median = examples[1]
    fit = median.track.fits["mano"][median.k]
    vertices, joints, faces = mano_mesh(fit)
    reference = reconstruct_hand(fit, models[("mano", median.track.side_name)])
    difference = float(np.abs(mesh_landmarks(vertices, joints) - reference).max())
    print(f"  mesh joints and tip vertices vs reconstruct_hand: max difference {difference:.2e} m")
    if difference > MESH_TOLERANCE_M:
        raise SystemExit(f"The mesh differs from reconstruct_hand by {difference:.2e} m")
    figure_mesh(median, difference, vertices, faces, reference, args.out_dir)

    figure_overlay(examples, clips, images, args.out_dir)
    report = json.loads(RESULTS_JSON.read_text())
    ratios = {f"{t['clip']} {t['side']}": t["half_span_ratio"] for t in report["tracks"]}
    spread = figure_size_drift(tracks, clips, ratios, args.out_dir)
    print(
        f"  fitted MANO hand length varies by at most {max(spread.values()):.2e} mm within a track"
    )
    print(f"\nDone in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
