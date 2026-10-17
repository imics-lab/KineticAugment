#!/usr/bin/env python3
"""
Visualise the body + hand round trip of one clip, stitched into one figure.

Body:  MediaPipe pose world landmarks -> SMPL-X (fit_shape, fit_body) -> reconstruct_body
Hands: MediaPipe hand world landmarks -> MANO (fit_hand_shape, fit_hands) -> reconstruct_hand

MediaPipe reports each hand in its own metric frame (origin near the knuckles), with
axes aligned to the pose's camera-aligned world frame, so a hand is stitched onto the
body by translating its wrist onto the body's wrist: the observed hand onto the
observed pose wrist, the reconstructed hand onto the reconstructed body wrist.

The video has four panels and an error strip:

    source + MediaPipe 2-D overlay | front view | side view | SMPL-X body figure
    BLUE  = MediaPipe input (pose arms + both hands)
    GREEN = round trip (SMPL-X arms + MANO hands), rebuilt from parameters alone

The body figure is the SMPL-X mesh posed by the round-trip parameters: the body fit's
global_orient / body_pose / betas, each hand's MANO finger pose as SMPL-X hand pose,
and a wrist rotation that turns the SMPL-X hand onto the MANO reconstruction's palm.
Joints the round trip does not model (spine, neck, head, legs) stay at rest. A hand
the detector missed in a frame is drawn flat and the panel says so.

Usage (from the repo root):
    .venv/bin/python scripts/view_body_hand_round_trip.py --clip 01988
    .venv/bin/python scripts/view_body_hand_round_trip.py --clip 01988 --landmark-dir <dir>
    .venv/bin/python scripts/view_body_hand_round_trip.py --clip 08917 \
        --landmark-dir data/round_trip_eval --video <path>/08917.mp4 --output-dir <dir>

Reads <landmark-dir>/<clip>_world.npz and <clip>_hands.npz (from
extract_world_landmarks.py and extract_hand_landmarks.py).
Output: data/test_output/videos/<clip>_body_hand_round_trip.mp4
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
from matplotlib.collections import PolyCollection  # noqa: E402

from view_round_trip import (  # noqa: E402
    ARM_BONES, CONTEXT_BONES, VideoSink, frame_errors, read_frames, round_trip_clip,
)
from kinetic_augment.body_model.assembly import attach_hands, body_rotations, pose_mesh  # noqa: E402
from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape, fit_hands, landmark_errors_mm, reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import (  # noqa: E402
    HAND_PARENTS, NUM_HAND_LANDMARKS, POSE_SLOT_LANDMARKS, SMPLX_FIRST_FINGER_JOINT, WRIST,
    HandModel,
)
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

LANDMARK_DIR = Path("data/test_output")
OUTPUT_DIR = Path("data/test_output/videos")
VISIBILITY_THRESHOLD = 0.5  # matches measure_round_trip.py
POSE_WRIST = {"left": 15, "right": 16}
SMPLX_WRIST = {"left": 20, "right": 21}
HAND_BONES = [(int(p), c) for c, p in enumerate(HAND_PARENTS) if p >= 0]
POSE_2D_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
INPUT, RECON = "#1f5fbf", "#2ca02c"


def load_observations(landmark_dir: Path, clip: str):
    data = np.load(landmark_dir / f"{clip}_world.npz")
    pose = data["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (data["visibility"] >= VISIBILITY_THRESHOLD)
    return pose, ok, load_hand_clip(landmark_dir / f"{clip}_hands.npz")


def hand_round_trip(hands, model_dir="models"):
    """MANO fit and reconstruction of each side; per-side fits, landmarks (T, 21, 3), errors."""
    fits, recon, errors = {}, {}, {}
    for s, side in enumerate(SIDES):
        model = HandModel("mano", side, model_dir=model_dir)
        observed = hands.world[:, s].astype(np.float64)
        valid = np.isfinite(observed).all(axis=-1)
        betas = fit_hand_shape(observed, valid, model)
        fits[side] = fit_hands(observed, valid, model, betas)
        recon[side] = np.stack([reconstruct_hand(f, model) for f in fits[side]])
        errors[side] = np.nanmean(landmark_errors_mm(recon[side], observed), axis=-1)
        print(f"  {side} hand: {int(valid.all(axis=1).sum())} frames, MANO round trip "
              f"median {np.nanmedian(errors[side]):.1f} mm (wrist-relative, landmarks 1-20)")
    return fits, recon, errors


def anchored(hand, wrist):
    """Hand landmarks translated so their wrist sits on `wrist`; NaN if either is missing."""
    if not (np.isfinite(hand).all() and np.isfinite(wrist).all()):
        return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
    return hand - hand[WRIST] + wrist


class BodyFigure:
    """Poses the full SMPL-X mesh from the body and hand round-trip parameters."""

    def __init__(self, skeleton: SMPLXSkeleton, model_dir="models"):
        self.skeleton = skeleton
        self.model = skeleton.model
        self.faces = self.model.faces.astype(np.int64)
        self.hand_models = {side: HandModel("smplx", side, model_dir=model_dir) for side in SIDES}

    def pose(self, body_fit, hand_fits, hand_recon):
        """
        Vertices (V, 3) in MediaPipe world, model joints (J, 3), and which hands were posed.
        """
        rotations, posed = attach_hands(body_rotations(body_fit), self.skeleton.parents,
                                        hand_fits, hand_recon, self.hand_models)
        mesh = pose_mesh(self.skeleton, rotations, body_fit.betas, body_fit.transl)
        return mesh.vertices, mesh.joints, posed


def _project(points, view):
    """MediaPipe world is y-down; plot with up = up. Front: (x, -y). Side: (z, -y)."""
    if view == "front":
        return points[..., 0], -points[..., 1]
    return points[..., 2], -points[..., 1]


def draw_bones(ax, points, bones, view, color, lw, alpha=1.0, label=None, size=0):
    first = True
    for a, b in bones:
        if np.isfinite(points[[a, b]]).all():
            xs, ys = _project(points[[a, b]], view)
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label if first else None,
                    solid_capstyle="round")
            first = False
    if size:
        pts = points[np.isfinite(points).all(axis=1)]
        xs, ys = _project(pts, view)
        ax.scatter(xs, ys, s=size, color=color, alpha=alpha, zorder=3, linewidths=0)


def draw_mesh(ax, vertices, faces, light=(0.3, -0.5, -1.0), color=(0.78, 0.66, 0.58)):
    """Orthographic front view of a mesh: back faces culled, painter's order, Lambert shade."""
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    front = normals[:, 2] < 0  # the camera looks along +z
    tri, normals = tri[front], normals[front]
    order = np.argsort(-tri[:, :, 2].mean(axis=1))  # farthest first
    tri, normals = tri[order], normals[order]
    light = np.asarray(light) / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normals @ light, 0.0, 1.0)
    colors = np.clip(shade[:, None] * np.asarray(color)[None], 0, 1)
    polys = np.stack([tri[..., 0], -tri[..., 1]], axis=-1)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=colors, linewidths=0.2))


def overlay_mediapipe(frame, hands, t):
    """The source frame with MediaPipe's own 2-D pose (upper body) and hand landmarks."""
    image = frame.copy()
    h, w = image.shape[:2]
    thick = max(1, round(h / 240))

    def px(p):
        return int(round(p[0] * w)), int(round(p[1] * h))

    pose = hands.pose_image[t]
    for a, b in POSE_2D_BONES:
        if np.isfinite(pose[[a, b], :2]).all() and min(hands.pose_visibility[t, [a, b]]) > 0.5:
            cv2.line(image, px(pose[a]), px(pose[b]), (90, 140, 255), thick + 1, cv2.LINE_AA)
    for s, color in enumerate(((255, 120, 40), (40, 200, 255))):
        points = hands.image[t, s]
        if not np.isfinite(points).all():
            continue
        for a, b in HAND_BONES:
            cv2.line(image, px(points[a]), px(points[b]), color, thick, cv2.LINE_AA)
        for p in points:
            cv2.circle(image, px(p), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def limits(clouds, view, pad=1.15):
    pts = np.concatenate([c.reshape(-1, 3) for c in clouds])
    pts = pts[np.isfinite(pts).all(axis=1)]
    xs, ys = _project(pts, view)
    lo = np.array([np.percentile(xs, 0.5), np.percentile(ys, 0.5)])
    hi = np.array([np.percentile(xs, 99.5), np.percentile(ys, 99.5)])
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * pad + 0.03
    return centre, half


def set_view(ax, lim, title):
    (cx, cy), half = lim
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def render(clip, landmark_dir: Path, out_dir: Path, model_dir="models", video_path=None):
    pose, ok, hands = load_observations(landmark_dir, clip)
    frames, fps = read_frames(clip, len(pose), video_path)
    skeleton = SMPLXSkeleton(model_path=model_dir)

    print(f"Clip {clip}: {len(pose)} frames")
    _, body_fits, body_recon = round_trip_clip(pose, ok, skeleton)
    body_err = np.array([np.mean(list(e.values())) if e else np.nan
                         for e in (frame_errors(pose[t], ok[t], body_recon[t])
                                   for t in range(len(pose)))])
    print(f"  body: {sum(f is not None for f in body_fits)}/{len(pose)} frames fitted, "
          f"SMPL-X round trip median {np.nanmedian(body_err):.2f}% of shoulder width")
    hand_fits, hand_recon, hand_err = hand_round_trip(hands, model_dir)

    # Stitch: hands onto the body's wrists, input onto input, round trip onto round trip.
    obs_pose = np.where(ok[..., None], pose, np.nan)
    stitched_in, stitched_rt = {}, {}
    for s, side in enumerate(SIDES):
        wrist = POSE_WRIST[side]
        stitched_in[side] = np.stack([anchored(hands.world[t, s], obs_pose[t, wrist])
                                      for t in range(len(pose))])
        stitched_rt[side] = np.stack([anchored(hand_recon[side][t], body_recon[t, wrist])
                                      for t in range(len(pose))])

    figure = BodyFigure(skeleton, model_dir)
    meshes, mesh_hands, pelvis, wrist_gap, finger_gap = [], [], [], [], []
    for t in range(len(pose)):
        if body_fits[t] is None:
            meshes.append(None)
            mesh_hands.append({side: False for side in SIDES})
            continue
        vertices, joints, posed = figure.pose(
            body_fits[t], {side: hand_fits[side][t] for side in SIDES},
            {side: hand_recon[side][t] for side in SIDES})
        meshes.append(vertices)
        mesh_hands.append(posed)
        pelvis.append(joints[0])
        for side in SIDES:  # the mesh's wrist joints should sit on the body round trip's
            gap = np.linalg.norm(joints[SMPLX_WRIST[side]] - body_recon[t, POSE_WRIST[side]])
            if np.isfinite(gap):
                wrist_gap.append(gap)
            if posed[side]:  # and its finger joints on the stitched MANO reconstruction's
                first = SMPLX_FIRST_FINGER_JOINT[side]
                mesh_fingers = joints[first:first + len(POSE_SLOT_LANDMARKS)]
                rt_fingers = stitched_rt[side][t][list(POSE_SLOT_LANDMARKS)]
                gaps = np.linalg.norm(mesh_fingers - rt_fingers, axis=1)
                finger_gap.extend(gaps[np.isfinite(gaps)])  # NaN: the arm was not placed
    print(f"  body figure: mesh wrist vs round-trip wrist max {1000 * max(wrist_gap):.3f} mm; "
          f"mesh finger joints vs stitched MANO hand median {1000 * np.median(finger_gap):.1f}"
          f" mm, p95 {1000 * np.percentile(finger_gap, 95):.1f} mm")

    arm_clouds = [obs_pose[:, 11:17], body_recon[:, 11:17], obs_pose[:, 23:25]]
    hand_clouds = list(stitched_in.values()) + list(stitched_rt.values())
    front = limits(arm_clouds + hand_clouds, "front")
    side_lim = limits(arm_clouds + hand_clouds, "side")
    hip_line = np.median(np.array(pelvis)[:, 1]) + 0.12  # y is down: keep the upper body
    mesh_pts = [m[m[:, 1] < hip_line] for m in meshes if m is not None]
    mesh_lim = limits([np.concatenate(mesh_pts)], "front", pad=1.05)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_body_hand_round_trip.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(19.2, 6.6), dpi=100)
    steps = np.arange(1, len(pose) + 1)

    for t in range(len(pose)):
        fig.clf()
        grid = fig.add_gridspec(2, 4, height_ratios=[4.2, 1], hspace=0.22, wspace=0.05,
                                left=0.02, right=0.98, top=0.94, bottom=0.08)
        ax_vid, ax_front, ax_side, ax_mesh = (fig.add_subplot(grid[0, i]) for i in range(4))
        ax_err = fig.add_subplot(grid[1, :])

        ax_vid.imshow(overlay_mediapipe(frames[t], hands, t))
        ax_vid.set_title(f"clip {clip}  frame {t + 1}/{len(pose)}  (MediaPipe 2-D)",
                         fontsize=10)
        ax_vid.axis("off")

        for ax, view, lim, title in ((ax_front, "front", front, "front: input vs round trip"),
                                     (ax_side, "side", side_lim, "side (depth →)")):
            draw_bones(ax, obs_pose[t], CONTEXT_BONES, view, "#7f9cc4", 1.2, alpha=0.6)
            draw_bones(ax, obs_pose[t], ARM_BONES, view, INPUT, 3.0, alpha=0.55,
                       label="MediaPipe (pose + hands)")
            for side in SIDES:
                draw_bones(ax, stitched_in[side][t], HAND_BONES, view, INPUT, 1.6, alpha=0.55,
                           size=6)
            draw_bones(ax, body_recon[t], ARM_BONES, view, RECON, 2.0,
                       label="round trip (SMPL-X arms + MANO hands)")
            for side in SIDES:
                draw_bones(ax, stitched_rt[side][t], HAND_BONES, view, RECON, 1.1, size=4)
            set_view(ax, lim, title)
        ax_front.legend(loc="lower left", fontsize=8, framealpha=0.85)

        lines = []
        lines.append("body : no fit" if body_fits[t] is None else
                     f"body : {body_err[t]:4.1f}% sh.w.")
        for side in SIDES:
            e = hand_err[side][t]
            lines.append(f"{side:<5}: " + ("not detected" if np.isnan(e) else f"{e:4.1f} mm"))
        ax_side.text(0.98, 0.02, "\n".join(lines), transform=ax_side.transAxes, ha="right",
                     va="bottom", fontsize=8, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        if meshes[t] is not None:
            draw_mesh(ax_mesh, meshes[t], figure.faces)
            missing = [side for side in SIDES if not mesh_hands[t][side]]
            if missing:
                ax_mesh.text(0.02, 0.98, f"{' + '.join(missing)} hand not detected: flat",
                             transform=ax_mesh.transAxes, fontsize=8, color="#a33", va="top",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        else:
            ax_mesh.text(0.5, 0.5, "no body fit", transform=ax_mesh.transAxes, ha="center")
        set_view(ax_mesh, mesh_lim, "SMPL-X figure from round-trip parameters")

        ax_err.plot(steps, body_err, color="#555", lw=1.2, label="body (% shoulder width)")
        ax_err.plot(steps, hand_err["left"], color="#e8762b", lw=1.0, label="left hand (mm)")
        ax_err.plot(steps, hand_err["right"], color="#2bb0e8", lw=1.0, label="right hand (mm)")
        ax_err.axvline(t + 1, color="#d62728", lw=1)
        ax_err.set_xlim(1, len(pose))
        top = np.nanmax(np.concatenate([body_err, hand_err["left"], hand_err["right"]]))
        ax_err.set_ylim(0, max(top * 1.05, 1.0))
        ax_err.set_ylabel("round-trip\nerror", fontsize=8)
        ax_err.set_xlabel("frame", fontsize=8)
        ax_err.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.85)
        ax_err.tick_params(labelsize=7)
        sink.add(fig)

    sink.close()
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True, help="clip id, e.g. 01988")
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--video", type=Path,
                        help="source video (default: data/sign_videos/<clip>.mp4)")
    args = parser.parse_args()
    render(args.clip, args.landmark_dir, args.output_dir, video_path=args.video)


if __name__ == "__main__":
    main()
