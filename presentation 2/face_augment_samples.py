#!/usr/bin/env python3
"""
Sample augmented clips with the face attached, for presentation.

Fits body + hands + head/neck/jaw with the V1.3 face pipeline (run_clip), carries MediaPipe's
478 face points on the fitted head (face_stream_local + head_transform), then applies the
"safe for every sign" augmentations at the evidence-based ranges:

    view turn      +20° about the vertical axis through the shoulder centre (Spoken2Sign ±20°)
    mirror         x reflected across the shoulder centre, hands swapped (left-handed signer)
    scale          ×0.85 about the shoulder centre (camera farther away)
    shift          whole scene moved (+0.3, +0.1) shoulder widths
    speed          ×1.25 uniform re-timing (clip ends early, then holds)
    arm joints     shoulders, elbows, wrists each picked with p = 0.3 (at least two), turned 4°
                   about a random axis (the edge of SPOTER's ±4°), re-posed through SMPL-X

Run from the feat/face-v1.3 worktree root:
    PYTHONPATH=src:scripts .venv/bin/python <this file> --clips 63229 17083 --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np

from view_body_hand_round_trip import draw_bones, draw_mesh, limits, overlay_mediapipe, set_view
import view_round_trip
from view_round_trip import VideoSink, read_frames
from view_face_round_trip import face_contours, overlay_face
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.assembly import pose_mesh
from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip
from kinetic_augment.body_model.hand_observations import load_hand_clip

LANDMARK_DIR = Path("data/round_trip_eval")
view_round_trip.VIDEO_DIR = Path.home() / "Downloads" / "raw_videos"  # source footage
SEED = 20261013
YAW_DEG, SCALE, SHIFT_SW, SPEED = 20.0, 0.85, (0.3, 0.1, 0.0), 1.25
ARM_JOINTS = {"L shoulder": 16, "R shoulder": 17, "L elbow": 18, "R elbow": 19,
              "L wrist": 20, "R wrist": 21}
ARM_P, ARM_MAX_DEG = 0.3, 4.0
SMPLX_ARM_BONES = [(16, 18), (18, 20), (17, 19), (19, 21), (16, 17)]
FACE_COLOURS = {"eye": "#1f77b4", "brow": "#8c564b", "lip": "#d62728", "nose": "#2ca02c"}
REFERENCE = "#1f5fbf"


class Variant:
    def __init__(self, title, meshes, faces3d, joints, tri):
        self.title, self.meshes, self.faces3d, self.joints, self.tri = (
            title, meshes, faces3d, joints, tri)


def face_world(result, i):
    if not result.face_attachment_valid[i]:
        return None
    m = result.head_transform[i]
    return result.face_stream_local[i] @ m[:3, :3].T + m[:3, 3]


def base_variant(result, skeleton, tri):
    body = skeleton.shaped(result.betas)
    meshes, faces3d, joints = [], [], []
    for i in range(len(result.body_ok)):
        if not result.body_ok[i]:
            meshes.append(None), faces3d.append(None), joints.append(None)
            continue
        mesh = pose_mesh(body, result.rotations[i], result.betas, result.transl[i])
        meshes.append(mesh.vertices)
        joints.append(mesh.joints)
        faces3d.append(face_world(result, i))
    return body, Variant("3-D copy with face (no change)", meshes, faces3d, joints, tri)


def pivot_of(base):
    return np.nanmedian(np.stack([(j[16] + j[17]) / 2 for j in base.joints if j is not None]), 0)


def width_of(base):
    return float(np.nanmedian([np.linalg.norm(j[16] - j[17]) for j in base.joints
                               if j is not None]))


def map_points(base, fn, title, tri=None):
    m = [None if v is None else fn(v) for v in base.meshes]
    f = [None if v is None else fn(v) for v in base.faces3d]
    j = [None if v is None else fn(v) for v in base.joints]
    return Variant(title, m, f, j, base.tri if tri is None else tri)


def yaw(base, deg):
    p, r = pivot_of(base), Rotation.from_euler("y", deg, degrees=True).as_matrix()
    return map_points(base, lambda x: (x - p) @ r.T + p, f"Camera turned {deg:.0f}°")


def mirror(base):
    cx = pivot_of(base)[0]

    def fn(x):
        out = x.copy()
        out[..., 0] = 2 * cx - out[..., 0]
        return out
    # Reflection reverses triangle winding; reversing the faces keeps normals outward.
    return map_points(base, fn, "Mirrored: a left-handed signer", tri=base.tri[:, ::-1].copy())


def scale(base, k):
    p = pivot_of(base)
    return map_points(base, lambda x: (x - p) * k + p, f"Camera farther: {100 * k:.0f}% size")


def shift(base, sw):
    off = np.asarray(sw) * width_of(base)
    return map_points(base, lambda x: x + off, "Shifted in the frame")


def speed(base, k):
    n = len(base.meshes)
    times = np.minimum(np.arange(n) * k, n - 1)

    def resample(seq):
        out = []
        for tau in times:
            lo = int(np.floor(tau)); hi = min(lo + 1, n - 1); w = tau - lo
            a, b = seq[lo], seq[hi]
            if a is not None and b is not None and np.isfinite(a).all() and np.isfinite(b).all():
                out.append((1 - w) * a + w * b)
            else:
                out.append(a if (a is not None and (w < 0.5 or b is None)) else b)
        return out
    return Variant(f"Played {k:g}× faster (then holds)", resample(base.meshes),
                   resample(base.faces3d), resample(base.joints), base.tri), times


def arm_nudge(result, body, base, rng):
    # Each joint is picked with p = 0.3 and turned by the full 4° (random sign and axis): the
    # edge of the safe range, so the change is visible. At least two joints are picked.
    names = list(ARM_JOINTS)
    chosen = [n for n in names if rng.random() < ARM_P]
    while len(chosen) < 2:
        chosen.append(str(rng.choice([n for n in names if n not in chosen])))
    picks = {}
    for name in names:
        if name in chosen:
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            picks[name] = (ARM_JOINTS[name], axis, float(rng.choice([-1.0, 1.0]) * ARM_MAX_DEG))
    meshes, faces3d, joints = [], [], []
    for i in range(len(result.body_ok)):
        if not result.body_ok[i]:
            meshes.append(None), faces3d.append(None), joints.append(None)
            continue
        rot = result.rotations[i].copy()
        for j, axis, deg in picks.values():  # same draw for every frame of the clip
            local = Rotation.from_rotvec(rot[j])
            rot[j] = (local * Rotation.from_rotvec(np.deg2rad(deg) * axis)).as_rotvec()
        mesh = pose_mesh(body, rot, result.betas, result.transl[i])
        meshes.append(mesh.vertices)
        joints.append(mesh.joints)
        faces3d.append(base.faces3d[i])  # arm edits leave the neck and head untouched
    desc = ", ".join(f"{n} {d:+.1f}°" for n, (_, _, d) in picks.items())
    return Variant("Arm joints nudged 4°", meshes, faces3d, joints, base.tri), desc


def draw_face(ax, pts, visible, contours, mp_indices):
    if pts is None:
        return
    show = visible & np.isfinite(pts).all(axis=1)
    ax.scatter(pts[show, 0], -pts[show, 1], s=0.5, color="#3a2a2a", alpha=0.55, linewidths=0,
               zorder=4)
    for feature, pairs in contours.items():
        for a, b in pairs:
            ia, ib = mp_indices[a], mp_indices[b]
            if show[ia] and show[ib]:
                ax.plot(pts[[ia, ib], 0], -pts[[ia, ib], 1], color=FACE_COLOURS[feature],
                        lw=1.0, zorder=5)


def render(clip, models, out_dir, rng, extra_note=""):
    hands = load_hand_clip(LANDMARK_DIR / f"{clip}_hands.npz")
    face = load_face_clip(LANDMARK_DIR / f"{clip}_face.npz")
    print(f"Clip {clip}: fitting body, hands and head")
    result = run_clip(hands, face, models)
    tri = models.skeleton.model.faces.astype(np.int64)
    body, base = base_variant(result, models.skeleton, tri)
    sp, times = speed(base, SPEED)
    arm, arm_desc = arm_nudge(result, body, base, rng)
    variants = [base, yaw(base, YAW_DEG), mirror(base), scale(base, SCALE),
                shift(base, SHIFT_SW), sp, arm]
    print(f"  arm joints drawn: {arm_desc}")

    n = len(base.meshes)
    frames, fps = read_frames(clip, n)
    contours = face_contours(models.corr, pose_mesh(body, np.zeros((55, 3)), result.betas,
                                                    np.zeros(3)).vertices)
    mp_indices = np.asarray(models.corr.mp_indices)
    # Frame head, shoulders and chest (MediaPipe y points down), so the face reads clearly.
    chest = np.nanmedian([j[[16, 17], 1].mean() for j in base.joints if j is not None]) + 0.28
    upper = [m[m[:, 1] < chest] for v in variants for m in v.meshes if m is not None]
    lim = limits([np.concatenate(upper)], "front", pad=1.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip}_face_augmentations.mp4"
    sink = VideoSink(path, fps)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    note = ("Face: MediaPipe's 478 face points carried on the fitted SMPL-X head (lips red, eyes "
            "blue, brows brown, nose green).  Blue lines: the unchanged arms, for reference.  "
            f"Arm joints drawn for this clip: {arm_desc}." + (f"\n{extra_note}" if extra_note else ""))
    for t in range(n):
        fig.clf()
        grid = fig.add_gridspec(2, 4, hspace=0.1, wspace=0.03, left=0.01, right=0.99,
                                top=0.95, bottom=0.05)
        axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(8)]
        image = overlay_mediapipe(frames[t], hands, t)
        if face.valid[t]:
            image = overlay_face(image, face.landmarks[t].astype(np.float64), contours,
                                 mp_indices)
        axes[0].imshow(image)
        axes[0].set_title(f"Original video, clip {clip}  (frame {t + 1}/{n})", fontsize=12)
        axes[0].axis("off")
        # Hidden face points (behind a hand in the source frame) stay hidden in every view.
        visible = result.face_point_visible
        for ax, v in zip(axes[1:], variants):
            src = int(round(times[t])) if v is sp else t
            if v.meshes[t] is not None:
                draw_mesh(ax, v.meshes[t], v.tri)
                draw_face(ax, v.faces3d[t], visible[min(src, n - 1)], contours, mp_indices)
            if v is not base and base.joints[t] is not None:
                draw_bones(ax, base.joints[t], SMPLX_ARM_BONES, "front", REFERENCE, 1.4,
                           alpha=0.6)
            set_view(ax, lim, v.title)
            ax.title.set_fontsize(13)
        fig.text(0.01, 0.008, note, fontsize=10, color="#333333", va="bottom")
        sink.add(fig)
    sink.close()
    plt.close(fig)
    print(f"  -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="+", default=["63229", "17083"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--note", default="", help="extra caption line under the panels")
    args = parser.parse_args()
    models = load_face_models("models")
    for k, clip in enumerate(args.clips):
        render(clip, models, args.out, np.random.default_rng(args.seed + k), args.note)


if __name__ == "__main__":
    main()
