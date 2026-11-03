#!/usr/bin/env python3
"""
Augment whole clips through the V1.3 body + hands + face round trip, carrying the observed face.

Each clip goes through `face_pipeline.run_clip`: one SMPL-X pose per frame (body, MANO hands,
neck / head / jaw) and the 478 observed FaceLandmarker points stored in the head joint's frame
(`face_stream_local`). An augmentation edits that pose (or moves its outputs), and the face is
put back with `face_stream.from_head_frame` on the augmented head, so the observed expression
travels with the head instead of being re-detected or replaced by the SMPL-X face.

Every augmentation draws its parameters once per clip and applies them to every frame
(V1 Problem Scope §11), as in augment_trials.py:

    yaw          extrinsic  the figure turned about the vertical axis through the shoulders
    mirror       extrinsic  left/right reflection with the hands swapped (a left-handed signer)
    scale        extrinsic  the whole scene scaled about the shoulders
    uncentre     extrinsic  the whole scene moved off-centre
    arm pose     intrinsic  elbows bent more, shoulders raised; the hands turn with the forearms
    body shape   intrinsic  SMPL-X betas[0] shifted; the same joint rotations
    head turn    intrinsic  the neck turned about the torso's vertical axis; the face rides along
    time warp    temporal   smooth monotonic re-timing; the clip keeps its length

Mirror reflects the face geometrically but does not relabel MediaPipe's left/right face indices,
so it is right for viewing and not yet a valid face stream for training.

Measured per augmentation: zero strength must reproduce the round trip; the face must stay rigid
on the head (face-to-head-joint distances), and the closest distance from each hand (wrist,
fingertips, palm on the SMPL-X mesh) to the observed face points, because an augmentation that
moves a hand away from the chin can change the sign. Distances are geometric only: the V1.3
results leave absolute hand-to-face depth uncalibrated, so they are not contact labels.

Each clip is trimmed to its signing span (first to last frame with both arms observed), since an
unobserved arm is left at the SMPL-X rest pose; `--full-clip` keeps every frame. With
`--depth-corrected` each wrist is re-placed from the image (run_clip's depth pass), which brings
hands that touch the face in the video to the face in 3-D; hands whose corrected wrist is not
reached are masked, as in the V1.3 results.

Usage (from the repo root):
    .venv/bin/python scripts/augment_face_trials.py --depth-corrected
    .venv/bin/python scripts/augment_face_trials.py --clips 57949 12328

Reads <landmark-dir>/<clip>_hands.npz and <clip>_face.npz and <video-dir>/<clip>.mp4. Writes
<output-dir>/<clip>_face_augmentations.mp4 and <output-dir>/augment_face_trials.json.
"""

import argparse
import dataclasses
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

from view_face_round_trip import (  # noqa: E402  (also sets up matplotlib and cv2)
    SKIN, draw_mesh_tinted, face_contours, overlay_face,
)
from view_body_hand_round_trip import _project, overlay_mediapipe, limits, set_view  # noqa: E402
import view_round_trip  # noqa: E402
from view_round_trip import VideoSink, read_frames  # noqa: E402
from demo_smplx_augmentation import (  # noqa: E402
    SPINE3, elbow_flex, rotate_in_world, shoulder_raise, world_pose,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kinetic_augment.body_model.assembly import (  # noqa: E402
    HEAD, NECK, global_rotations, pose_mesh,
)
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip  # noqa: E402
from kinetic_augment.body_model.face_stream import from_head_frame  # noqa: E402
from kinetic_augment.body_model.hand_face_geometry import hand_points  # noqa: E402
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402

LANDMARK_DIR = Path("data/round_trip_eval")
VIDEO_DIR = Path.home() / "Downloads" / "raw_videos"
OUTPUT_DIR = Path("data/augment_trials")
CLIPS = ("57949", "12328", "63669", "63229")

YAW_DEG = 25.0
SCALE = 0.85
UNCENTRE_SW = (0.6, 0.2, 0.0)   # shoulder widths: right and down (MediaPipe y is down)
ELBOW_FLEX_DEG = 15.0
SHOULDER_RAISE_DEG = 10.0
SHAPE_SIGMAS = 2.0
HEAD_TURN_DEG = 20.0            # about the torso's vertical axis, at the neck
WARP_STRENGTH = 0.5
NEAR_FACE = 0.05                # metres: "hand near the face" for the frame counts

SMPLX_SHOULDER, SMPLX_ELBOW, SMPLX_WRIST = (16, 17), (18, 19), (20, 21)
ARM_JOINTS = [(16, 18), (18, 20), (17, 19), (19, 21), (16, 17), (12, 15)]
FACE_BLUE, BASE_BLUE, GAP_GREY = "#1f5fbf", "#1f5fbf", "#444444"
FACE_LINE = {"eye": "#0b3d91", "brow": "#0b3d91", "lip": "#b0127a", "nose": "#2a7f3f"}


@dataclasses.dataclass
class Variant:
    """One version of a clip, per frame: mesh vertices, joints, observed face, hand points."""

    name: str
    title: str
    vertices: list      # (V, 3) or None where the body was not fitted
    joints: np.ndarray  # (T, 55, 3)
    face: np.ndarray    # (T, 478, 3); NaN where the face was not attached
    hands: np.ndarray   # (T, 2, 7, 3) wrist, fingertips, palm; NaN where the hand is not usable
    faces: np.ndarray   # (F, 3) mesh triangles


@dataclasses.dataclass
class _Fit:
    """The two fields the demo edits read: SMPL-X root and body rotations."""

    global_orient: np.ndarray
    body_pose: np.ndarray


# --------------------------------------------------------------------------- parameter space

def from_rotations(name, title, result, rotations, betas, triangles, skeleton):
    """Mesh, joints, carried face and hand points for every frame of (edited) rotations."""
    n = len(rotations)
    out = Variant(name, title, [None] * n, np.full((n, 55, 3), np.nan),
                  np.full((n, 478, 3), np.nan), np.full((n, 2, 7, 3), np.nan), triangles)
    for i in range(n):
        if not result.body_ok[i]:
            continue
        mesh = pose_mesh(skeleton, rotations[i], betas, result.transl[i])
        out.vertices[i], out.joints[i] = mesh.vertices, mesh.joints
        if result.face_attachment_valid[i]:
            head = global_rotations(rotations[i], skeleton.parents)[HEAD]
            out.face[i] = from_head_frame(result.face_stream_local[i], head, mesh.joints[HEAD])
        usable = result.geometry_valid[i]
        out.hands[i, usable] = hand_points(mesh)[usable]
    return out


def edit_arms(rotations, body, flex, raise_):
    """Arm-pose edit of the body joints; hand, head and jaw rotations are kept as they are."""
    out = rotations.copy()
    for i, r in enumerate(rotations):
        if not np.isfinite(r).all():
            continue
        fit = _Fit(r[0].copy(), r[1:22].reshape(63).copy())
        new = elbow_flex(shoulder_raise(fit, body, raise_), body, flex)
        out[i, 1:22] = np.asarray(new.body_pose).reshape(21, 3)
    return out


def turn_head(rotations, body, degrees):
    """The neck (and so head, jaw and face) turned about the upper torso's vertical axis."""
    out = rotations.copy()
    for i, r in enumerate(rotations):
        if not np.isfinite(r).all():
            continue
        rot, _ = world_pose(r, body)
        up = rot[SPINE3] @ np.array([0.0, 1.0, 0.0])  # SMPL-X is y-up at rest
        delta = Rotation.from_rotvec(up * np.radians(degrees)).as_matrix()
        rotate_in_world(out[i], rot, NECK, body.parents[NECK], delta)
    return out


# --------------------------------------------------------------------------- output space

def _map(variant, fn, name, title, triangles=None, swap=False):
    hands = fn(variant.hands)
    if swap:
        hands = hands[:, ::-1]
    return Variant(name, title, [None if v is None else fn(v) for v in variant.vertices],
                   fn(variant.joints), fn(variant.face), hands,
                   variant.faces if triangles is None else triangles)


def shoulder_centre(variant):
    return np.nanmedian(variant.joints[:, list(SMPLX_SHOULDER)].mean(axis=1), axis=0)


def shoulder_width(variant):
    a, b = SMPLX_SHOULDER
    return float(np.nanmedian(np.linalg.norm(variant.joints[:, a] - variant.joints[:, b], axis=-1)))


def yaw(variant, degrees):
    pivot = shoulder_centre(variant)
    rot = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    return _map(variant, lambda p: (p - pivot) @ rot.T + pivot, "yaw",
                f"yaw {degrees:+.0f}° (extrinsic)")


def mirror(variant):
    flip_x = 2 * shoulder_centre(variant)[0]

    def fn(p):
        out = p.copy()
        out[..., 0] = flip_x - out[..., 0]
        return out

    # Reflection reverses triangle winding; reversing the faces keeps normals outward.
    return _map(variant, fn, "mirror", "mirror, hands swapped (extrinsic)",
                triangles=variant.faces[:, ::-1].copy(), swap=True)


def scale(variant, factor):
    pivot = shoulder_centre(variant)
    return _map(variant, lambda p: (p - pivot) * factor + pivot, "scale",
                f"scale ×{factor:.2f} (extrinsic)")


def uncentre(variant, shift_sw):
    offset = np.asarray(shift_sw, dtype=np.float64) * shoulder_width(variant)
    return _map(variant, lambda p: p + offset, "uncentre",
                f"off-centre ({shift_sw[0]:+.1f}, {shift_sw[1]:+.1f}) sh.w. (extrinsic)")


def warp_times(count, strength):
    u = np.linspace(0.0, 1.0, count)
    return (count - 1) * (u + strength * np.sin(2 * np.pi * u) / (2 * np.pi))


def _resample(frames, times):
    """Per-frame linear interpolation; the nearer frame if the other is missing."""
    out = []
    for tau in times:
        lo = int(np.floor(tau))
        hi = min(lo + 1, len(frames) - 1)
        w = tau - lo
        a, b = frames[lo], frames[hi]
        a_ok = a is not None and np.isfinite(a).all()
        b_ok = b is not None and np.isfinite(b).all()
        if a_ok and b_ok:
            out.append((1 - w) * a + w * b)
        elif a_ok or b_ok:
            out.append(a if (a_ok and (w < 0.5 or not b_ok)) else b)
        else:
            out.append(a)
    return out


def _resample_rows(seq, times):
    """Resample (T, N, ...) row by row, so one missing row stays local to its frame."""
    out = np.full_like(seq, np.nan)
    for j in range(seq.shape[1]):
        out[:, j] = np.stack(_resample(list(seq[:, j]), times))
    return out


def time_warp(variant, strength):
    times = warp_times(len(variant.joints), strength)
    return Variant("time_warp", f"time warp k={strength:.1f} (temporal)",
                   _resample(variant.vertices, times), _resample_rows(variant.joints, times),
                   np.stack(_resample(list(variant.face), times)),
                   _resample_rows(variant.hands, times), variant.faces), times


# --------------------------------------------------------------------------- measurement

def _max_diff(a, b):
    diffs = [np.nanmax(np.abs(a.joints - b.joints))]
    if np.isfinite(a.face).any():
        diffs.append(np.nanmax(np.abs(a.face - b.face)))
    diffs += [np.max(np.abs(x - y)) for x, y in zip(a.vertices, b.vertices) if x is not None]
    return float(max(diffs))


def hand_face_distance(variant):
    """(T, 2) closest distance (m) from each hand's points to the observed face points."""
    n = len(variant.joints)
    out = np.full((n, 2), np.nan)
    for t in range(n):
        face = variant.face[t][np.isfinite(variant.face[t]).all(axis=1)]
        if not len(face):
            continue
        for s in range(2):
            pts = variant.hands[t, s]
            if np.isfinite(pts).all():
                out[t, s] = np.linalg.norm(pts[:, None] - face[None], axis=-1).min()
    return out


def face_rigidity_mm(base, variant):
    """Max change (mm) of the face points' distances to the head joint: 0 = carried rigidly."""
    def radii(v):
        return np.linalg.norm(v.face - v.joints[:, HEAD][:, None], axis=-1)
    return float(1000 * np.nanmax(np.abs(radii(variant) - radii(base))))


def measure(base, variant, compare_frames=True):
    row = {}
    d = hand_face_distance(variant)
    closest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    closest[~np.isfinite(closest)] = np.nan
    row["hand_face_cm_median"] = float(100 * np.nanmedian(closest))
    row["hand_face_cm_min"] = float(100 * np.nanmin(closest))
    row["frames_hand_near_face"] = int(np.sum(closest < NEAR_FACE))
    if compare_frames:
        row["face_moved_mm_median"] = float(1000 * np.nanmedian(
            np.linalg.norm(variant.face - base.face, axis=-1)))
        row["face_rigidity_mm"] = face_rigidity_mm(base, variant)
        wrists = list(SMPLX_WRIST)
        row["wrist_moved_mm_median"] = float(1000 * np.nanmedian(
            np.linalg.norm(variant.joints[:, wrists] - base.joints[:, wrists], axis=-1)))
    return row, closest


# --------------------------------------------------------------------------- clip

def signing_span(hands):
    """
    Source frames from the first to the last in which both arms (elbow and wrist) are observed.

    An unobserved arm is left at the SMPL-X rest pose (arms out), which in the lead-in and
    lead-out of a clip shows a T-pose instead of the arms at rest; the sign itself is inside.
    """
    pose = hands.pose_world
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= 0.5)
    both = np.flatnonzero(ok[:, [13, 15, 14, 16]].all(axis=1))
    return np.arange(both[0], both[-1] + 1) if len(both) else None


def augment_clip(clip, landmark_dir, models, depth_corrected=False, full_clip=False):
    started = time.time()
    hands = load_hand_clip(landmark_dir / f"{clip}_hands.npz")
    face = load_face_clip(landmark_dir / f"{clip}_face.npz")
    span = None if full_clip else signing_span(hands)
    result = run_clip(hands, face, models, frames=span, depth_corrected=depth_corrected)
    skeleton = models.skeleton
    body = skeleton.shaped(result.betas)
    triangles = skeleton.model.faces.astype(np.int64)
    n = len(result.frames)
    print(f"Clip {clip}: {n} frames, {int(result.body_ok.sum())} bodies, "
          f"{int(result.face_attachment_valid.sum())} faces attached "
          f"({time.time() - started:.0f} s)")

    rotations = result.rotations

    def params(name, title, rots, betas=result.betas):
        return from_rotations(name, title, result, rots, betas, triangles, skeleton)

    base = params("round_trip", "round trip (no augmentation)", rotations)
    shaped = result.betas.copy()
    shaped[0] += SHAPE_SIGMAS
    warped, times = time_warp(base, WARP_STRENGTH)
    variants = [
        yaw(base, YAW_DEG),
        mirror(base),
        scale(base, SCALE),
        uncentre(base, UNCENTRE_SW),
        params("arm_pose", f"elbows +{ELBOW_FLEX_DEG:.0f}°, shoulders "
               f"+{SHOULDER_RAISE_DEG:.0f}° (intrinsic)",
               edit_arms(rotations, body, ELBOW_FLEX_DEG, SHOULDER_RAISE_DEG)),
        params("body_shape", f"betas[0] {SHAPE_SIGMAS:+.0f} (intrinsic)", rotations, shaped),
        params("head_turn", f"head turn {HEAD_TURN_DEG:+.0f}° (intrinsic, face carried)",
               turn_head(rotations, body, HEAD_TURN_DEG)),
        warped,
    ]

    # Zero strength must give back the round trip exactly; mirroring twice must too. The base
    # face must also match the archive's own stored transform (head-local -> world).
    stored = np.full_like(base.face, np.nan)
    for i in np.flatnonzero(result.face_attachment_valid):
        h = result.head_transform[i]
        stored[i] = result.face_stream_local[i] @ h[:3, :3].T + h[:3, 3]
    identity = {
        "face_vs_stored_transform": float(np.nanmax(np.abs(stored - base.face))),
        "yaw": _max_diff(base, yaw(base, 0.0)),
        "mirror": _max_diff(base, mirror(mirror(base))),
        "scale": _max_diff(base, scale(base, 1.0)),
        "uncentre": _max_diff(base, uncentre(base, (0.0, 0.0, 0.0))),
        "arm_pose": _max_diff(base, params("z", "", edit_arms(rotations, body, 0.0, 0.0))),
        "body_shape": _max_diff(base, params("z", "", rotations, result.betas.copy())),
        "head_turn": _max_diff(base, params("z", "", turn_head(rotations, body, 0.0))),
        "time_warp": _max_diff(base, time_warp(base, 0.0)[0]),
    }
    report = {"frames": n, "source_frames": [int(result.frames[0]), int(result.frames[-1])],
              "bodies": int(result.body_ok.sum()),
              "faces_attached": int(result.face_attachment_valid.sum()),
              "identity_max_abs_diff_m": identity}
    closest = {}
    report["round_trip"], closest["round_trip"] = measure(base, base, compare_frames=False)
    for v in variants:
        report[v.name], closest[v.name] = measure(base, v, compare_frames=v.name not in
                                                  ("mirror", "time_warp"))
        print(f"  {v.name:<10} zero-strength {identity[v.name]:.1e} m  " + ", ".join(
            f"{k} {x:.2f}" for k, x in report[v.name].items()))
    report["time_warp"]["source_frame_range"] = [float(times.min()), float(times.max())]
    return result, hands, face, base, variants, report, closest


# --------------------------------------------------------------------------- rendering

def draw_face(ax, points, contours, mp_indices, alpha=1.0):
    ok = np.isfinite(points).all(axis=1)
    if not ok.any():
        return
    xs, ys = _project(points[ok], "front")
    ax.scatter(xs, ys, s=0.6, color=FACE_BLUE, alpha=0.6 * alpha, linewidths=0, zorder=6)
    for feature, pairs in contours.items():
        for a, b in pairs:
            seg = points[[mp_indices[a], mp_indices[b]]]
            if np.isfinite(seg).all():
                xs, ys = _project(seg, "front")
                ax.plot(xs, ys, color=FACE_LINE[feature], lw=0.9, alpha=alpha, zorder=7)


def draw_reference_arms(ax, joints):
    for a, b in ARM_JOINTS:
        seg = joints[[a, b]]
        if np.isfinite(seg).all():
            xs, ys = _project(seg, "front")
            ax.plot(xs, ys, color=BASE_BLUE, lw=1.2, alpha=0.55, zorder=8)


def draw_closest_gap(ax, variant, t):
    """A grey line from the hand point nearest the face to the nearest face point."""
    face = variant.face[t]
    ok = np.isfinite(face).all(axis=1)
    if not ok.any():
        return None
    best = None
    for s in range(2):
        pts = variant.hands[t, s]
        if not np.isfinite(pts).all():
            continue
        d = np.linalg.norm(pts[:, None] - face[ok][None], axis=-1)
        k, j = np.unravel_index(np.argmin(d), d.shape)
        if best is None or d[k, j] < best[0]:
            best = (d[k, j], pts[k], face[ok][j])
    if best is None:
        return None
    xs, ys = _project(np.stack(best[1:]), "front")
    ax.plot(xs, ys, color=GAP_GREY, lw=1.0, zorder=9)
    ax.scatter(xs[:1], ys[:1], s=12, color=GAP_GREY, zorder=10, linewidths=0)
    return best[0]


def render(clip, hands, face, base, variants, report, closest, models, result, out_dir):
    n = len(base.joints)
    src = result.frames
    frames, fps = read_frames(clip, int(src[-1]) + 1)
    corr = models.corr
    body = models.skeleton.shaped(result.betas)
    albedo = np.tile(SKIN, (len(base.faces), 1))
    rest = pose_mesh(body, np.zeros((55, 3)), result.betas, np.zeros(3)).vertices
    contours = face_contours(corr, rest)
    mp_indices = corr.mp_indices

    everything = [base] + variants
    shoulder_y = np.nanmedian(base.joints[:, list(SMPLX_SHOULDER), 1])
    upper = [m[m[:, 1] < shoulder_y + 0.40] for v in everything for m in v.vertices
             if m is not None]
    lim = limits([np.concatenate(upper)], "front", pad=1.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_face_augmentations.mp4"
    sink = VideoSink(out_path, fps)
    fig = plt.figure(figsize=(20.0, 15.6), dpi=90)
    text = metrics_text(clip, report)
    for t in range(n):
        fig.clf()
        grid = fig.add_gridspec(3, 4, hspace=0.10, wspace=0.03,
                                left=0.005, right=0.995, top=0.965, bottom=0.005)
        source = fig.add_subplot(grid[0, 0])
        image = overlay_mediapipe(frames[src[t]], hands, src[t])
        if face.valid[src[t]]:
            image = overlay_face(image, face.landmarks[src[t]].astype(np.float64), contours,
                                 mp_indices)
        source.imshow(image)
        source.set_title(f"clip {clip}  frame {src[t] + 1} (span {t + 1}/{n})  (MediaPipe 2-D)", fontsize=11)
        source.axis("off")
        cells = [(r, c) for r in range(3) for c in range(4)][1:10]
        for (r, c), v in zip(cells, everything):
            ax = fig.add_subplot(grid[r, c])
            if v.vertices[t] is not None:
                draw_mesh_tinted(ax, v.vertices[t], v.faces, albedo)
            if v is not base:  # the unaugmented arms and neck, for reference
                draw_reference_arms(ax, base.joints[t])
            draw_face(ax, v.face[t], contours, mp_indices)
            gap = draw_closest_gap(ax, v, t)
            readout = ("\nhand–face: masked" if gap is None
                       else f"\nhand–face {100 * gap:.1f} cm")
            set_view(ax, lim, v.title + readout)
        panel = fig.add_subplot(grid[2, 2:])
        panel.axis("off")
        panel.text(0.0, 1.0, text, family="monospace", fontsize=9.5, va="top",
                   transform=panel.transAxes)
        strip = panel.inset_axes([0.62, 0.08, 0.37, 0.84])
        for name, colour in (("round_trip", "k"), ("arm_pose", "#d62728"),
                             ("head_turn", "#9467bd"), ("body_shape", "#ff7f0e")):
            strip.plot(np.arange(1, n + 1), 100 * closest[name], color=colour, lw=1.3,
                       label=name.replace("_", " "))
        strip.axvline(t + 1, color="#888888", lw=0.8)
        strip.axhline(100 * NEAR_FACE, color="#888888", lw=0.6, ls=":")
        strip.set_xlim(1, n)
        strip.set_title("closest hand to observed face (cm)", fontsize=9)
        strip.set_xlabel("frame", fontsize=8)
        strip.tick_params(labelsize=7)
        strip.legend(fontsize=7, loc="upper right")
        sink.add(fig)
    sink.close()
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


def metrics_text(clip, report):
    lines = [f"clip {clip}: per-clip parameters, every frame",
             f"{report['faces_attached']}/{report['frames']} frames with an attached face", "",
             "mesh: SMPL-X posed from the augmented parameters",
             "BLUE dots/lines: observed MediaPipe face,",
             "  carried in the head frame",
             "BLUE bones: unaugmented arms (reference)",
             "GREY line: closest hand point to the face", "",
             f"{'':<11}{'face':>6}{'rigid':>6}{'h-f':>6}{'near':>5}",
             f"{'':<11}{'mm':>6}{'mm':>6}{'cm':>6}{'fr':>5}"]
    for name in ("round_trip", "yaw", "mirror", "scale", "uncentre", "arm_pose", "body_shape",
                 "head_turn", "time_warp"):
        r = report[name]
        lines.append(f"{name:<11}{r.get('face_moved_mm_median', float('nan')):>6.0f}"
                     f"{r.get('face_rigidity_mm', float('nan')):>6.1f}"
                     f"{r['hand_face_cm_median']:>6.1f}{r['frames_hand_near_face']:>5d}")
    lines += ["", "face mm: median face-point move",
              "rigid mm: max change of face-to-head",
              "  distances (0 = rigid; scale shrinks it)",
              "h-f cm: median closest hand-face gap",
              f"near fr: frames with a gap < {100 * NEAR_FACE:.0f} cm",
              "masked: hand geometry invalid (V1.3 mask)",
              "geometric only; not contact labels", "",
              "zero strength reproduces the round trip:",
              f"  max diff {max(report['identity_max_abs_diff_m'].values()):.0e} m (all 8 + face)"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="*", default=list(CLIPS))
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--full-clip", action="store_true",
                        help="keep the lead-in and lead-out where an arm is not observed")
    parser.add_argument("--depth-corrected", action="store_true",
                        help="re-place each wrist from the image (run_clip's depth pass)")
    args = parser.parse_args()
    view_round_trip.VIDEO_DIR = args.video_dir

    models = load_face_models(args.model_dir)
    out = {"clips": args.clips,
           "parameters": {"yaw_deg": YAW_DEG, "scale": SCALE, "uncentre_sw": UNCENTRE_SW,
                          "elbow_flex_deg": ELBOW_FLEX_DEG,
                          "shoulder_raise_deg": SHOULDER_RAISE_DEG,
                          "shape_sigmas_betas0": SHAPE_SIGMAS, "head_turn_deg": HEAD_TURN_DEG,
                          "warp_strength": WARP_STRENGTH, "near_face_m": NEAR_FACE},
           "depth_corrected": args.depth_corrected, "full_clip": args.full_clip,
           "results": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        result, hands, face, base, variants, report, closest = augment_clip(
            clip, args.landmark_dir, models, args.depth_corrected, args.full_clip)
        out["results"][clip] = report
        if not args.no_render:
            report["video"] = str(render(clip, hands, face, base, variants, report, closest,
                                         models, result, args.output_dir))
        (args.output_dir / "augment_face_trials.json").write_text(json.dumps(out, indent=2))
    print(f"-> {args.output_dir / 'augment_face_trials.json'}")


if __name__ == "__main__":
    main()
