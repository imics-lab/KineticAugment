#!/usr/bin/env python3
"""
Visualise the body + hands + face round trip of one clip (V1.3), stitched into one figure.

Body:  MediaPipe pose world landmarks -> SMPL-X (fit_shape, fit_body)
Hands: MediaPipe hand world landmarks -> MANO (fit_hand_shape, fit_hands)
Head:  MediaPipe FaceLandmarker -> SMPL-X neck / head / jaw (face_pipeline.run_clip)

The point is to judge by eye whether the signer's hand really comes to the face: each frame
shows the hand-to-face distance two ways, in the image and in 3-D from the side, where the
depth gap between hand and face shows up. BLUE = MediaPipe input, GREEN = round trip rebuilt
from SMPL-X parameters alone.

    1. source + MediaPipe 2-D overlay (pose, hands, face landmarks and contours) with, per hand,
       the signed distance of its closest point to the head surface and the ASL-LEX region
    2. front view and 3. side view (MediaPipe world, metres): arms and hands as in
       view_body_hand_round_trip.py, plus the 105 embedded face points and the nose and ears
       of both sides, and a line from each hand's closest point to the head surface
    4. the SMPL-X mesh posed from body + hands + head/neck/jaw, head tinted by region
    5. error strips: arm / hand / face errors, and each hand's closest distance to the face

The observed face points are placed in the body frame by a similarity fit of the observed rigid
points onto the posed SMPL-X rigid points, so blue and green faces agree by construction up to
expression and fit error. That fit removes head *placement*, so the nose and ear markers (blue:
MediaPipe pose, green: SMPL-X vertices) show it. Hand-to-face distance is measured on the SMPL-X
mesh (the green side). A signed distance is negative inside the head.

Usage (from the repo root):
    .venv/bin/python scripts/view_face_round_trip.py --clip 57949 12328
    .venv/bin/python scripts/view_face_round_trip.py --clip 57949 --stills 3

Reads <landmark-dir>/<clip>_hands.npz and <clip>_face.npz and the video <video-dir>/<clip>.mp4.
Output: <output-dir>/<clip>_body_hand_face_round_trip.mp4, and with --stills N the N frames of
closest hand-face approach as <clip>_frame<N>.png.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from view_body_hand_round_trip import (  # noqa: E402  (also sets up matplotlib and cv2)
    HAND_BONES, INPUT, POSE_WRIST, RECON, _project, anchored, draw_bones, hand_round_trip,
    limits, overlay_mediapipe, set_view,
)
import view_round_trip  # noqa: E402
from view_round_trip import (  # noqa: E402
    ARM_BONES, CONTEXT_BONES, VideoSink, frame_errors, read_frames, round_trip_clip,
)
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from kinetic_augment.body_model.alignment import (  # noqa: E402
    apply_similarity, similarity_transform,
)
from kinetic_augment.body_model.assembly import pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import (  # noqa: E402
    INNER_LIP, LEFT_EYE, LEFT_EYEBROW, NOSE, OUTER_LIP, RIGHT_EYE, RIGHT_EYEBROW, RIGID,
    LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
)
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import (  # noqa: E402
    SIDES, load_face_models, run_clip,
)
from kinetic_augment.body_model.face_regions import (  # noqa: E402
    REGIONS, SUBREGIONS, build_head_surface,
)
from kinetic_augment.body_model.hand_face_geometry import (  # noqa: E402
    CONTACT_DISTANCE, HAND_POINTS, hand_points, signed_distances,
)
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402
from kinetic_augment.body_model.head_converter import (  # noqa: E402
    OCCLUSION_PAD, hand_outlines, occluded,
)

LANDMARK_DIR = Path("data/round_trip_eval")
OUTPUT_DIR = Path("data/round_trip_eval/videos")
VIDEO_DIR = Path.home() / "Downloads" / "raw_videos"
CONTACT_RED, GAP_GREY = "#d62728", "#444444"
SKIN = np.array([0.78, 0.66, 0.58])
TINT = 0.5  # weight of the region colour over the skin colour on the head surface
FACE_COLOURS_2D = {"eye": (255, 255, 0), "brow": (255, 170, 0), "lip": (255, 60, 255),
                   "nose": (60, 255, 120)}  # RGB, drawn over the source frame
NOSE_BRIDGE, NOSE_BASE = (168, 6, 197, 195, 5, 4), (129, 98, 97, 2, 326, 327, 358)
SMPLX_NOSE, SMPLX_EAR = 9120, {"left": 6, "right": 616}  # smplx.vertex_ids['smplx']
POSE_NOSE, POSE_EAR = 0, {"left": 7, "right": 8}


def face_contours(corr, rest_vertices):
    """
    Contour line segments of the embedded face points, as {feature: [(row, row), ...]} with rows
    into the 105-point array. The MediaPipe index groups are sets, not curves, so eyes and brows
    are ordered into closed loops by angle about their centre on the rest mesh (front view);
    the lips are already closed loops, and the nose is a bridge and a base.
    """
    def ordered(group):
        rows = corr.rows(group)
        pts = corr.points(rest_vertices, group)[:, :2]
        d = pts - pts.mean(axis=0)
        return rows[np.argsort(np.arctan2(d[:, 1], d[:, 0]))], True

    def segments(rows, closed):
        pairs = list(zip(rows[:-1], rows[1:]))
        return pairs + ([(rows[-1], rows[0])] if closed else [])

    out = {"eye": [], "brow": [], "lip": [], "nose": []}
    for group in (RIGHT_EYE, LEFT_EYE):
        out["eye"] += segments(*ordered(group))
    for group in (RIGHT_EYEBROW, LEFT_EYEBROW):
        out["brow"] += segments(*ordered(group))
    for group in (OUTER_LIP, INNER_LIP):
        out["lip"] += segments(corr.rows(group), True)
    assert set(NOSE) == set(NOSE_BRIDGE) | set(NOSE_BASE)
    for group in (NOSE_BRIDGE, NOSE_BASE):
        out["nose"] += segments(corr.rows(group), False)
    return {k: [(int(a), int(b)) for a, b in v] for k, v in out.items()}


def overlay_face(image, face_t, contours, mp_indices):
    """The source frame with the FaceLandmarker points (small dots) and the feature contours."""
    image = image.copy()
    h, w = image.shape[:2]
    thick = max(1, round(h / 400))
    if not np.isfinite(face_t[:, :2]).all():
        return image

    def px(p):
        return int(round(p[0] * w)), int(round(p[1] * h))

    for p in face_t:
        cv2.circle(image, px(p), 1, (255, 255, 255), -1, cv2.LINE_AA)
    for feature, pairs in contours.items():
        for a, b in pairs:
            cv2.line(image, px(face_t[mp_indices[a]]), px(face_t[mp_indices[b]]),
                     FACE_COLOURS_2D[feature], thick, cv2.LINE_AA)
    return image


def carry_face(face_px_t, mesh_vertices, head, corr):
    """
    The 105 observed face points in the body frame (metres), by a similarity fit of the observed
    rigid points onto the posed SMPL-X rigid points weighted by the head fit's own weights.
    None when the frame has no usable head fit.
    """
    weights = head.rigid_weights
    if face_px_t is None or not head.valid or weights is None or (weights > 0).sum() < 3:
        return None
    source = face_px_t[list(RIGID)]
    if not np.isfinite(source).all():
        return None
    scale, rotation, translation = similarity_transform(
        source, corr.points(mesh_vertices, RIGID), weights=weights)
    return apply_similarity(face_px_t[corr.mp_indices], scale, rotation, translation)


def face_residuals_mm(blue, green, corr, weights, hidden):
    """
    RMS distance (mm) between the carried observed points and the SMPL-X ones: over the rigid
    points the head fit used, and over the other (non-rigid) points not under a hand.
    """
    rigid = corr.rows(RIGID)
    gap = 1000.0 * np.linalg.norm(blue - green, axis=1)
    used = np.zeros(len(gap), bool)
    used[rigid[weights > 0]] = True
    other = np.ones(len(gap), bool)
    other[rigid] = False
    other &= ~hidden & np.isfinite(gap)

    def rms(mask):
        return float(np.sqrt(np.mean(gap[mask] ** 2))) if mask.any() else np.nan

    return rms(used), rms(other)


def closest_to_head(mesh, surface):
    """Per hand point: signed distance (2, 7), the closest head-surface point (2, 7, 3)."""
    points = hand_points(mesh)
    head = mesh.vertices[surface.vertex_ids]
    signed, closest, _ = signed_distances(points.reshape(-1, 3), head, surface.faces)
    return points, signed.reshape(2, -1), closest.reshape(2, -1, 3)


def region_tints(faces, surface, n_vertices):
    """(F, 3) albedo: skin colour, the head surface blended with its region colour."""
    region = np.full(n_vertices, -1)
    region[surface.vertex_ids] = surface.region
    on_head = (region[faces] >= 0).all(axis=1)
    palette = np.asarray(plt.get_cmap("tab10").colors)[:, :3]
    albedo = np.tile(SKIN, (len(faces), 1))
    albedo[on_head] = (1 - TINT) * SKIN + TINT * palette[region[faces[on_head, 0]]]
    return albedo


def draw_mesh_tinted(ax, vertices, faces, albedo, light=(0.3, -0.5, -1.0)):
    """
    Same orthographic Lambert view as view_body_hand_round_trip.draw_mesh, with a colour per
    triangle (that function takes one colour for the whole mesh).
    """
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    front = normals[:, 2] < 0
    tri, normals, albedo = tri[front], normals[front], albedo[front]
    order = np.argsort(-tri[:, :, 2].mean(axis=1))
    tri, normals, albedo = tri[order], normals[order], albedo[order]
    light = np.asarray(light) / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(normals @ light, 0.0, 1.0)
    colours = np.clip(shade[:, None] * albedo, 0, 1)
    polys = np.stack([tri[..., 0], -tri[..., 1]], axis=-1)
    ax.add_collection(PolyCollection(polys, facecolors=colours, edgecolors=colours,
                                     linewidths=0.2))


def draw_gap(ax, view, hand, head, contact):
    """The line from a hand point to its closest head-surface point."""
    colour = CONTACT_RED if contact else GAP_GREY
    xs, ys = _project(np.stack([hand, head]), view)
    ax.plot(xs, ys, color=colour, lw=1.4 if contact else 1.0, zorder=5)
    ax.scatter(xs[:1], ys[:1], s=14, color=colour, zorder=6, linewidths=0)
    ax.scatter(xs[1:], ys[1:], s=14, color=colour, zorder=6, marker="x", linewidths=1)


def choose_stills(closest, count, spacing=8):
    """The `count` frames (0-based) of smallest |hand-face distance|, at least `spacing` apart."""
    order = np.argsort(np.where(np.isfinite(closest), closest, np.inf))
    chosen = []
    for t in order:
        if not np.isfinite(closest[t]):
            break
        if all(abs(int(t) - c) >= spacing for c in chosen):
            chosen.append(int(t))
        if len(chosen) == count:
            break
    return sorted(chosen)


def render(clip, landmark_dir: Path, out_dir: Path, models, stills=3, frame_range=None,
           still_frames=None):
    started = time.time()
    hands = load_hand_clip(landmark_dir / f"{clip}_hands.npz")
    face = load_face_clip(landmark_dir / f"{clip}_face.npz")
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= 0.5)
    n = len(pose)
    frames, fps = read_frames(clip, n)
    skeleton, corr = models.skeleton, models.corr
    print(f"Clip {clip}: {n} frames")

    # Round trip of the body and the hands, as in view_body_hand_round_trip.
    _, body_fits, body_recon = round_trip_clip(pose, ok, skeleton)
    body_err = np.array([np.mean(list(e.values())) if e else np.nan
                         for e in (frame_errors(pose[t], ok[t], body_recon[t]) for t in range(n))])
    hand_fits, hand_recon, hand_err = hand_round_trip(hands, models.model_dir)
    t_hands = time.time()

    # Head, hands and body assembled into one pose per frame, and hand-to-face geometry.
    result = run_clip(hands, face, models)
    body = skeleton.shaped(result.betas)
    surface = build_head_surface(body, corr, models.model_dir)
    faces = skeleton.model.faces.astype(np.int64)
    albedo = region_tints(faces, surface, skeleton.model.v_template.shape[0])
    print(f"  face pipeline: {int(result.body_ok.sum())}/{n} body fits, "
          f"{sum(f.valid for f in result.head_fits)} faces, "
          f"{sum(f.reliable for f in result.head_fits)} reliable heads, "
          f"{int(result.hand_near_face.sum())} frames with a hand near the face "
          f"({time.time() - t_hands:.0f} s)")
    contours = face_contours(corr, pose_mesh(body, np.zeros((55, 3)), result.betas,
                                              np.zeros(3)).vertices)

    face_px = face.pixels()
    nan105 = np.full((len(corr.mp_indices), 3), np.nan)
    meshes, joints0, green, blue = [None] * n, [], np.full((n,) + nan105.shape, np.nan), \
        np.full((n,) + nan105.shape, np.nan)
    rms = np.full((n, 2), np.nan)
    hand_pt = np.full((n, 2, len(HAND_POINTS), 3), np.nan)
    gap_pt = np.full_like(hand_pt, np.nan)
    signed = np.full((n, 2, len(HAND_POINTS)), np.nan)
    markers = {k: np.full((n, 3), np.nan) for k in ("g_nose", "g_lear", "g_rear")}
    for i in range(n):
        if not result.body_ok[i]:
            continue
        mesh = pose_mesh(body, result.rotations[i], result.betas, result.transl[i])
        meshes[i] = mesh.vertices
        joints0.append(mesh.joints[0])
        green[i] = corr.points(mesh.vertices)
        for key, vid in (("g_nose", SMPLX_NOSE), ("g_lear", SMPLX_EAR["left"]),
                         ("g_rear", SMPLX_EAR["right"])):
            markers[key][i] = mesh.vertices[vid]
        head = result.head_fits[i]
        observed = face_px[i] if face.valid[i] else None
        carried = carry_face(observed, mesh.vertices, head, corr)
        if carried is not None:
            blue[i] = carried
            outlines = hand_outlines(hands.image[i], hands.width, hands.height)
            reference = np.linalg.norm(observed[RIGHT_EYE_OUTER] - observed[LEFT_EYE_OUTER])
            hidden = occluded(observed[corr.mp_indices][:, :2], outlines,
                              OCCLUSION_PAD * reference)
            rms[i] = face_residuals_mm(carried, green[i], corr, head.rigid_weights, hidden)
        points, d, closest = closest_to_head(mesh, surface)
        for s in range(2):
            if np.isfinite(result.geometry[i].distance[s]).all():  # the hand was posed
                hand_pt[i, s], gap_pt[i, s], signed[i, s] = points[s], closest[s], d[s]
    if signed.size and np.isfinite(signed).any():
        mine = np.where(np.isfinite(signed), signed, 0)
        theirs = np.array([np.where(np.isfinite(g.distance), g.distance, 0) if g else
                           np.zeros((2, len(HAND_POINTS))) for g in result.geometry])
        print(f"  closest-point distances vs run_clip's: max diff "
              f"{1000 * np.abs(mine - theirs).max():.4f} mm")

    # The closest hand point per side and frame, by |distance|.
    with np.errstate(invalid="ignore"):
        absd = np.where(np.isfinite(signed), np.abs(signed), np.inf)
    best = absd.argmin(axis=2)                              # (n, 2)
    dmin = np.take_along_axis(signed, best[..., None], 2)[..., 0]  # signed, NaN: no hand
    both = np.nanmin(np.where(np.isfinite(dmin), np.abs(dmin), np.nan), axis=1) \
        if np.isfinite(dmin).any() else np.full(n, np.nan)

    # Stitch the hands onto the wrists: input onto input, round trip onto round trip.
    obs_pose = np.where(ok[..., None], pose, np.nan)
    stitched_in, stitched_rt = {}, {}
    for s, side in enumerate(SIDES):
        wrist = POSE_WRIST[side]
        stitched_in[side] = np.stack([anchored(hands.world[t, s], obs_pose[t, wrist])
                                      for t in range(n)])
        stitched_rt[side] = np.stack([anchored(hand_recon[side][t], body_recon[t, wrist])
                                      for t in range(n)])
    obs_markers = {"o_nose": obs_pose[:, POSE_NOSE], "o_lear": obs_pose[:, POSE_EAR["left"]],
                   "o_rear": obs_pose[:, POSE_EAR["right"]]}

    # Frame the shoulders, wrists, hands and face (not the elbows or hips): that is where the
    # hand-to-face gap is, and it keeps the head large enough to compare the two faces.
    wrists = [11, 12, 15, 16]
    arm_clouds = [obs_pose[:, wrists], body_recon[:, wrists]]
    hand_clouds = list(stitched_in.values()) + list(stitched_rt.values())
    face_clouds = [blue, green] + list(obs_markers.values()) + list(markers.values())
    front = limits(arm_clouds + hand_clouds + face_clouds, "front", pad=1.05)
    side_lim = limits(arm_clouds + hand_clouds + face_clouds, "side", pad=1.05)
    hip_line = np.median(np.array(joints0)[:, 1]) + 0.12  # y is down: keep the upper body
    mesh_pts = [m[m[:, 1] < hip_line] for m in meshes if m is not None]
    mesh_lim = limits([np.concatenate(mesh_pts)], "front", pad=1.05) if mesh_pts else None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_body_hand_face_round_trip.mp4"
    sink = VideoSink(out_path, fps)
    still_frames = {f - 1 for f in still_frames} if still_frames else \
        set(choose_stills(both, stills)) if stills else set()
    first, last = (frame_range[0] - 1, frame_range[1]) if frame_range else (0, n)
    fig = plt.figure(figsize=(21.6, 8.4), dpi=100)
    steps = np.arange(1, n + 1)
    region_legend = [Patch(color=(1 - TINT) * SKIN + TINT * np.asarray(plt.get_cmap("tab10")(r))[:3],
                           label=name) for r, name in enumerate(REGIONS)]
    mp_indices = corr.mp_indices

    for t in range(first, last):
        fig.clf()
        grid = fig.add_gridspec(3, 4, height_ratios=[7, 1, 1], hspace=0.55, wspace=0.05,
                                left=0.045, right=0.98, top=0.95, bottom=0.06)
        ax_vid, ax_front, ax_side, ax_mesh = (fig.add_subplot(grid[0, i]) for i in range(4))
        ax_err, ax_dist = fig.add_subplot(grid[1, :]), fig.add_subplot(grid[2, :])
        head = result.head_fits[t]
        geometry = result.geometry[t]

        # 1. source + MediaPipe 2-D overlay, with the hand-to-face readout.
        image = overlay_mediapipe(frames[t], hands, t)
        if face.valid[t]:
            image = overlay_face(image, face.landmarks[t].astype(np.float64), contours,
                                 mp_indices)
        ax_vid.imshow(image)
        ax_vid.set_title(f"clip {clip}  frame {t + 1}/{n}  (MediaPipe 2-D)", fontsize=10)
        ax_vid.axis("off")
        y = 0.985
        for s, side in enumerate(SIDES):
            if geometry is None or not np.isfinite(dmin[t, s]):
                text, colour = f"{side:<5}: hand not detected", "#555555"
            else:
                d, k = dmin[t, s], best[t, s]
                region = REGIONS[geometry.region[s, k]] if geometry.region[s, k] >= 0 else "-"
                sub = geometry.subregion[s, k]
                where = region + (f"/{SUBREGIONS[sub]}" if sub >= 0 else "")
                contact = abs(d) < CONTACT_DISTANCE
                text = (f"{side:<5}: {100 * d:+6.1f} cm  {HAND_POINTS[k]:<6} {where}"
                        + ("  CONTACT" if contact else ""))
                colour = CONTACT_RED if contact else "black"
            ax_vid.text(0.01, y, text, transform=ax_vid.transAxes, fontsize=8, va="top",
                        family="monospace", color=colour,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", alpha=0.8, lw=0))
            y -= 0.045
        if not face.valid[t]:
            head_text = "head : no face detected"
        elif not result.body_ok[t]:
            head_text = "head : no body fit"
        else:
            head_text = (f"head : {'reliable' if head.reliable else 'UNRELIABLE'}, jaw "
                         f"{'ok' if head.jaw_reliable else 'hidden'}, {head.points_used}/"
                         f"{len(RIGID)} rigid pts")
        ax_vid.text(0.01, 0.01, head_text, transform=ax_vid.transAxes, fontsize=8, va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", alpha=0.8, lw=0))
        if result.hand_near_face[t]:
            ax_vid.text(0.99, 0.985, "HAND NEAR FACE (2-D)", transform=ax_vid.transAxes,
                        fontsize=9, va="top", ha="right", color="white", weight="bold",
                        bbox=dict(boxstyle="square,pad=0.25", fc=CONTACT_RED, lw=0))

        # 2. front and 3. side views.
        for ax, view, lim, title in ((ax_front, "front", front, "front: input vs round trip"),
                                     (ax_side, "side", side_lim, "side (depth →)")):
            draw_bones(ax, obs_pose[t], CONTEXT_BONES, view, "#7f9cc4", 1.2, alpha=0.6)
            draw_bones(ax, obs_pose[t], ARM_BONES, view, INPUT, 3.0, alpha=0.55,
                       label="MediaPipe input")
            for side in SIDES:
                draw_bones(ax, stitched_in[side][t], HAND_BONES, view, INPUT, 1.6, alpha=0.55,
                           size=6)
            draw_bones(ax, body_recon[t], ARM_BONES, view, RECON, 2.0,
                       label="round trip (from parameters)")
            for side in SIDES:
                draw_bones(ax, stitched_rt[side][t], HAND_BONES, view, RECON, 1.1, size=4)
            for feature_pairs in contours.values():
                draw_bones(ax, blue[t], feature_pairs, view, INPUT, 0.8, alpha=0.7)
                draw_bones(ax, green[t], feature_pairs, view, RECON, 0.8, alpha=0.9)
            draw_bones(ax, blue[t], [], view, INPUT, 0, alpha=0.7, size=5)
            draw_bones(ax, green[t], [], view, RECON, 0, alpha=0.9, size=3)
            for key in ("o_nose", "o_lear", "o_rear"):
                if np.isfinite(obs_markers[key][t]).all():
                    xs, ys = _project(obs_markers[key][t][None], view)
                    ax.scatter(xs, ys, s=60, color=INPUT, edgecolors="white", linewidths=0.6,
                               zorder=7, marker="o")
            for key in ("g_nose", "g_lear", "g_rear"):
                if np.isfinite(markers[key][t]).all():
                    xs, ys = _project(markers[key][t][None], view)
                    ax.scatter(xs, ys, s=60, color=RECON, edgecolors="black", linewidths=0.6,
                               zorder=8, marker="^")
            for s in range(2):
                if np.isfinite(dmin[t, s]):
                    k = best[t, s]
                    draw_gap(ax, view, hand_pt[t, s, k], gap_pt[t, s, k],
                             abs(dmin[t, s]) < CONTACT_DISTANCE)
            set_view(ax, lim, title)
        ax_front.scatter([], [], s=40, color=INPUT, marker="o", label="nose, ears: MediaPipe")
        ax_front.scatter([], [], s=40, color=RECON, marker="^", edgecolors="black",
                         label="nose, ears: SMPL-X")
        ax_front.plot([], [], color=GAP_GREY, label="closest hand pt → head (red: contact)")
        ax_front.legend(loc="lower left", fontsize=7, framealpha=0.85)

        lines = ["body : no fit" if body_fits[t] is None else f"body : {body_err[t]:4.1f}% sh.w."]
        for side in SIDES:
            e = hand_err[side][t]
            lines.append(f"{side:<5}: " + ("not detected" if np.isnan(e) else f"{e:4.1f} mm"))
        lines.append("face : " + ("n/a" if np.isnan(rms[t, 0]) else
                                  f"{rms[t, 0]:3.1f} rigid / "
                                  + ("-" if np.isnan(rms[t, 1]) else f"{rms[t, 1]:3.1f}")
                                  + " non-rigid mm"))
        ax_side.text(0.98, 0.02, "\n".join(lines), transform=ax_side.transAxes, ha="right",
                     va="bottom", fontsize=8, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        # 4. the posed SMPL-X mesh, head tinted by region.
        if meshes[t] is not None:
            draw_mesh_tinted(ax_mesh, meshes[t], faces, albedo)
            for s in range(2):
                if np.isfinite(dmin[t, s]):
                    k = best[t, s]
                    draw_gap(ax_mesh, "front", hand_pt[t, s, k], gap_pt[t, s, k],
                             abs(dmin[t, s]) < CONTACT_DISTANCE)
            missing = [side for s, side in enumerate(SIDES) if not np.isfinite(dmin[t, s])]
            if missing:
                ax_mesh.text(0.02, 0.98, f"{' + '.join(missing)} hand not detected: flat",
                             transform=ax_mesh.transAxes, fontsize=8, color="#a33", va="top",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
            ax_mesh.legend(handles=region_legend, loc="lower left", fontsize=6, ncol=2,
                           framealpha=0.85, handlelength=1, columnspacing=0.8)
        else:
            ax_mesh.text(0.5, 0.5, "no body fit", transform=ax_mesh.transAxes, ha="center")
        if mesh_lim is not None:
            set_view(ax_mesh, mesh_lim, "SMPL-X figure, head tinted by ASL-LEX region")
        else:
            ax_mesh.axis("off")

        # 5. error strips.
        ax_err.plot(steps, body_err, color="#555", lw=1.2, label="body (% shoulder width)")
        ax_err.plot(steps, hand_err["left"], color="#e8762b", lw=1.0, label="left hand (mm)")
        ax_err.plot(steps, hand_err["right"], color="#2bb0e8", lw=1.0, label="right hand (mm)")
        ax_err.plot(steps, rms[:, 0], color="#8c4fbf", lw=1.0, label="face rigid RMS (mm)")
        ax_err.plot(steps, rms[:, 1], color="#c43f9a", lw=1.0, ls="--",
                    label="face non-rigid RMS (mm)")
        ax_err.axvline(t + 1, color=CONTACT_RED, lw=1)
        ax_err.set_xlim(1, n)
        top = np.nanmax(np.concatenate([body_err, hand_err["left"], hand_err["right"], rms[:, 0],
                                        rms[:, 1]]))
        ax_err.set_ylim(0, max(top * 1.05, 1.0))
        ax_err.set_ylabel("round-trip\nerror", fontsize=8)
        ax_err.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=5,
                      frameon=False)
        ax_err.tick_params(labelsize=7)
        ax_err.set_xticklabels([])

        ax_dist.axhspan(-100 * CONTACT_DISTANCE, 100 * CONTACT_DISTANCE, color=CONTACT_RED,
                        alpha=0.15, lw=0, label=f"contact (|d| < {100 * CONTACT_DISTANCE:.0f} cm)")
        ax_dist.axhline(0, color="#999", lw=0.6)
        ax_dist.plot(steps, 100 * dmin[:, 0], color="#e8762b", lw=1.0, label="left hand")
        ax_dist.plot(steps, 100 * dmin[:, 1], color="#2bb0e8", lw=1.0, label="right hand")
        ax_dist.axvline(t + 1, color=CONTACT_RED, lw=1)
        ax_dist.set_xlim(1, n)
        finite = dmin[np.isfinite(dmin)]
        ax_dist.set_ylim(min(-5.0, 100 * finite.min() - 1) if len(finite) else -5,
                         min(60.0, 100 * finite.max() + 2) if len(finite) else 60)
        ax_dist.set_ylabel("closest hand\nto head (cm)", fontsize=8)
        ax_dist.set_xlabel("frame", fontsize=8)
        ax_dist.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=3,
                       frameon=False)
        ax_dist.tick_params(labelsize=7)

        sink.add(fig)
        if t in still_frames:
            fig.savefig(out_dir / f"{clip}_frame{t + 1}.png", dpi=100)

    sink.close()
    plt.close(fig)
    capture = cv2.VideoCapture(str(out_path))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = last - first
    capture.release()
    nearest = int(np.nanargmin(both)) if np.isfinite(both).any() else -1
    summary = {
        "clip": clip, "frames": n, "written": written, "seconds": time.time() - started,
        "rigid_median_mm": float(np.nanmedian(rms[:, 0])),
        "nonrigid_median_mm": float(np.nanmedian(rms[:, 1])),
        "min_distance_cm": float(100 * dmin[nearest, np.nanargmin(np.abs(dmin[nearest]))])
        if nearest >= 0 else np.nan,
        "min_frame": nearest + 1, "stills": sorted(t + 1 for t in still_frames),
    }
    print(f"  face RMS median {summary['rigid_median_mm']:.1f} mm rigid / "
          f"{summary['nonrigid_median_mm']:.1f} mm non-rigid; closest hand-to-head "
          f"{summary['min_distance_cm']:+.1f} cm at frame {summary['min_frame']}")
    print(f"  -> {out_path} ({written}/{n_out} frames, {summary['seconds']:.0f} s)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True, nargs="+", help="clip id(s), e.g. 57949")
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--stills", type=int, default=3, metavar="N",
                        help="also save the N frames of closest hand-face approach as PNGs")
    parser.add_argument("--frame-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="debugging: render only these frames (1-based, inclusive)")
    parser.add_argument("--still-frames", type=int, nargs="+", metavar="N",
                        help="save these frames (1-based) as stills instead of the closest ones")
    args = parser.parse_args()
    view_round_trip.VIDEO_DIR = args.video_dir
    models = load_face_models(args.model_dir)
    for clip in args.clip:
        render(clip, args.landmark_dir, args.output_dir, models, args.stills, args.frame_range,
               args.still_frames)


if __name__ == "__main__":
    main()
