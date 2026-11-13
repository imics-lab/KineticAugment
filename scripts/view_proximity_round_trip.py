#!/usr/bin/env python3
"""
Compare three hand-to-face proximity measures of one clip side by side (V1.3), to judge by eye
whether the two fixes for MediaPipe's depth error work.

Each clip is run through face_pipeline.run_clip twice: as before ("original": the hand sits where
MediaPipe's pose-world puts the wrist) and with depth_corrected=True ("corrected": the wrist is
moved to where the image says it is, the body re-fitted). BLUE = MediaPipe input, GREEN = SMPL-X
round trip, as in view_face_round_trip.py. Three numbers per hand and frame:

    image      distance of the nearest hand point to the projected SMPL-X head in the video frame
               (2-D, at the head's depth; 0 when it overlaps); NEAR below 2 cm
    3-D orig   signed distance to the head surface of the original SMPL-X fit; CONTACT below 2 cm
    3-D corr   the same on the depth-corrected fit, with whether the SMPL-X wrist reached the
               image-derived target (and by how much it missed). Contact on a hand that did not
               reach is not trustworthy

Panels:
    1. video, MediaPipe pose / hands / face, and the projected SMPL-X head (front-facing
       triangles, Neck left out, 35% region colours): it should sit on the real face. Per hand,
       the three readouts above.
    2. front and 3. side view, 3-D (side: depth to the right): the corrected pass. Light grey is
       MediaPipe's own wrist and hand, the orange x the image-derived wrist target, green the
       corrected SMPL-X arms and hands; a grey line joins grey wrist to green wrist (the shift).
    4. the corrected SMPL-X figure, head tinted by ASL-LEX region.
    5. strip: the closest hand-to-head distance per frame (minimum over hands) for the three
       measures, the 2 cm band, markers where image NEAR (triangle) and corrected 3-D CONTACT
       (square, hollow when the wrist did not reach) fire, and the corrected line dashed where it
       did not reach.

Usage (from the repo root):
    .venv/bin/python scripts/view_proximity_round_trip.py --clip 57949 08917 63229 63669 05631 12328

Reads <landmark-dir>/<clip>_hands.npz and <clip>_face.npz and the video <video-dir>/<clip>.mp4.
Output: <output-dir>/<clip>_proximity.mp4 and 2-3 stills <clip>_proximity_frame<N>.png at the
frames where image NEAR or corrected 3-D CONTACT first occurs (else the smallest corrected 3-D
distance).
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from view_face_round_trip import (  # noqa: E402  (also sets up matplotlib and cv2)
    CONTACT_RED, GAP_GREY, LANDMARK_DIR, OUTPUT_DIR, SKIN, TINT, VIDEO_DIR, carry_face,
    choose_stills, closest_to_head, draw_gap, draw_mesh_tinted, face_contours, overlay_face,
    region_tints,
)
from view_body_hand_round_trip import (  # noqa: E402
    HAND_BONES, INPUT, POSE_WRIST, RECON, _project, anchored, draw_bones, hand_round_trip,
    limits, overlay_mediapipe, set_view,
)
import view_round_trip  # noqa: E402
from view_round_trip import (  # noqa: E402
    ARM_BONES, CONTEXT_BONES, VideoSink, read_frames,
)
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from kinetic_augment.body_model.assembly import pose_mesh  # noqa: E402
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE  # noqa: E402
from kinetic_augment.body_model.face_correspondence import RIGID  # noqa: E402
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import (  # noqa: E402
    SIDES, load_face_models, run_clip,
)
from kinetic_augment.body_model.face_regions import (  # noqa: E402
    REGIONS, SUBREGIONS, build_head_surface,
)
from kinetic_augment.body_model.hand_face_geometry import (  # noqa: E402
    CONTACT_DISTANCE, HAND_POINTS,
)
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402
from kinetic_augment.body_model.image_proximity import NECK, project_head  # noqa: E402

ORIG_GREY, IMAGE_PURPLE, CORR_GREEN, TARGET_ORANGE = "#8a8a8a", "#8c4fbf", "#2ca02c", "#ff7f0e"
LIGHT_GREY = "#c4c4c4"
HEAD_ALPHA = 0.35
TARGET_HEIGHT = 480  # the source frame is enlarged to at least this many pixels tall


def readout(distance, region, subregion, flag):
    """
    Per side, the closest point by |distance|: (distance m, "Region/Sub", flag); NaN and "-"
    where the hand has no measurement.
    """
    out = []
    for s in range(2):
        d = np.asarray(distance[s], dtype=np.float64)
        if not np.isfinite(d).any():
            out.append((np.nan, "-", False))
            continue
        k = int(np.argmin(np.where(np.isfinite(d), np.abs(d), np.inf)))
        r, sub = int(region[s, k]), int(subregion[s, k])
        where = (REGIONS[r] if r >= 0 else "-") + (f"/{SUBREGIONS[sub]}" if sub >= 0 else "")
        out.append((float(d[k]), where, bool(np.asarray(flag[s]).any())))
    return out


def per_frame_readouts(result):
    """Per frame: the (image, original/corrected 3-D) readouts as lists of two sides."""
    image, three = [], []
    for g, ig in zip(result.geometry, result.image_geometry):
        image.append(readout(ig.distance, ig.region, ig.subregion, ig.near) if ig is not None
                     else [(np.nan, "-", False)] * 2)
        three.append(readout(g.distance, g.region, g.subregion, g.contact) if g is not None
                     else [(np.nan, "-", False)] * 2)
    return image, three


def distances(readouts):
    """(T, 2) distance column of `per_frame_readouts` lists."""
    return np.array([[side[0] for side in frame] for frame in readouts], dtype=np.float64)


def flags(readouts):
    """(T, 2) bool flag column of `per_frame_readouts` lists."""
    return np.array([[side[2] for side in frame] for frame in readouts], dtype=bool)


def overlay_head(image, projected, surface, scale):
    """
    The head surface projected into the image (pixels) coloured by ASL-LEX region at HEAD_ALPHA:
    front-facing triangles only, Neck-only triangles left out (as in the image test), far to
    near. `scale` maps the projection's pixels onto this (enlarged) frame.
    """
    all_faces = np.asarray(surface.faces)
    region = np.asarray(surface.region)
    neck = (region[all_faces] == NECK).all(axis=1)
    keep = np.asarray(projected.visible) & ~neck & (region[all_faces[:, 0]] >= 0)
    faces = all_faces[keep]
    tri = projected.points[faces]                                  # (F, 3, 3)
    order = np.argsort(-tri[:, :, 2].mean(axis=1))
    palette = (np.asarray(plt.get_cmap("tab10").colors)[:, :3] * 255).astype(np.uint8)
    layer = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    corners = np.round(tri[order][:, :, :2] * scale * 16).astype(np.int32)  # 4 fractional bits
    colours = palette[region[faces[order, 0]]]
    for poly, colour in zip(corners, colours):
        c = tuple(int(v) for v in colour)
        cv2.fillConvexPoly(layer, poly, c, cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(mask, poly, 1, cv2.LINE_AA, shift=4)
    blended = image.astype(np.float32)
    on = mask.astype(bool)
    blended[on] = (1 - HEAD_ALPHA) * blended[on] + HEAD_ALPHA * layer[on].astype(np.float32)
    return blended.astype(np.uint8)


def reliable_mask(result):
    """(T, 2) bool: the SMPL-X wrist is where the pose wants it (uncorrected, or reached)."""
    return ~(result.wrist_corrected & ~result.wrist_reached)


def choose_proximity_stills(image_near, corr_contact, corr_dist, count=3, spacing=4):
    """
    Frames (0-based) for the stills: where image NEAR first occurs, where corrected 3-D CONTACT
    first occurs, then the smallest corrected 3-D distances, `spacing` frames apart.
    """
    chosen = []
    for frames in (np.flatnonzero(image_near), np.flatnonzero(corr_contact)):
        if len(frames) and all(abs(int(frames[0]) - c) >= spacing for c in chosen):
            chosen.append(int(frames[0]))
    for t in choose_stills(corr_dist, count + len(chosen), spacing):
        if len(chosen) >= count:
            break
        if all(abs(t - c) >= spacing for c in chosen):
            chosen.append(t)
    return sorted(chosen[:count])


def render(clip, landmark_dir: Path, out_dir: Path, models, frame_range=None, still_frames=None,
           stills=3):
    started = time.time()
    hands = load_hand_clip(landmark_dir / f"{clip}_hands.npz")
    face = load_face_clip(landmark_dir / f"{clip}_face.npz")
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= 0.5)
    n = len(pose)
    frames, fps = read_frames(clip, n)
    skeleton, corr = models.skeleton, models.corr
    print(f"Clip {clip}: {n} frames")

    # Hand round trip (green hands), then the two passes of the face pipeline.
    hand_fits, hand_recon, _ = hand_round_trip(hands, models.model_dir)
    t0 = time.time()
    orig = run_clip(hands, face, models)
    t1 = time.time()
    corrd = run_clip(hands, face, models, depth_corrected=True)
    print(f"  run_clip: original {t1 - t0:.0f} s, corrected {time.time() - t1:.0f} s; "
          f"{int(corrd.wrist_corrected.sum())} hand-frames corrected, "
          f"{int((corrd.wrist_corrected & corrd.wrist_reached).sum())} reached")

    body = skeleton.shaped(corrd.betas)
    surface = build_head_surface(body, corr, models.model_dir)
    faces = skeleton.model.faces.astype(np.int64)
    albedo = region_tints(faces, surface, skeleton.model.v_template.shape[0])
    contours = face_contours(corr, pose_mesh(body, np.zeros((55, 3)), corrd.betas,
                                              np.zeros(3)).vertices)

    # Readouts of the three measures.
    _, orig_ro = per_frame_readouts(orig)
    image_ro, corr_ro = per_frame_readouts(corrd)  # the image measure: corrected pass's head
    d_orig, d_image, d_corr = distances(orig_ro), distances(image_ro), distances(corr_ro)
    near, contact_orig, contact_corr = flags(image_ro), flags(orig_ro), flags(corr_ro)
    reliable = reliable_mask(corrd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s_orig = np.nanmin(np.abs(d_orig), axis=1)
        s_image = np.nanmin(d_image, axis=1)
        s_corr = np.nanmin(np.abs(d_corr), axis=1)
    near_any = near.any(axis=1)
    contact_any = contact_corr.any(axis=1)
    trusted_contact = (contact_corr & reliable).any(axis=1)
    # reliability of the hand that gives the strip's corrected minimum
    best_hand = np.array([int(np.argmin(np.where(np.isfinite(r), np.abs(r), np.inf)))
                          for r in d_corr])
    strip_reliable = reliable[np.arange(n), best_hand]

    # The corrected pass: meshes, SMPL-X arms, face carry, hand-to-head gap lines.
    nan105 = np.full((len(corr.mp_indices), 3), np.nan)
    meshes = [None] * n
    green, blue = np.full((n,) + nan105.shape, np.nan), np.full((n,) + nan105.shape, np.nan)
    recon_arm = np.full((n, 33, 3), np.nan)
    hand_pt = np.full((n, 2, len(HAND_POINTS), 3), np.nan)
    gap_pt = np.full_like(hand_pt, np.nan)
    signed = np.full((n, 2, len(HAND_POINTS)), np.nan)
    projections = [None] * n
    joints0 = []
    face_px = face.pixels()
    for i in range(n):
        if not corrd.body_ok[i]:
            continue
        mesh = pose_mesh(body, corrd.rotations[i], corrd.betas, corrd.transl[i])
        meshes[i] = mesh.vertices
        joints0.append(mesh.joints[0])
        green[i] = corr.points(mesh.vertices)
        for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
            recon_arm[i, mp_idx] = mesh.joints[smplx_idx]
        head = corrd.head_fits[i]
        observed = face_px[i] if face.valid[i] else None
        carried = carry_face(observed, mesh.vertices, head, corr)
        if carried is not None:
            blue[i] = carried
        if corrd.image_geometry[i] is not None:
            projections[i] = project_head(
                mesh.vertices[surface.vertex_ids], surface.faces,
                corr.points(mesh.vertices, RIGID), observed[list(RIGID)], head.rigid_weights)
        points, d, closest = closest_to_head(mesh, surface)
        for s in range(2):
            if corrd.geometry[i] is not None and np.isfinite(corrd.geometry[i].distance[s]).all():
                hand_pt[i, s], gap_pt[i, s], signed[i, s] = points[s], closest[s], d[s]
    with np.errstate(invalid="ignore"):
        absd = np.where(np.isfinite(signed), np.abs(signed), np.inf)
    best = absd.argmin(axis=2)

    obs_pose = np.where(ok[..., None], pose, np.nan)
    stitched_in, stitched_rt = {}, {}
    for s, side in enumerate(SIDES):
        wrist = POSE_WRIST[side]
        stitched_in[side] = np.stack([anchored(hands.world[t, s], obs_pose[t, wrist])
                                      for t in range(n)])
        stitched_rt[side] = np.stack([anchored(hand_recon[side][t], recon_arm[t, wrist])
                                      for t in range(n)])
    targets = corrd.wrist_target                                   # (n, 2, 3)

    wrists = [11, 12, 15, 16]
    clouds = [obs_pose[:, wrists], recon_arm[:, wrists], targets, blue, green] \
        + list(stitched_in.values()) + list(stitched_rt.values())
    front, side_lim = limits(clouds, "front", pad=1.05), limits(clouds, "side", pad=1.05)
    hip_line = np.median(np.array(joints0)[:, 1]) + 0.12
    mesh_pts = [m[m[:, 1] < hip_line] for m in meshes if m is not None]
    mesh_lim = limits([np.concatenate(mesh_pts)], "front", pad=1.05) if mesh_pts else None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip}_proximity.mp4"
    sink = VideoSink(out_path, fps)
    if still_frames:
        still_set = {f - 1 for f in still_frames}
    else:
        still_set = set(choose_proximity_stills(near_any, contact_any, s_corr, stills)) \
            if stills else set()
    first, last = (frame_range[0] - 1, frame_range[1]) if frame_range else (0, n)
    fig = plt.figure(figsize=(24, 8.6), dpi=100)
    steps = np.arange(1, n + 1)
    region_legend = [
        Patch(color=(1 - TINT) * SKIN + TINT * np.asarray(plt.get_cmap("tab10")(r))[:3],
              label=name) for r, name in enumerate(REGIONS)]
    top = np.nanmax(np.concatenate([s_orig, s_image, s_corr])) \
        if np.isfinite(np.concatenate([s_orig, s_image, s_corr])).any() else 10.0
    strip_top = min(100 * top * 1.05, 60.0)

    for t in range(first, last):
        fig.clf()
        grid = fig.add_gridspec(2, 4, height_ratios=[7, 1.9], width_ratios=[1.7, 1, 1, 1],
                                hspace=0.22, wspace=0.05,
                                left=0.045, right=0.98, top=0.95, bottom=0.07)
        ax_vid, ax_front, ax_side, ax_mesh = (fig.add_subplot(grid[0, i]) for i in range(4))
        ax_strip = fig.add_subplot(grid[1, :])

        # 1. video, MediaPipe overlay, projected head, readouts.
        frame = frames[t]
        k = max(1, int(round(TARGET_HEIGHT / frame.shape[0])))
        if k > 1:
            frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        if projections[t] is not None:
            frame = overlay_head(frame, projections[t], surface, frame.shape[1] / hands.width)
        image = overlay_mediapipe(frame, hands, t)
        if face.valid[t]:
            image = overlay_face(image, face.landmarks[t].astype(np.float64), contours,
                                 corr.mp_indices)
        ax_vid.imshow(image)
        ax_vid.set_title(f"clip {clip}  frame {t + 1}/{n}  (video, projected SMPL-X head)",
                         fontsize=10)
        ax_vid.axis("off")
        y = 0.985
        for s, side in enumerate(SIDES):
            d_i, where_i, near_i = image_ro[t][s]
            d_o, where_o, con_o = orig_ro[t][s]
            d_c, where_c, con_c = corr_ro[t][s]
            if corrd.wrist_corrected[t, s]:
                tail = (f"  ({'reached' if corrd.wrist_reached[t, s] else 'NOT reached'}, "
                        f"residual {100 * corrd.wrist_residual_m[t, s]:.1f} cm)")
            else:
                tail = "  (wrist not corrected)"
            rows = [
                (f"{side:<5} image  : " + ("n/a" if not np.isfinite(d_i) else
                 f"{100 * d_i:5.1f} cm {where_i} " + ("NEAR" if near_i else "-")),
                 CONTACT_RED if near_i else "black"),
                ("      3-D orig: " + ("n/a" if not np.isfinite(d_o) else
                 f"{100 * d_o:+5.1f} cm {where_o} " + ("CONTACT" if con_o else "-")),
                 CONTACT_RED if con_o else "black"),
                ("      3-D corr: " + ("n/a" if not np.isfinite(d_c) else
                 f"{100 * d_c:+5.1f} cm {where_c} " + ("CONTACT" if con_c else "-") + tail),
                 ("#b35c00" if con_c and corrd.wrist_corrected[t, s]
                  and not corrd.wrist_reached[t, s] else CONTACT_RED) if con_c else "black"),
            ]
            for text, colour in rows:
                ax_vid.text(0.01, y, text, transform=ax_vid.transAxes, fontsize=7.5, va="top",
                            family="monospace", color=colour,
                            bbox=dict(boxstyle="square,pad=0.12", fc="white", alpha=0.85, lw=0))
                y -= 0.037
            y -= 0.012
        if not face.valid[t]:
            note = "no face detected"
        elif projections[t] is None:
            note = "no head projection"
        else:
            note = f"head scale {corrd.head_scale_px_per_m[t]:.0f} px/m"
        ax_vid.text(0.01, 0.01, note, transform=ax_vid.transAxes, fontsize=8, va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", alpha=0.8, lw=0))

        # 2. front and 3. side view of the corrected pass, with the original wrist in grey.
        for ax, view, lim, title in (
                (ax_front, "front", front, "front: input, original, corrected"),
                (ax_side, "side", side_lim, "side (depth →)")):
            draw_bones(ax, obs_pose[t], CONTEXT_BONES, view, "#7f9cc4", 1.2, alpha=0.6)
            draw_bones(ax, obs_pose[t], ARM_BONES, view, INPUT, 3.0, alpha=0.5,
                       label="MediaPipe input arms")
            for side in SIDES:
                draw_bones(ax, stitched_in[side][t], HAND_BONES, view, LIGHT_GREY, 1.8, alpha=0.95,
                           size=7)
            draw_bones(ax, recon_arm[t], ARM_BONES, view, RECON, 2.0,
                       label="corrected SMPL-X arms")
            for side in SIDES:
                draw_bones(ax, stitched_rt[side][t], HAND_BONES, view, RECON, 1.1, size=4)
            for feature_pairs in contours.values():
                draw_bones(ax, blue[t], feature_pairs, view, INPUT, 0.8, alpha=0.7)
                draw_bones(ax, green[t], feature_pairs, view, RECON, 0.8, alpha=0.9)
            draw_bones(ax, blue[t], [], view, INPUT, 0, alpha=0.7, size=5)
            draw_bones(ax, green[t], [], view, RECON, 0, alpha=0.9, size=3)
            for s, side in enumerate(SIDES):
                wrist = POSE_WRIST[side]
                if np.isfinite(obs_pose[t, wrist]).all():
                    xs, ys = _project(obs_pose[t, wrist][None], view)
                    ax.scatter(xs, ys, s=55, color=LIGHT_GREY, edgecolors="#555", linewidths=0.7,
                               zorder=7)
                if corrd.wrist_corrected[t, s] and np.isfinite(targets[t, s]).all():
                    pts = np.stack([obs_pose[t, wrist], targets[t, s]])
                    xs, ys = _project(pts, view)
                    ax.plot(xs, ys, color=LIGHT_GREY, lw=1.0, ls=":", zorder=5)
                    ax.scatter(xs[1:], ys[1:], s=70, color=TARGET_ORANGE, marker="x",
                               linewidths=2.2, zorder=9)
            for s in range(2):
                if np.isfinite(signed[t, s]).any():
                    kk = best[t, s]
                    draw_gap(ax, view, hand_pt[t, s, kk], gap_pt[t, s, kk],
                             abs(signed[t, s, kk]) < CONTACT_DISTANCE)
            set_view(ax, lim, title)
        ax_front.scatter([], [], s=45, color=LIGHT_GREY, edgecolors="#555",
                         label="MediaPipe wrist + hand (original)")
        ax_front.scatter([], [], s=50, color=TARGET_ORANGE, marker="x",
                         label="wrist target (image)")
        ax_front.plot([], [], color=GAP_GREY, label="closest hand pt → head (red: contact)")
        ax_front.legend(loc="lower left", fontsize=7, framealpha=0.85)
        shifts = []
        for s, side in enumerate(SIDES):
            if corrd.wrist_corrected[t, s]:
                shifts.append(f"{side:<5}: shift {100 * corrd.wrist_shift_m[t, s]:4.1f} cm, "
                              f"{'reached' if corrd.wrist_reached[t, s] else 'NOT reached'} "
                              f"(res {100 * corrd.wrist_residual_m[t, s]:.1f} cm)")
            else:
                shifts.append(f"{side:<5}: not corrected")
        ax_side.text(0.98, 0.02, "\n".join(shifts), transform=ax_side.transAxes, ha="right",
                     va="bottom", fontsize=8, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        # 4. the corrected SMPL-X figure, head tinted by region.
        if meshes[t] is not None:
            draw_mesh_tinted(ax_mesh, meshes[t], faces, albedo)
            for s in range(2):
                if np.isfinite(signed[t, s]).any():
                    kk = best[t, s]
                    draw_gap(ax_mesh, "front", hand_pt[t, s, kk], gap_pt[t, s, kk],
                             abs(signed[t, s, kk]) < CONTACT_DISTANCE)
            ax_mesh.legend(handles=region_legend, loc="lower left", fontsize=6, ncol=2,
                           framealpha=0.85, handlelength=1, columnspacing=0.8)
        else:
            ax_mesh.text(0.5, 0.5, "no body fit", transform=ax_mesh.transAxes, ha="center")
        if mesh_lim is not None:
            set_view(ax_mesh, mesh_lim, "corrected SMPL-X figure, head tinted by region")
        else:
            ax_mesh.axis("off")

        # 5. strip: closest hand-to-head distance, three measures.
        ax_strip.axhspan(0, 100 * CONTACT_DISTANCE, color=CONTACT_RED, alpha=0.15, lw=0,
                         label=f"{100 * CONTACT_DISTANCE:.0f} cm threshold")
        ax_strip.plot(steps, 100 * s_orig, color=ORIG_GREY, lw=1.4, label="3-D original")
        ax_strip.plot(steps, 100 * s_image, color=IMAGE_PURPLE, lw=1.4,
                      label="image (2-D, at head depth)")
        ax_strip.plot(steps, 100 * s_corr, color=CORR_GREEN, lw=1.4, ls="--",
                      label="3-D corrected (dashed: wrist did not reach)")
        ax_strip.plot(steps, 100 * np.where(strip_reliable, s_corr, np.nan), color=CORR_GREEN,
                      lw=1.6)
        marks = np.flatnonzero(near_any)
        ax_strip.scatter(marks + 1, 100 * s_image[marks], marker="^", s=26, color=IMAGE_PURPLE,
                         zorder=5, label="image NEAR")
        for m in np.flatnonzero(contact_any):
            ok_m = trusted_contact[m]
            ax_strip.scatter([m + 1], [100 * s_corr[m]], marker="s", s=30, zorder=6,
                             facecolors=CORR_GREEN if ok_m else "white", edgecolors=CORR_GREEN,
                             linewidths=1.2)
        ax_strip.scatter([], [], marker="s", s=30, color=CORR_GREEN, label="corrected CONTACT")
        ax_strip.scatter([], [], marker="s", s=30, facecolors="white", edgecolors=CORR_GREEN,
                         label="CONTACT, wrist not reached")
        ax_strip.axvline(t + 1, color=CONTACT_RED, lw=1)
        ax_strip.set_xlim(1, n)
        ax_strip.set_ylim(0, strip_top)
        ax_strip.set_ylabel("closest hand\nto head (cm)", fontsize=8)
        ax_strip.set_xlabel("frame", fontsize=8)
        ax_strip.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=7,
                        frameon=False)
        ax_strip.tick_params(labelsize=7)

        sink.add(fig)
        if t in still_set:
            fig.savefig(out_dir / f"{clip}_proximity_frame{t + 1}.png", dpi=100)

    sink.close()
    plt.close(fig)
    capture = cv2.VideoCapture(str(out_path))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    summary = {
        "clip": clip, "frames": n, "written": written, "expected": last - first,
        "seconds": time.time() - started, "stills": sorted(t + 1 for t in still_set),
        "image_near_frames": int(near_any.sum()), "corr_contact_frames": int(contact_any.sum()),
        "corr_contact_trusted_frames": int(trusted_contact.sum()),
        "orig_contact_frames": int(contact_orig.any(axis=1).sum()),
        "min_orig_cm": float(100 * np.nanmin(s_orig)) if np.isfinite(s_orig).any() else np.nan,
        "min_image_cm": float(100 * np.nanmin(s_image)) if np.isfinite(s_image).any() else np.nan,
        "min_corr_cm": float(100 * np.nanmin(s_corr)) if np.isfinite(s_corr).any() else np.nan,
        "corrected_hand_frames": int(corrd.wrist_corrected.sum()),
        "reached_hand_frames": int((corrd.wrist_corrected & corrd.wrist_reached).sum()),
    }
    print(f"  frames with image NEAR {summary['image_near_frames']}, corrected CONTACT "
          f"{summary['corr_contact_frames']} ({summary['corr_contact_trusted_frames']} reached), "
          f"original CONTACT {summary['orig_contact_frames']}; min cm orig/image/corr "
          f"{summary['min_orig_cm']:.1f}/{summary['min_image_cm']:.1f}/"
          f"{summary['min_corr_cm']:.1f}")
    print(f"  -> {out_path} ({written}/{summary['expected']} frames, {summary['seconds']:.0f} s)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True, nargs="+", help="clip id(s), e.g. 57949")
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--stills", type=int, default=3, metavar="N",
                        help="save N stills (default 3): first image NEAR, first corrected "
                             "CONTACT, then the smallest corrected 3-D distance")
    parser.add_argument("--frame-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="debugging: render only these frames (1-based, inclusive)")
    parser.add_argument("--still-frames", type=int, nargs="+", metavar="N",
                        help="save these frames (1-based) as stills instead")
    args = parser.parse_args()
    view_round_trip.VIDEO_DIR = args.video_dir
    models = load_face_models(args.model_dir)
    for clip in args.clip:
        render(clip, args.landmark_dir, args.output_dir, models, args.frame_range,
               args.still_frames, args.stills)


if __name__ == "__main__":
    main()
