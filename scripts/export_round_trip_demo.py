#!/usr/bin/env python3
"""
Export one clip's full MediaPipe -> SMPL-X -> MediaPipe round trip for the web viewer.

Per clip it:

  1. Extracts MediaPipe pose + hands (legacy Holistic + Hands, `extract_hand_landmarks`) and the
     face mesh (Tasks FaceLandmarker, `extract_face_landmarks`), cached as hands.npz / face.npz.
  2. Runs the V1.3 pipeline (`face_pipeline.run_clip`): one SMPL-X per frame with the arms from
     the pose, MANO hands attached at the wrists and the head / neck / jaw from the face mesh.
  3. Reads MediaPipe's points back off that one posed SMPL-X mesh -- arms 11-16 from the body
     joints, both hands' 21 points from the hand joints and fingertip vertices, the 105 embedded
     face points from the FLAME correspondence -- and scores them against the extraction:
        body   % of shoulder width, per arm landmark (MediaPipe pose world)
        hands  mm, wrist-relative landmarks 1-20 (SMPL-X hand; the MANO hand alone for reference)
        face   mm RMS, rigid / non-rigid points, observed face carried onto the SMPL-X head
  4. Measures how well the three MediaPipe streams agree with each other in the image, before
     any fitting: pose wrist vs hand wrist, and pose face points vs the face mesh's equivalents.
  5. Applies the Tier 1 augmentations (`kinetic_augment.augmentations.tier1`) to the unsmoothed
     round trip, each drawn once per clip: view, mirror, duration, arm joints, scale, uncentre,
     and all of them combined. Arm joints edit the SMPL-X rotations and re-pose the mesh; the
     others act on the mesh and the read-back points. A weak-perspective camera fitted to the
     clip projects the augmented points back into the image.

Writes <out>/clips/<clip>/{v/<smoothing>/, aug/<augmentation>/}{clip.json, data.bin} and
<out>/clips/<clip>/frames/NNNN.jpg, and updates <out>/clips.json. data.bin holds the arrays listed
in clip.json["arrays"] (little-endian, C order).

Usage (from the repo root; see demo/round_trip/README.md):
    python scripts/export_round_trip_demo.py --clips 20983 --video-dir path/to/videos
    python scripts/export_round_trip_demo.py --sample 50 --video-dir path/to/videos
    python scripts/export_round_trip_demo.py --reindex
"""

import argparse
import dataclasses
import json
import random
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

import cv2  # noqa: E402

from kinetic_augment.augmentations.tier1 import (  # noqa: E402
    ARM_JOINTS,
    TIER1,
    ArmJointDraw,
    WeakCamera,
    about,
    duration_times,
    fit_weak_camera,
    frame_loss,
    in_frame,
    perturb_arm_joints,
    reflect_x,
    resample,
    resample_points,
    sample_tier1,
    uncentre_shift,
    view_rotation,
)
from kinetic_augment.body_model.alignment import (  # noqa: E402
    apply_similarity,
    similarity_transform,
)
from kinetic_augment.body_model.assembly import pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import (  # noqa: E402
    INNER_LIP,
    LEFT_EYE,
    LEFT_EYE_OUTER,
    LEFT_EYEBROW,
    NOSE,
    OUTER_LIP,
    RIGHT_EYE,
    RIGHT_EYE_OUTER,
    RIGHT_EYEBROW,
    RIGID,
)
from kinetic_augment.body_model.face_observations import (  # noqa: E402
    load_face_clip,
    save_face_clip,
)
from kinetic_augment.body_model import face_pipeline  # noqa: E402
from kinetic_augment.body_model.face_pipeline import SIDES, load_face_models  # noqa: E402
from kinetic_augment.body_model.hand_converter import landmark_errors_mm  # noqa: E402
from kinetic_augment.body_model.hand_model import (  # noqa: E402
    POSE_SLOT_LANDMARKS,
    SMPLX_TIP_VERTICES,
    SMPLX_WRIST_JOINT,
    TIPS,
)
from kinetic_augment.body_model.hand_observations import (  # noqa: E402
    load_hand_clip,
    save_hand_clip,
)
from kinetic_augment.utils.smoothing import SMOOTHING  # noqa: E402
from kinetic_augment.body_model.head_converter import (  # noqa: E402
    OCCLUSION_PAD,
    hand_outlines,
    occluded,
)

VIDEO_DIR = Path("data/metsis_new_unseen_sign_videos")
OUTPUT_DIR = Path("demo/round_trip")
SEED = 20261017
VISIBILITY = 0.5
ARM_MP = (11, 12, 13, 14, 15, 16)
ARM_SMPLX = (16, 17, 18, 19, 20, 21)  # converter.ARM_CORRESPONDENCE
POSE_WRISTS = (15, 16)
HAND_FIRST_JOINT = {"left": 25, "right": 40}
# MediaPipe pose face point -> face-mesh point at roughly the same spot (subject's left/right).
POSE_FACE_PAIRS = (
    (0, 1, "nose"),
    (1, 362, "L eye inner"),
    (3, 263, "L eye outer"),
    (4, 133, "R eye inner"),
    (6, 33, "R eye outer"),
    (9, 291, "mouth L"),
    (10, 61, "mouth R"),
)
SCALE_I16 = 10000.0  # mesh vertices stored as int16 in 0.1 mm
NOSE_BRIDGE, NOSE_BASE = (168, 6, 197, 195, 5, 4), (129, 98, 97, 2, 326, 327, 358)
ARM_MIRROR = [1, 0, 3, 2, 5, 4]  # ARM_MP with left and right swapped
AUGMENTATIONS = {
    "view": ("view", "yaw ±20° about the vertical axis through the shoulders, pitch ±5°"),
    "mirror": ("mirror", "left/right reflection with the sides swapped (always applied here)"),
    "duration": ("duration", "the clip's duration scaled by one factor, 0.8-1.25×; nothing cropped"),
    "arm_joints": ("arm joints", "shoulder, elbow and wrist of each arm, each p = 0.3, ±4°"),
    "scale": ("scale", "the whole scene about the shoulders, 0.85-1.15 (camera distance)"),
    "uncentre": ("uncentre", "the whole scene shifted up to ±0.1 of the frame, hands kept in it"),
    "combined": ("combined", "arm joints, mirror (if drawn), view, scale, uncentre, duration"),
}


def face_contours(corr, rest_vertices):
    """
    Contour segments of the 105 embedded face points, {feature: [(row, row), ...]} with rows into
    the 105-point array. Eyes and brows are ordered into loops by angle about their centre on the
    rest mesh; the lips are already loops; the nose is a bridge and a base.
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


def face_residuals_mm(blue, green, corr, weights, hidden):
    """
    RMS distance (mm) between the carried observed face points and the SMPL-X ones: over the
    rigid points the head fit used, and over the other (non-rigid) points not under a hand.
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


def extract(video: Path, cache: Path, reextract: bool):
    """MediaPipe hands + pose and face mesh for one clip, cached in `cache`."""
    from extract_face_landmarks import extract_face
    from extract_hand_landmarks import HANDS_SETTINGS, HOLISTIC_SETTINGS, extract_hands
    import mediapipe as mp

    hands_path, face_path = cache / "hands.npz", cache / "face.npz"
    if reextract or not hands_path.exists():
        with (
            mp.solutions.hands.Hands(**HANDS_SETTINGS) as hands,
            mp.solutions.holistic.Holistic(**HOLISTIC_SETTINGS) as holistic,
        ):
            clip = extract_hands(video, hands, holistic)
        clip.meta = {"source": video.name, "mediapipe": mp.__version__}
        save_hand_clip(hands_path, clip)
    if reextract or not face_path.exists():
        clip = extract_face(video)
        clip.meta = {"source": video.name, "mediapipe": mp.__version__}
        save_face_clip(face_path, clip)
    return load_hand_clip(hands_path), load_face_clip(face_path)


def mesh_hand_points(mesh, side):
    """MediaPipe's 21 hand points read off the posed SMPL-X mesh (joints + fingertip vertices)."""
    out = np.full((21, 3), np.nan)
    out[0] = mesh.joints[SMPLX_WRIST_JOINT[side]]
    first = HAND_FIRST_JOINT[side]
    for slot, landmark in enumerate(POSE_SLOT_LANDMARKS):
        out[landmark] = mesh.joints[first + slot]
    out[list(TIPS)] = mesh.vertices[list(SMPLX_TIP_VERTICES[side])]
    return out


def carry_all(face_px_t, mesh_vertices, head, corr):
    """All 478 observed face points in the body frame, by the head fit's rigid similarity."""
    weights = head.rigid_weights
    if face_px_t is None or not head.valid or weights is None or (weights > 0).sum() < 3:
        return None
    source = face_px_t[list(RIGID)]
    if not np.isfinite(source).all():
        return None
    scale, rotation, translation = similarity_transform(
        source, corr.points(mesh_vertices, RIGID), weights=weights
    )
    return apply_similarity(face_px_t, scale, rotation, translation)


def read_frames(video: Path, out_dir: Path, n: int, quality=85):
    """Decode every frame (the same cv2 decode the extraction used) to NNNN.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{count:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        count += 1
    capture.release()
    if count != n:
        raise ValueError(f"{video}: decoded {count} frames, landmarks have {n}")


def json_safe(obj):
    """Nested dicts / lists with every non-finite float replaced by None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def connections():
    import mediapipe as mp

    def pairs(conn):
        return sorted([int(a), int(b)] for a, b in conn)

    return {
        "pose": pairs(mp.solutions.pose.POSE_CONNECTIONS),
        "hand": pairs(mp.solutions.hands.HAND_CONNECTIONS),
        "face_contours": pairs(mp.solutions.face_mesh.FACEMESH_CONTOURS),
        "face_irises": pairs(mp.solutions.face_mesh.FACEMESH_IRISES),
    }


def nan_list(a, digits=3):
    """JSON-safe nested list; NaN -> None."""
    a = np.asarray(a, dtype=np.float64)
    return (
        [None if not np.isfinite(v) else round(float(v), digits) for v in a.ravel()]
        if a.ndim <= 1
        else [nan_list(row, digits) for row in a]
    )


def smoothed(hands, face, name):
    """Copies of the hand (with pose) and face clips with every landmark track smoothed."""
    if name == "off":
        return hands, face
    apply, fps = SMOOTHING[name].apply, hands.fps

    def f(x):
        return apply(x, fps).astype(np.asarray(x).dtype)

    hands = replace(
        hands,
        world=f(hands.world),
        image=f(hands.image),
        pose_world=f(hands.pose_world),
        pose_image=f(hands.pose_image),
    )
    return hands, replace(face, landmarks=f(face.landmarks))


def run_with_mano(hands, face, models):
    """`run_clip`, keeping the MANO reconstruction it fits on the way (saves a second fit)."""
    captured = {}
    original = face_pipeline._fit_hands

    def capture(*args):
        captured["fits"], captured["recon"] = original(*args)
        return captured["fits"], captured["recon"]

    face_pipeline._fit_hands = capture
    try:
        result = face_pipeline.run_clip(hands, face, models)
    finally:
        face_pipeline._fit_hands = original
    return result, captured["recon"]


def jitter_mm(points):
    """(T,) mean over points of |second difference| in mm per frame²; NaN at the ends."""
    p = np.asarray(points, dtype=np.float64).reshape(len(points), -1, 3)
    out = np.full(len(p), np.nan)
    acc = 1000 * np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
    with np.errstate(all="ignore"):
        out[1:-1] = np.nanmean(acc, axis=1)
    return out


def export_clip(clip_id, video_dir: Path, out_root: Path, models, variants, reextract=False,
                augmentations=(), seed=SEED, sample=0):
    started = time.time()
    video = video_dir / f"{clip_id}.mp4"
    out = out_root / "clips" / clip_id
    out.mkdir(parents=True, exist_ok=True)
    raw_hands, raw_face = extract(video, out, reextract)
    n = len(raw_face.valid)
    read_frames(video, out / "frames", n)
    print(
        f"clip {clip_id}: {n} frames, {raw_hands.width}x{raw_hands.height} @ "
        f"{raw_hands.fps:.1f} fps (extracted {time.time() - started:.0f} s)"
    )
    trips = {}
    for name in variants:
        hands, face = smoothed(raw_hands, raw_face, name)
        _, trips[name] = export_variant(
            clip_id, name, hands, face, raw_hands, raw_face, out / "v" / name, models
        )
    if augmentations:
        source = "off" if "off" in trips else variants[0]
        export_augmentations(clip_id, trips[source], source, out / "aug", models, augmentations,
                             seed, sample)


def write_data(path: Path, arrays):
    """Write {name: (array, dtype)} back to back, 4-byte aligned; the manifest and total size."""
    manifest, offset = {}, 0
    with open(path, "wb") as fh:
        for name, (a, dtype) in arrays.items():
            a = np.ascontiguousarray(a)
            pad = (-offset) % 4
            fh.write(b"\0" * pad)
            offset += pad
            fh.write(a.tobytes())
            manifest[name] = {"dtype": dtype, "shape": list(a.shape), "offset": offset}
            offset += a.nbytes
    return manifest, offset


def export_variant(clip_id, variant, hands, face, raw_hands, raw_face, out: Path, models):
    """
    One smoothing setting of one clip: fit, read back, score, write clip.json + data.bin.

    Returns the summary and the round trip (`RoundTrip`) the augmentations start from.
    """
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    n, W, H = len(face.valid), hands.width, hands.height
    skeleton, corr = models.skeleton, models.corr
    result, mano_recon = run_with_mano(hands, face, models)
    body = skeleton.shaped(result.betas)
    face_px = face.pixels()

    pose_w = hands.pose_world.astype(np.float64)
    vis = hands.pose_visibility.astype(np.float64)
    ok = np.isfinite(pose_w).all(axis=-1) & (vis >= VISIBILITY)

    V = skeleton.model.v_template.shape[0]
    verts = np.zeros((n, V, 3), np.int16)
    verts_m = np.full((n, V, 3), np.nan)
    rt_arm = np.full((n, 6, 3), np.nan)
    rt_hands = np.full((n, 2, 21, 3), np.nan)
    rt_face = np.full((n, len(corr.mp_indices), 3), np.nan)
    in_face = np.full((n, 478, 3), np.nan)
    in_hands = np.full((n, 2, 21, 3), np.nan)
    err_arm = np.full((n, 6), np.nan)
    err_hand_pts = np.full((n, 2, 21), np.nan)
    err_face_pts = np.full((n, len(corr.mp_indices)), np.nan)
    face_rms = np.full((n, 2), np.nan)
    hand_mm = np.full((n, 2), np.nan)
    mano_mm = np.full((n, 2), np.nan)
    shoulder_m = np.full(n, np.nan)
    head_reliable = np.zeros(n, bool)
    for t in range(n):
        for s, side in enumerate(SIDES):
            wrist = POSE_WRISTS[s]
            observed = hands.world[t, s].astype(np.float64)
            if np.isfinite(observed).all() and ok[t, wrist]:
                in_hands[t, s] = observed - observed[0] + pose_w[t, wrist]
            if np.isfinite(observed).all():
                mano_mm[t, s] = np.nanmean(landmark_errors_mm(mano_recon[side][t], observed))
        if not result.body_ok[t]:
            continue
        mesh = pose_mesh(body, result.rotations[t], result.betas, result.transl[t])
        verts[t] = np.clip(np.round(mesh.vertices * SCALE_I16), -32767, 32767)
        verts_m[t] = mesh.vertices
        rt_arm[t] = mesh.joints[list(ARM_SMPLX)]
        if ok[t, 11] and ok[t, 12]:
            shoulder_m[t] = np.linalg.norm(pose_w[t, 11] - pose_w[t, 12])
            for k, mp_id in enumerate(ARM_MP):
                if ok[t, mp_id]:
                    err_arm[t, k] = (
                        100 * np.linalg.norm(rt_arm[t, k] - pose_w[t, mp_id]) / shoulder_m[t]
                    )
        for s, side in enumerate(SIDES):
            observed = hands.world[t, s].astype(np.float64)
            if not np.isfinite(observed).all():
                continue  # no hand: the SMPL-X hand stays flat, nothing to compare
            rt_hands[t, s] = mesh_hand_points(mesh, side)
            e = landmark_errors_mm(rt_hands[t, s], observed)
            err_hand_pts[t, s, 1:] = e
            hand_mm[t, s] = np.nanmean(e)
        rt_face[t] = corr.points(mesh.vertices)
        head = result.head_fits[t]
        head_reliable[t] = bool(head.valid and head.reliable)
        observed = face_px[t] if face.valid[t] else None
        carried = carry_all(observed, mesh.vertices, head, corr)
        if carried is not None:
            in_face[t] = carried
            blue = carried[corr.mp_indices]
            err_face_pts[t] = 1000 * np.linalg.norm(blue - rt_face[t], axis=1)
            outlines = hand_outlines(hands.image[t], W, H)
            ref = np.linalg.norm(observed[RIGHT_EYE_OUTER] - observed[LEFT_EYE_OUTER])
            hidden = occluded(observed[corr.mp_indices][:, :2], outlines, OCCLUSION_PAD * ref)
            face_rms[t] = face_residuals_mm(blue, rt_face[t], corr, head.rigid_weights, hidden)

    # Stream agreement in the image, before any fitting (pixels).
    wh = np.array([W, H], np.float64)
    pose_px = hands.pose_image[..., :2].astype(np.float64) * wh
    hand_px = hands.image[..., :2].astype(np.float64) * wh
    shoulder_px = np.linalg.norm(pose_px[:, 11] - pose_px[:, 12], axis=-1)
    wrist_gap_px = np.full((n, 2), np.nan)
    face_gap_px = np.full((n, len(POSE_FACE_PAIRS)), np.nan)
    for t in range(n):
        for s in range(2):
            if np.isfinite(hand_px[t, s, 0]).all() and vis[t, POSE_WRISTS[s]] >= VISIBILITY:
                wrist_gap_px[t, s] = np.linalg.norm(hand_px[t, s, 0] - pose_px[t, POSE_WRISTS[s]])
        if face.valid[t]:
            for k, (p, f, _) in enumerate(POSE_FACE_PAIRS):
                if vis[t, p] >= VISIBILITY:
                    face_gap_px[t, k] = np.linalg.norm(face_px[t, f, :2] - pose_px[t, p])

    with np.errstate(all="ignore"):
        body_pct = np.nanmean(err_arm, axis=1)
        face_gap_rms = np.sqrt(np.nanmean(face_gap_px**2, axis=1))

    # What smoothing moved, in the image: mean px per stream (pose, hands, face) against raw.
    raw_pose_px = raw_hands.pose_image[..., :2].astype(np.float64) * wh
    raw_hand_px = raw_hands.image[..., :2].astype(np.float64) * wh
    raw_face_px = raw_face.landmarks[..., :2].astype(np.float64) * wh
    shift_px = np.full((n, 3), np.nan)
    with np.errstate(all="ignore"):
        d = np.linalg.norm(pose_px - raw_pose_px, axis=-1)
        d[raw_hands.pose_visibility < VISIBILITY] = np.nan
        shift_px[:, 0] = np.nanmean(d, axis=1)
        shift_px[:, 1] = np.nanmean(
            np.linalg.norm(hand_px - raw_hand_px, axis=-1).reshape(n, -1), axis=1
        )
        shift_px[:, 2] = np.nanmean(np.linalg.norm(face_px[..., :2] - raw_face_px, axis=-1), axis=1)

    # Jitter of each stream (arms, hands, face) in 3-D: the fitted input and the round trip.
    arm_in = np.where(ok[:, list(ARM_MP), None], pose_w[:, list(ARM_MP)], np.nan)
    jitter_in = np.stack(
        [jitter_mm(arm_in), jitter_mm(in_hands), jitter_mm(in_face[:, corr.mp_indices])], axis=1
    )
    jitter_rt = np.stack([jitter_mm(rt_arm), jitter_mm(rt_hands), jitter_mm(rt_face)], axis=1)

    arrays = {
        "img_pose": (hands.pose_image.astype(np.float32), "float32"),
        "pose_vis": (vis.astype(np.float32), "float32"),
        "img_hands": (hands.image.astype(np.float32), "float32"),
        "img_face": (face.landmarks[..., :2].astype(np.float32), "float32"),
        "w_pose": (pose_w.astype(np.float32), "float32"),
        "w_hands_in": (in_hands.astype(np.float32), "float32"),
        "w_face_in": (in_face.astype(np.float32), "float32"),
        "rt_arm": (rt_arm.astype(np.float32), "float32"),
        "rt_hands": (rt_hands.astype(np.float32), "float32"),
        "rt_face": (rt_face.astype(np.float32), "float32"),
        "err_arm_pct": (err_arm.astype(np.float32), "float32"),
        "err_hand_mm": (err_hand_pts.astype(np.float32), "float32"),
        "err_face_mm": (err_face_pts.astype(np.float32), "float32"),
        "mesh_verts": (verts, "int16"),
        "mesh_faces": (skeleton.model.faces.astype(np.uint16), "uint16"),
    }
    manifest, offset = write_data(out / "data.bin", arrays)

    rest = pose_mesh(body, np.zeros((55, 3)), result.betas, np.zeros(3)).vertices
    with np.errstate(all="ignore"):
        summary = {
            "body_pct_median": float(np.nanmedian(body_pct)),
            "hand_mm_median": nan_list(np.nanmedian(hand_mm, axis=0), 1),
            "mano_mm_median": nan_list(np.nanmedian(mano_mm, axis=0), 1),
            "face_rigid_mm_median": float(np.nanmedian(face_rms[:, 0])),
            "face_nonrigid_mm_median": float(np.nanmedian(face_rms[:, 1])),
            "wrist_gap_px_median": nan_list(np.nanmedian(wrist_gap_px, axis=0), 1),
            "face_gap_px_median": float(np.nanmedian(face_gap_rms)),
            "frames_body_fit": int(result.body_ok.sum()),
            "frames_face": int(face.valid.sum()),
            "frames_head_reliable": int(head_reliable.sum()),
            "frames_hand": nan_list(hands.present.sum(axis=0), 0),
            "jitter_in_median": nan_list(np.nanmedian(jitter_in, axis=0), 2),
            "jitter_rt_median": nan_list(np.nanmedian(jitter_rt, axis=0), 2),
            "shift_px_median": nan_list(np.nanmedian(shift_px, axis=0), 2),
        }
    meta = {
        "clip": clip_id,
        "variant": variant,
        "n": n,
        "width": W,
        "height": H,
        "fps": hands.fps,
        "scale_i16": SCALE_I16,
        "arrays": manifest,
        "connections": connections(),
        "face105": {
            "mp_indices": [int(i) for i in corr.mp_indices],
            "contours": face_contours(corr, rest),
        },
        "arm_mp": list(ARM_MP),
        "pose_face_pairs": [[p, f, name] for p, f, name in POSE_FACE_PAIRS],
        "flags": {
            "body_ok": result.body_ok.astype(int).tolist(),
            "face_valid": face.valid.astype(int).tolist(),
            "head_reliable": head_reliable.astype(int).tolist(),
            "hand_present": hands.present.astype(int).tolist(),
            "hand_near_face": result.hand_near_face.astype(int).tolist(),
        },
        "series": {
            "body_pct": nan_list(body_pct, 2),
            "hand_mm": nan_list(hand_mm, 1),
            "mano_mm": nan_list(mano_mm, 1),
            "face_rigid_mm": nan_list(face_rms[:, 0], 1),
            "face_nonrigid_mm": nan_list(face_rms[:, 1], 1),
            "shoulder_m": nan_list(shoulder_m, 4),
            "shoulder_px": nan_list(shoulder_px, 1),
            "wrist_gap_px": nan_list(wrist_gap_px, 1),
            "face_gap_px": nan_list(face_gap_px, 1),
            "face_gap_rms_px": nan_list(face_gap_rms, 1),
            "jitter_in": nan_list(jitter_in, 2),
            "jitter_rt": nan_list(jitter_rt, 2),
            "shift_px": nan_list(shift_px, 2),
        },
        "summary": summary,
    }
    (out / "clip.json").write_text(
        json.dumps(json_safe(meta), separators=(",", ":"), allow_nan=False)
    )
    print(
        f"  {variant:<8} wrote {out} ({offset / 1e6:.1f} MB data, {time.time() - started:.0f} s): "
        + json.dumps(json_safe(summary))
    )
    trip = RoundTrip(
        Seq(verts_m, rt_arm, rt_hands, rt_face, None, np.arange(n, dtype=np.float64), False),
        result, body, hands, face, np.isfinite(hands.world).all(axis=(-1, -2)),
    )
    return summary, trip


# --------------------------------------------------------------------------- augmentations


@dataclasses.dataclass
class Seq:
    """
    A sequence of SMPL-X output: mesh vertices and the read-back points, in MediaPipe world.

    `times` are the source frames each frame shows (fractional after a re-timing); `camera`
    projects the sequence into the source image.
    """

    verts: np.ndarray  # (T, V, 3), NaN where the body was not fitted
    arm: np.ndarray  # (T, 6, 3), ARM_MP
    hands: np.ndarray  # (T, 2, 21, 3)
    face: np.ndarray  # (T, 105, 3)
    camera: WeakCamera
    times: np.ndarray  # (T,)
    mirrored: bool


@dataclasses.dataclass
class RoundTrip:
    """What the augmentations need from one exported smoothing setting."""

    seq: Seq
    result: object  # face_pipeline.ClipResult
    body: object  # the shaped SMPLXSkeleton
    hands: object  # HandClip the fit used
    face: object  # FaceClip the fit used
    hand_seen: np.ndarray  # (T, 2): a hand observation to read the SMPL-X hand back against


def _map(seq, fn):
    return dataclasses.replace(seq, verts=fn(seq.verts), arm=fn(seq.arm), hands=fn(seq.hands),
                               face=fn(seq.face))


def _pivot(seq):
    return np.nanmedian((seq.arm[:, 0] + seq.arm[:, 1]) / 2, axis=0)


def _hand_points(seq):
    """(T, 44, 3): both hands' points and the two pose wrists, the points kept in the frame."""
    return np.concatenate([seq.hands.reshape(len(seq.hands), 42, 3), seq.arm[:, 4:6]], axis=1)


def aug_view(seq, yaw, pitch):
    pivot, rot = _pivot(seq), view_rotation(yaw, pitch)
    return _map(seq, lambda p: about(p, pivot, rot))


def aug_mirror(seq):
    centre_x = _pivot(seq)[0]
    out = _map(seq, lambda p: reflect_x(p, centre_x))
    # Face points are reflected but keep their indices: the face mesh has no left/right table.
    return dataclasses.replace(out, arm=out.arm[:, ARM_MIRROR], hands=out.hands[:, ::-1],
                               mirrored=not seq.mirrored)


def aug_scale(seq, factor):
    pivot = _pivot(seq)
    return _map(seq, lambda p: about(p, pivot, factor * np.eye(3)))


def aug_uncentre(seq, u, limit):
    shift = uncentre_shift(seq.camera.project(_hand_points(seq)), u, limit)
    offset = seq.camera.world_shift(shift)
    return _map(seq, lambda p: p + offset), shift


def aug_duration(seq, factor):
    count = len(seq.times)
    times = duration_times(count, factor)
    hands = resample_points(seq.hands.reshape(count, 42, 3), times).reshape(len(times), 2, 21, 3)
    verts = np.stack(resample(list(seq.verts), times))
    return Seq(verts, resample_points(seq.arm, times), hands, resample_points(seq.face, times),
               seq.camera.resampled(times), np.interp(times, np.arange(count), seq.times),
               seq.mirrored)


def aug_arm_joints(trip, joints, corr):
    """Re-pose every fitted frame with the arm-joint draw and read the points back off it."""
    seq, result, body = trip.seq, trip.result, trip.body
    out = dataclasses.replace(seq, verts=seq.verts.copy(), arm=seq.arm.copy(),
                              hands=seq.hands.copy(), face=seq.face.copy())
    if not joints:
        return out
    for t in range(len(seq.times)):
        if not result.body_ok[t]:
            continue
        rotations = perturb_arm_joints(result.rotations[t], body.rest_joints, body.parents, joints)
        mesh = pose_mesh(body, rotations, result.betas, result.transl[t])
        out.verts[t] = mesh.vertices
        out.arm[t] = mesh.joints[list(ARM_SMPLX)]
        for s, side in enumerate(SIDES):
            if trip.hand_seen[t, s]:
                out.hands[t, s] = mesh_hand_points(mesh, side)
        out.face[t] = corr.points(mesh.vertices)
    return out


def _max_diff(a, b):
    return float(max(np.nanmax(np.abs(x - y)) for x, y in
                     ((a.verts, b.verts), (a.arm, b.arm), (a.hands, b.hands), (a.face, b.face))
                     if np.isfinite(x - y).any()))


def _in_frame_pct(seq):
    uv = seq.camera.project(_hand_points(seq))
    present = np.isfinite(uv).all(-1)
    return float(100 * in_frame(uv).sum() / present.sum()) if present.any() else float("nan")


def _arm_title(joints):
    if not joints:
        return f"none drawn (p = {TIER1.arm_joint_p} each)"
    return ", ".join(f"{j.side[0].upper()} {j.joint} {j.angle_deg:+.1f}°" for j in joints)


def fit_camera(trip, corr):
    """
    Weak-perspective camera from the round trip's read-back points to MediaPipe's image points.

    Fitted on all three streams (arms, hands, face) with one scale for both axes. Fitted on the
    arms alone, it put the face some 60 px too high on clip 20983.
    """
    seq, hands, face = trip.seq, trip.hands, trip.face
    n = len(seq.times)
    arm = hands.pose_image[:, list(ARM_MP), :2].astype(np.float64)
    arm[hands.pose_visibility[:, list(ARM_MP)] < VISIBILITY] = np.nan
    hand = hands.image[..., :2].astype(np.float64).reshape(n, 42, 2)
    mesh = face.landmarks[:, list(corr.mp_indices), :2].astype(np.float64)
    mesh[~np.asarray(face.valid, bool)] = np.nan
    world = np.concatenate([seq.arm, seq.hands.reshape(n, 42, 3), seq.face], axis=1)
    image = np.concatenate([arm, hand, mesh], axis=1)
    return fit_weak_camera(world, image, pixel_size=(hands.width, hands.height))


def export_augmentations(clip_id, trip, smoothing, out_root: Path, models, names, seed, sample=0):
    """
    The Tier 1 augmentations of one clip's round trip, each written as aug/<name>/.

    `sample` picks which of the clip's draws to use; draw 0 is the clip's default.
    """
    started = time.time()
    camera = fit_camera(trip, models.corr)
    base = dataclasses.replace(trip.seq, camera=camera)
    trip = dataclasses.replace(trip, seq=base)
    key = int(clip_id) if clip_id.isdigit() else zlib.crc32(clip_id.encode())
    entropy = [seed, key] if sample == 0 else [seed, key, sample]
    draw = sample_tier1(np.random.default_rng(entropy))

    arms = aug_arm_joints(trip, draw.arm_joints, models.corr)
    uncentred, shift = aug_uncentre(base, draw.uncentre_u, TIER1.uncentre)
    chain = aug_mirror(arms) if draw.mirror else arms
    chain, chain_shift = aug_uncentre(
        aug_scale(aug_view(chain, draw.yaw_deg, draw.pitch_deg), draw.scale),
        draw.uncentre_u, TIER1.uncentre)
    built = {
        "view": (aug_view(base, draw.yaw_deg, draw.pitch_deg),
                 f"yaw {draw.yaw_deg:+.1f}°, pitch {draw.pitch_deg:+.1f}°"),
        "mirror": (aug_mirror(base), "sides swapped" + (" (drawn)" if draw.mirror else
                                                         " (not drawn for combined)")),
        "duration": (aug_duration(base, draw.duration), f"×{draw.duration:.3f}"),
        "arm_joints": (arms, _arm_title(draw.arm_joints)),
        "scale": (aug_scale(base, draw.scale), f"×{draw.scale:.3f}"),
        "uncentre": (uncentred, f"({shift[0]:+.3f}, {shift[1]:+.3f}) of the frame"),
        "combined": (aug_duration(chain, draw.duration),
                     ("mirrored" if draw.mirror else "not mirrored")
                     + f", yaw {draw.yaw_deg:+.1f}°, ×{draw.duration:.2f}"),
    }

    # Zero strength must give back the round trip exactly; mirroring twice must too.
    zero_joints = tuple(ArmJointDraw(s, j, 0.0, (1.0, 0.0, 0.0)) for s in SIDES for j in ARM_JOINTS)
    zero_arms = aug_arm_joints(trip, zero_joints, models.corr)
    identity = {
        "view": lambda: aug_view(base, 0.0, 0.0),
        "mirror": lambda: aug_mirror(aug_mirror(base)),
        "duration": lambda: aug_duration(base, 1.0),
        "arm_joints": lambda: zero_arms,
        "scale": lambda: aug_scale(base, 1.0),
        "uncentre": lambda: aug_uncentre(base, (0.5, 0.5), 0.0)[0],
        "combined": lambda: aug_duration(aug_uncentre(aug_scale(aug_view(
            zero_arms, 0.0, 0.0), 1.0), (0.5, 0.5), 0.0)[0], 1.0),
    }
    base_uv = camera.project(_hand_points(base))
    for name in names:
        seq, detail = built[name]
        n = len(seq.times)
        metrics = {
            "identity_max_abs_diff_m": _max_diff(base, identity[name]()),
            "hand_points_in_frame_pct": _in_frame_pct(seq),
            "round_trip_hand_points_in_frame_pct": _in_frame_pct(base),
        }
        if n == len(base.times) and not seq.mirrored:
            moved = 1000 * np.linalg.norm(seq.arm[:, 4:6] - base.arm[:, 4:6], axis=-1)
            loss = frame_loss(base_uv, seq.camera.project(_hand_points(seq)))
            with np.errstate(all="ignore"):
                metrics["wrist_moved_mm_median"] = float(np.nanmedian(moved))
                metrics["wrist_moved_mm_max"] = float(np.nanmax(moved))
            metrics["hand_points_leaving_frame_pct"] = None if loss is None else 100 * loss
        verts = np.where(np.isfinite(seq.verts), seq.verts, 0.0)
        arrays = {
            "times": (seq.times.astype(np.float32), "float32"),
            "mesh_verts": (np.clip(np.round(verts * SCALE_I16), -32767, 32767).astype(np.int16),
                           "int16"),
            "aug_arm": (seq.arm.astype(np.float32), "float32"),
            "aug_hands": (seq.hands.astype(np.float32), "float32"),
            "aug_face": (seq.face.astype(np.float32), "float32"),
            "img_arm": (seq.camera.project(seq.arm).astype(np.float32), "float32"),
            "img_hands": (seq.camera.project(seq.hands).astype(np.float32), "float32"),
            "img_face": (seq.camera.project(seq.face).astype(np.float32), "float32"),
        }
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        manifest, size = write_data(out / "data.bin", arrays)
        label, description = AUGMENTATIONS[name]
        meta = {
            "clip": clip_id,
            "augmentation": name,
            "label": label,
            "description": description,
            "title": f"{label}: {detail}" + (f" [draw {sample}]" if sample else ""),
            "smoothing": smoothing,
            "n": n,
            "mirrored": seq.mirrored,
            "scale_i16": SCALE_I16,
            "arrays": manifest,
            "body_ok": np.isfinite(seq.verts).all(axis=(1, 2)).astype(int).tolist(),
            "draw": draw.as_dict(),
            "sample": sample,
            "uncentre_shift": [float(x) for x in (chain_shift if name == "combined" else shift)],
            "camera": {"scale_per_m": list(camera.scale), "residual_frame_pct": 100 * camera.residual},
            "metrics": metrics,
        }
        (out / "clip.json").write_text(
            json.dumps(json_safe(meta), separators=(",", ":"), allow_nan=False)
        )
        print(f"  aug {name:<10} {n} frames, {size / 1e6:.1f} MB: {meta['title']}; "
              + json.dumps(json_safe(metrics)))
    print(f"  augmentations done ({time.time() - started:.0f} s)")


def update_index(out_root: Path, order):
    """clips.json: the review order, the smoothing settings, each clip's summary per setting."""
    index_path = out_root / "clips.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index["seed"], index["order"] = SEED, order
    index["smoothing"] = {
        name: {"label": s.label, "description": s.description} for name, s in SMOOTHING.items()
    }
    index["augmentation"] = {
        name: {"label": label, "description": description}
        for name, (label, description) in AUGMENTATIONS.items()
    }
    clips = index["clips"] = {}
    for clip_dir in sorted((out_root / "clips").iterdir()):
        variants = {}
        for meta_path in sorted(clip_dir.glob("v/*/clip.json")):
            meta = json.loads(meta_path.read_text())
            variants[meta["variant"]] = json_safe(meta["summary"])
        if variants:
            clips[clip_dir.name] = {"n": meta["n"], "variants": variants}
            augs = {}
            for meta_path in sorted(clip_dir.glob("aug/*/clip.json")):
                aug = json.loads(meta_path.read_text())
                augs[aug["augmentation"]] = {"n": aug["n"], "title": aug["title"]}
            if augs:
                clips[clip_dir.name]["augmentations"] = augs
    index_path.write_text(json.dumps(index, indent=1, allow_nan=False))


def sample(video_dir: Path, count: int, seed: int):
    clips = sorted(p.stem for p in video_dir.glob("*.mp4"))
    return random.Random(seed).sample(clips, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--clips", nargs="*", default=[], help="clip ids to export")
    parser.add_argument(
        "--sample", type=int, default=0, help="export the first N of the seeded random review order"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--reindex", action="store_true", help="only rebuild clips.json")
    parser.add_argument(
        "--smoothing",
        nargs="*",
        choices=list(SMOOTHING),
        default=list(SMOOTHING),
        help="smoothing settings to export (default: all)",
    )
    parser.add_argument(
        "--augment",
        nargs="*",
        choices=list(AUGMENTATIONS),
        default=list(AUGMENTATIONS),
        help="Tier 1 augmentations to export, from the unsmoothed round trip (default: all)",
    )
    parser.add_argument(
        "--aug-draw",
        type=int,
        default=0,
        help="which of each clip's Tier 1 draws to export (default 0, the clip's own)",
    )
    args = parser.parse_args()
    video_dir = args.video_dir.expanduser()
    order = sample(video_dir, 50, args.seed)
    clips = list(args.clips) + order[: args.sample]
    if args.reindex and not clips:
        update_index(args.output_dir, order)
        return
    if not clips:
        sys.exit("nothing to export: pass --clips or --sample")
    models = load_face_models()
    for clip_id in clips:
        export_clip(clip_id, video_dir, args.output_dir, models, args.smoothing, args.reextract,
                    args.augment, args.seed, args.aug_draw)
    update_index(args.output_dir, order)


if __name__ == "__main__":
    main()
