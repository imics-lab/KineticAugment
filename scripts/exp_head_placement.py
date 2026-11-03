#!/usr/bin/env python3
"""
Head placement from the 11 MediaPipe pose face points (08925 head-placement spike, task 2).

Runs the face pipeline's first pass (no depth correction) on a clip, then re-places the head three
ways on the same body fit, hands and head world rotation R_head (the FaceLandmarker rigid fit):

    rest   neck at rest; the head takes all of R_head
    ear    the current placement: the neck swings so the SMPL-X ear midpoint hits the pose ear
           midpoint (`head_converter.split_neck_head`, taken from the pipeline's own fit)
    11pt   new: the neck swings so the head joint hits the weighted mean of the per-point votes
           observed_k - R_head o_k over pose points 0-10 (`head_placement.fit_neck_to_points`)

Pose point -> SMPL-X point: 0 nose vertex; 2 / 5 "leye" / "reye" vertices; 1 / 3 and 4 / 6 eye
inner / outer corners = face-mesh 362 / 263 and 133 / 33; 7 / 8 ear vertices; 9 / 10 mouth corners
= face-mesh 291 / 61. The side convention is checked on the data before anything else (pose pixels
vs FaceLandmarker pixels, direct vs mirrored) and a mirrored pair is swapped.

Weight of a point: 1 if its pose visibility is >= 0.5, it is finite, and it lies outside the padded
hand outlines (pad as in `fit_head`); else 0. Per frame and placement the script records the
weighted RMS and per-point residual of the 11 points against the posed SMPL-X mesh (cm and % of
shoulder width), the ear-midpoint residual, the head-joint shift between ear and 11pt, and the
minimum signed hand-to-head-surface distance. An independent head rotation is fitted by weighted
Kabsch from the SMPL-X rest points to the observed points and compared with R_head (degrees).

Usage (from the repo root):
    PYTHONPATH=src:scripts .venv/bin/python scripts/exp_head_placement.py 08925 --video-dir DIR

Output: <output-dir>/<clip>_head_placement.npz, .mp4 and stills <clip>_head_placement_frame<N>.png.
Panels: 1. head crop of the video with observed pose face points (orange, red cross = weight 0),
rest (grey square), ear (blue +) and 11pt (green circle) SMPL-X points, the right hand and padded
hand outlines; 2. front (x, y) and 3. side (z, y) views, observed vs ear and 11pt; 4. strip of the
11-point RMS residual (three placements) and min hand-to-head distance (ear, 11pt), with a cursor.
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
from scipy.spatial.transform import Rotation  # noqa: E402
from smplx.vertex_ids import vertex_ids  # noqa: E402

from kinetic_augment.body_model.assembly import HEAD, NECK, pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import (  # noqa: E402
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
)
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip  # noqa: E402
from kinetic_augment.body_model.face_regions import build_head_surface  # noqa: E402
from kinetic_augment.body_model.hand_face_geometry import hand_face_geometry  # noqa: E402
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402
from kinetic_augment.body_model.head_converter import (  # noqa: E402
    EAR_VERTICES,
    OCCLUSION_PAD,
    _segment_distance,
    hand_outlines,
    occluded,
)
from kinetic_augment.body_model.head_placement import (  # noqa: E402
    fit_neck_to_points,
    rest_neck,
)
from view_pose_face_proximity import read_frames  # noqa: E402
from view_round_trip import VideoSink  # noqa: E402

VIDEO_DIR = Path("data/metsis_new_unseen_sign_videos")
DATA_DIR = Path("data/test_output/exp_08925")
MODEL_DIR = "models"
VISIBILITY = 0.5
POINTS = 11
PLACEMENTS = ("rest", "ear", "11pt")
PLACEMENT_COLOURS = {"rest": "#8a8a8a", "ear": "#1f77b4", "11pt": "#2ca02c"}
EAR_PAIR = (7, 8)
SHOULDERS = (11, 12)
POSE_NAMES = [
    "nose",
    "L eye in",
    "L eye",
    "L eye out",
    "R eye in",
    "R eye",
    "R eye out",
    "L ear",
    "R ear",
    "mouth L",
    "mouth R",
]
# Pose points that sit on a transferred face-mesh point: pose index -> face-mesh index.
MESH_POINTS = {1: 362, 3: 263, 4: 133, 6: 33, 9: 291, 10: 61}
MIRROR = {1: 4, 4: 1, 3: 6, 6: 3, 9: 10, 10: 9}  # pose index of the mirrored counterpart


def check_sides(pose_px: np.ndarray, face_px: np.ndarray, face_valid: np.ndarray):
    """
    Median pixel offset of each mesh-paired pose point to its mapped and its mirrored face-mesh
    point; returns (rows, mapping), mapping being pose index -> face-mesh index after swapping any
    pair whose mirrored counterpart is closer.
    """
    rows, mapping = [], dict(MESH_POINTS)
    for k, mesh_index in MESH_POINTS.items():
        mirrored = MESH_POINTS[MIRROR[k]]
        direct = np.linalg.norm(
            pose_px[face_valid, k] - face_px[face_valid, mesh_index, :2], axis=1
        )
        flipped = np.linalg.norm(pose_px[face_valid, k] - face_px[face_valid, mirrored, :2], axis=1)
        d, f = float(np.nanmedian(direct)), float(np.nanmedian(flipped))
        rows.append((k, mesh_index, d, mirrored, f))
        if f < d:
            mapping[k] = mirrored
    return rows, mapping


def smplx_face_points(mesh, corr, mapping) -> np.ndarray:
    """(11, 3) SMPL-X points in pose-landmark order 0..10."""
    ids = vertex_ids["smplx"]
    out = np.empty((POINTS, 3))
    out[0] = mesh.vertices[ids["nose"]]
    out[2], out[5] = mesh.vertices[ids["leye"]], mesh.vertices[ids["reye"]]
    out[7], out[8] = mesh.vertices[EAR_VERTICES[0]], mesh.vertices[EAR_VERTICES[1]]
    ks = sorted(mapping)
    out[ks] = corr.points(mesh.vertices, [mapping[k] for k in ks])
    return out


def weighted_kabsch(source, target, weights) -> np.ndarray:
    """Rotation minimising the weighted |R (source - mean) - (target - mean)|; NaN if < 3 points."""
    use = weights > 0
    if use.sum() < 3:
        return np.full((3, 3), np.nan)
    w = weights[use] / weights[use].sum()
    s, t = source[use] - w @ source[use], target[use] - w @ target[use]
    u, _, vt = np.linalg.svd((s * w[:, None]).T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def outline_gap(points, outlines) -> float:
    """Smallest pixel distance from any point to any hand outline edge; NaN without a hand."""
    gaps = [
        _segment_distance(points, o[j], o[(j + 1) % len(o)])
        for o in outlines
        for j in range(len(o))
    ]
    return float(np.min(gaps)) if gaps else float("nan")


def similarity(world_xy, image_px):
    """(a, b) of the least-squares complex similarity p = a w + b; NaN pair if degenerate."""
    ok = np.isfinite(world_xy).all(axis=1) & np.isfinite(image_px).all(axis=1)
    if ok.sum() < 2:
        return np.nan + 0j, np.nan + 0j
    w = world_xy[ok, 0] + 1j * world_xy[ok, 1]
    p = image_px[ok, 0] + 1j * image_px[ok, 1]
    wc, pc = w - w.mean(), p - p.mean()
    denom = float(np.sum(np.abs(wc) ** 2))
    if denom < 1e-12:
        return np.nan + 0j, np.nan + 0j
    a = np.sum(np.conj(wc) * pc) / denom
    return a, p.mean() - a * w.mean()


def project(points, a, b) -> np.ndarray:
    z = a * (points[:, 0] + 1j * points[:, 1]) + b
    return np.stack([z.real, z.imag], axis=1)


def compute(clip, data_dir: Path):
    hands = load_hand_clip(data_dir / f"{clip}_hands.npz")
    face = load_face_clip(data_dir / f"{clip}_face.npz")
    models = load_face_models(MODEL_DIR)
    result = run_clip(hands, face, models, attach_face_stream=False)
    n, W, H = len(result.frames), hands.width, hands.height
    body = models.skeleton.shaped(result.betas)
    surface = build_head_surface(body, models.corr, models.model_dir)
    pose_world = result.pose_world.astype(np.float64)[:, :POINTS]
    pose_px = result.pose_image.astype(np.float64)[..., :2] * [W, H]
    vis = result.pose_visibility
    face_px = result.face_pixels

    side_rows, mapping = check_sides(pose_px, face_px, result.face_observed)
    rest_mesh = pose_mesh(body, np.zeros((55, 3)), result.betas, np.zeros(3))
    rest_points = smplx_face_points(rest_mesh, models.corr, mapping)
    offsets = rest_points - rest_mesh.joints[HEAD]

    shape = (n, len(PLACEMENTS))
    nan = lambda *s: np.full(s, np.nan)  # noqa: E731
    out = dict(
        weights=nan(n, POINTS),
        observed=pose_world,
        shoulder_width_cm=nan(n),
        smplx_points=nan(n, 3, POINTS, 3),
        head_joint=nan(n, 3, 3),
        resid_cm=nan(n, 3, POINTS),
        resid_pct=nan(n, 3, POINTS),
        rms_cm=nan(*shape),
        rms_pct=nan(*shape),
        ear_resid_cm=nan(*shape),
        hand_min_cm=nan(*shape),
        head_shift_cm=nan(n),
        kabsch_deg=nan(n),
        ear_swing_mode=np.zeros(n, "U8"),
        pt_fallback=np.zeros(n, bool),
        rotations=nan(n, 3, 55, 3),
        transl=result.transl,
        head_world=nan(n, 3, 3),
        pose_visibility=vis[:, :POINTS],
        pose_px=pose_px,
        sim_a=nan(n).astype(complex),
        sim_b=nan(n).astype(complex),
        pad_px=nan(n),
        outline_gap_px=nan(n),
    )
    for i in range(n):
        head = result.head_fits[i]
        if not result.body_ok[i] or not np.isfinite(head.head_world).all():
            continue
        t = result.frames[i]
        head_world, rotations, transl = head.head_world, result.rotations[i], result.transl[i]
        outlines = hand_outlines(hands.image[t], W, H)
        if face.valid[t]:
            ref = np.linalg.norm(face_px[i, RIGHT_EYE_OUTER, :2] - face_px[i, LEFT_EYE_OUTER, :2])
        else:
            ref = np.linalg.norm(pose_px[i, 3] - pose_px[i, 6])
        weights = (
            (vis[i, :POINTS] >= VISIBILITY)
            & np.isfinite(pose_world[i]).all(axis=1)
            & ~occluded(pose_px[i, :POINTS], outlines, OCCLUSION_PAD * ref)
        ).astype(np.float64)
        out["weights"][i] = weights
        out["pad_px"][i] = OCCLUSION_PAD * ref
        out["outline_gap_px"][i] = outline_gap(pose_px[i, :POINTS], outlines)
        width = np.linalg.norm(
            result.pose_world[i, SHOULDERS[0]] - result.pose_world[i, SHOULDERS[1]]
        )
        out["shoulder_width_cm"][i] = 100 * width
        ears_ok = (vis[i, list(EAR_PAIR)] >= VISIBILITY).all()
        ear_obs = pose_world[i, list(EAR_PAIR)].mean(axis=0) if ears_ok else None

        placed = {
            "rest": rest_neck(rotations, transl, body, head_world),
            "ear": None,
            "11pt": fit_neck_to_points(
                rotations, transl, body, offsets, pose_world[i], weights, head_world
            ),
        }
        out["ear_swing_mode"][i] = head.neck_mode
        placed["ear"] = type("Ear", (), {"neck": head.neck_pose, "head": head.head_pose})
        if placed["11pt"] is None:
            placed["11pt"], out["pt_fallback"][i] = placed["ear"], True
        posed = {s: bool(result.geometry_valid[i, k]) for k, s in enumerate(("left", "right"))}
        for p, name in enumerate(PLACEMENTS):
            rot = rotations.copy()
            rot[NECK], rot[HEAD] = placed[name].neck, placed[name].head
            mesh = pose_mesh(body, rot, result.betas, transl)
            pts = smplx_face_points(mesh, models.corr, mapping)
            out["rotations"][i, p], out["smplx_points"][i, p] = rot, pts
            out["head_joint"][i, p] = mesh.joints[HEAD]
            r = np.linalg.norm(pts - pose_world[i], axis=1)
            out["resid_cm"][i, p], out["resid_pct"][i, p] = 100 * r, 100 * r / width
            if weights.sum() > 0:
                out["rms_cm"][i, p] = 100 * np.sqrt(weights @ r**2 / weights.sum())
                out["rms_pct"][i, p] = out["rms_cm"][i, p] / (100 * width) * 100
            if ear_obs is not None:
                ear_mid = mesh.vertices[list(EAR_VERTICES)].mean(axis=0)
                out["ear_resid_cm"][i, p] = 100 * np.linalg.norm(ear_mid - ear_obs)
            dist = hand_face_geometry(mesh, surface, posed).distance
            if np.isfinite(dist).any():
                out["hand_min_cm"][i, p] = 100 * np.nanmin(dist)
        out["head_shift_cm"][i] = 100 * np.linalg.norm(
            out["head_joint"][i, 2] - out["head_joint"][i, 1]
        )
        out["head_world"][i] = head_world
        kabsch = weighted_kabsch(offsets, pose_world[i], weights)
        if np.isfinite(kabsch).all():
            out["kabsch_deg"][i] = np.degrees(
                Rotation.from_matrix(kabsch.T @ head_world).magnitude()
            )
        seen = np.where(vis[i, :POINTS] >= VISIBILITY)[0]
        if len(seen) < 3:
            seen = np.arange(POINTS)
        out["sim_a"][i], out["sim_b"][i] = similarity(
            result.pose_world[i, seen, :2].astype(np.float64), pose_px[i, seen]
        )
    out["outlines"] = [hand_outlines(hands.image[t], W, H) for t in result.frames]
    out["pad_ref"] = [
        (
            np.linalg.norm(face_px[i, RIGHT_EYE_OUTER, :2] - face_px[i, LEFT_EYE_OUTER, :2])
            if face.valid[t]
            else np.linalg.norm(pose_px[i, 3] - pose_px[i, 6])
        )
        for i, t in enumerate(result.frames)
    ]
    meta = dict(
        clip=clip,
        width=W,
        height=H,
        fps=hands.fps,
        n=n,
        mapping=mapping,
        side_rows=side_rows,
        offsets=offsets,
        pose_world_full=result.pose_world,
    )
    return out, meta


def summarise(out, meta):
    n = meta["n"]
    ok = np.isfinite(out["rms_cm"]).all(axis=1)
    med = lambda v: float(np.nanmedian(v))  # noqa: E731
    print(
        f"{meta['clip']}: {n} frames, {int(np.isfinite(out['head_shift_cm']).sum())} placed, "
        f"{int(out['pt_fallback'].sum())} 11pt fallbacks to ear"
    )
    print("side convention: median px offset, mapped vs mirrored face-mesh point")
    for k, m, d, mm, f in meta["side_rows"]:
        chosen = meta["mapping"][k]
        print(
            f"  pose {k:2d} ({POSE_NAMES[k]:9s}) -> mesh {m:3d}: {d:5.2f} px   "
            f"mirrored mesh {mm:3d}: {f:5.2f} px   "
            f"{'SWAPPED to ' + str(chosen) if chosen != m else 'ok'}"
        )
    print(
        f"closest pose face point to a hand outline: {np.nanmin(out['outline_gap_px']):.1f} px "
        f"(median {med(out['outline_gap_px']):.1f}); occlusion pad {med(out['pad_px']):.1f} px"
    )
    w = out["weights"]
    print(
        f"weighted points per frame: median {med(w.sum(axis=1)):.0f}, min {w.sum(axis=1).min():.0f}"
        f"; per-point weight-1 frames: {w.sum(axis=0).astype(int).tolist()}"
    )
    print(f"{'median over frames':<40}" + "".join(f"{p:>9}" for p in PLACEMENTS))
    for label, key in (
        ("11-pt weighted RMS residual, cm", "rms_cm"),
        ("11-pt weighted RMS residual, % shoulder", "rms_pct"),
        ("ear-midpoint residual, cm", "ear_resid_cm"),
        ("min hand-to-head distance, cm", "hand_min_cm"),
    ):
        print(f"{label:<40}" + "".join(f"{med(out[key][:, p]):9.2f}" for p in range(3)))
    print(f"{'per-point residual, cm':<40}" + "".join(f"{p:>9}" for p in PLACEMENTS))
    for k in range(POINTS):
        print(
            f"  {k:2d} {POSE_NAMES[k]:<35}"
            + "".join(f"{med(out['resid_cm'][:, p, k]):9.2f}" for p in range(3))
        )
    print(
        f"head-joint shift ear -> 11pt, cm: median {med(out['head_shift_cm']):.2f}, "
        f"max {np.nanmax(out['head_shift_cm']):.2f}"
    )
    print(
        f"Kabsch head rotation vs R_head, deg: median {med(out['kabsch_deg']):.2f}, "
        f"max {np.nanmax(out['kabsch_deg']):.2f}"
    )
    better = (out["rms_cm"][:, 2] < out["rms_cm"][:, 1]).sum()
    print(f"frames where 11pt RMS < ear RMS: {int(better)} / {int(ok.sum())}")


def pick_stills(out):
    shift = out["head_shift_cm"]
    gain = out["rms_cm"][:, 1] - out["rms_cm"][:, 2]
    picks = [
        int(np.nanargmax(shift)),
        int(np.nanargmax(gain)),
        int(np.nanargmin(gain)),
        int(np.nanargmin(out["hand_min_cm"][:, 1])),
        0,
    ]
    chosen = []
    for t in picks:
        if t not in chosen:
            chosen.append(t)
    return chosen[:3]


def render(clip, out, meta, video_dir: Path, out_dir: Path):
    n, W, H = meta["n"], meta["width"], meta["height"]
    frames = read_frames(video_dir / f"{clip}.mp4", n)
    world_full = meta["pose_world_full"].astype(np.float64)
    face_world = out["observed"]
    ok_frames = np.isfinite(out["rms_cm"]).all(axis=1)
    lo = np.nanmin(face_world[ok_frames].reshape(-1, 3), axis=0)
    hi = np.nanmax(face_world[ok_frames].reshape(-1, 3), axis=0)
    centre, half = (lo + hi) / 2, float((hi - lo).max() / 2) + 0.07
    pose_px = out["pose_px"]
    stills = set(pick_stills(out))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip}_head_placement.mp4"
    sink = VideoSink(path, meta["fps"])
    fig = plt.figure(figsize=(14, 7.4), dpi=100)
    xs = np.arange(1, n + 1)
    for t in range(n):
        fig.clf()
        grid = fig.add_gridspec(
            2,
            3,
            height_ratios=[3.6, 1.3],
            hspace=0.42,
            wspace=0.14,
            left=0.05,
            right=0.95,
            top=0.94,
            bottom=0.07,
        )
        ax_vid, ax_front, ax_side = (fig.add_subplot(grid[0, i]) for i in range(3))
        ax_strip = fig.add_subplot(grid[1, :])
        placed = np.isfinite(out["rms_cm"][t]).all()

        ax_vid.imshow(frames[t])
        weights = out["weights"][t]
        face_centre = np.nanmean(pose_px[t, :POINTS], axis=0)
        for outline in out["outlines"][t]:
            ax_vid.add_patch(plt.Polygon(outline, fill=False, ec="#2ca02c", lw=1, ls="--"))
        ax_vid.scatter(
            pose_px[t, :POINTS, 0],
            pose_px[t, :POINTS, 1],
            s=26,
            c="#ff8c00",
            zorder=3,
            label="observed",
        )
        masked = np.flatnonzero(weights == 0) if placed else []
        ax_vid.scatter(
            pose_px[t, masked, 0],
            pose_px[t, masked, 1],
            s=60,
            marker="x",
            c="#d62728",
            zorder=5,
            label="weight 0",
        )
        if placed:
            proj = {
                p: project(out["smplx_points"][t, p], out["sim_a"][t], out["sim_b"][t])
                for p in range(3)
            }
            for p, (name, marker, size) in enumerate(
                (("rest", "s", 34), ("ear", "+", 55), ("11pt", "o", 30))
            ):
                kw = (
                    dict(c=PLACEMENT_COLOURS[name])
                    if marker == "+"
                    else dict(facecolors="none", edgecolors=PLACEMENT_COLOURS[name])
                )
                ax_vid.scatter(
                    proj[p][:, 0], proj[p][:, 1], s=size, marker=marker, zorder=4, label=name, **kw
                )
        ax_vid.set_xlim(face_centre[0] - 42, face_centre[0] + 42)
        ax_vid.set_ylim(face_centre[1] + 42, face_centre[1] - 42)
        ax_vid.set_title(f"{clip}  frame {t + 1}/{n}", fontsize=9)
        ax_vid.legend(fontsize=6, loc="lower left", ncol=2)
        ax_vid.tick_params(labelsize=6)
        if placed:
            r, e, h = out["rms_cm"][t], out["ear_resid_cm"][t], out["hand_min_cm"][t]
            ax_vid.set_xlabel(
                f"RMS cm  rest {r[0]:.2f}  ear {r[1]:.2f}  11pt {r[2]:.2f}  "
                f"(pts used {int(weights.sum())})\n"
                f"ear-mid cm  {e[0]:.2f} / {e[1]:.2f} / {e[2]:.2f}   head shift "
                f"{out['head_shift_cm'][t]:.2f} cm\n"
                f"hand-head cm  {h[0]:.1f} / {h[1]:.1f} / {h[2]:.1f}   Kabsch "
                f"{out['kabsch_deg'][t]:.1f} deg",
                fontsize=7,
                family="monospace",
                loc="left",
            )

        for ax, u, label in (
            (ax_front, 0, "front (x, y)"),
            (ax_side, 2, "side (z, y), depth right"),
        ):
            ax.scatter(face_world[t, :, u], face_world[t, :, 1], s=26, c="#ff8c00", zorder=3)
            ax.scatter(
                face_world[t, masked, u] if placed else [],
                face_world[t, masked, 1] if placed else [],
                s=60,
                marker="x",
                c="#d62728",
                zorder=5,
            )
            if placed:
                for p, name, marker in ((1, "ear", "+"), (2, "11pt", "o")):
                    pts = out["smplx_points"][t, p]
                    kw = (
                        dict(c=PLACEMENT_COLOURS[name])
                        if marker == "+"
                        else dict(facecolors="none", edgecolors=PLACEMENT_COLOURS[name])
                    )
                    ax.scatter(
                        pts[:, u], pts[:, 1], s=45, marker=marker, zorder=4, label=name, **kw
                    )
                    for k in range(POINTS):
                        ax.plot(
                            [face_world[t, k, u], pts[k, u]],
                            [face_world[t, k, 1], pts[k, 1]],
                            color=PLACEMENT_COLOURS[name],
                            lw=0.6,
                            alpha=0.6,
                        )
            hand = world_full[t, [16, 18, 20, 22]]
            ax.scatter(hand[:, u], hand[:, 1], s=22, c="#2ca02c", zorder=3, marker="^")
            ax.set_xlim(centre[u] - half, centre[u] + half)
            ax.set_ylim(centre[1] + half, centre[1] - half)
            ax.set_aspect("equal")
            ax.set_title(label + "  (orange observed, green ^ right hand)", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)
        ax_front.legend(fontsize=6, loc="lower left")

        for p, name in enumerate(PLACEMENTS):
            ax_strip.plot(
                xs,
                out["rms_cm"][:, p],
                color=PLACEMENT_COLOURS[name],
                lw=1.5,
                label=f"11-pt RMS, {name}",
            )
        ax_strip.axvline(t + 1, color="k", lw=1)
        ax_strip.set_xlim(1, n)
        ax_strip.set_ylim(0, float(np.nanmax(out["rms_cm"])) * 1.4)
        ax_strip.set_ylabel("11-pt RMS residual (cm)", fontsize=8)
        ax_strip.set_xlabel("frame", fontsize=8)
        ax_strip.tick_params(labelsize=7)
        ax_hand = ax_strip.twinx()
        for p, name in ((1, "ear"), (2, "11pt")):
            ax_hand.plot(
                xs,
                out["hand_min_cm"][:, p],
                color=PLACEMENT_COLOURS[name],
                lw=1.3,
                ls="--",
                label=f"min hand-head, {name}",
            )
        ax_hand.set_ylim(0, float(np.nanmax(out["hand_min_cm"][:, 1:])) * 1.4)
        ax_hand.set_ylabel("min hand-to-head (cm)", fontsize=8)
        ax_hand.tick_params(labelsize=7)
        handles = ax_strip.get_legend_handles_labels()
        extra = ax_hand.get_legend_handles_labels()
        ax_strip.legend(
            handles[0] + extra[0], handles[1] + extra[1], fontsize=6, ncol=5, loc="upper right"
        )
        sink.add(fig)
        if t in stills:
            fig.savefig(out_dir / f"{clip}_head_placement_frame{t + 1}.png", dpi=100)
    sink.close()
    plt.close(fig)
    capture = cv2.VideoCapture(str(path))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return path, written, sorted(t + 1 for t in stills)


def save(out, meta, out_dir: Path):
    skip = {"outlines", "pad_ref"}
    arrays = {k: v for k, v in out.items() if k not in skip}
    arrays["placements"] = np.array(PLACEMENTS)
    arrays["rest_offsets"] = meta["offsets"]
    arrays["side_mapping"] = np.array(sorted(meta["mapping"].items()))
    arrays["side_check"] = np.array(meta["side_rows"], dtype=np.float64)
    path = out_dir / f"{meta['clip']}_head_placement.npz"
    np.savez_compressed(path, **arrays)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clip", help="clip id, e.g. 08925")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    out, meta = compute(args.clip, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz = save(out, meta, args.output_dir)
    summarise(out, meta)
    mp4, written, stills = render(args.clip, out, meta, args.video_dir, args.output_dir)
    print(f"npz {npz}\nmp4 {mp4} ({written} frames), stills at frames {stills}")


if __name__ == "__main__":
    main()
