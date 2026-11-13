#!/usr/bin/env python3
"""
Pose-world vs image / FaceLandmarker frame alignment across clips (ledger assumption O11).

The V1.3 face pipeline applies FaceLandmarker's head rotation R_head, fitted in the image (pixel)
frame, directly in MediaPipe's pose-world frame. That is only right if the two frames are aligned.
On 08925 the spike (`exp_head_placement.py`) found a constant ~7.5 deg world->image rotation and a
~10 deg pose-face Kabsch vs R_head disagreement, almost all roll. This script measures both on
every clip, per frame, from the face pipeline's first pass (`run_clip`, no depth correction):

    sim_all   signed angle of the 2-D least-squares similarity pose-world (x, y) -> pose pixels
              over all visible (>= 0.5) in-frame pose points (0-32). A 2-D similarity's rotation
              axis is the camera axis by construction; the sign is from +x towards +y (y down),
              i.e. clockwise on screen.
    sim_face  the same over the visible face points 0-10 only (the spike's choice).
    cam3d     angle, axis and roll (twist about the camera z axis) of a 3-D Umeyama similarity
              pose-world -> pose image (x W, y H, z W) over the same points as sim_all; checks
              whether the frame offset has pitch / yaw besides roll. The image z is MediaPipe's
              relative depth, the noisiest coordinate.
    ortho     angle, roll and swing of a scaled-orthographic camera fitted pose-world (x, y, z) ->
              pose pixels (x, y) over the same points: a free 2x3 least-squares map, snapped to the
              nearest scaled pair of orthonormal rows. Pitch / yaw without the image z; weakly
              conditioned when the visible body is nearly planar.
    kabsch    angle between a weighted Kabsch head rotation K (SMPL-X rest face points -> pose
              face points 0-10 in pose-world, weights as in the spike: visible, finite, outside
              the padded hand outlines) and R_head. The offset is expressed in the world frame,
              D = R_head K^T, so R_head ~ D K; its roll is the twist of D about z (same sign
              convention as sim_all), its swing the rest.
    kabsch_rollfix  angle between K and Rz(-sim_all) R_head: what is left if R_head is rotated
              back by the measured world->image roll.
    shoulder  tilt of the shoulder line 12 -> 11 in the image (pixels) and in pose-world (x, y),
              degrees, same sign convention.
    eye_*     tilt of the outer eye-corner line (pose 6 -> 3, face-mesh 33 -> 263) as seen by each
              source: FaceLandmarker pixels (facemesh), pose pixels (pose_img), pose-world (x, y)
              (pose_world), and SMPL-X rest corners rotated by R_head (rhead) or by K (kabsch). An
              image roll is directly visible in the image, so the pixel sources are the reference.
    head_cam  the pipeline's own `head_camera_rotation_deg` (posed head rigid points -> face
              pixels). The head is posed with R_head, so this is ~0 by construction and cannot
              detect an O11 roll; it is kept to show why the earlier ~1 deg check missed it.

Usage (from the repo root):
    PYTHONPATH=src:scripts .venv/bin/python scripts/exp_frame_alignment.py \
        --model-dir MODELS [--clips 08925 05631 ...] [--out OUT.npz]

Without --clips: every clip with <clip>_hands.npz and <clip>_face.npz under data/round_trip_eval,
plus 08925 from data/test_output/exp_08925. Per-frame arrays go to --out (one entry per clip and
quantity, keys "<clip>/<name>"); the per-clip table is printed as markdown.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from exp_head_placement import (  # noqa: E402
    POINTS,
    VISIBILITY,
    check_sides,
    similarity,
    smplx_face_points,
    weighted_kabsch,
)
from scipy.spatial.transform import Rotation  # noqa: E402

from kinetic_augment.body_model.alignment import similarity_transform  # noqa: E402
from kinetic_augment.body_model.assembly import HEAD, pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import (  # noqa: E402
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
)
from kinetic_augment.body_model.face_observations import load_face_clip  # noqa: E402
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip  # noqa: E402
from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402
from kinetic_augment.body_model.head_converter import (  # noqa: E402
    OCCLUSION_PAD,
    hand_outlines,
    occluded,
)

EVAL_DIR = Path("data/round_trip_eval")
SPIKE_DIR = Path("data/test_output/exp_08925")
SHOULDERS = (11, 12)
Z = np.array([0.0, 0.0, 1.0])
EYES = ("eye_facemesh", "eye_pose_img", "eye_pose_world", "eye_rhead", "eye_kabsch")


def clip_dirs(clips):
    """clip id -> data dir; default every eval clip with hands and face plus 08925."""
    found = {}
    for path in sorted(EVAL_DIR.glob("*_hands.npz")):
        clip = path.name[: -len("_hands.npz")]
        if (EVAL_DIR / f"{clip}_face.npz").exists():
            found[clip] = EVAL_DIR
    if (SPIKE_DIR / "08925_face.npz").exists():
        found["08925"] = SPIKE_DIR
    if clips:
        missing = [c for c in clips if c not in found]
        if missing:
            raise SystemExit(f"no hands + face data for {missing}")
        found = {c: found[c] for c in clips}
    return found


def roll_swing(rotation):
    """(signed twist about z, swing angle), degrees, of a rotation matrix."""
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    twist = 2 * np.arctan2(z, w)
    twist = (twist + np.pi) % (2 * np.pi) - np.pi
    swing = Rotation.from_matrix(rotation) * Rotation.from_rotvec(-twist * Z)
    return np.degrees(twist), np.degrees(swing.magnitude())


def angle_deg(rotation):
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


def tilt_deg(a, b):
    """Signed angle of the 2-D vector b -> a from +x towards +y, degrees."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def ortho_rotation(world, pixels):
    """Rotation of the least-squares scaled-orthographic projection world (N, 3) -> pixels (N, 2)."""
    x, p = world - world.mean(axis=0), pixels - pixels.mean(axis=0)
    m = np.linalg.lstsq(x, p, rcond=None)[0].T  # (2, 3)
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    rows = u @ vt
    return np.vstack([rows, np.cross(rows[0], rows[1])])


def measure(clip, data_dir: Path, models):
    hands = load_hand_clip(data_dir / f"{clip}_hands.npz")
    face = load_face_clip(data_dir / f"{clip}_face.npz")
    result = run_clip(hands, face, models, attach_face_stream=False)
    n, W, H = len(result.frames), hands.width, hands.height
    body = models.skeleton.shaped(result.betas)
    world = result.pose_world.astype(np.float64)
    image = result.pose_image.astype(np.float64)
    px = image[..., :2] * [W, H]
    img3 = image * [W, H, W]
    vis = result.pose_visibility
    face_px = result.face_pixels

    _, mapping = check_sides(px, face_px, result.face_observed)
    rest_mesh = pose_mesh(body, np.zeros((55, 3)), result.betas, np.zeros(3))
    offsets = smplx_face_points(rest_mesh, models.corr, mapping) - rest_mesh.joints[HEAD]

    keys = (
        "sim_all sim_face cam3d_angle cam3d_roll cam3d_swing kabsch kabsch_roll kabsch_swing "
        "kabsch_rollfix shoulder_img shoulder_world head_cam n_points n_face_weight "
        "ortho_angle ortho_roll ortho_swing "
        "eye_facemesh eye_pose_img eye_pose_world eye_rhead eye_kabsch"
    ).split()
    out = {k: np.full(n, np.nan) for k in keys}
    out["cam3d_axis"] = np.full((n, 3), np.nan)
    out["kabsch_axis"] = np.full((n, 3), np.nan)
    out["ortho_R"] = np.full((n, 3, 3), np.nan)
    out["kabsch_D"] = np.full((n, 3, 3), np.nan)
    out["head_cam"] = np.asarray(result.head_camera_rotation_deg, np.float64)
    for i in range(n):
        t = result.frames[i]
        good = (
            (vis[i] >= VISIBILITY)
            & np.isfinite(world[i]).all(axis=1)
            & np.isfinite(img3[i]).all(axis=1)
            & (image[i, :, 0] >= 0)
            & (image[i, :, 0] <= 1)
            & (image[i, :, 1] >= 0)
            & (image[i, :, 1] <= 1)
        )
        out["n_points"][i] = good.sum()
        if good.sum() >= 3:
            a, _ = similarity(world[i, good, :2], px[i, good])
            out["sim_all"][i] = np.degrees(np.angle(a))
            _, r3, _ = similarity_transform(world[i, good], img3[i, good])
            out["cam3d_angle"][i] = angle_deg(r3)
            out["cam3d_roll"][i], out["cam3d_swing"][i] = roll_swing(r3)
            rotvec = Rotation.from_matrix(r3).as_rotvec()
            if np.linalg.norm(rotvec) > 0:
                out["cam3d_axis"][i] = rotvec / np.linalg.norm(rotvec)
            r_ortho = ortho_rotation(world[i, good], px[i, good])
            out["ortho_R"][i] = r_ortho
            out["ortho_angle"][i] = angle_deg(r_ortho)
            out["ortho_roll"][i], out["ortho_swing"][i] = roll_swing(r_ortho)
        face_good = good[:POINTS]
        if face_good.sum() >= 3:
            a, _ = similarity(world[i, :POINTS][face_good, :2], px[i, :POINTS][face_good])
            out["sim_face"][i] = np.degrees(np.angle(a))
        if good[list(SHOULDERS)].all():
            out["shoulder_img"][i] = tilt_deg(px[i, SHOULDERS[0]], px[i, SHOULDERS[1]])
            out["shoulder_world"][i] = tilt_deg(
                world[i, SHOULDERS[0], :2], world[i, SHOULDERS[1], :2]
            )

        if result.face_observed[i]:
            out["eye_facemesh"][i] = tilt_deg(face_px[i, mapping[3], :2], face_px[i, mapping[6], :2])
        if good[[3, 6]].all():
            out["eye_pose_img"][i] = tilt_deg(px[i, 3], px[i, 6])
            out["eye_pose_world"][i] = tilt_deg(world[i, 3, :2], world[i, 6, :2])

        head = result.head_fits[i]
        if not result.body_ok[i] or not np.isfinite(head.head_world).all():
            continue
        outlines = hand_outlines(hands.image[t], W, H)
        if face.valid[t]:
            ref = np.linalg.norm(face_px[i, RIGHT_EYE_OUTER, :2] - face_px[i, LEFT_EYE_OUTER, :2])
        else:
            ref = np.linalg.norm(px[i, 3] - px[i, 6])
        weights = (
            (vis[i, :POINTS] >= VISIBILITY)
            & np.isfinite(world[i, :POINTS]).all(axis=1)
            & ~occluded(px[i, :POINTS], outlines, OCCLUSION_PAD * ref)
        ).astype(np.float64)
        out["n_face_weight"][i] = weights.sum()
        kabsch = weighted_kabsch(offsets, world[i, :POINTS], weights)
        if not np.isfinite(kabsch).all():
            continue
        r_head = head.head_world
        out["eye_rhead"][i] = tilt_deg((r_head @ offsets[3])[:2], (r_head @ offsets[6])[:2])
        out["eye_kabsch"][i] = tilt_deg((kabsch @ offsets[3])[:2], (kabsch @ offsets[6])[:2])
        delta = r_head @ kabsch.T
        out["kabsch_D"][i] = delta
        out["kabsch"][i] = angle_deg(delta)
        out["kabsch_roll"][i], out["kabsch_swing"][i] = roll_swing(delta)
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        if np.linalg.norm(rotvec) > 0:
            out["kabsch_axis"][i] = rotvec / np.linalg.norm(rotvec)
        if np.isfinite(out["sim_all"][i]):
            unroll = Rotation.from_rotvec(-np.radians(out["sim_all"][i]) * Z).as_matrix()
            out["kabsch_rollfix"][i] = angle_deg(unroll @ r_head @ kabsch.T)
    return out


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def stats(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 95))


def table(results):
    head = (
        "| clip | frames | sim_all deg med (p95 abs) | sim_face med | cam3d angle med / roll / "
        "swing | ortho angle med / roll / swing | Kabsch vs R_head deg med (p95) | its roll / swing med | roll share | "
        "after roll fix med (p95) | shoulder img / world med | r(sim, shoulder img) | "
        "head_cam med | eye tilt facemesh / pose img / pose world / R_head / K med |"
    )
    rows = [head, "|" + "---|" * 14]
    for clip, o in results.items():
        k_med, k_p95 = stats(o["kabsch"])
        f_med, f_p95 = stats(o["kabsch_rollfix"])
        s_med = stats(o["sim_all"])[0]
        s_p95 = stats(np.abs(o["sim_all"]))[1]
        kr, ks = np.nanmedian(o["kabsch_roll"]), np.nanmedian(o["kabsch_swing"])
        share = np.nanmedian(np.abs(o["kabsch_roll"]) / o["kabsch"])
        rows.append(
            f"| {clip} | {np.isfinite(o['kabsch']).sum()}/{len(o['kabsch'])} "
            f"| {s_med:+.1f} ({s_p95:.1f}) | {np.nanmedian(o['sim_face']):+.1f} "
            f"| {np.nanmedian(o['cam3d_angle']):.1f} / {np.nanmedian(o['cam3d_roll']):+.1f} / "
            f"{np.nanmedian(o['cam3d_swing']):.1f} "
            f"| {np.nanmedian(o['ortho_angle']):.1f} / {np.nanmedian(o['ortho_roll']):+.1f} / "
            f"{np.nanmedian(o['ortho_swing']):.1f} "
            f"| {k_med:.1f} ({k_p95:.1f}) | {kr:+.1f} / {ks:.1f} | {share:.2f} "
            f"| {f_med:.1f} ({f_p95:.1f}) "
            f"| {np.nanmedian(o['shoulder_img']):+.1f} / {np.nanmedian(o['shoulder_world']):+.1f} "
            f"| {corr(o['sim_all'], o['shoulder_img']):+.2f} | {np.nanmedian(o['head_cam']):.1f} "
            f"| {' / '.join(f'{np.nanmedian(o[k]):+.1f}' for k in EYES)} |"
        )
    return "\n".join(rows)


def pooled(results):
    med = lambda k: np.array(  # noqa: E731
        [np.nanmedian(o[k]) if np.isfinite(o[k]).any() else np.nan for o in results.values()]
    )
    cat = lambda k: np.concatenate([o[k] for o in results.values()])  # noqa: E731
    lines = [
        "across clips (per-clip medians):",
        f"  r(sim_all, shoulder_img) = {corr(med('sim_all'), med('shoulder_img')):+.2f}",
        f"  r(sim_all, shoulder_world) = {corr(med('sim_all'), med('shoulder_world')):+.2f}",
        f"  r(sim_all, kabsch_roll) = {corr(med('sim_all'), med('kabsch_roll')):+.2f}",
        f"  r(sim_all, cam3d_roll) = {corr(med('sim_all'), med('cam3d_roll')):+.2f}",
        f"  r(sim_all, ortho_roll) = {corr(med('sim_all'), med('ortho_roll')):+.2f}",
        f"  r(kabsch_swing, ortho_swing) = {corr(med('kabsch_swing'), med('ortho_swing')):+.2f}",
        f"  r(kabsch_swing, cam3d_swing) = {corr(med('kabsch_swing'), med('cam3d_swing')):+.2f}",
        f"  median of clip medians: |sim_all| {np.nanmedian(np.abs(med('sim_all'))):.1f}, "
        f"kabsch {np.nanmedian(med('kabsch')):.1f}, "
        f"rollfix {np.nanmedian(med('kabsch_rollfix')):.1f}",
        "pooled frames:",
        f"  r(sim_all, shoulder_img) = {corr(cat('sim_all'), cat('shoulder_img')):+.2f}",
        f"  r(sim_all, shoulder_world) = {corr(cat('sim_all'), cat('shoulder_world')):+.2f}",
        f"  r(sim_all, kabsch_roll) = {corr(cat('sim_all'), cat('kabsch_roll')):+.2f}",
        f"  sim_all - (shoulder_img - shoulder_world): median "
        f"{np.nanmedian(cat('sim_all') - (cat('shoulder_img') - cat('shoulder_world'))):+.2f}",
        f"  r(kabsch_roll, eye_pose_world - eye_pose_img) = "
        f"{corr(cat('kabsch_roll'), cat('eye_pose_img') - cat('eye_pose_world')):+.2f}",
        "  eye tilt minus FaceLandmarker eye tilt, median (p95 abs) over frames:",
        *(
            f"    {k:<15} {np.nanmedian(cat(k) - cat('eye_facemesh')):+.2f} "
            f"({np.nanpercentile(np.abs(cat(k) - cat('eye_facemesh')), 95):.2f})"
            for k in EYES[1:]
        ),
        f"  kabsch: median {np.nanmedian(cat('kabsch')):.1f}, p95 "
        f"{np.nanpercentile(cat('kabsch'), 95):.1f}; after roll fix median "
        f"{np.nanmedian(cat('kabsch_rollfix')):.1f}, p95 "
        f"{np.nanpercentile(cat('kabsch_rollfix'), 95):.1f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", type=Path, default=Path("data/test_output/frame_alignment.npz"))
    args = parser.parse_args()
    models = load_face_models(args.model_dir)
    results = {}
    for clip, data_dir in clip_dirs(args.clips).items():
        results[clip] = measure(clip, data_dir, models)
        print(f"{clip}: done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, **{f"{c}/{k}": v for c, o in results.items() for k, v in o.items()}
    )
    print(table(results))
    print(pooled(results))
    print(f"npz {args.out}")


if __name__ == "__main__":
    main()
