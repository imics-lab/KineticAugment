#!/usr/bin/env python3
"""
The V1.3 phase-A gate (design §7) over every clip with <clip>_hands.npz and <clip>_face.npz.

Every metric is reported for hand-near-face frames and for the rest:
  1  rigid-set RMS residual, % outer-eye-corner distance, against the correspondence floor
  2  fitted head rotation vs FaceLandmarker's, relative to each clip's mean (degrees)
  3  per-clip Spearman correlation, fitted jaw pitch vs the jawOpen blendshape
  4  ear-midpoint residual, % shoulder width, with the neck swing and with the neck at rest
  5  fraction of frames with a hand point more than 1 cm inside the head, plus how close hands
     get: minimum signed distance, fraction of frames in contact, and (near-face frames only)
     the gap between the closest hand point and its closest head point split into camera-axis
     depth and image-plane parts
  6  fraction of frames flagged invalid / unreliable / jaw unreliable

Usage:
    python scripts/evaluate_face.py --landmark-dir data/round_trip_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from kinetic_augment.body_model.assembly import pose_mesh
from kinetic_augment.body_model.face_regions import build_head_surface
from kinetic_augment.body_model.face_correspondence import RIGID
from kinetic_augment.body_model.face_metrics import (
    correspondence_floor, gap_components, head_pose_errors_deg, load_obj_vertices,
)
from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import (
    VISIBILITY_THRESHOLD, load_face_models, run_clip, save_face_features,
)
from kinetic_augment.body_model.hand_face_geometry import hand_points, signed_distances
from kinetic_augment.body_model.hand_observations import load_hand_clip

SHOULDERS = (11, 12)
PENETRATION = -0.01
MIN_JAW_FRAMES, MIN_JAW_SPREAD = 10, 0.02
TARGETS = {"head_pose_median_deg": 5.0, "head_pose_p95_deg": 10.0, "jaw_spearman": 0.7}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "mean": float(values.mean())}


def nearest_gap(result, i, body, surface):
    """(|Δz|, |Δxy|) from the hand point nearest the head surface in frame i to that surface."""
    g = result.geometry[i]
    if g is None or not np.isfinite(g.distance).any():
        return np.nan, np.nan
    mesh = pose_mesh(body, result.rotations[i], result.betas, result.transl[i])
    points = hand_points(mesh).reshape(-1, 3)
    _, closest, _ = signed_distances(points, mesh.vertices[surface.vertex_ids], surface.faces)
    nearest = np.nanargmin(np.abs(g.distance.reshape(-1)))
    return gap_components(points[nearest], closest[nearest])


def clip_rows(hands, face, result, body, surface):
    """Per-frame metric values for one clip, keyed by metric, plus the near-face split."""
    fits = result.head_fits
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(-1) & (hands.pose_visibility >= VISIBILITY_THRESHOLD)
    both = ok[:, SHOULDERS[0]] & ok[:, SHOULDERS[1]]
    width = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    frames = result.frames
    valid = np.array([f.valid for f in fits])
    fitted = np.stack([f.head_world for f in fits])
    head_pose = head_pose_errors_deg(fitted, face.head_matrix[frames], valid & face.valid[frames])
    penetrating = np.array([g is not None and bool(np.nanmin(np.append(g.distance, np.inf))
                                                   < PENETRATION) for g in result.geometry])
    posed = np.array([g is not None and np.isfinite(g.distance).any() for g in result.geometry])
    n = len(fits)
    min_distance = np.full(n, np.nan)
    contact = np.full(n, np.nan)
    gaps = np.full((n, 2), np.nan)
    for i, g in enumerate(result.geometry):
        if g is not None and np.isfinite(g.distance).any():
            min_distance[i] = np.nanmin(g.distance)
            contact[i] = float(g.contact.any())
            if result.hand_near_face[i]:
                gaps[i] = nearest_gap(result, i, body, surface)
    rows = {
        "rigid_rms_pct": 100 * np.array([f.rms_fraction for f in fits]),
        "head_pose_deg": head_pose,
        "ear_swing_pct": 100 * np.array([f.ear_residual_m for f in fits]) / width,
        "ear_rest_pct": 100 * np.array([f.ear_residual_rest_m for f in fits]) / width,
        "penetrating": np.where(posed, penetrating.astype(float), np.nan),
        "min_distance_m": min_distance,
        "contact": contact,
        "gap_depth_m": gaps[:, 0],
        "gap_plane_m": gaps[:, 1],
        "invalid": (~valid).astype(float),
        "unreliable": np.array([not f.reliable for f in fits], float),
        "jaw_unreliable": np.array([not f.jaw_reliable for f in fits], float),
    }
    jaw_ok = valid & np.array([f.jaw_reliable for f in fits])
    names = list(face.blendshape_names)  # empty for a clip with no detected face
    jaw_open = face.blendshapes[frames, names.index("jawOpen")] if "jawOpen" in names else None
    if (jaw_open is not None and jaw_ok.sum() >= MIN_JAW_FRAMES
            and np.nanstd(jaw_open[jaw_ok]) >= MIN_JAW_SPREAD):
        pitch = np.array([f.jaw_pose[0] for f in fits])
        jaw = float(spearmanr(pitch[jaw_ok], jaw_open[jaw_ok]).correlation)
    else:
        jaw = None
    return rows, result.hand_near_face, jaw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-dir", type=Path, default=Path("data/round_trip_eval"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.landmark_dir / "face_report.json"
    models = load_face_models()
    rest = pose_mesh(models.skeleton, np.zeros((55, 3)), models.skeleton.betas,
                     np.zeros(3)).vertices
    floor = correspondence_floor(load_obj_vertices("models/mediapipe/canonical_face_model.obj"),
                                 models.corr.points(rest, RIGID))
    pooled = {"near": {}, "far": {}}
    jaw, clips = {}, sorted(p.name[: -len("_face.npz")]
                            for p in args.landmark_dir.glob("*_face.npz"))
    for clip in clips:
        hands = load_hand_clip(args.landmark_dir / f"{clip}_hands.npz")
        face = load_face_clip(args.landmark_dir / f"{clip}_face.npz")
        result = run_clip(hands, face, models)
        save_face_features(args.landmark_dir / f"{clip}_face_features.npz", result, face)
        body = models.skeleton.shaped(result.betas)
        surface = build_head_surface(body, models.corr, models.model_dir)
        rows, near, jaw[clip] = clip_rows(hands, face, result, body, surface)
        for key, values in rows.items():
            pooled["near"].setdefault(key, []).extend(values[near])
            pooled["far"].setdefault(key, []).extend(values[~near])
        print(f"{clip}: {len(result.frames)} frames, {int(near.sum())} hand-near-face")
    report = {
        "clips": clips,
        "floor_rigid_rms_pct": 100 * floor,
        "targets": TARGETS,
        "jaw_spearman": jaw,
        "jaw_spearman_summary": stats([v for v in jaw.values() if v is not None]),
        "metrics": {split: {key: stats(v) for key, v in values.items()}
                    for split, values in pooled.items()},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("floor_rigid_rms_pct", "jaw_spearman_summary")},
                     indent=2))
    for split in ("far", "near"):
        print(f"\n{split}:")
        for key, s in report["metrics"][split].items():
            if s.get("n"):
                print(f"  {key:<16} n={s['n']:<6} median={s['median']:.3f} "
                      f"p95={s['p95']:.3f} mean={s['mean']:.3f}")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
