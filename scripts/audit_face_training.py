"""Read-only probes for the V1.3 training-data audit; writes JSON only to --output.

Run from the repository root with PYTHONPATH=src. Existing caches are never changed.
"""

import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


def provenance():
    """Pin the audited implementation and selected raw inputs without copying assets."""
    import kinetic_augment

    files = [Path(__file__), *Path("src/kinetic_augment/body_model").glob("*.py")]
    for directory, clip in (
        ("data/test_output", "01986"),
        ("data/test_output", "01991"),
        ("data/round_trip_eval", "09849"),
        ("data/round_trip_eval", "63665"),
    ):
        files.extend(Path(directory) / f"{clip}_{kind}.npz" for kind in ("face", "hands"))
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "package_import": kinetic_augment.__file__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "smplx", "mediapipe", "pytest")
        },
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def stats(x):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    return {
        "n": int(len(x)),
        **(
            {
                "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            if len(x)
            else {}
        ),
    }


def cache_audit(directory, suffix):
    records, distance, residual, shift = [], [], [], []
    for path in sorted(Path(directory).glob(f"*{suffix}")):
        with np.load(path) as d:
            valid, reliable = d["valid"], d["reliable"]
            finite = np.isfinite(d["distance"]).any(axis=(1, 2))
            record = {
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames": len(valid),
                "invalid": int((~valid).sum()),
                "invalid_with_geometry": int((~valid & finite).sum()),
                "unreliable_with_geometry": int((~reliable & finite).sum()),
                "contact_frames": int(d["contact"].any(axis=(1, 2)).sum()),
                "near_frames": int(d["hand_near_face"].sum()),
                "has_face_stream_local": "face_stream_local" in d.files,
                "has_timestamp": "timestamp_ms" in d.files,
                "metadata_keys": sorted(json.loads(str(d["meta"])).keys()),
            }
            distance.extend(d["distance"].ravel())
            if "wrist_corrected" in d.files:
                corrected, reached = d["wrist_corrected"], d["wrist_reached"]
                contact = d["contact"].any(axis=2)
                record.update(
                    corrected_hands=int(corrected.sum()),
                    reached_hands=int(reached.sum()),
                    contact_hands=int(contact.sum()),
                    contact_hands_unreached=int((contact & corrected & ~reached).sum()),
                    contact_hands_reached=int((contact & reached).sum()),
                )
                residual.extend(d["wrist_residual_m"][corrected])
                shift.extend(d["wrist_shift_m"][corrected])
            records.append(record)
    counts = {
        k: sum(r.get(k, 0) for r in records)
        for k in (
            "frames",
            "invalid",
            "invalid_with_geometry",
            "unreliable_with_geometry",
            "contact_frames",
            "near_frames",
            "corrected_hands",
            "reached_hands",
            "contact_hands",
            "contact_hands_unreached",
            "contact_hands_reached",
        )
    }
    return {
        "files": records,
        "totals": counts,
        "all_point_distance_m": stats(distance),
        "corrected_wrist_residual_m": stats(residual),
        "wrist_shift_m": stats(shift),
    }


def synthetic_probes():
    from kinetic_augment.body_model.face_pipeline import hand_near_face
    from kinetic_augment.body_model.face_metrics import GL_TO_IMAGE, head_pose_errors_deg
    from kinetic_augment.body_model.hand_depth import focal_px, wrist_targets

    face = np.zeros((478, 3))
    face[:, :2] = 150
    face[:4, :2] = [[100, 100], [200, 100], [200, 200], [100, 200]]
    crossing = np.array([[50, 145], [250, 145], [250, 155], [50, 155]])
    crossing_result = hand_near_face(face, [crossing], pad=0)

    # A constant template-side error cancels out of clip-relative rotation scoring.
    rng = np.random.default_rng(42)
    gl = Rotation.from_rotvec(rng.normal(0, 0.15, (30, 3))).as_matrix()
    correct = GL_TO_IMAGE @ gl
    biased = correct @ Rotation.from_euler("y", 45, degrees=True).as_matrix()
    matrices = np.repeat(np.eye(4)[None], len(gl), axis=0)
    matrices[:, :3, :3] = gl
    errors = head_pose_errors_deg(biased, matrices, np.ones(len(gl), bool))

    # Exactly weak-perspective synthetic observations: no perspective/noise confound.
    w, h = 640, 480
    focal = focal_px(h)
    offsets = rng.uniform(-0.04, 0.04, (21, 3))
    offsets[:, 2] = 0
    offsets -= offsets[0]
    wrist_camera = np.array([0.03, 0.10, 0.90])
    hand = np.full((2, 21, 3), np.nan)
    hand[0] = offsets
    pixels = np.full((2, 21, 2), np.nan)
    pixels[0] = (offsets[:, :2] + wrist_camera[:2]) * focal / 0.90 + [w / 2, h / 2]
    base = wrist_targets([w / 2, h / 2], np.zeros(3), focal, hand, pixels, w, h)
    sensitivity = []
    for hand_factor, head_factor, fov in (
        (1, 1, 63),
        (0.95, 1, 63),
        (1.05, 1, 63),
        (1, 0.95, 63),
        (1, 1.05, 63),
        (1, 1, 45),
        (1, 1, 80),
    ):
        result = wrist_targets(
            [w / 2, h / 2],
            np.zeros(3),
            focal / head_factor,
            hand * hand_factor,
            pixels,
            w,
            h,
            fov_deg=fov,
        )
        sensitivity.append(
            {
                "hand_metric_scale_factor": hand_factor,
                "head_metric_scale_factor": head_factor,
                "assumed_fov_deg": fov,
                "wrist_z_m": float(result.target[0, 2]),
                "delta_from_baseline_m": float(np.linalg.norm(result.target[0] - base.target[0])),
            }
        )
    return {
        "crossing_polygon": {"expected_overlap": True, "reported_near": crossing_result},
        "constant_45_degree_head_bias": {"relative_gate_error_deg": stats(errors)},
        "depth_sensitivity": sensitivity,
    }


def fresh_runs():
    import torch

    torch.set_num_threads(1)
    from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    models = load_face_models()
    records = []
    cases = [
        ("data/test_output", "01986", list(range(14, 22))),
        ("data/test_output", "01991", list(range(0, 20))),
        ("data/round_trip_eval", "09849", [0, 20, 40, 60, 80]),
        ("data/round_trip_eval", "63665", None),
    ]
    for directory, clip, frames in cases:
        hands = load_hand_clip(Path(directory) / f"{clip}_hands.npz")
        face = load_face_clip(Path(directory) / f"{clip}_face.npz")
        if frames is None:
            frames = np.unique(np.linspace(0, len(face.valid) - 1, 12).astype(int)).tolist()
        for corrected in (False, True):
            result = run_clip(hands, face, models, frames=frames, depth_corrected=corrected)
            contact = np.array(
                [g.contact.any(axis=1) if g else [False, False] for g in result.geometry]
            )
            valid = np.array([f.valid for f in result.head_fits])
            geometry = np.array(
                [g is not None and np.isfinite(g.distance).any() for g in result.geometry]
            )
            record = {
                "clip": clip,
                "frames": frames,
                "depth_corrected": corrected,
                "face_valid_frames": int(valid.sum()),
                "invalid_with_geometry": int((~valid & geometry).sum()),
                "contact_hands": int(contact.sum()),
                "corrected_hands": int(result.wrist_corrected.sum()),
                "reached_hands": int(result.wrist_reached.sum()),
                "contact_hands_unreached": int(
                    (contact & result.wrist_corrected & ~result.wrist_reached).sum()
                ),
                "residual_m": stats(result.wrist_residual_m),
                "head_camera_rotation_deg": stats(result.head_camera_rotation_deg),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = {
        "audit_date": "2026-10-11",
        "provenance": provenance(),
        "synthetic": synthetic_probes(),
        "cache_snapshots": {},
    }
    for directory in ("data/test_output", "data/round_trip_eval"):
        for suffix in ("_face_features.npz", "_face_features_dc.npz"):
            key = directory + "/" + suffix
            result["cache_snapshots"][key] = cache_audit(directory, suffix)
    if args.fresh:
        result["fresh_runs"] = fresh_runs()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
