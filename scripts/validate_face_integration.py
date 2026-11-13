#!/usr/bin/env python3
"""Run real cached clips through the V1.3 face pipeline and record validation evidence.

Inputs are read-only ``*_hands.npz`` and ``*_face.npz`` archives. Feature archives and the JSON
report are written beneath ``--output-dir``, which must be outside every input directory.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_INPUT_DIRS = (Path("data/test_output"), Path("data/round_trip_eval"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(input_dirs, selected):
    clips = {}
    for directory in input_dirs:
        root = directory.resolve(strict=True)
        for face_path in sorted(root.glob("*_face.npz")):
            clip = face_path.name[: -len("_face.npz")]
            hands_path = root / (clip + "_hands.npz")
            if hands_path.is_file():
                clips.setdefault(clip, (root, face_path, hands_path))
    if selected:
        missing = sorted(set(selected) - clips.keys())
        if missing:
            raise ValueError("selected clips lack paired face/hand caches: {}".format(missing))
        clips = {clip: clips[clip] for clip in selected}
    return clips


def assert_output_is_separate(output_dir, input_dirs):
    output = output_dir.resolve()
    for directory in input_dirs:
        source = directory.resolve(strict=True)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("output directory must be separate from input directories")


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_regression_values(
    reference, candidate, reference_valid=None, candidate_valid=None, atol=1e-10
):
    """Compare arrays only after requiring identical validity and finite-value masks."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    same_shape = reference.shape == candidate.shape
    if not same_shape:
        return {
            "same_shape": False,
            "valid_mask_equal": False,
            "finite_mask_equal": False,
            "compared_finite_count": 0,
            "max_abs_delta": None,
            "pass": False,
        }

    def expanded_valid(mask):
        if mask is None:
            return np.ones(reference.shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim < reference.ndim and mask.shape == reference.shape[: mask.ndim]:
            mask = mask.reshape(mask.shape + (1,) * (reference.ndim - mask.ndim))
        return np.broadcast_to(mask, reference.shape)

    ref_valid = expanded_valid(reference_valid)
    cand_valid = expanded_valid(candidate_valid)
    ref_finite = np.isfinite(reference)
    cand_finite = np.isfinite(candidate)
    valid_equal = np.array_equal(ref_valid, cand_valid)
    finite_equal = np.array_equal(ref_finite, cand_finite)
    comparable = ref_valid & cand_valid & ref_finite & cand_finite
    count = int(comparable.sum())
    delta = (
        float(np.max(np.abs(reference[comparable] - candidate[comparable])))
        if count
        else None
    )
    return {
        "same_shape": True,
        "valid_mask_equal": valid_equal,
        "finite_mask_equal": finite_equal,
        "compared_finite_count": count,
        "max_abs_delta": delta,
        "pass": bool(
            valid_equal and finite_equal and count > 0 and delta is not None and delta <= atol
        ),
    }


def archive_metrics(features, result):
    from kinetic_augment.body_model.face_stream import from_head_frame, transform_points
    from scipy.spatial.transform import Rotation

    attached = np.asarray(features["face_attachment_valid"], dtype=bool)
    face_valid = np.asarray(features["face_observed"], dtype=bool)
    geometry_valid = np.asarray(features["geometry_valid"], dtype=bool)
    geometry_distance = np.asarray(features["distance"], dtype=np.float64)
    finite_dist = np.isfinite(geometry_distance).any(axis=(1, 2))
    invalid_face_geometry_accepted = (~face_valid) & geometry_valid.any(axis=1)

    inverse_errors = []
    pixel_inverse_errors = []
    reposed_local_errors = []
    for i in np.flatnonzero(attached):
        local = np.asarray(features["face_stream_local"][i])
        head = np.asarray(features["head_transform"][i])
        pixels = np.asarray(features["face_pixels"][i])
        pixel_to_world = np.asarray(features["face_pixel_to_world"][i])
        reconstructed = from_head_frame(local, head[:3, :3], head[:3, 3])
        expected = transform_points(pixels, pixel_to_world)
        finite = np.isfinite(reconstructed).all(axis=1) & np.isfinite(expected).all(axis=1)
        if finite.any():
            world_delta = reconstructed[finite] - expected[finite]
            inverse_errors.extend(np.linalg.norm(world_delta, axis=1))
            recovered_pixels = transform_points(reconstructed, np.linalg.inv(pixel_to_world))
            finite_pixels = np.isfinite(recovered_pixels).all(axis=1) & np.isfinite(
                pixels
            ).all(axis=1)
            pixel_delta = recovered_pixels[finite_pixels] - pixels[finite_pixels]
            pixel_inverse_errors.extend(np.linalg.norm(pixel_delta, axis=1))
            reposed_rotation = Rotation.from_rotvec([0.23, -0.31, 0.17]).as_matrix()
            reposed_position = np.array([0.41, -0.28, 0.63])
            reposed_world = from_head_frame(local, reposed_rotation, reposed_position)
            recovered_local = (reposed_world - reposed_position) @ reposed_rotation
            finite_local = np.isfinite(recovered_local).all(axis=1) & np.isfinite(local).all(axis=1)
            reposed_local_errors.extend(
                np.linalg.norm(recovered_local[finite_local] - local[finite_local], axis=1)
            )

    corrected_unreached = np.asarray(features["wrist_corrected"], bool) & ~np.asarray(
        features["wrist_reached"], bool
    )
    return {
        "frames": int(len(result.frames)),
        "observed_face_frames": int(face_valid.sum()),
        "head_attachment_frames": int(attached.sum()),
        "attachment_coverage_of_observed_face": (
            float(attached.sum() / face_valid.sum()) if face_valid.any() else None
        ),
        "body_valid_frames": int(np.asarray(features["body_ok"], bool).sum()),
        "geometry_valid_hand_frames": int(geometry_valid.sum()),
        "frames_with_finite_geometry": int(finite_dist.sum()),
        "invalid_face_frames_with_accepted_geometry": int(invalid_face_geometry_accepted.sum()),
        "invalid_face_geometry_mask_pass": not bool(invalid_face_geometry_accepted.any()),
        "attachment_reconstruction_error_m": finite_stats(inverse_errors),
        "attachment_inverse_pixel_error_px": finite_stats(pixel_inverse_errors),
        "arbitrary_rigid_repose_inverse_error_m": finite_stats(reposed_local_errors),
        "corrected_unreached_hands": int(corrected_unreached.sum()),
        "unreached_hands_with_valid_geometry": int(
            (corrected_unreached & geometry_valid).sum()
        ),
    }


def run_training_wiring_probe(feature_paths, seed=913):
    """Probe fresh feature archives with synthetic labels, never recognition accuracy."""
    import torch
    from kinetic_augment.evaluation.datasets.face_features import (
        FaceTrainingSample,
        collate_face_samples,
        sample_from_face_features,
    )
    from kinetic_augment.body_model.face_features import load_face_features
    from kinetic_augment.evaluation.models.face_classifier import FaceStreamClassifier

    torch.manual_seed(seed)
    feature_paths = [Path(path) for path in feature_paths]
    features = [load_face_features(path) for path in feature_paths]
    result = {
        "synthetic_labels_only": True,
        "accuracy_measured": False,
        "samples": len(feature_paths),
        "clips": [path.name.removesuffix("_face_features.npz") for path in feature_paths],
        "no_face_clips": [
            path.name.removesuffix("_face_features.npz")
            for path, archive in zip(feature_paths, features)
            if not np.asarray(archive["face_point_valid"], dtype=bool).any()
        ],
        "modes": {},
    }
    for mode, include_blendshapes in (("multimodal", True), ("body_only", False)):
        samples = [
            sample_from_face_features(archive, include_blendshapes=include_blendshapes)
            for archive in features
        ]
        if mode == "body_only":
            samples = [
                FaceTrainingSample(
                    streams={"body_hands": sample.streams["body_hands"]},
                    masks={"body_hands": sample.masks["body_hands"]},
                    timestamp_ms=sample.timestamp_ms,
                    source_frames=sample.source_frames,
                    source_time_ms=sample.source_time_ms,
                    interpolated=sample.interpolated,
                    metadata=sample.metadata,
                )
                for sample in samples
            ]
        batch = [(sample, index % 2) for index, sample in enumerate(samples)]
        inputs, lengths, labels = collate_face_samples(batch)
        stream_dims = {name: value.shape[-1] for name, value in inputs["streams"].items()}
        model = FaceStreamClassifier(num_classes=2, stream_dims=stream_dims, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        optimizer.step()
        updates = [
            parameter.detach() - initial
            for parameter, initial in zip(model.parameters(), before)
        ]
        result["modes"][mode] = {
            "streams": sorted(inputs["streams"]),
            "all_stream_values_finite": all(
                torch.isfinite(value).all().item() for value in inputs["streams"].values()
            ),
            "all_logits_finite": bool(torch.isfinite(logits).all().item()),
            "finite_gradients": bool(gradients_finite),
            "optimizer_step_completed": True,
            "updated_parameter_count": sum(bool(torch.any(delta != 0).item()) for delta in updates),
            "all_parameter_updates_finite": all(
                torch.isfinite(delta).all().item() for delta in updates
            ),
            "loss_before_step": float(loss.detach().item()),
            "padded_frames": int(lengths.max().item()),
            "blendshape_valid_values": int(
                inputs["masks"].get(
                    "blendshapes", torch.zeros((), dtype=torch.bool)
                ).sum().item()
            ),
        }
    return result


def face_only_adapter_probe(hands, face, models, features, output_dir):
    """Fail one real body fit, then confirm observed image-face input survives in the adapter."""
    from kinetic_augment.body_model.face_pipeline import run_clip, save_face_features
    from kinetic_augment.evaluation.datasets.face_features import sample_from_face_features
    from kinetic_augment.body_model.face_features import load_face_features
    from kinetic_augment.evaluation.datasets.face_features import FACE_SUBSET_IDS

    point_valid = np.asarray(features["face_point_valid"], dtype=bool)
    point_visible = np.asarray(features["face_point_visible"], dtype=bool)
    point_rows = (point_valid[:, FACE_SUBSET_IDS] & point_visible[:, FACE_SUBSET_IDS]).any(axis=1)
    candidates = np.flatnonzero(point_rows)
    if not len(candidates):
        return {"available": False, "reason": "no observed face rows in selected clip"}
    source_frame = int(np.asarray(features["frames"])[candidates[0]])
    body_failure_hands = copy.copy(hands)
    body_failure_hands.pose_world = np.asarray(hands.pose_world).copy()
    body_failure_hands.pose_visibility = np.asarray(hands.pose_visibility).copy()
    body_failure_hands.pose_world[source_frame] = np.nan
    body_failure_hands.pose_visibility[source_frame] = 0
    failed = run_clip(
        body_failure_hands,
        face,
        models,
        frames=np.array([source_frame], dtype=np.int64),
    )
    probe_path = output_dir / "body_failure_face_only_probe.npz"
    save_face_features(probe_path, failed, face)
    failed_features = load_face_features(probe_path)
    sample = sample_from_face_features(failed_features, face_space="image")
    return {
        "available": True,
        "source_frame": source_frame,
        "producer_body_fit_failed": bool(not failed.body_ok[0]),
        "producer_preserved_observed_face": bool(failed.face_point_valid[0].any()),
        "body_hand_valid_values": int(sample.masks["body_hands"][0].sum()),
        "face_valid_values": int(sample.masks["face"][0].sum()),
        "face_survives_without_body_hand": bool(
            not sample.masks["body_hands"][0].any() and sample.masks["face"][0].any()
        ),
        "pass": bool(
            not failed.body_ok[0]
            and failed.face_point_valid[0].any()
            and not sample.masks["body_hands"][0].any()
            and sample.masks["face"][0].any()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="directory containing paired *_hands.npz and *_face.npz (repeatable)",
    )
    parser.add_argument("--clip", action="append", help="clip id; repeatable (default: all)")
    parser.add_argument(
        "--probe-feature-dir",
        type=Path,
        help="probe existing *_face_features.npz archives without rerunning fitting",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="separate destination for features/report"
    )
    parser.add_argument(
        "--max-clips", type=int, help="explicitly cap sorted clip coverage for a runtime probe"
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="process every Nth frame; default is all"
    )
    args = parser.parse_args()
    if args.probe_feature_dir is not None:
        assert_output_is_separate(args.output_dir, [args.probe_feature_dir])
        feature_paths = sorted(args.probe_feature_dir.glob("*_face_features.npz"))
        if args.clip:
            by_clip = {path.name[: -len("_face_features.npz")]: path for path in feature_paths}
            missing = sorted(set(args.clip) - by_clip.keys())
            if missing:
                parser.error("requested feature archives not found: {}".format(missing))
            feature_paths = [by_clip[clip] for clip in args.clip]
        if not feature_paths:
            parser.error("no saved *_face_features.npz archives found")
        import torch

        torch.set_num_threads(1)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        probe = run_training_wiring_probe(feature_paths)
        report = {
            "schema": 1,
            "probe_feature_dir": str(args.probe_feature_dir.resolve(strict=True)),
            "classifier_real_archive_probe": probe,
            "recognition_accuracy": {
                "measured": False,
                "reason": "Classifier labels in this wiring probe are synthetic.",
            },
        }
        report_path = args.output_dir / "face_classifier_archive_probe.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("probed {} feature archives; wrote {}".format(len(feature_paths), report_path))
        return

    input_dirs = args.input_dir or list(DEFAULT_INPUT_DIRS)
    if args.frame_stride < 1 or (args.max_clips is not None and args.max_clips < 1):
        parser.error("--frame-stride and --max-clips must be positive")
    assert_output_is_separate(args.output_dir, input_dirs)
    clips = discover(input_dirs, args.clip)
    if args.max_clips is not None:
        clips = dict(list(clips.items())[: args.max_clips])
    if not clips:
        parser.error("no paired real face/hand caches found")

    import torch
    from kinetic_augment.body_model.face_features import load_face_features
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import (
        load_face_models,
        run_clip,
        save_face_features,
    )
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {
        str(path): sha256(path)
        for _, face_path, hands_path in clips.values()
        for path in (face_path, hands_path)
    }
    models = load_face_models()
    records, total_frames, feature_paths = [], 0, []
    pilot_sample = None
    regression_samples = {}
    for clip, (source_dir, face_path, hands_path) in clips.items():
        started = time.monotonic()
        hands = load_hand_clip(hands_path)
        face = load_face_clip(face_path)
        frames = np.arange(0, len(face.valid), args.frame_stride, dtype=np.int64)
        result = run_clip(hands, face, models, frames=frames)
        output_path = args.output_dir / (clip + "_face_features.npz")
        save_face_features(output_path, result, face)
        feature_paths.append(output_path)
        features = load_face_features(output_path)
        metrics = archive_metrics(features, result)
        records.append(
            {
                "clip": clip,
                "source_dir": str(source_dir),
                "source_face_sha256": file_hashes[str(face_path)],
                "source_hands_sha256": file_hashes[str(hands_path)],
                "output": str(output_path),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                **metrics,
            }
        )
        total_frames += len(frames)
        if pilot_sample is None:
            pilot_sample = (clip, hands, face, result, features, output_path)
        if clip in ("01986", "01991"):
            regression_samples[clip] = (clip, hands, face, result)

    if any(
        sha256(Path(path)) != digest for path, digest in file_hashes.items()
    ):
        raise RuntimeError("an input cache changed during validation")

    # Attachment is observational: disabling only that stream must leave body and hand outputs
    # unchanged. Keep each clip's full fitting context, then compare three geometry-valid rows.
    regression_results = []
    selected_regressions = [regression_samples[name] for name in ("01986", "01991")
                            if name in regression_samples]
    if not selected_regressions:
        selected_regressions = [(pilot_sample[0], *pilot_sample[1:4])]
    for clip, hands, face, reference_result in selected_regressions:
        finite_geometry_rows = [
            row
            for row, geometry in enumerate(reference_result.geometry)
            if geometry is not None and np.isfinite(geometry.distance).any()
        ]
        probe_rows = np.asarray(
            finite_geometry_rows[:3]
            if finite_geometry_rows
            else list(range(min(3, len(reference_result.frames)))),
            dtype=np.int64,
        )
        probe_frames = reference_result.frames[probe_rows]
        detached_result = run_clip(
            hands,
            face,
            models,
            frames=reference_result.frames,
            attach_face_stream=False,
        )
        reference_distances = np.asarray(
            [
                reference_result.geometry[row].distance
                if reference_result.geometry[row] is not None
                else np.full((2, 7), np.nan)
                for row in probe_rows
            ]
        )
        detached_distances = np.asarray(
            [
                detached_result.geometry[row].distance
                if detached_result.geometry[row] is not None
                else np.full((2, 7), np.nan)
                for row in probe_rows
            ]
        )
        reference_body_ok = reference_result.body_ok[probe_rows]
        detached_body_ok = detached_result.body_ok[probe_rows]
        body_ok_equal = np.array_equal(reference_body_ok, detached_body_ok)
        reference_geometry_valid = reference_result.geometry_valid[probe_rows]
        detached_geometry_valid = detached_result.geometry_valid[probe_rows]
        rotation_comparison = compare_regression_values(
            reference_result.rotations[probe_rows],
            detached_result.rotations[probe_rows],
            reference_body_ok,
            detached_body_ok,
        )
        translation_comparison = compare_regression_values(
            reference_result.transl[probe_rows],
            detached_result.transl[probe_rows],
            reference_body_ok,
            detached_body_ok,
        )
        distance_comparison = compare_regression_values(
            reference_distances,
            detached_distances,
            reference_geometry_valid[:, :, None],
            detached_geometry_valid[:, :, None],
        )
        geometry_valid_equal = np.array_equal(
            reference_geometry_valid, detached_geometry_valid
        )
        regression_results.append(
            {
                "clip": clip,
                "frames": probe_frames.tolist(),
                "body_ok_equal": body_ok_equal,
                "geometry_valid_mask_equal": geometry_valid_equal,
                "body_rotations": rotation_comparison,
                "translation_m": translation_comparison,
                "hand_face_distance_m": distance_comparison,
                "face_attachment_disabled": bool(not detached_result.face_attachment_valid.any()),
                "pass": bool(
                    body_ok_equal
                    and geometry_valid_equal
                    and rotation_comparison["pass"]
                    and translation_comparison["pass"]
                    and distance_comparison["pass"]
                ),
            }
        )

    # Run the plan's explicit corrected-but-unreached regression on its known pilot interval.
    unreached_regression = {"available": False}
    target_clip = "01986"
    if target_clip in clips:
        _, source_face_path, source_hands_path = clips[target_clip]
        hands_dc = load_hand_clip(source_hands_path)
        face_dc = load_face_clip(source_face_path)
        target_frames = np.arange(14, min(22, len(face_dc.valid)), dtype=np.int64)
        corrected = run_clip(hands_dc, face_dc, models, frames=target_frames, depth_corrected=True)
        corrected_features_path = args.output_dir / (target_clip + "_corrected_probe.npz")
        save_face_features(corrected_features_path, corrected, face_dc)
        corrected_features = load_face_features(corrected_features_path)
        corrected_unreached = corrected_features["wrist_corrected"] & ~corrected_features[
            "wrist_reached"
        ]
        invalid_geometry_leaks = int(
            (corrected_unreached & corrected_features["geometry_valid"]).sum()
        )
        unreached_regression = {
            "available": True,
            "clip": target_clip,
            "frames": target_frames.tolist(),
            "corrected_unreached_hands": int(corrected_unreached.sum()),
            "unreached_hands_with_valid_geometry": invalid_geometry_leaks,
            "pass": invalid_geometry_leaks == 0,
        }

    training_probe = run_training_wiring_probe(feature_paths)
    report = {
        "schema": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coverage": {
            "input_directories": [str(path.resolve()) for path in input_dirs],
            "clip_count": len(records),
            "clips": [record["clip"] for record in records],
            "processed_frames": total_frames,
            "frame_stride": args.frame_stride,
            "full_frame_coverage": args.frame_stride == 1,
        },
        "clips": records,
        "attachment_on_off_body_hand_regression": regression_results,
        "corrected_unreached_geometry_mask_regression": unreached_regression,
        "face_only_adapter_probe": face_only_adapter_probe(
            pilot_sample[1], pilot_sample[2], models, pilot_sample[4], args.output_dir
        ),
        "classifier_real_archive_probe": training_probe,
        "source_hashes_before": file_hashes,
        "source_hashes_unchanged": True,
        "recognition_accuracy": {
            "measured": False,
            "reason": "No usable real WLASL label annotations were found for these cached clips.",
        },
    }
    report_path = args.output_dir / "face_integration_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "validated {} clips / {} frames; wrote {}".format(
            len(records), total_frames, report_path
        )
    )


if __name__ == "__main__":
    main()
