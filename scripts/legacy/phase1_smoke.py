#!/usr/bin/env python3
"""
Phase 1 Testing Script for KineticAugment.

This script tests the body model module by:
1. Extracting MediaPipe landmarks from sign language videos
2. Testing the joint mapping functionality
3. Testing SMPL-X fitting (if models are available)
4. Validating round-trip conversion
5. Generating visualization and statistics

Usage:
    python scripts/legacy/phase1_smoke.py --video-dir data/sign_videos --output-dir data/test_output
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

# Set up headless environment before importing GUI libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Configure matplotlib for headless operation BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")

# Import KineticAugment modules
from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    MEDIAPIPE_POSE_LANDMARKS,
    SMPLX_TO_MEDIAPIPE_POSE,
    SMPLX_JOINT_LIMITS,
    get_corresponding_joints,
    get_kinematic_chain,
    clamp_joint_angles,
    LIMB_GROUPS,
)
from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    TOTAL_LANDMARKS,
)

# Check for SMPL-X
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MediaPipeExtractor:
    """Extract landmarks from video using MediaPipe Holistic."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for landmark extraction")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_video(self, video_path: str) -> dict:
        """
        Extract landmarks from a video file.

        Returns:
            Dictionary with:
                - landmarks: numpy array (num_frames, 543, 3)
                - frame_count: number of frames
                - fps: frames per second
                - detection_rate: percentage of frames with detected landmarks
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_landmarks = []
        detected_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)

            # Extract landmarks
            frame_landmarks = self._extract_frame_landmarks(results)
            all_landmarks.append(frame_landmarks)

            # Check if pose was detected
            if results.pose_landmarks is not None:
                detected_frames += 1

        cap.release()

        landmarks_array = np.array(all_landmarks)
        detection_rate = detected_frames / len(all_landmarks) * 100 if all_landmarks else 0

        return {
            'landmarks': landmarks_array,
            'frame_count': len(all_landmarks),
            'fps': fps,
            'detection_rate': detection_rate,
        }

    def _extract_frame_landmarks(self, results) -> np.ndarray:
        """Extract all landmarks for a single frame."""
        landmarks = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)

        # Pose landmarks (33)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[i] = [lm.x, lm.y, lm.z]

        # Face landmarks (468)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                landmarks[NUM_POSE_LANDMARKS + i] = [lm.x, lm.y, lm.z]

        # Left hand landmarks (21)
        lh_start = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                landmarks[lh_start + i] = [lm.x, lm.y, lm.z]

        # Right hand landmarks (21)
        rh_start = lh_start + NUM_HAND_LANDMARKS
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                landmarks[rh_start + i] = [lm.x, lm.y, lm.z]

        return landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()


def test_joint_mapping():
    """Test the joint mapping module."""
    print("\n" + "=" * 60)
    print("TEST: Joint Mapping Module")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Test 1: SMPL-X body joints count
    tests_total += 1
    if len(SMPLX_BODY_JOINTS) == 22:
        print("✓ SMPL-X has 22 body joints")
        tests_passed += 1
    else:
        print(f"✗ SMPL-X body joints: expected 22, got {len(SMPLX_BODY_JOINTS)}")

    # Test 2: MediaPipe pose landmarks count
    tests_total += 1
    if len(MEDIAPIPE_POSE_LANDMARKS) == 33:
        print("✓ MediaPipe has 33 pose landmarks")
        tests_passed += 1
    else:
        print(f"✗ MediaPipe landmarks: expected 33, got {len(MEDIAPIPE_POSE_LANDMARKS)}")

    # Test 3: Joint correspondence mapping
    tests_total += 1
    mp_to_smplx = get_corresponding_joints('mediapipe')
    if len(mp_to_smplx) > 0:
        print(f"✓ MediaPipe→SMPL-X mapping has {len(mp_to_smplx)} correspondences")
        tests_passed += 1
    else:
        print("✗ No joint correspondences found")

    # Test 4: Kinematic chain
    tests_total += 1
    chain = get_kinematic_chain('left_wrist', 'pelvis')
    expected_joints = ['left_shoulder', 'left_elbow', 'left_wrist']
    if all(j in chain for j in expected_joints):
        print(f"✓ Kinematic chain for left arm: {' → '.join(chain)}")
        tests_passed += 1
    else:
        print(f"✗ Kinematic chain missing expected joints")

    # Test 5: Joint limits clamping
    tests_total += 1
    test_angles = np.array([5.0, 0.0, 0.0])  # Exceeds elbow limit
    clamped = clamp_joint_angles('left_elbow', test_angles)
    if clamped[0] <= 2.62:
        print(f"✓ Joint limit clamping works (5.0 → {clamped[0]:.2f})")
        tests_passed += 1
    else:
        print(f"✗ Joint limit clamping failed")

    # Test 6: Limb groups
    tests_total += 1
    if 'left_arm' in LIMB_GROUPS and 'right_arm' in LIMB_GROUPS:
        print(f"✓ Limb groups defined: {list(LIMB_GROUPS.keys())}")
        tests_passed += 1
    else:
        print("✗ Limb groups not properly defined")

    print(f"\nJoint Mapping Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_landmark_extraction(video_path: str, extractor: MediaPipeExtractor) -> dict:
    """Test MediaPipe landmark extraction on a video."""
    print(f"\n  Processing: {Path(video_path).name}")

    start_time = time.time()
    result = extractor.extract_from_video(video_path)
    elapsed = time.time() - start_time

    landmarks = result['landmarks']

    # Compute statistics
    pose_landmarks = landmarks[:, :NUM_POSE_LANDMARKS, :]

    # Check for valid landmarks (non-zero)
    valid_mask = np.any(pose_landmarks != 0, axis=(1, 2))
    valid_frames = valid_mask.sum()

    # Compute landmark ranges
    if valid_frames > 0:
        valid_landmarks = pose_landmarks[valid_mask]
        x_range = (valid_landmarks[:, :, 0].min(), valid_landmarks[:, :, 0].max())
        y_range = (valid_landmarks[:, :, 1].min(), valid_landmarks[:, :, 1].max())
        z_range = (valid_landmarks[:, :, 2].min(), valid_landmarks[:, :, 2].max())
    else:
        x_range = y_range = z_range = (0, 0)

    # Check hand detection
    lh_start = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
    rh_start = lh_start + NUM_HAND_LANDMARKS
    left_hand_detected = np.any(landmarks[:, lh_start:lh_start + NUM_HAND_LANDMARKS, :] != 0, axis=(1, 2)).sum()
    right_hand_detected = np.any(landmarks[:, rh_start:rh_start + NUM_HAND_LANDMARKS, :] != 0, axis=(1, 2)).sum()

    stats = {
        'video': Path(video_path).name,
        'frames': result['frame_count'],
        'fps': result['fps'],
        'detection_rate': result['detection_rate'],
        'valid_frames': int(valid_frames),
        'left_hand_frames': int(left_hand_detected),
        'right_hand_frames': int(right_hand_detected),
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'processing_time': elapsed,
        'landmarks': landmarks,
    }

    print(f"    Frames: {result['frame_count']}, Detection: {result['detection_rate']:.1f}%")
    print(f"    Left hand: {left_hand_detected}/{result['frame_count']}, Right hand: {right_hand_detected}/{result['frame_count']}")
    print(f"    Time: {elapsed:.2f}s")

    return stats


def test_body_model_conversion(landmarks: np.ndarray) -> dict:
    """Test the body model conversion (MediaPipe ↔ SMPL-X)."""
    print("\n" + "=" * 60)
    print("TEST: Body Model Conversion")
    print("=" * 60)

    results = {
        'fitter': None,
        'smplx_available': False,
        'round_trip_error': None,
    }

    # Test SMPLXWrapper (requires models)
    print("\n2. Testing SMPL-X integration...")
    # Note: smplx.create() expects the parent directory containing 'smplx/' folder
    model_path = Path(__file__).resolve().parents[2] / 'models'
    smplx_model_dir = model_path / 'smplx'


    if not smplx_model_dir.exists():
        print(f"   ⊘ SMPL-X models not found at {smplx_model_dir}")
        print("     Run: python scripts/download_models.py --setup-dir models")
        return results

    try:
        from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
        from kinetic_augment.legacy.mp_to_smplx import MediaPipeToSMPLX
        from kinetic_augment.legacy.smplx_to_mp import SMPLXToMediaPipe

        print("   Loading SMPL-X model...")
        wrapper = SMPLXWrapper(
            model_path=Path(__file__).resolve().parents[2] / "models",
            gender="neutral",
            use_pca=False,
        )
        print(f"   ✓ SMPL-X model loaded ({wrapper.device})")
        results['smplx_available'] = True

        # Test fitting
        print("\n   Testing MediaPipe → SMPL-X fitting...")
        fitter = MediaPipeToSMPLX()

        test_frame = landmarks[0]
        smplx_params = fitter.fit(test_frame)

        print(f"   ✓ Fitting completed")
        print(f"     - body_pose shape: {smplx_params['body_pose'].shape}")

        # Test projection
        print("\n   Testing SMPL-X → MediaPipe projection...")
        projector = SMPLXToMediaPipe(wrapper)

        # Add batch dimension
        batch_params = {k: v[np.newaxis, ...] if v.ndim == 1 else v
                       for k, v in smplx_params.items() if not k.startswith('_')}

        reconstructed = projector.project(batch_params, target_mp=test_frame[np.newaxis], return_full_frame=False)
        print(f"   ✓ Projection completed")
        print(f"     - Output shape: {reconstructed.shape}")

        # Compute round-trip error
        original_pose = test_frame[:NUM_POSE_LANDMARKS]
        reconstructed_pose = reconstructed[0]

        # Only compare valid landmarks (non-zero in original)
        valid_mask = np.any(original_pose != 0, axis=1)
        if valid_mask.sum() > 0:
            error = np.sqrt(((original_pose[valid_mask] - reconstructed_pose[valid_mask]) ** 2).mean())
            results['round_trip_error'] = float(error)
            print(f"\n   Round-trip error (RMSE): {error:.4f}")

            # Per-joint analysis
            print("\n   Per-joint errors (top 5):")
            joint_errors = []
            for name, idx in MEDIAPIPE_POSE_LANDMARKS.items():
                if valid_mask[idx]:
                    err = np.linalg.norm(original_pose[idx] - reconstructed_pose[idx])
                    joint_errors.append((name, err))

            joint_errors.sort(key=lambda x: x[1], reverse=True)
            for name, err in joint_errors[:5]:
                print(f"     - {name}: {err:.4f}")

    except FileNotFoundError as e:
        print(f"   ⊘ SMPL-X model files not found: {e}")
    except Exception as e:
        print(f"   ✗ SMPL-X test failed: {e}")
        import traceback
        traceback.print_exc()

    return results


def analyze_sign_motion(landmarks: np.ndarray, video_name: str):
    """Analyze motion characteristics of sign language video."""
    print(f"\n  Motion Analysis for {video_name}:")

    pose_landmarks = landmarks[:, :NUM_POSE_LANDMARKS, :]

    # Compute frame-to-frame motion
    if len(pose_landmarks) > 1:
        motion = np.diff(pose_landmarks, axis=0)
        motion_magnitude = np.sqrt((motion ** 2).sum(axis=2))

        # Average motion per landmark
        avg_motion = motion_magnitude.mean(axis=0)

        # Find most active landmarks
        landmark_names = list(MEDIAPIPE_POSE_LANDMARKS.keys())
        motion_by_name = [(landmark_names[i], avg_motion[i]) for i in range(len(avg_motion))]
        motion_by_name.sort(key=lambda x: x[1], reverse=True)

        print("    Most active landmarks:")
        for name, motion_val in motion_by_name[:5]:
            print(f"      - {name}: {motion_val:.4f}")

        # Hand motion specifically
        left_wrist_idx = MEDIAPIPE_POSE_LANDMARKS['left_wrist']
        right_wrist_idx = MEDIAPIPE_POSE_LANDMARKS['right_wrist']

        left_wrist_motion = motion_magnitude[:, left_wrist_idx].mean()
        right_wrist_motion = motion_magnitude[:, right_wrist_idx].mean()

        print(f"    Wrist motion - Left: {left_wrist_motion:.4f}, Right: {right_wrist_motion:.4f}")

        # Temporal smoothness (jerk)
        if len(motion) > 1:
            acceleration = np.diff(motion, axis=0)
            jerk = np.sqrt((acceleration ** 2).sum(axis=2)).mean()
            print(f"    Motion smoothness (avg jerk): {jerk:.6f}")


def visualize_landmarks(landmarks: np.ndarray, output_path: str, video_name: str):
    """Create visualization of landmark trajectories."""
    if not MATPLOTLIB_AVAILABLE:
        print("    Matplotlib not available - skipping visualization")
        return

    pose_landmarks = landmarks[:, :NUM_POSE_LANDMARKS, :]
    num_frames = len(pose_landmarks)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Landmark Analysis: {video_name}', fontsize=14)

    # Plot 1: Wrist trajectories (X-Y)
    ax = axes[0, 0]
    left_wrist = pose_landmarks[:, MEDIAPIPE_POSE_LANDMARKS['left_wrist'], :]
    right_wrist = pose_landmarks[:, MEDIAPIPE_POSE_LANDMARKS['right_wrist'], :]

    # Filter out zero frames
    left_valid = np.any(left_wrist != 0, axis=1)
    right_valid = np.any(right_wrist != 0, axis=1)

    if left_valid.any():
        ax.plot(left_wrist[left_valid, 0], left_wrist[left_valid, 1], 'b-', label='Left Wrist', alpha=0.7)
        ax.scatter(left_wrist[left_valid, 0][0], left_wrist[left_valid, 1][0], c='b', s=100, marker='o', zorder=5)
        ax.scatter(left_wrist[left_valid, 0][-1], left_wrist[left_valid, 1][-1], c='b', s=100, marker='x', zorder=5)

    if right_valid.any():
        ax.plot(right_wrist[right_valid, 0], right_wrist[right_valid, 1], 'r-', label='Right Wrist', alpha=0.7)
        ax.scatter(right_wrist[right_valid, 0][0], right_wrist[right_valid, 1][0], c='r', s=100, marker='o', zorder=5)
        ax.scatter(right_wrist[right_valid, 0][-1], right_wrist[right_valid, 1][-1], c='r', s=100, marker='x', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Wrist Trajectories (o=start, x=end)')
    ax.legend()
    ax.invert_yaxis()  # MediaPipe Y increases downward
    ax.set_aspect('equal')

    # Plot 2: Wrist position over time
    ax = axes[0, 1]
    frames = np.arange(num_frames)

    if left_valid.any():
        ax.plot(frames[left_valid], left_wrist[left_valid, 0], 'b-', label='Left X', alpha=0.7)
        ax.plot(frames[left_valid], left_wrist[left_valid, 1], 'b--', label='Left Y', alpha=0.7)

    if right_valid.any():
        ax.plot(frames[right_valid], right_wrist[right_valid, 0], 'r-', label='Right X', alpha=0.7)
        ax.plot(frames[right_valid], right_wrist[right_valid, 1], 'r--', label='Right Y', alpha=0.7)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Position')
    ax.set_title('Wrist Position Over Time')
    ax.legend()

    # Plot 3: Skeleton visualization (first frame)
    ax = axes[1, 0]
    first_valid = np.where(np.any(pose_landmarks != 0, axis=(1, 2)))[0]
    if len(first_valid) > 0:
        frame_idx = first_valid[0]
        frame = pose_landmarks[frame_idx]

        # Plot pose landmarks
        valid = np.any(frame != 0, axis=1)
        ax.scatter(frame[valid, 0], frame[valid, 1], c='blue', s=30, alpha=0.7)

        # Draw connections
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
        ]
        for c1, c2 in connections:
            if valid[c1] and valid[c2]:
                ax.plot([frame[c1, 0], frame[c2, 0]], [frame[c1, 1], frame[c2, 1]], 'b-', alpha=0.5)

        ax.set_title(f'Skeleton (Frame {frame_idx})')
    else:
        ax.text(0.5, 0.5, 'No valid frames', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Plot 4: Motion magnitude over time
    ax = axes[1, 1]
    if num_frames > 1:
        motion = np.diff(pose_landmarks, axis=0)
        motion_mag = np.sqrt((motion ** 2).sum(axis=2))

        # Overall motion
        total_motion = motion_mag.sum(axis=1)
        ax.plot(range(1, num_frames), total_motion, 'g-', label='Total', alpha=0.7)

        # Wrist motion
        ax.plot(range(1, num_frames), motion_mag[:, MEDIAPIPE_POSE_LANDMARKS['left_wrist']], 'b-', label='Left Wrist', alpha=0.7)
        ax.plot(range(1, num_frames), motion_mag[:, MEDIAPIPE_POSE_LANDMARKS['right_wrist']], 'r-', label='Right Wrist', alpha=0.7)

        ax.set_xlabel('Frame')
        ax.set_ylabel('Motion Magnitude')
        ax.set_title('Motion Over Time')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Testing for KineticAugment")
    parser.add_argument('--video-dir', type=Path, default=Path('data/sign_videos'),
                       help='Directory containing sign language videos')
    parser.add_argument('--output-dir', type=Path, default=Path('data/test_output'),
                       help='Directory for test outputs')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip video extraction (use cached landmarks)')
    parser.add_argument('--max-videos', type=int, default=5,
                       help='Maximum number of videos to process')

    args = parser.parse_args()

    print("=" * 60)
    print("KineticAugment Phase 1 Testing")
    print("=" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Joint Mapping Module
    mapping_ok = test_joint_mapping()

    # Test 2: MediaPipe Landmark Extraction
    print("\n" + "=" * 60)
    print("TEST: MediaPipe Landmark Extraction")
    print("=" * 60)

    video_files = sorted(args.video_dir.glob('*.mp4'))[:args.max_videos]

    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} video(s)")

    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available - cannot extract landmarks")
        return

    extractor = MediaPipeExtractor()
    all_stats = []

    for video_path in video_files:
        landmarks_cache = args.output_dir / f"{video_path.stem}_landmarks.npy"

        if args.skip_extraction and landmarks_cache.exists():
            print(f"\n  Loading cached: {video_path.name}")
            landmarks = np.load(landmarks_cache)
            stats = {'video': video_path.name, 'landmarks': landmarks, 'frames': len(landmarks)}
        else:
            stats = test_landmark_extraction(str(video_path), extractor)

            # Cache landmarks
            np.save(landmarks_cache, stats['landmarks'])

        all_stats.append(stats)

        # Analyze motion
        analyze_sign_motion(stats['landmarks'], stats['video'])

        # Create visualization
        viz_path = args.output_dir / f"{video_path.stem}_analysis.png"
        visualize_landmarks(stats['landmarks'], str(viz_path), stats['video'])

    extractor.close()

    # Test 3: Body Model Conversion
    # Use the video with best detection rate
    best_stats = max(all_stats, key=lambda x: x.get('detection_rate', 0))
    print(f"\nUsing {best_stats['video']} for body model tests (best detection)")

    conversion_results = test_body_model_conversion(best_stats['landmarks'])

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print("\n1. Joint Mapping: " + ("✓ PASSED" if mapping_ok else "✗ FAILED"))

    print("\n2. Landmark Extraction:")
    for stats in all_stats:
        det_rate = stats.get('detection_rate', 0)
        print(f"   - {stats['video']}: {stats['frames']} frames, {det_rate:.1f}% detection")

    print("\n3. Body Model Conversion:")
    print(f"   - MediaPipeToSMPLX fitter: {conversion_results['fitter']}")
    print(f"   - SMPL-X available: {conversion_results['smplx_available']}")
    if conversion_results['round_trip_error'] is not None:
        print(f"   - Round-trip error: {conversion_results['round_trip_error']:.4f}")

    # Save summary
    summary = {
        'joint_mapping': 'passed' if mapping_ok else 'failed',
        'videos': [{k: v for k, v in s.items() if k != 'landmarks'} for s in all_stats],
        'conversion': conversion_results,
    }

    summary_path = args.output_dir / 'test_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Phase 1 Testing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
