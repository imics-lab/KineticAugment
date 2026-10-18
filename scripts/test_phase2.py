#!/usr/bin/env python3
"""
Phase 2 Testing Script for KineticAugment.

Tests the new augmentation architecture and constraint system:
1. Test base augmentation classes
2. Test intrinsic (SMPL-X) augmentations
3. Test extrinsic (landmark) augmentations
4. Test temporal augmentations
5. Test constraint enforcement
6. End-to-end test with sign language video

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/test_phase2.py
"""

import os
import sys
from pathlib import Path

from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX

# Set up headless environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_augmentation_base():
    """Test base augmentation classes."""
    print("\n" + "=" * 60)
    print("TEST: Augmentation Base Classes")
    print("=" * 60)

    from kinetic_augment.augmentations.base import (
        BaseAugmentation,
        LandmarkAugmentation,
        SMPLXAugmentation,
        AugmentationRegistry,
    )

    tests_passed = 0
    tests_total = 0

    # Test 1: Registry exists
    tests_total += 1
    try:
        available = AugmentationRegistry.list_available()
        print(f"✓ Registry has {len(available)} registered augmentations")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Registry test failed: {e}")

    # Test 2: Can list by category
    tests_total += 1
    try:
        intrinsic = AugmentationRegistry.list_by_category('intrinsic')
        print(f"✓ Found {len(intrinsic)} intrinsic augmentations: {intrinsic}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Category listing failed: {e}")

    print(f"\nBase Classes Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_intrinsic_augmentations():
    """Test SMPL-X based augmentations."""
    print("\n" + "=" * 60)
    print("TEST: Intrinsic (SMPL-X) Augmentations")
    print("=" * 60)

    from kinetic_augment.augmentations.intrinsic import (
        JointAnglePerturbation,
        HandPosePerturbation,
        JointCoupledNoise,
        GlobalOrientPerturbation,
    )

    tests_passed = 0
    tests_total = 0

    # Create test SMPL-X parameters
    test_params = {
        'body_pose': np.zeros((1, 63), dtype=np.float32),
        'left_hand_pose': np.zeros((1, 45), dtype=np.float32),
        'right_hand_pose': np.zeros((1, 45), dtype=np.float32),
        'betas': np.zeros((1, 10), dtype=np.float32),
        'global_orient': np.zeros((1, 3), dtype=np.float32),
    }

    # Test 1: JointAnglePerturbation
    tests_total += 1
    try:
        aug = JointAnglePerturbation({
            'stddev': 0.2,
            'joint_groups': ['left_arm', 'right_arm'],
        })
        result = aug.apply(test_params)

        # Check that body_pose changed
        assert not np.allclose(result['body_pose'], test_params['body_pose'])
        print(f"✓ JointAnglePerturbation works")
        print(f"    Max angle change: {np.abs(result['body_pose'] - test_params['body_pose']).max():.3f} rad")
        tests_passed += 1
    except Exception as e:
        print(f"✗ JointAnglePerturbation failed: {e}")

    # Test 2: HandPosePerturbation
    tests_total += 1
    try:
        aug = HandPosePerturbation({
            'stddev': 0.1,
            'couple_joints': True,
        })
        result = aug.apply(test_params)

        assert not np.allclose(result['left_hand_pose'], test_params['left_hand_pose'])
        print(f"✓ HandPosePerturbation works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ HandPosePerturbation failed: {e}")

    # Test 3: JointCoupledNoise
    tests_total += 1
    try:
        aug = JointCoupledNoise({
            'stddev': 0.15,
            'coupling_mode': 'mirror',
            'affected_limbs': ['arms'],
        })
        result = aug.apply(test_params)

        assert not np.allclose(result['body_pose'], test_params['body_pose'])
        print(f"✓ JointCoupledNoise works (mirror mode)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ JointCoupledNoise failed: {e}")

    # Test 4: GlobalOrientPerturbation
    tests_total += 1
    try:
        aug = GlobalOrientPerturbation({
            'max_angle_deg': {'x': 15, 'y': 30, 'z': 10},
        })
        result = aug.apply(test_params)

        assert not np.allclose(result['global_orient'], test_params['global_orient'])
        print(f"✓ GlobalOrientPerturbation works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ GlobalOrientPerturbation failed: {e}")

    print(f"\nIntrinsic Augmentation Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_extrinsic_augmentations():
    """Test landmark-based augmentations."""
    print("\n" + "=" * 60)
    print("TEST: Extrinsic (Landmark) Augmentations")
    print("=" * 60)

    from kinetic_augment.augmentations.extrinsic import (
        GlobalRotation,
        GlobalScaling,
        PoseFlipping,
        TrajectoryJittering,
        GaussianNoise,
    )

    tests_passed = 0
    tests_total = 0

    # Create test landmarks (30 frames, 543 landmarks, 3D)
    test_landmarks = np.random.randn(30, 543, 3).astype(np.float32) * 0.1

    # Test 1: GlobalRotation
    tests_total += 1
    try:
        aug = GlobalRotation({'max_angle_deg': {'x': 20, 'y': 20, 'z': 20}})
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        assert not np.allclose(result, test_landmarks)
        print(f"✓ GlobalRotation works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ GlobalRotation failed: {e}")

    # Test 2: GlobalScaling
    tests_total += 1
    try:
        aug = GlobalScaling({'scale_range': [0.8, 1.2]})
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        print(f"✓ GlobalScaling works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ GlobalScaling failed: {e}")

    # Test 3: PoseFlipping
    tests_total += 1
    try:
        aug = PoseFlipping({'flip_probability': 1.0})  # Always flip for test
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        # X coordinates should be negated (before label swap)
        print(f"✓ PoseFlipping works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ PoseFlipping failed: {e}")

    # Test 4: TrajectoryJittering
    tests_total += 1
    try:
        aug = TrajectoryJittering({
            'amplitude': 0.02,
            'frequency': 0.5,
        })
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        print(f"✓ TrajectoryJittering works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ TrajectoryJittering failed: {e}")

    # Test 5: GaussianNoise
    tests_total += 1
    try:
        aug = GaussianNoise({
            'stddev': 0.01,
            'landmark_groups': ['pose', 'left_hand', 'right_hand'],
        })
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        print(f"✓ GaussianNoise works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ GaussianNoise failed: {e}")

    print(f"\nExtrinsic Augmentation Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_temporal_augmentations():
    """Test temporal augmentations."""
    print("\n" + "=" * 60)
    print("TEST: Temporal Augmentations")
    print("=" * 60)

    from kinetic_augment.augmentations.temporal import (
        TimeWarping,
        SpeedVariation,
        FrameDropping,
    )

    tests_passed = 0
    tests_total = 0

    # Create test landmarks
    test_landmarks = np.random.randn(50, 543, 3).astype(np.float32) * 0.1

    # Test 1: TimeWarping
    tests_total += 1
    try:
        aug = TimeWarping({
            'max_warp_factor': 0.2,
            'num_knots': 5,
        })
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        assert not np.allclose(result, test_landmarks)
        print(f"✓ TimeWarping works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ TimeWarping failed: {e}")

    # Test 2: SpeedVariation (same frames)
    tests_total += 1
    try:
        aug = SpeedVariation({
            'speed_range': [0.8, 1.2],
            'output_frames': 'same',
        })
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape
        print(f"✓ SpeedVariation (same frames) works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ SpeedVariation failed: {e}")

    # Test 3: SpeedVariation (proportional frames)
    tests_total += 1
    try:
        aug = SpeedVariation({
            'speed_range': [0.5, 0.5],  # Fixed 2x slow down
            'output_frames': 'proportional',
        })
        result = aug.apply(test_landmarks)

        # Should have more frames (slower = more frames)
        assert result.shape[0] > test_landmarks.shape[0] * 1.5
        print(f"✓ SpeedVariation (proportional) works: {test_landmarks.shape[0]} → {result.shape[0]} frames")
        tests_passed += 1
    except Exception as e:
        print(f"✗ SpeedVariation (proportional) failed: {e}")

    # Test 4: FrameDropping
    tests_total += 1
    try:
        aug = FrameDropping({
            'drop_probability': 0.2,
            'max_consecutive_drops': 2,
        })
        result = aug.apply(test_landmarks)

        assert result.shape == test_landmarks.shape  # Output same length (interpolated)
        print(f"✓ FrameDropping works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FrameDropping failed: {e}")

    print(f"\nTemporal Augmentation Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_constraints():
    """Test constraint enforcement."""
    print("\n" + "=" * 60)
    print("TEST: Constraint System")
    print("=" * 60)

    from kinetic_augment.constraints import (
        ConstraintEngine,
        JointLimitConstraint,
        VelocityConstraint,
    )

    tests_passed = 0
    tests_total = 0

    # Test 1: Joint limit constraint
    tests_total += 1
    try:
        constraint = JointLimitConstraint()

        # Create params with a violation (elbow hyperextended)
        bad_params = {
            'body_pose': np.zeros((1, 63), dtype=np.float32),
        }
        # Elbow is joint index 18, pose index (18-1)*3 = 51
        bad_params['body_pose'][0, 51] = 5.0  # Way beyond 2.62 limit

        violations = constraint.check(bad_params)
        assert len(violations) > 0
        print(f"✓ JointLimitConstraint detects violations ({len(violations)} found)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ JointLimitConstraint check failed: {e}")

    # Test 2: Joint limit enforcement
    tests_total += 1
    try:
        fixed_params = constraint.enforce(bad_params)
        violations_after = constraint.check(fixed_params)

        assert len(violations_after) == 0
        assert fixed_params['body_pose'][0, 51] <= 2.62
        print(f"✓ JointLimitConstraint enforcement works (clamped to {fixed_params['body_pose'][0, 51]:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"✗ JointLimitConstraint enforcement failed: {e}")

    # Test 3: Velocity constraint
    tests_total += 1
    try:
        vel_constraint = VelocityConstraint(use_sign_language_limits=True)

        current = {'body_pose': np.zeros((1, 63), dtype=np.float32)}
        previous = {'body_pose': np.zeros((1, 63), dtype=np.float32)}

        # Create a big jump in elbow angle (velocity violation)
        current['body_pose'][0, 51] = 2.0  # 2 radians in 1/30 second = 60 rad/s

        violations = vel_constraint.check(current, previous, dt=1/30)
        assert len(violations) > 0
        print(f"✓ VelocityConstraint detects violations ({len(violations)} found)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ VelocityConstraint check failed: {e}")

    # Test 4: Constraint engine
    tests_total += 1
    try:
        engine = ConstraintEngine(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
        )

        bad_params = {'body_pose': np.zeros((1, 63), dtype=np.float32)}
        bad_params['body_pose'][0, 51] = 5.0

        fixed = engine.enforce(bad_params)
        is_valid, violations = engine.validate(fixed, return_violations=True)

        print(f"✓ ConstraintEngine works (valid={is_valid}, violations={len(violations)})")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ConstraintEngine failed: {e}")

    print(f"\nConstraint Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_end_to_end():
    """End-to-end test with real sign language data."""
    print("\n" + "=" * 60)
    print("TEST: End-to-End Pipeline")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Check if we have cached landmarks from Phase 1
    landmarks_path = Path(__file__).parent.parent / 'data' / 'test_output' / '01986_landmarks.npy'

    if not landmarks_path.exists():
        print("⊘ Skipping end-to-end test (run test_phase1.py first)")
        return True

    # Load cached landmarks
    landmarks = np.load(landmarks_path)
    print(f"Loaded landmarks: {landmarks.shape}")

    # Test 1: Apply extrinsic augmentations to landmarks
    tests_total += 1
    try:
        from kinetic_augment.augmentations.extrinsic import GlobalRotation, TrajectoryJittering

        aug1 = GlobalRotation({'max_angle_deg': {'x': 10, 'y': 15, 'z': 5}})
        aug2 = TrajectoryJittering({'amplitude': 0.01})

        result = aug1.apply(landmarks)
        result = aug2.apply(result)

        print(f"✓ Extrinsic pipeline works on real data")
        print(f"    Input range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
        print(f"    Output range: [{result.min():.3f}, {result.max():.3f}]")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Extrinsic pipeline failed: {e}")

    # Test 2: Fit SMPL-X and apply intrinsic augmentations
    tests_total += 1
    try:
        import kinetic_augment.body_model.mp_to_smplx
        from kinetic_augment.augmentations.intrinsic import JointAnglePerturbation

        fitter = MediaPipeToSMPLX()
        first_frame = landmarks[0]
        smplx_params = fitter.fit(first_frame)

        aug = JointAnglePerturbation({
            'stddev': 0.15,
            'joint_groups': ['left_arm', 'right_arm'],
        })

        augmented_params = aug.apply(smplx_params)

        print(f"✓ Intrinsic pipeline works on real data")
        print(f"    Body pose change: {np.abs(augmented_params['body_pose'] - smplx_params['body_pose']).max():.3f} rad")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Intrinsic pipeline failed: {e}")

    # Test 3: Apply constraints
    tests_total += 1
    try:
        from kinetic_augment.constraints import ConstraintEngine

        engine = ConstraintEngine(joint_limits=True, velocity_limits=False)

        # Apply heavy perturbation that might violate limits
        heavy_aug = JointAnglePerturbation({'stddev': 0.5})
        heavy_result = heavy_aug.apply(smplx_params)

        # Check violations before
        _, violations_before = engine.validate(heavy_result, return_violations=True)

        # Enforce constraints
        fixed = engine.enforce(heavy_result)
        _, violations_after = engine.validate(fixed, return_violations=True)

        print(f"✓ Constraint enforcement works on real data")
        print(f"    Violations before: {len(violations_before)}")
        print(f"    Violations after: {len(violations_after)}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Constraint enforcement failed: {e}")

    print(f"\nEnd-to-End Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def main():
    print("=" * 60)
    print("KineticAugment Phase 2 Testing")
    print("=" * 60)

    results = {
        'base': test_augmentation_base(),
        'intrinsic': test_intrinsic_augmentations(),
        'extrinsic': test_extrinsic_augmentations(),
        'temporal': test_temporal_augmentations(),
        'constraints': test_constraints(),
        'e2e': test_end_to_end(),
    }

    print("\n" + "=" * 60)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All Phase 2 tests passed! Ready for Phase 3.")
    else:
        print("Some tests failed. Please review the output above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
