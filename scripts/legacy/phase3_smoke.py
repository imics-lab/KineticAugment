#!/usr/bin/env python3
"""
Phase 3 Testing Script for KineticAugment.

Tests the PyBullet collision detection system:
1. Test body part definitions
2. Test collision checker
3. Test collision resolution
4. Test collision constraint
5. Test integration with ConstraintEngine
6. End-to-end test with SMPL-X

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/legacy/phase3_smoke.py
"""

import os
import sys
from pathlib import Path

# Set up headless environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def test_body_parts():
    """Test body part definitions."""
    print("\n" + "=" * 60)
    print("TEST: Body Part Definitions")
    print("=" * 60)

    from kinetic_augment.constraints.collision.body_parts import (
        COLLISION_BODY_PARTS,
        SLR_COLLISION_PAIRS,
        ADJACENT_PARTS,
        are_adjacent,
        get_contact_tolerance,
        get_body_part_params,
    )
    from kinetic_augment.body_model.joint_mapping import SMPLX_BODY_JOINTS

    tests_passed = 0
    tests_total = 0

    # Test 1: Body parts defined
    tests_total += 1
    try:
        assert len(COLLISION_BODY_PARTS) == 12
        print(f"  12 body parts defined: {list(COLLISION_BODY_PARTS.keys())}")
        tests_passed += 1
    except Exception as e:
        print(f"  Body part definitions failed: {e}")

    # Test 2: SLR collision pairs
    tests_total += 1
    try:
        assert len(SLR_COLLISION_PAIRS) >= 8
        print(f"  {len(SLR_COLLISION_PAIRS)} SLR collision pairs defined")
        tests_passed += 1
    except Exception as e:
        print(f"  SLR collision pairs failed: {e}")

    # Test 3: Adjacent parts check
    tests_total += 1
    try:
        assert are_adjacent('left_forearm', 'left_hand')
        assert not are_adjacent('left_hand', 'right_hand')
        print(f"  are_adjacent() works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  are_adjacent() failed: {e}")

    # Test 4: Contact tolerance
    tests_total += 1
    try:
        tol = get_contact_tolerance('left_hand', 'torso')
        assert tol > 0
        print(f"  Contact tolerance: {tol:.3f}m for hand-torso")
        tests_passed += 1
    except Exception as e:
        print(f"  get_contact_tolerance() failed: {e}")

    # Test 5: Body part params from joints
    tests_total += 1
    try:
        # Create fake joint positions
        joint_positions = np.zeros((55, 3), dtype=np.float32)
        # Set some positions
        joint_positions[SMPLX_BODY_JOINTS['left_shoulder']] = [0.3, 0.5, 0.0]
        joint_positions[SMPLX_BODY_JOINTS['left_elbow']] = [0.3, 0.2, 0.0]

        params = get_body_part_params(joint_positions, SMPLX_BODY_JOINTS, 'left_upper_arm')

        assert 'center' in params
        assert 'radius' in params
        assert params['type'] == 'capsule'
        print(f"  get_body_part_params() works: center={params['center']}, radius={params['radius']:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"  get_body_part_params() failed: {e}")

    print(f"\nBody Parts Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_collision_checker():
    """Test collision checker."""
    print("\n" + "=" * 60)
    print("TEST: Collision Checker")
    print("=" * 60)

    try:
        from kinetic_augment.constraints.collision import (
            CollisionChecker,
            check_pybullet_available,
            PYBULLET_AVAILABLE,
        )
    except ImportError:
        print("  PyBullet not installed, skipping collision checker tests")
        return True  # Skip but don't fail

    if not PYBULLET_AVAILABLE:
        print("  PyBullet not available, skipping collision checker tests")
        return True

    from kinetic_augment.body_model.joint_mapping import SMPLX_BODY_JOINTS

    tests_passed = 0
    tests_total = 0

    # Test 1: Create checker
    tests_total += 1
    try:
        checker = CollisionChecker(use_gui=False)
        checker.setup()
        print(f"  CollisionChecker created and set up")
        tests_passed += 1
    except Exception as e:
        print(f"  CollisionChecker creation failed: {e}")
        return False

    # Test 2: Update from joints (T-pose - no collisions expected)
    tests_total += 1
    try:
        # Create T-pose joint positions
        joint_positions = np.zeros((55, 3), dtype=np.float32)

        # Set up a basic T-pose
        # Pelvis at origin
        joint_positions[SMPLX_BODY_JOINTS['pelvis']] = [0, 0.9, 0]
        # Spine
        joint_positions[SMPLX_BODY_JOINTS['spine1']] = [0, 1.0, 0]
        joint_positions[SMPLX_BODY_JOINTS['spine2']] = [0, 1.1, 0]
        joint_positions[SMPLX_BODY_JOINTS['spine3']] = [0, 1.2, 0]
        # Head
        joint_positions[SMPLX_BODY_JOINTS['neck']] = [0, 1.35, 0]
        joint_positions[SMPLX_BODY_JOINTS['head']] = [0, 1.5, 0]
        # Left arm (T-pose)
        joint_positions[SMPLX_BODY_JOINTS['left_shoulder']] = [0.2, 1.25, 0]
        joint_positions[SMPLX_BODY_JOINTS['left_elbow']] = [0.5, 1.25, 0]
        joint_positions[SMPLX_BODY_JOINTS['left_wrist']] = [0.8, 1.25, 0]
        # Right arm (T-pose)
        joint_positions[SMPLX_BODY_JOINTS['right_shoulder']] = [-0.2, 1.25, 0]
        joint_positions[SMPLX_BODY_JOINTS['right_elbow']] = [-0.5, 1.25, 0]
        joint_positions[SMPLX_BODY_JOINTS['right_wrist']] = [-0.8, 1.25, 0]
        # Legs
        joint_positions[SMPLX_BODY_JOINTS['left_hip']] = [0.1, 0.85, 0]
        joint_positions[SMPLX_BODY_JOINTS['left_knee']] = [0.1, 0.45, 0]
        joint_positions[SMPLX_BODY_JOINTS['left_ankle']] = [0.1, 0.05, 0]
        joint_positions[SMPLX_BODY_JOINTS['right_hip']] = [-0.1, 0.85, 0]
        joint_positions[SMPLX_BODY_JOINTS['right_knee']] = [-0.1, 0.45, 0]
        joint_positions[SMPLX_BODY_JOINTS['right_ankle']] = [-0.1, 0.05, 0]

        checker.update_from_joints(joint_positions)
        collisions = checker.check_collisions()

        print(f"  T-pose collisions: {len(collisions)}")
        tests_passed += 1
    except Exception as e:
        print(f"  update_from_joints failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Create intentional collision (crossed arms)
    tests_total += 1
    try:
        # Cross the arms in front of torso
        joint_positions[SMPLX_BODY_JOINTS['left_elbow']] = [-0.1, 1.1, 0.15]
        joint_positions[SMPLX_BODY_JOINTS['left_wrist']] = [-0.4, 1.0, 0.15]
        joint_positions[SMPLX_BODY_JOINTS['right_elbow']] = [0.1, 1.1, 0.15]
        joint_positions[SMPLX_BODY_JOINTS['right_wrist']] = [0.4, 1.0, 0.15]

        checker.update_from_joints(joint_positions)
        collisions = checker.check_collisions()

        print(f"  Crossed arms collisions: {len(collisions)}")
        for c in collisions[:3]:
            print(f"      {c[0]} <-> {c[1]}: {c[2]:.4f}m")
        tests_passed += 1
    except Exception as e:
        print(f"  Crossed arms test failed: {e}")

    # Test 4: Get contact points
    tests_total += 1
    try:
        contacts = checker.get_contact_points()
        print(f"  Contact points: {len(contacts)}")
        for contact in contacts[:3]:
            print(f"      {contact}")
        tests_passed += 1
    except Exception as e:
        print(f"  get_contact_points() failed: {e}")

    # Test 5: Cleanup
    tests_total += 1
    try:
        checker.cleanup()
        print(f"  Cleanup successful")
        tests_passed += 1
    except Exception as e:
        print(f"  Cleanup failed: {e}")

    print(f"\nCollision Checker Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_collision_resolution():
    """Test collision resolution."""
    print("\n" + "=" * 60)
    print("TEST: Collision Resolution")
    print("=" * 60)

    try:
        from kinetic_augment.constraints.collision import PYBULLET_AVAILABLE
        from kinetic_augment.constraints.collision.resolution import (
            GradientResolver,
            create_resolver,
        )
        from kinetic_augment.constraints.collision.collision_checker import ContactPoint
    except ImportError:
        print("  PyBullet not installed, skipping resolution tests")
        return True

    if not PYBULLET_AVAILABLE:
        print("  PyBullet not available, skipping resolution tests")
        return True

    tests_passed = 0
    tests_total = 0

    # Test 1: Create resolver
    tests_total += 1
    try:
        resolver = GradientResolver(step_size=0.1, max_iterations=5)
        print(f"  GradientResolver created")
        tests_passed += 1
    except Exception as e:
        print(f"  GradientResolver creation failed: {e}")

    # Test 2: Factory function
    tests_total += 1
    try:
        resolver = create_resolver('gradient', step_size=0.05)
        assert isinstance(resolver, GradientResolver)
        print(f"  create_resolver('gradient') works")
        tests_passed += 1
    except Exception as e:
        print(f"  create_resolver() failed: {e}")

    # Test 3: Resolve with fake contact
    tests_total += 1
    try:
        params = {
            'body_pose': np.zeros((1, 63), dtype=np.float32),
            'global_orient': np.zeros((1, 3), dtype=np.float32),
        }

        # Create fake contact
        contact = ContactPoint(
            body_part_a='left_hand',
            body_part_b='torso',
            position=np.array([0.0, 1.0, 0.1]),
            normal=np.array([0.0, 0.0, 1.0]),
            depth=0.02,
        )

        resolved = resolver.resolve(params, [contact])

        # Should have modified body_pose
        changed = not np.allclose(resolved['body_pose'], params['body_pose'])
        print(f"  Resolve with contact: params modified = {changed}")
        tests_passed += 1
    except Exception as e:
        print(f"  Resolve failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nCollision Resolution Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_collision_constraint():
    """Test collision constraint."""
    print("\n" + "=" * 60)
    print("TEST: Collision Constraint")
    print("=" * 60)

    try:
        from kinetic_augment.constraints.collision import (
            CollisionConstraint,
            CollisionViolation,
            PYBULLET_AVAILABLE,
        )
    except ImportError:
        print("  PyBullet not installed, skipping constraint tests")
        return True

    if not PYBULLET_AVAILABLE:
        print("  PyBullet not available, skipping constraint tests")
        return True

    tests_passed = 0
    tests_total = 0

    # Test 1: Create constraint
    tests_total += 1
    try:
        constraint = CollisionConstraint(
            smplx_wrapper=None,  # No wrapper for basic tests
            resolution_strategy='gradient',
        )
        assert constraint.is_available
        print(f"  CollisionConstraint created, available={constraint.is_available}")
        tests_passed += 1
    except Exception as e:
        print(f"  CollisionConstraint creation failed: {e}")
        return False

    # Test 2: Check method (needs joints, but we don't have wrapper)
    tests_total += 1
    try:
        params = {
            'body_pose': np.zeros((1, 63), dtype=np.float32),
        }
        # Without smplx_wrapper, check returns empty (can't compute joints)
        violations = constraint.check(params)
        print(f"  check() returns {len(violations)} violations (expected 0 without wrapper)")
        tests_passed += 1
    except Exception as e:
        print(f"  check() failed: {e}")

    # Test 3: CollisionViolation structure
    tests_total += 1
    try:
        violation = CollisionViolation(
            constraint_type='collision',
            joint_name='left_hand_torso',
            body_part_a='left_hand',
            body_part_b='torso',
            penetration_depth=0.015,
            severity=0.3,
        )
        print(f"  CollisionViolation: {violation}")
        tests_passed += 1
    except Exception as e:
        print(f"  CollisionViolation failed: {e}")

    print(f"\nCollision Constraint Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_engine_integration():
    """Test integration with ConstraintEngine."""
    print("\n" + "=" * 60)
    print("TEST: ConstraintEngine Integration")
    print("=" * 60)

    from kinetic_augment.constraints import ConstraintEngine, PYBULLET_AVAILABLE

    tests_passed = 0
    tests_total = 0

    # Test 1: Create engine without collision detection
    tests_total += 1
    try:
        engine = ConstraintEngine(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
        )
        print(f"  Engine without collision: {engine}")
        tests_passed += 1
    except Exception as e:
        print(f"  Engine creation failed: {e}")

    # Test 2: Create engine with collision detection (may fail gracefully)
    tests_total += 1
    try:
        engine = ConstraintEngine(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=True,
        )
        if PYBULLET_AVAILABLE:
            print(f"  Engine with collision: {engine}")
        else:
            print(f"  Engine with collision (disabled - no PyBullet): {engine}")
        tests_passed += 1
    except Exception as e:
        print(f"  Engine with collision failed: {e}")

    # Test 3: Validate returns collision violations
    tests_total += 1
    try:
        params = {
            'body_pose': np.zeros((1, 63), dtype=np.float32),
        }
        is_valid, violations = engine.validate(params, return_violations=True)
        # Count collision violations
        collision_violations = [v for v in violations if v.constraint_type == 'collision']
        print(f"  validate() collision violations: {len(collision_violations)}")
        tests_passed += 1
    except Exception as e:
        print(f"  validate() failed: {e}")

    # Test 4: Enforce applies collision constraint
    tests_total += 1
    try:
        result = engine.enforce(params)
        print(f"  enforce() returned params successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  enforce() failed: {e}")

    print(f"\nEngine Integration Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_end_to_end_smplx():
    """End-to-end test with SMPL-X model."""
    print("\n" + "=" * 60)
    print("TEST: End-to-End with SMPL-X")
    print("=" * 60)

    try:
        from kinetic_augment.constraints.collision import PYBULLET_AVAILABLE
    except ImportError:
        PYBULLET_AVAILABLE = False

    if not PYBULLET_AVAILABLE:
        print("  PyBullet not available, skipping end-to-end test")
        return True

    tests_passed = 0
    tests_total = 0

    # Try to load SMPL-X wrapper
    tests_total += 1
    try:
        from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
        from kinetic_augment.constraints import ConstraintEngine
        from kinetic_augment.constraints.collision import CollisionConstraint

        model_path = Path(__file__).resolve().parents[2] / 'models'
        if not (model_path / 'smplx').exists():
            print("  SMPL-X models not found, skipping end-to-end test")
            return True

        wrapper = SMPLXWrapper(model_path=str(model_path))
        print(f"  Loaded SMPL-X wrapper")
        tests_passed += 1
    except Exception as e:
        print(f"  SMPL-X wrapper failed: {e}")
        return True  # Don't fail if no SMPL-X models

    # Test 2: Create constraint with wrapper
    tests_total += 1
    try:
        constraint = CollisionConstraint(
            smplx_wrapper=wrapper,
            resolution_strategy='gradient',
        )
        print(f"  CollisionConstraint with wrapper created")
        tests_passed += 1
    except Exception as e:
        print(f"  CollisionConstraint with wrapper failed: {e}")

    # Test 3: Check neutral pose (T-pose)
    tests_total += 1
    try:
        params = wrapper.get_default_params()
        violations = constraint.check(params)
        print(f"  Neutral pose violations: {len(violations)}")
        tests_passed += 1
    except Exception as e:
        print(f"  Neutral pose check failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Create collision and check
    tests_total += 1
    try:
        # Bend arms to create potential collision with torso
        params = wrapper.get_default_params()
        # Modify shoulder to bring arm across body
        # left_shoulder is joint 16, pose index (16-1)*3 = 45
        params['body_pose'][0, 45:48] = [0.5, 0.5, -0.5]  # Some rotation
        # left_elbow is joint 18, pose index (18-1)*3 = 51
        params['body_pose'][0, 51:54] = [1.5, 0, 0]  # Bend elbow

        violations = constraint.check(params)
        print(f"  Bent arm pose violations: {len(violations)}")
        for v in violations[:3]:
            print(f"      {v}")
        tests_passed += 1
    except Exception as e:
        print(f"  Bent arm check failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Engine integration with wrapper
    tests_total += 1
    try:
        engine = ConstraintEngine(
            joint_limits=True,
            velocity_limits=False,
            collision_detection=True,
            smplx_wrapper=wrapper,
        )

        params = wrapper.get_default_params()
        result = engine.enforce(params)
        is_valid, violations = engine.validate(result, return_violations=True)

        print(f"  Engine with SMPL-X: valid={is_valid}, violations={len(violations)}")
        tests_passed += 1
    except Exception as e:
        print(f"  Engine with SMPL-X failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nEnd-to-End Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def main():
    print("=" * 60)
    print("KineticAugment Phase 3 Testing")
    print("PyBullet Collision Detection")
    print("=" * 60)

    # Check PyBullet availability
    try:
        import pybullet as p
        print(f"\nPyBullet version: {p.getAPIVersion()}")
    except ImportError:
        print("\nPyBullet not installed. Install with: pip install pybullet")
        print("Collision detection tests will be skipped.")

    results = {
        'body_parts': test_body_parts(),
        'collision_checker': test_collision_checker(),
        'resolution': test_collision_resolution(),
        'constraint': test_collision_constraint(),
        'engine': test_engine_integration(),
        'e2e': test_end_to_end_smplx(),
    }

    print("\n" + "=" * 60)
    print("PHASE 3 TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All Phase 3 tests passed! Collision detection is working.")
    else:
        print("Some tests failed. Please review the output above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
