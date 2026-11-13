"""Reproduce the frozen-root counterexample for the V1 arm swing IK."""

import numpy as np

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik


def main():
    skeleton = SMPLXSkeleton(model_path="models")

    # Same exact skeleton and limb lengths; only rotate the ground-truth root.
    truth = np.zeros((skeleton.num_joints, 3))
    truth[0] = [0.0, 0.0, 0.7]
    true_joints = forward_kinematics(truth, skeleton)

    correspondence = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
    landmarks = np.zeros((33, 3))
    for joint, landmark in correspondence:
        landmarks[landmark] = true_joints[joint]

    # Match the harness: solve without a global_orient estimate.
    fitted = solve_swing_ik(landmarks, skeleton, SMPLX_ARM_BONES)
    fitted_joints = forward_kinematics(fitted, skeleton)

    true_positions = np.array([landmarks[m] for _, m in correspondence])
    fitted_positions = np.array([fitted_joints[j] for j, _ in correspondence])
    raw_errors = np.linalg.norm(fitted_positions - true_positions, axis=1)

    scale, rotation, translation = similarity_transform(fitted_positions, true_positions)
    aligned_positions = apply_similarity(fitted_positions, scale, rotation, translation)
    aligned_errors = np.linalg.norm(aligned_positions - true_positions, axis=1)

    direction_errors = []
    for _, _, mp_parent, mp_child in SMPLX_ARM_BONES:
        observed = landmarks[mp_child] - landmarks[mp_parent]
        reconstructed = fitted_joints[
            next(b[1] for b in SMPLX_ARM_BONES if b[2] == mp_parent and b[3] == mp_child)
        ] - fitted_joints[
            next(b[0] for b in SMPLX_ARM_BONES if b[2] == mp_parent and b[3] == mp_child)
        ]
        cosine = np.dot(observed, reconstructed) / (
            np.linalg.norm(observed) * np.linalg.norm(reconstructed)
        )
        direction_errors.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    print("Synthetic same-skeleton case: ground-truth root rotvec [0, 0, 0.7] rad")
    print(f"Fitted root rotvec: {fitted[0]}")
    print("Position errors over shoulder, elbow, wrist correspondences (m):")
    print(f"  raw:     max={raw_errors.max():.9f}, mean={raw_errors.mean():.9f}")
    print(f"  aligned: max={aligned_errors.max():.9f}, mean={aligned_errors.mean():.9f}")
    print("Arm-segment direction errors (degrees):")
    for spec, error in zip(SMPLX_ARM_BONES, direction_errors):
        print(f"  {spec[2]}->{spec[3]}: {error:.9f}")
    print(f"  max={max(direction_errors):.9f}, mean={np.mean(direction_errors):.9f}")


if __name__ == "__main__":
    main()
