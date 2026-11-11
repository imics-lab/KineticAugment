"""
Legacy Augmentation Pipeline (DEPRECATED).

This module is deprecated. Please use the new Pipeline class instead:

    # Old way (deprecated)
    from kinetic_augment import AugmentationPipeline
    pipeline = AugmentationPipeline("config.yaml")

    # New way (recommended)
    from kinetic_augment import Pipeline
    pipeline = Pipeline.from_yaml("config.yaml")
    # or
    pipeline = Pipeline.from_preset("moderate")

The new Pipeline class provides:
- Built-in presets (conservative, moderate, aggressive)
- PyTorch Dataset integration
- Configurable constraint enforcement
- Better reproducibility with seed support
"""

import warnings
import yaml
import numpy as np
from pathlib import Path

from . import canonicalization
from . import augmentations


class AugmentationPipeline:
    """
    DEPRECATED: Use `kinetic_augment.Pipeline` instead.

    Legacy augmentation pipeline for backwards compatibility.
    """

    def __init__(self, profile_path: Path):
        warnings.warn(
            "AugmentationPipeline is deprecated. Use Pipeline.from_yaml() instead:\n"
            "  from kinetic_augment import Pipeline\n"
            "  pipeline = Pipeline.from_yaml('config.yaml')",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            with open(profile_path, 'r') as f:
                self.profile = yaml.safe_load(f)
            print(f"Augmentation profile '{self.profile.get('profile_name', 'N/A')}' loaded successfully.")
        except Exception as e:
            print(f"Error loading profile from {profile_path}: {e}")
            raise

        # Map operation names from YAML to actual functions
        self.augmentation_map = {
            "GlobalRotation": augmentations.global_rotation,
            "GlobalScaling": augmentations.global_scaling,
            "PoseFlipping": augmentations.pose_flipping,
            "TimeWarping": augmentations.time_warping,
            "DynamicTrajectoryJittering": augmentations.dynamic_trajectory_jittering,
            # Placeholders for advanced augmentations
            "JointAnglePerturbation": augmentations.joint_angle_perturbation,
            "JointCoupledNoise": augmentations.joint_coupled_noise,
        }

    def process(self, raw_sequence_data: np.ndarray, output_space: str = 'canonical') -> np.ndarray:
        """
        Processes a raw data sequence through the full pipeline.

        Args:
            raw_sequence_data (np.ndarray): The input data of shape (num_frames, 543, 3).
            output_space (str): The desired output space. Either 'canonical' or 'original'.

        Returns:
            np.ndarray: The processed data in the specified output space.
        """
        # 1. Convert to Canonical Space and store transformation params
        print("  -> Step 1: Converting to canonical space...")
        canonical_data, params_per_frame = canonicalization.canonicalize_sequence(raw_sequence_data)

        augmented_data = canonical_data.copy()

        # 2. Apply Augmentation Plan
        print("  -> Step 2: Applying augmentation plan...")
        plan = self.profile.get('plan', [])
        for step in plan:
            operation_name = step.get('operation')
            probability = step.get('probability', 1.0)
            params = step.get('params', {})

            if np.random.rand() < probability:
                if operation_name in self.augmentation_map:
                    print(f"    - Applying '{operation_name}'...")
                    aug_func = self.augmentation_map[operation_name]
                    augmented_data = aug_func(augmented_data, **params)
                else:
                    print(f"    - Warning: Augmentation '{operation_name}' not implemented. Skipping.")

        # 3. Handle Output Space
        if output_space == 'original':
            print("  -> Step 3: De-canonicalizing back to original space...")
            final_data = canonicalization.decanonicalize_sequence(augmented_data, params_per_frame)
        elif output_space == 'canonical':
            print("  -> Step 3: Keeping data in canonical space.")
            final_data = augmented_data
        else:
            raise ValueError(f"Unknown output_space: '{output_space}'. Must be 'canonical' or 'original'.")

        print("  -> Augmentation pipeline finished.")
        return final_data
