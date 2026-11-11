"""
Main Pipeline class for KineticAugment.

Provides a unified interface for augmentation with constraint enforcement,
supporting both landmark and SMPL-X modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import numpy as np

from kinetic_augment.pipeline.config import PipelineConfig, ConstraintConfig
from kinetic_augment.pipeline.chain import AugmentationChain
from kinetic_augment.pipeline.presets import get_preset, PRESET_NAMES
from kinetic_augment.constraints.engine import ConstraintEngine

if TYPE_CHECKING:
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper


class Pipeline:
    """
    Main augmentation pipeline with constraint enforcement.

    Integrates AugmentationChain with ConstraintEngine to provide a complete
    augmentation workflow with anatomical validity guarantees.

    Supports two modes:
    - 'landmark': Direct landmark augmentation (fast)
    - 'smplx': SMPL-X parameter augmentation (accurate)

    Limitation: constraint enforcement (joint / velocity / collision limits) only acts in
    mode='smplx'. In mode='landmark' it is currently a no-op -- the data is returned unchanged
    because landmark-space constraint enforcement (IK -> enforce -> FK) is not implemented.

    Example:
        >>> # From preset
        >>> pipeline = Pipeline.from_preset('moderate')
        >>> augmented = pipeline.process(landmarks)

        >>> # From YAML
        >>> pipeline = Pipeline.from_yaml('configs/slr_profile.yaml')
        >>> augmented = pipeline.process(landmarks, enforce_constraints=True)

        >>> # With SMPL-X
        >>> pipeline = Pipeline.from_preset('aggressive', smplx_wrapper=wrapper)
        >>> augmented = pipeline.process(smplx_params)

    Attributes:
        config: Pipeline configuration
        chain: Augmentation chain
        constraint_engine: Constraint enforcement engine
        mode: Operating mode ('landmark' or 'smplx')
    """

    def __init__(
        self,
        config: PipelineConfig,
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            smplx_wrapper: SMPL-X wrapper for intrinsic augmentations
                          and collision detection
        """
        self.config = config
        self.smplx_wrapper = smplx_wrapper
        self.mode = config.mode

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Build augmentation chain
        self.chain = AugmentationChain.from_config(
            config=config.plan,
            mode=config.mode,
            seed=config.seed,
        )

        # Build constraint engine
        self.constraint_engine = self._build_constraint_engine()

    def _build_constraint_engine(self) -> Optional[ConstraintEngine]:
        """Build constraint engine from config."""
        cc = self.config.constraints

        if not cc.is_any_enabled():
            return None

        return ConstraintEngine(
            **cc.to_engine_kwargs(),
            smplx_wrapper=self.smplx_wrapper,
        )

    def process(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray], List],
        enforce_constraints: bool = True,
        dt: float = None,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], List]:
        """
        Process data through the augmentation pipeline.

        Args:
            data: Input data:
                - For mode='landmark': np.ndarray (frames, landmarks, 3)
                - For mode='smplx': Dict of SMPL-X parameters or List of dicts
            enforce_constraints: Whether to enforce constraints after augmentation
            dt: Time step for velocity constraints (default: 1/frame_rate)
            **kwargs: Additional arguments passed to augmentations

        Returns:
            Augmented data in the same format as input
        """
        if dt is None:
            dt = 1.0 / self.config.frame_rate

        # Apply augmentation chain
        augmented = self.chain(data, **kwargs)

        # Enforce constraints if enabled
        if enforce_constraints and self.constraint_engine is not None:
            augmented = self._enforce_constraints(augmented, dt)

        return augmented

    def _enforce_constraints(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray], List],
        dt: float,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], List]:
        """
        Enforce constraints on augmented data.

        Only acts in mode='smplx'. In mode='landmark' this is a NO-OP and returns `data`
        unchanged, whether or not a wrapper is available (landmark -> IK -> enforce -> FK is
        not implemented; see the TODO below).
        """
        if self.mode == 'smplx':
            # SMPL-X mode: enforce directly on parameters
            if isinstance(data, list):
                return self.constraint_engine.enforce_sequence(data, dt)
            else:
                return self.constraint_engine.enforce(data)

        elif self.mode == 'landmark':
            # Landmark mode: enforce on landmarks if we have a wrapper
            # Otherwise, skip constraint enforcement
            if self.smplx_wrapper is None:
                return data

            # For landmark mode with wrapper, we need IK -> enforce -> FK
            # This is expensive, so we only do it if configured
            # For now, just return data unchanged in landmark mode
            # TODO: Implement landmark constraint enforcement via IK/FK
            return data

        return data

    def process_batch(
        self,
        batch: List[Union[np.ndarray, Dict[str, np.ndarray]]],
        enforce_constraints: bool = True,
        **kwargs,
    ) -> List[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Process a batch of samples.

        Args:
            batch: List of input samples
            enforce_constraints: Whether to enforce constraints
            **kwargs: Additional arguments

        Returns:
            List of augmented samples
        """
        return [
            self.process(sample, enforce_constraints=enforce_constraints, **kwargs)
            for sample in batch
        ]

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """
        Reset the random seed.

        Args:
            seed: New seed (uses config seed if None)
        """
        self.chain.reset_rng(seed if seed is not None else self.config.seed)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
    ) -> 'Pipeline':
        """
        Create pipeline from YAML configuration file.

        Args:
            path: Path to YAML file
            smplx_wrapper: SMPL-X wrapper for intrinsic augmentations

        Returns:
            Configured Pipeline instance
        """
        config = PipelineConfig.from_yaml(path)
        return cls(config, smplx_wrapper=smplx_wrapper)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        seed: Optional[int] = None,
    ) -> 'Pipeline':
        """
        Create pipeline from a built-in preset.

        Args:
            preset_name: Name of preset ('none', 'conservative', 'moderate', 'aggressive')
            smplx_wrapper: SMPL-X wrapper for intrinsic augmentations
            seed: Override preset seed

        Returns:
            Configured Pipeline instance
        """
        config = get_preset(preset_name)
        if seed is not None:
            config.seed = seed
        return cls(config, smplx_wrapper=smplx_wrapper)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
    ) -> 'Pipeline':
        """
        Create pipeline from configuration dictionary.

        Args:
            config_dict: Configuration dictionary
            smplx_wrapper: SMPL-X wrapper

        Returns:
            Configured Pipeline instance
        """
        config = PipelineConfig.from_dict(config_dict)
        return cls(config, smplx_wrapper=smplx_wrapper)

    def get_info(self) -> Dict[str, Any]:
        """
        Get pipeline information.

        Returns:
            Dictionary with pipeline details
        """
        return {
            'profile_name': self.config.profile_name,
            'description': self.config.description,
            'mode': self.mode,
            'num_augmentations': len(self.chain),
            'augmentations': self.chain.get_step_info(),
            'constraints': {
                'joint_limits': self.config.constraints.joint_limits,
                'velocity_limits': self.config.constraints.velocity_limits,
                'collision_detection': self.config.constraints.collision_detection,
            },
            'seed': self.config.seed,
            'frame_rate': self.config.frame_rate,
        }

    def __repr__(self) -> str:
        return (
            f"Pipeline(profile={self.config.profile_name}, "
            f"mode={self.mode}, augmentations={len(self.chain)})"
        )


def list_presets() -> List[str]:
    """Get list of available preset names."""
    return PRESET_NAMES.copy()
