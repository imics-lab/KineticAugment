"""
Pipeline Configuration for KineticAugment.

Provides dataclasses for pipeline and constraint configuration,
with YAML serialization and validation support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import yaml


@dataclass
class ConstraintConfig:
    """
    Configuration for constraint enforcement in the pipeline.

    Controls which constraints are applied and their parameters.
    """

    # Enable/disable each constraint type
    joint_limits: bool = True
    velocity_limits: bool = True
    collision_detection: bool = False

    # Constraint behavior
    strict_mode: bool = False  # True = raise errors, False = auto-correct

    # Custom limits (override defaults)
    custom_joint_limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    custom_velocity_limits: Optional[Dict[str, float]] = None

    # Collision detection settings
    collision_strategy: str = 'gradient'  # 'gradient', 'rejection', 'interpolation'
    collision_use_tiered: bool = True
    collision_pairs: Optional[List[Tuple[str, str]]] = None

    # Sign language specific
    use_sign_language_limits: bool = True

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for ConstraintEngine.__init__()."""
        return {
            'joint_limits': self.joint_limits,
            'velocity_limits': self.velocity_limits,
            'collision_detection': self.collision_detection,
            'strict_mode': self.strict_mode,
            'custom_joint_limits': self.custom_joint_limits,
            'custom_velocity_limits': self.custom_velocity_limits,
            'collision_config': {
                'strategy': self.collision_strategy,
                'use_tiered': self.collision_use_tiered,
                'collision_pairs': self.collision_pairs,
            } if self.collision_detection else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            'joint_limits': self.joint_limits,
            'velocity_limits': self.velocity_limits,
            'collision_detection': self.collision_detection,
            'strict_mode': self.strict_mode,
            'use_sign_language_limits': self.use_sign_language_limits,
        }

        if self.custom_joint_limits:
            result['custom_joint_limits'] = self.custom_joint_limits
        if self.custom_velocity_limits:
            result['custom_velocity_limits'] = self.custom_velocity_limits

        if self.collision_detection:
            result['collision'] = {
                'strategy': self.collision_strategy,
                'use_tiered': self.collision_use_tiered,
            }
            if self.collision_pairs:
                result['collision']['pairs'] = self.collision_pairs

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstraintConfig':
        """Create from dictionary."""
        collision = data.get('collision', {})
        return cls(
            joint_limits=data.get('joint_limits', True),
            velocity_limits=data.get('velocity_limits', True),
            collision_detection=data.get('collision_detection', False),
            strict_mode=data.get('strict_mode', False),
            custom_joint_limits=data.get('custom_joint_limits'),
            custom_velocity_limits=data.get('custom_velocity_limits'),
            collision_strategy=collision.get('strategy', 'gradient'),
            collision_use_tiered=collision.get('use_tiered', True),
            collision_pairs=collision.get('pairs'),
            use_sign_language_limits=data.get('use_sign_language_limits', True),
        )

    def is_any_enabled(self) -> bool:
        """Check if any constraint is enabled."""
        return self.joint_limits or self.velocity_limits or self.collision_detection


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    Defines the augmentation plan, constraint settings, and processing options.
    """

    # Profile metadata
    profile_name: str = "default"
    description: str = ""

    # Operating mode
    mode: str = 'landmark'  # 'landmark' or 'smplx'

    # Augmentation plan
    plan: List[Dict[str, Any]] = field(default_factory=list)

    # Constraint configuration
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    # Processing options
    # NOTE: `canonicalize` and `output_space` are accepted and validated but currently ignored
    # by Pipeline (only the legacy AugmentationPipeline canonicalizes).
    canonicalize: bool = False  # Apply canonicalization (not implemented in Pipeline)
    output_space: str = 'canonical'  # 'canonical' or 'original' (not implemented in Pipeline)

    # Reproducibility
    seed: Optional[int] = None

    # SMPL-X settings (only for mode='smplx')
    smplx_model_path: Optional[str] = None
    smplx_gender: str = 'neutral'

    # Temporal settings
    frame_rate: float = 30.0  # fps

    # Dataset options
    cache_augmentations: bool = False
    num_augmented_versions: int = 1

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'PipelineConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            PipelineConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        constraints_data = data.get('constraints', {})
        constraints = ConstraintConfig.from_dict(constraints_data)

        return cls(
            profile_name=data.get('profile_name', 'default'),
            description=data.get('description', ''),
            mode=data.get('mode', 'landmark'),
            plan=data.get('plan', []),
            constraints=constraints,
            canonicalize=data.get('canonicalize', False),
            output_space=data.get('output_space', 'canonical'),
            seed=data.get('seed'),
            smplx_model_path=data.get('smplx_model_path'),
            smplx_gender=data.get('smplx_gender', 'neutral'),
            frame_rate=data.get('frame_rate', 30.0),
            cache_augmentations=data.get('cache_augmentations', False),
            num_augmented_versions=data.get('num_augmented_versions', 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            'profile_name': self.profile_name,
            'mode': self.mode,
            'plan': self.plan,
            'constraints': self.constraints.to_dict(),
            'frame_rate': self.frame_rate,
        }

        if self.description:
            result['description'] = self.description
        if self.canonicalize:
            result['canonicalize'] = self.canonicalize
            result['output_space'] = self.output_space
        if self.seed is not None:
            result['seed'] = self.seed
        if self.mode == 'smplx':
            result['smplx_model_path'] = self.smplx_model_path
            result['smplx_gender'] = self.smplx_gender
        if self.cache_augmentations:
            result['cache_augmentations'] = self.cache_augmentations
        if self.num_augmented_versions > 1:
            result['num_augmented_versions'] = self.num_augmented_versions

        return result

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate mode
        if self.mode not in ['landmark', 'smplx']:
            errors.append(f"Invalid mode: {self.mode}. Must be 'landmark' or 'smplx'")

        # Validate output_space
        if self.output_space not in ['canonical', 'original']:
            errors.append(f"Invalid output_space: {self.output_space}")

        # SMPL-X mode requires model path
        if self.mode == 'smplx' and not self.smplx_model_path:
            errors.append("smplx_model_path required when mode='smplx'")

        # Validate plan
        try:
            from kinetic_augment.augmentations.base import AugmentationRegistry
            available = set(AugmentationRegistry.list_available())

            for i, step in enumerate(self.plan):
                if 'operation' not in step:
                    errors.append(f"Plan step {i}: missing 'operation' key")
                elif step['operation'] not in available:
                    errors.append(
                        f"Plan step {i}: unknown operation '{step['operation']}'. "
                        f"Available: {sorted(available)}"
                    )

                prob = step.get('probability', 1.0)
                if not 0 <= prob <= 1:
                    errors.append(f"Plan step {i}: probability must be in [0, 1], got {prob}")
        except ImportError:
            pass  # Skip augmentation validation if not available

        # Validate constraint strategy
        valid_strategies = ['gradient', 'rejection', 'interpolation']
        if self.constraints.collision_strategy not in valid_strategies:
            errors.append(
                f"Invalid collision strategy: {self.constraints.collision_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(profile={self.profile_name}, mode={self.mode}, "
            f"augmentations={len(self.plan)})"
        )
