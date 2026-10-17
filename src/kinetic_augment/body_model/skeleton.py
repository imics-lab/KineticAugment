"""
Rest-pose skeleton sources for forward and inverse kinematics.

Fitting needs only two things from a body model: where each joint sits at rest,
and which joint is its parent. Isolating that behind a small interface lets the
kinematics run against a hand-checkable synthetic chain as well as the real
SMPL-X skeleton, so a failure can be localised to the maths or to SMPL-X's
specific rest geometry.
"""

import copy
from typing import List, Optional

import numpy as np


class SyntheticSkeleton:
    """
    A straight 3-bone chain of unit-length bones along +x.

    Deliberately trivial: every posed position is computable by hand, so a
    failing kinematics test can be checked on paper.
    """

    def __init__(self):
        self.rest_joints = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        self.parents = np.array([-1, 0, 1, 2])
        self.names: List[str] = ["root", "a", "b", "c"]

    @property
    def num_joints(self) -> int:
        return len(self.parents)


class SMPLXSkeleton:
    """
    The real SMPL-X rest skeleton: 55 model joints and their kinematic tree.

    Rest joints are the shape-blended template regressed to joint centres, which
    is the same quantity SMPL-X's own skinning poses. Only the 55 model joints
    are exposed; the 72 extra landmarks SMPL-X appends to its `joints` output are
    derived from mesh vertices and are not part of the kinematic chain.

    Joint centres are linear in the shape coefficients:

        rest_joints = joint_template + joint_shapedirs @ betas

    which is what lets a body shape be fitted to observed proportions, and lets
    `shaped` re-shape the skeleton without reloading the model.

    The full shape space is used by default -- 300 coefficients for SMPL-X v1.1.
    The common ten truncate the shape prior, which then understates how much real
    bodies vary in the arm proportions MediaPipe observes.
    """

    NUM_MODEL_JOINTS = 55
    FULL_SHAPE_SPACE = 300

    def __init__(self, model_path: str = "models", gender: str = "neutral",
                 betas: Optional[np.ndarray] = None, num_betas: int = FULL_SHAPE_SPACE):
        import smplx
        from smplx.vertex_ids import vertex_ids

        # flat_hand_mean=True so a zero rotation genuinely means no rotation.
        # SMPL-X otherwise adds a non-zero mean hand pose (up to ~0.85 rad) to
        # whatever hand parameters it is given, which would make "zero pose"
        # silently differ from the rest skeleton and corrupt round-trip error.
        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
            num_betas=num_betas,
        )

        regressor = self.model.J_regressor.detach().numpy().astype(np.float64)
        template = self.model.v_template.detach().numpy().astype(np.float64)
        shapedirs = self.model.shapedirs.detach().numpy().astype(np.float64)
        regressor = regressor[: self.NUM_MODEL_JOINTS]

        self.joint_template = regressor @ template
        self.joint_shapedirs = (regressor @ shapedirs.reshape(len(template), -1)).reshape(
            self.NUM_MODEL_JOINTS, 3, -1)

        # The nose tip on the mesh surface: the same anatomical point MediaPipe
        # reports, unlike the head joint, which sits inside the skull and points in
        # a direction ~30 degrees different from the shoulders.
        nose = vertex_ids["smplx"]["nose"]
        self._nose_template = template[nose]
        self._nose_shapedirs = shapedirs[nose]

        self.parents = self.model.parents[: self.NUM_MODEL_JOINTS].detach().numpy().astype(int)
        self.names: List[str] = [f"joint_{i}" for i in range(self.NUM_MODEL_JOINTS)]
        self._set_shape(betas)

    @property
    def num_joints(self) -> int:
        return self.NUM_MODEL_JOINTS

    @property
    def num_betas(self) -> int:
        return self.joint_shapedirs.shape[-1]

    def shaped(self, betas: np.ndarray) -> "SMPLXSkeleton":
        """
        The same model in another body shape. Shorter `betas` leave the remaining
        coefficients at zero; the original skeleton is unchanged.
        """
        other = copy.copy(self)
        other._set_shape(betas)
        return other

    def _set_shape(self, betas: Optional[np.ndarray]) -> None:
        coefficients = np.zeros(self.num_betas)
        if betas is not None:
            betas = np.asarray(betas, dtype=np.float64).ravel()
            if len(betas) > self.num_betas:
                raise ValueError(f"{len(betas)} betas given, but the shape space holds "
                                 f"{self.num_betas}")
            coefficients[: len(betas)] = betas

        self.betas = coefficients
        self.rest_joints = self.joint_template + self.joint_shapedirs @ coefficients
        self.rest_nose = self._nose_template + self._nose_shapedirs @ coefficients
