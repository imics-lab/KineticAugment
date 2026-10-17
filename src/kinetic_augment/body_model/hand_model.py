"""
MediaPipe's 21 hand landmarks evaluated on MANO or SMPL-X (V1.2).

MediaPipe reports the wrist, three joints per finger and the five fingertips.
Both models have the wrist and the fifteen finger joints as skeleton joints but
no fingertip joints, so the tips are read from fixed mesh vertices.
`HandModel.points` computes exactly those 21 points of the real model -- shape
blend shapes, pose correctives and linear blend skinning, through `smplx.lbs`
-- without skinning the rest of the mesh. It equals the full forward pass.

Conventions:
  * Hand pose is the full 45-value axis-angle pose of the fifteen finger joints
    with `flat_hand_mean=True`, so zero is the flat template hand. Slots follow
    the models' shared order -- index, middle, pinky, ring, thumb; proximal
    joint first -- and `POSE_SLOT_LANDMARKS` names the landmark at each joint.
    A joint's rotation orients the bone to its child landmark.
  * `wrist_orient` is the wrist joint's world rotation. In MANO that is the
    root rotation. In SMPL-X it is the wrist's local rotation with every other
    body joint at rest, so the two coincide; attaching the hand to a posed arm
    is a separate step.
  * Positions are metres in the model frame, before any translation.
"""

from types import SimpleNamespace

import numpy as np
import torch

NUM_HAND_LANDMARKS = 21
NUM_FINGER_JOINTS = 15
WRIST = 0
TIPS = (4, 8, 12, 16, 20)
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
LANDMARK_NAMES = (
    "wrist",
    "thumb CMC", "thumb MCP", "thumb IP", "thumb tip",
    "index MCP", "index PIP", "index DIP", "index tip",
    "middle MCP", "middle PIP", "middle DIP", "middle tip",
    "ring MCP", "ring PIP", "ring DIP", "ring tip",
    "pinky MCP", "pinky PIP", "pinky DIP", "pinky tip",
)

# Parent of each MediaPipe hand landmark; the wrist has none.
HAND_PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

# The landmark at the joint whose local rotation fills each 3-value slot of the
# 45-value hand pose (index, middle, pinky, ring, thumb; proximal first).
POSE_SLOT_LANDMARKS = (5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)

# Fingertip vertices in MediaPipe tip order: thumb, index, middle, ring, pinky.
MANO_TIP_VERTICES = (744, 320, 443, 554, 671)
SMPLX_TIP_VERTICES = {
    "left": (5361, 4933, 5058, 5169, 5286),
    "right": (8079, 7669, 7794, 7905, 8022),
}
# Each SMPL-X hand's wrist joint and first finger joint (index1).
SMPLX_WRIST_JOINT = {"left": 20, "right": 21}
SMPLX_FIRST_FINGER_JOINT = {"left": 25, "right": 40}


class HandModel:
    """The 21 MediaPipe hand points of one side of MANO or SMPL-X."""

    def __init__(self, backend: str, side: str, model_dir: str = "models", num_betas: int = 10,
                 dtype=torch.float64):
        import smplx

        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if backend == "mano":
            model = smplx.create(model_dir, model_type="mano", is_rhand=side == "right",
                                 use_pca=False, flat_hand_mean=True, num_betas=num_betas,
                                 ext="npz", dtype=dtype)
            wrist, first_finger, tips = 0, 1, MANO_TIP_VERTICES
        elif backend == "smplx":
            model = smplx.create(model_dir, model_type="smplx", gender="neutral",
                                 use_pca=False, flat_hand_mean=True, num_betas=num_betas,
                                 dtype=dtype)
            wrist = SMPLX_WRIST_JOINT[side]
            first_finger = SMPLX_FIRST_FINGER_JOINT[side]
            tips = SMPLX_TIP_VERTICES[side]
        else:
            raise ValueError(f"backend must be 'mano' or 'smplx', got {backend!r}")
        if torch.any(model.pose_mean != 0):
            raise ValueError("expected flat_hand_mean: zero pose must be the template hand")

        self.backend, self.side, self.dtype = backend, side, dtype
        self.num_betas = int(model.shapedirs.shape[-1])
        self._wrist = wrist
        self._fingers = slice(first_finger, first_finger + NUM_FINGER_JOINTS)
        self._parents = model.parents
        self._num_joints = int(model.J_regressor.shape[0])

        with torch.no_grad():
            regressor = model.J_regressor.to(dtype)
            self._joint_template = regressor @ model.v_template
            self._joint_shapedirs = torch.einsum("jv,vck->jck", regressor, model.shapedirs)
            tip_ids = torch.tensor(tips)
            self._tip_template = model.v_template[tip_ids]
            self._tip_shapedirs = model.shapedirs[tip_ids]
            columns = (3 * tip_ids[:, None] + torch.arange(3)).reshape(-1)
            self._tip_posedirs = model.posedirs[:, columns]
            self._tip_weights = model.lbs_weights[tip_ids]

        joints = {WRIST: wrist}
        for slot, landmark in enumerate(POSE_SLOT_LANDMARKS):
            joints[landmark] = first_finger + slot
        order = sorted(joints)
        self._joint_landmarks = torch.tensor(order)
        self._landmark_joints = torch.tensor([joints[i] for i in order])

    def points(self, wrist_orient, hand_pose, betas) -> torch.Tensor:
        """
        Model-frame positions of the 21 MediaPipe hand points.

        Args:
            wrist_orient: (B, 3) or (3,) axis-angle wrist rotation.
            hand_pose: (B, 45) or (45,) finger joint rotations, axis-angle.
            betas: (B, K) or (K,) shape coefficients, K = num_betas.

        Returns:
            (B, 21, 3) metres in MediaPipe order, untranslated. Differentiable.
        """
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        wrist_orient = torch.as_tensor(wrist_orient, dtype=self.dtype).reshape(-1, 3)
        hand_pose = torch.as_tensor(hand_pose, dtype=self.dtype).reshape(-1, 3 * NUM_FINGER_JOINTS)
        betas = torch.as_tensor(betas, dtype=self.dtype).reshape(-1, self.num_betas)
        batch = max(len(wrist_orient), len(hand_pose), len(betas))
        betas = betas.expand(batch, -1)

        pose = wrist_orient.new_zeros(batch, self._num_joints, 3)
        pose[:, self._wrist] = wrist_orient.expand(batch, -1)
        pose[:, self._fingers] = hand_pose.expand(batch, -1).reshape(batch, NUM_FINGER_JOINTS, 3)
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, -1, 3, 3)

        rest = self._joint_template + torch.einsum("jck,bk->bjc", self._joint_shapedirs, betas)
        joints, transforms = batch_rigid_transform(rotations, rest, self._parents,
                                                   dtype=self.dtype)

        corrective = (rotations[:, 1:] - torch.eye(3, dtype=self.dtype)).reshape(batch, -1)
        tips = (self._tip_template
                + torch.einsum("vck,bk->bvc", self._tip_shapedirs, betas)
                + (corrective @ self._tip_posedirs).reshape(batch, -1, 3))
        skin = torch.einsum("vj,bjrc->bvrc", self._tip_weights, transforms)
        tips = torch.einsum("bvrc,bvc->bvr", skin[..., :3, :3], tips) + skin[..., :3, 3]

        out = joints.new_zeros(batch, NUM_HAND_LANDMARKS, 3)
        out[:, self._joint_landmarks] = joints[:, self._landmark_joints]
        out[:, list(TIPS)] = tips
        return out

    def rest_points(self, betas=None) -> np.ndarray:
        """(21, 3) points of the flat template hand in shape `betas` (neutral by default)."""
        betas = np.zeros(self.num_betas) if betas is None else betas
        with torch.no_grad():
            return self.points(np.zeros(3), np.zeros(3 * NUM_FINGER_JOINTS), betas)[0].numpy()

    def skeleton(self, betas=None):
        """Rest skeleton over the 21 landmarks, in the form `solve_swing_ik` and FK take."""
        return SimpleNamespace(rest_joints=self.rest_points(betas), parents=HAND_PARENTS,
                               num_joints=NUM_HAND_LANDMARKS)
