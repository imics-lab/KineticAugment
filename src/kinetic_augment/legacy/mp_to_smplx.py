"""
MediaPipe → SMPL-X parameter encoding.

Encoding strategy
-----------------
Rather than computing a rotation *relative to a rest direction* (which causes
the back-projector to mis-apply the rotation when the original bone isn't
aligned with that rest direction), we store the observed bone direction
directly inside the 3-D axis-angle slots.

For each driven bone we store:
    body_pose[slot*3 : slot*3+3] = normalize(child - parent) * bone_length_scale

where bone_length_scale = ||child - parent|| (the original bone length in MP
screen units).  This means the axis-angle vector encodes both direction and
length of the bone in one compact 3-D vector — no rotation math needed.

The back-projector (smplx_to_mp) reads this back as:
    new_child = new_parent + body_pose[slot*3:slot*3+3]
                            (= direction * length = the bone vector directly)

This is a bespoke encoding that lives inside the SMPL-X parameter format for
the purpose of this pipeline.  It is NOT standard SMPL-X axis-angle, so you
cannot feed these params directly into an SMPL-X model and expect meaningful
output — but for the MP→SMPL-X→(augment)→MP round-trip it is exact.

Augmentation code that wants to modify poses should operate on the stored
bone vectors directly (rotate them, scale them, etc.) and the back-projector
will faithfully reconstruct the new landmark positions.
"""

import warnings
from typing import Dict
import numpy as np

from kinetic_augment.body_model.joint_mapping import (
    MEDIAPIPE_POSE_LANDMARKS,
    SMPLX_BODY_JOINTS,
)
from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    LH_START, LH_END,
    RH_START, RH_END,
)


# ---------------------------------------------------------------------------
# Bone definitions
# (parent_mp_name, child_mp_name, child_smplx_name)
# Order: parents before children so FK propagates correctly.
# ---------------------------------------------------------------------------
BODY_BONES = [
    ("left_shoulder",  "left_elbow",   "left_elbow"),
    ("left_elbow",     "left_wrist",   "left_wrist"),
    ("right_shoulder", "right_elbow",  "right_elbow"),
    ("right_elbow",    "right_wrist",  "right_wrist"),
    ("left_hip",       "left_knee",    "left_knee"),
    ("left_knee",      "left_ankle",   "left_ankle"),
    ("right_hip",      "right_knee",   "right_knee"),
    ("right_knee",     "right_ankle",  "right_ankle"),
]

# Hand finger chains: list of (parent_idx, child_idx) in MP hand space
HAND_CHAINS = [
    [(1,2),(2,3),(3,4)],      # thumb
    [(5,6),(6,7),(7,8)],      # index
    [(9,10),(10,11),(11,12)], # middle
    [(13,14),(14,15),(15,16)],# ring
    [(17,18),(18,19),(19,20)],# pinky
]


def _body_pose_slot(smplx_name: str) -> int | None:
    idx = SMPLX_BODY_JOINTS.get(smplx_name)
    if idx is None or idx == 0:
        return None
    slot = idx - 1
    return slot if 0 <= slot < 21 else None


class MediaPipeToSMPLX:
    """
    Encodes MediaPipe landmarks into SMPL-X parameter arrays.

    The body_pose and hand_pose arrays store raw bone vectors
    (direction × length) rather than axis-angles.  See module docstring.
    """

    def __init__(self):
        warnings.warn(
            "MediaPipeToSMPLX stores bone displacements in SMPL-X-shaped slots; its "
            "output is not valid SMPL-X input. For genuine SMPL-X parameters use "
            "kinetic_augment.body_model.converter.fit_body on world landmarks.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mp    = MEDIAPIPE_POSE_LANDMARKS
        self.smplx = SMPLX_BODY_JOINTS

    def fit(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis]

        B = landmarks.shape[0]
        pose       = landmarks[:, :NUM_POSE_LANDMARKS]
        left_hand  = landmarks[:, LH_START:LH_END]
        right_hand = landmarks[:, RH_START:RH_END]

        body_pose       = np.zeros((B, 63), dtype=np.float32)
        global_orient   = np.zeros((B, 3),  dtype=np.float32)
        left_hand_pose  = np.zeros((B, 45), dtype=np.float32)
        right_hand_pose = np.zeros((B, 45), dtype=np.float32)

        for b in range(B):
            body_pose[b] = self._encode_body(pose[b])
            if np.any(left_hand[b] != 0):
                left_hand_pose[b]  = self._encode_hand(left_hand[b])
            if np.any(right_hand[b] != 0):
                right_hand_pose[b] = self._encode_hand(right_hand[b])

        return {
            "body_pose":       body_pose,
            "global_orient":   global_orient,
            "betas":           np.zeros((B, 10), dtype=np.float32),
            "left_hand_pose":  left_hand_pose,
            "right_hand_pose": right_hand_pose,
        }

    def _encode_body(self, L: np.ndarray) -> np.ndarray:
        """Store each bone vector (child - parent) in the corresponding slot."""
        pose = np.zeros(63, dtype=np.float32)
        for parent_mp, child_mp, child_smplx in BODY_BONES:
            p = L[self.mp[parent_mp]]
            c = L[self.mp[child_mp]]
            slot = _body_pose_slot(child_smplx)
            if slot is not None:
                # Store the raw bone vector: direction + length in one vec3
                pose[slot*3 : slot*3+3] = c - p
        return pose

    def _encode_hand(self, H: np.ndarray) -> np.ndarray:
        """Store each finger-segment bone vector in the corresponding slot."""
        pose = np.zeros(45, dtype=np.float32)
        slot = 0
        for finger in HAND_CHAINS:
            for (p_idx, c_idx) in finger:
                if slot >= 15:
                    break
                pose[slot*3 : slot*3+3] = H[c_idx] - H[p_idx]
                slot += 1
        return pose