"""
SMPL-X → MediaPipe back-projection.

Reads the bone vectors stored by mp_to_smplx and reconstructs MediaPipe
landmark positions by walking the kinematic chain:

    child_pos = parent_pos + bone_vector   (stored in params slot)

For undriven joints (bone vector == 0) the original position is kept.
For driven joints the stored bone vector replaces the original bone,
which means after augmentation the new rotated/scaled bone vector is
used automatically — no rotation math required here at all.

smplx_wrapper is accepted for API compatibility but never called.
"""

import warnings
from typing import Dict, Optional
import numpy as np

from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    MEDIAPIPE_POSE_LANDMARKS as MP,
)
from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    TOTAL_LANDMARKS,
    LH_START, LH_END,
    RH_START, RH_END,
)

# ---------------------------------------------------------------------------
# Kinematic chains — must match mp_to_smplx exactly
# ---------------------------------------------------------------------------

BODY_BONES = [
    (MP["left_shoulder"],  MP["left_elbow"],   "left_elbow"),
    (MP["left_elbow"],     MP["left_wrist"],   "left_wrist"),
    (MP["right_shoulder"], MP["right_elbow"],  "right_elbow"),
    (MP["right_elbow"],    MP["right_wrist"],  "right_wrist"),
    (MP["left_hip"],       MP["left_knee"],    "left_knee"),
    (MP["left_knee"],      MP["left_ankle"],   "left_ankle"),
    (MP["right_hip"],      MP["right_knee"],   "right_knee"),
    (MP["right_knee"],     MP["right_ankle"],  "right_ankle"),
]

HAND_CHAINS = [
    [(1,2),(2,3),(3,4)],
    [(5,6),(6,7),(7,8)],
    [(9,10),(10,11),(11,12)],
    [(13,14),(14,15),(15,16)],
    [(17,18),(18,19),(19,20)],
]


def _body_pose_slot(smplx_name: str) -> Optional[int]:
    idx = SMPLX_BODY_JOINTS.get(smplx_name)
    if idx is None or idx == 0:
        return None
    slot = idx - 1
    return slot if 0 <= slot < 21 else None


class SMPLXToMediaPipe:

    def __init__(self, smplx_wrapper=None):
        warnings.warn(
            "SMPLXToMediaPipe re-adds stored displacements to the original landmarks "
            "rather than evaluating SMPL-X, so it needs `target_mp` to reconstruct "
            "anything. For reconstruction from parameters alone use "
            "kinetic_augment.body_model.converter.reconstruct_body.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.smplx = smplx_wrapper  # unused, kept for API compatibility

    def project(
        self,
        smplx_params: Dict[str, np.ndarray],
        target_mp: np.ndarray,
        return_full_frame: bool = True,
    ) -> np.ndarray:
        """
        Args
            smplx_params    : dict with 'body_pose' (B,63) and optionally
                              'left_hand_pose'/'right_hand_pose' (B,45)
            target_mp       : (B, TOTAL_LANDMARKS, 3) original MediaPipe frame
            return_full_frame: True  → (B, TOTAL_LANDMARKS, 3)
                               False → (B, NUM_POSE_LANDMARKS, 3)
        """
        body_pose       = smplx_params["body_pose"]
        left_hand_pose  = smplx_params.get("left_hand_pose")
        right_hand_pose = smplx_params.get("right_hand_pose")
        B = body_pose.shape[0]

        if return_full_frame:
            out = np.zeros((B, TOTAL_LANDMARKS, 3), dtype=np.float32)
        else:
            out = np.zeros((B, NUM_POSE_LANDMARKS, 3), dtype=np.float32)

        for b in range(B):
            orig = target_mp[b]

            # Pose
            out[b, :NUM_POSE_LANDMARKS] = self._reconstruct_body(
                orig[:NUM_POSE_LANDMARKS], body_pose[b]
            )

            if return_full_frame:
                # Face — copy unchanged
                fs, fe = NUM_POSE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
                out[b, fs:fe] = orig[fs:fe]

                # Hands
                lh_orig = orig[LH_START:LH_END]
                rh_orig = orig[RH_START:RH_END]

                if left_hand_pose is not None and np.any(lh_orig != 0):
                    out[b, LH_START:LH_END] = self._reconstruct_hand(
                        lh_orig, left_hand_pose[b]
                    )
                else:
                    out[b, LH_START:LH_END] = lh_orig

                if right_hand_pose is not None and np.any(rh_orig != 0):
                    out[b, RH_START:RH_END] = self._reconstruct_hand(
                        rh_orig, right_hand_pose[b]
                    )
                else:
                    out[b, RH_START:RH_END] = rh_orig

        return out

    # ------------------------------------------------------------------

    def _reconstruct_body(
        self, original_pose: np.ndarray, body_pose: np.ndarray
    ) -> np.ndarray:
        """
        Walk the kinematic chain.  For each bone:
          - If the stored bone vector is non-zero, use it.
          - Otherwise keep the original child position.
        Parent positions are updated first so child bones attach correctly.
        """
        out = original_pose.copy()

        for parent_idx, child_idx, smplx_name in BODY_BONES:
            slot = _body_pose_slot(smplx_name)
            if slot is None:
                continue

            bone_vec = body_pose[slot*3 : slot*3+3]

            if np.linalg.norm(bone_vec) < 1e-8:
                # Undriven — preserve relative to (possibly updated) parent
                orig_bone   = original_pose[child_idx] - original_pose[parent_idx]
                out[child_idx] = out[parent_idx] + orig_bone
            else:
                # Driven — attach stored bone vector to current parent
                out[child_idx] = out[parent_idx] + bone_vec

        return out

    def _reconstruct_hand(
        self, original_hand: np.ndarray, hand_pose: np.ndarray
    ) -> np.ndarray:
        out  = original_hand.copy()
        slot = 0
        for finger in HAND_CHAINS:
            for (p_idx, c_idx) in finger:
                if slot >= 15:
                    break
                bone_vec = hand_pose[slot*3 : slot*3+3]
                if np.linalg.norm(bone_vec) < 1e-8:
                    orig_bone      = original_hand[c_idx] - original_hand[p_idx]
                    out[c_idx]     = out[p_idx] + orig_bone
                else:
                    out[c_idx] = out[p_idx] + bone_vec
                slot += 1
        return out