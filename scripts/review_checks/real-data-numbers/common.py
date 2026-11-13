"""Shared helpers for the real-data-numbers audit (read-only use of the snapshot)."""
import glob
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

MP = [11, 12, 13, 14, 15, 16]
NAMES = {11: "L shoulder", 12: "R shoulder", 13: "L elbow", 14: "R elbow", 15: "L wrist", 16: "R wrist"}
BONES = [(11, 13, "L upper arm"), (13, 15, "L forearm"), (12, 14, "R upper arm"), (14, 16, "R forearm")]
# SMPL-X joint for each MP landmark
SX = {m: s for s, m in ARM_CORRESPONDENCE}
VIS = 0.5


def load_clips():
    out = []
    for path in sorted(glob.glob("data/test_output/*_world.npz")):
        d = np.load(path)
        pose = d["landmarks"].astype(np.float64)
        ok = np.isfinite(pose).all(axis=-1) & (d["visibility"] >= VIS)
        both = ok[:, 11] & ok[:, 12]
        width = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
        out.append((Path(path).stem.replace("_world", ""), pose, ok, width))
    return out


def ik_points(frame, ok, skeleton, root=None):
    """Swing IK (optionally relative to a given root rotvec) -> FK -> {mp: xyz} in model frame."""
    masked = np.where(ok[:, None], frame, np.nan)
    rots = solve_swing_ik(masked, skeleton, SMPLX_ARM_BONES, global_orient=root)
    joints = forward_kinematics(rots, skeleton)
    return {m: joints[SX[m]] for m in MP}, rots


def align(src, tgt, mode):
    """mode: 'sim' (scale+R+t), 'rigid' (R+t), 'none'. Returns aligned src and params."""
    if mode == "none":
        return src.copy(), (1.0, np.eye(3), np.zeros(3))
    s, R, t = similarity_transform(src, tgt, allow_scale=(mode == "sim"))
    return apply_similarity(src, s, R, t), (s, R, t)


class EditableSkeleton:
    """Copy of a skeleton whose arm bone lengths can be overridden (descendants shifted)."""

    def __init__(self, base):
        self.base_rest = np.array(base.rest_joints, dtype=np.float64)
        self.parents = np.array(base.parents, dtype=int)
        self.rest_joints = self.base_rest.copy()
        self._children = {j: [] for j in range(len(self.parents))}
        for j, p in enumerate(self.parents):
            if p >= 0:
                self._children[p].append(j)

    @property
    def num_joints(self):
        return len(self.parents)

    def _descendants(self, j):
        stack, out = list(self._children[j]), []
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self._children[k])
        return out

    def set_lengths(self, lengths):
        """lengths: dict smplx_child_joint -> new length of bone (parent->child). None keeps rest."""
        rest = self.base_rest.copy()
        # process in chain order: shoulder->elbow before elbow->wrist
        for child in sorted(lengths):
            L = lengths[child]
            if L is None or not np.isfinite(L):
                continue
            p = self.parents[child]
            v = rest[child] - rest[p]
            new = rest[p] + v / np.linalg.norm(v) * L
            shift = new - rest[child]
            rest[child] = new
            for d in self._descendants(child):
                rest[d] = rest[d] + shift
        self.rest_joints = rest

    def set_shoulder_width(self, width):
        """Move both shoulder joints (16,17) symmetrically along their connecting line."""
        rest = self.rest_joints.copy()
        mid = (rest[16] + rest[17]) / 2
        half = (rest[16] - rest[17]) / 2
        new_half = half / np.linalg.norm(half) * width / 2
        for j, sign in ((16, 1), (17, -1)):
            new = mid + sign * new_half
            shift = new - rest[j]
            rest[j] = new
            for d in self._descendants(j):
                rest[d] = rest[d] + shift
        self.rest_joints = rest


# MP bone -> SMPL-X child joint whose rest offset is that bone
BONE_CHILD = {(11, 13): 18, (13, 15): 20, (12, 14): 19, (14, 16): 21}


def obs_lengths(frame, ok):
    out = {}
    for (a, b), child in BONE_CHILD.items():
        out[child] = np.linalg.norm(frame[b] - frame[a]) if (ok[a] and ok[b]) else None
    return out


def pct(v, q=50):
    v = np.asarray(v)
    return float(np.percentile(v, q) * 100) if len(v) else float("nan")


def angle_deg(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))))
