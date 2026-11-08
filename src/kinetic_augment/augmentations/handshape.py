"""
Handshape features and finger-offset helpers for the week-4 finger augmentations.

Layout. The SMPL-X left hand is joints 25-39 and the right hand 40-54, hung from the wrists
(20 and 21). Offsets within a hand: index 0-2, middle 3-5, pinky 6-8, ring 9-11, thumb 12-14, each
finger in MCP -> PIP -> DIP order (the thumb is CMC -> MCP -> IP). Fingertips are not joints.

Features (`handshape_features`). Per frame the hand is posed by forward kinematics inside the hand
(the wrist is the origin; angles between bones do not depend on the wrist's world rotation). For a
finger with joints J0, J1, J2 the bones are b0 = J0 - wrist (the metacarpal), b1 = J1 - J0 (the
proximal bone) and b2 = J2 - J1 (the middle bone). The bend sums the two inter-bone angles of the
chain wrist -> MCP -> PIP -> DIP: angle(b0, b1) (the metacarpal against the MCP -> PIP bone) and
angle(b1, b2). Each angle is signed and measured about the hinge: the turn from one bone to the
next projected on the flexion axis of the joint between them (the MCP's axis in the wrist frame,
the PIP's axis in the MCP's world frame), positive into the palm. This is a deliberate change from
a plain unsigned angle: the SMPL-X rest hand is slightly hyperextended at the MCP, so an unsigned
sum barely moves for a 20 degree flexion (21.5 -> 22.6 degrees) and the rest thumb reads 55
degrees; the signed sum is about 0 at rest and rises by the flexion. (The DIP joint's own rotation
turns a bone that ends at the fingertip, which is not a joint, so it does not enter the bend.)
The spread is the angle between the index and pinky proximal bones (b1); the thumb abduction is
the angle between the thumb proximal bone (b1) and the index metacarpal (b0).

Classes. A finger is extended when its bend is below `EXTENDED_MAX_DEG` (40), flexed from
`FLEXED_MIN_DEG` (110) up, and curved in between (states 0, 1, 2). The spread is wide from
`SPREAD_THRESHOLD_DEG` (25) up and the thumb is abducted from `THUMB_THRESHOLD_DEG` (45) up.
Thresholds are the brief's values; they were not tuned (see the task report for the check on the
cached clips).

Offset convention (`apply_finger_offsets`). Each finger joint turns in its own right-handed basis
(a, b, n) at rest: a = the flexion axis (`flexion_axes`), b = the bone direction, n = the palm
normal made orthogonal to both. A joint's offsets (flexion, twist about b, abduction about n),
in degrees, are applied in the joint's parent frame as `R_new = R(axis, angle) @ R_old`, so the
rest parent frame (all rotations zero) is the world frame. The flexion axis is a pseudovector, so
the left and right hands share the sign of flexion; twist and abduction axes are true vectors, so
they are negated on the right hand. The same offsets therefore move both hands as mirror images.
The two SMPL-X hands are mirror images up to about 1 degree of rest asymmetry.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
# Offsets (into a hand's 15 joints) of each finger's first, second and third joint.
FINGER_SLOTS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (12, 13, 14),
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "ring": (9, 10, 11),
    "pinky": (6, 7, 8),
}
HAND_START = {"left": 25, "right": 40}
WRIST = {"left": 20, "right": 21}
INDEX = FINGERS.index("index")

EXTENDED_MAX_DEG = 40.0
FLEXED_MIN_DEG = 110.0
SPREAD_THRESHOLD_DEG = 25.0
THUMB_THRESHOLD_DEG = 45.0
_SIDE_SIGN = {"left": 1.0, "right": -1.0}  # sign of the palm-normal cross product


@dataclass
class HandshapeFeatures:
    """Per-frame handshape features of one hand. Frames that are not finite hold NaN / -1."""

    bend_deg: np.ndarray  # (T, 5) in FINGERS order
    state: np.ndarray  # (T, 5) int: 0 extended, 1 curved, 2 flexed, -1 not finite
    spread_deg: np.ndarray  # (T,) index-pinky proximal bone angle
    thumb_abduction_deg: np.ndarray  # (T,)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(u * v, axis=-1)))


def _signed_angle_deg(u: np.ndarray, v: np.ndarray, hinge: np.ndarray) -> np.ndarray:
    """Angle from u to v about the unit `hinge` axis (only the part of the turn about it)."""
    return np.degrees(np.arctan2(np.sum(np.cross(u, v) * hinge, axis=-1), np.sum(u * v, axis=-1)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _hand_pose(rotations, rest_joints, parents, side: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hand joint positions (T, 16, 3) relative to the wrist (index 0 is the wrist itself) and the
    joints' world rotations (T, 15, 3, 3), with the wrist's own rotation taken as identity.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    rest = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    start, wrist = HAND_START[side], WRIST[side]
    frames = len(rotations)
    local = Rotation.from_rotvec(rotations[:, start : start + 15].reshape(-1, 3))
    local = local.as_matrix().reshape(frames, 15, 3, 3)
    world = np.zeros((frames, 15, 3, 3))
    pos = np.zeros((frames, 16, 3))
    for i in range(15):
        j, p = start + i, int(parents[start + i])
        if p == wrist:
            world[:, i] = local[:, i]
            pos[:, i + 1] = rest[j] - rest[wrist]
        else:
            k = p - start
            world[:, i] = world[:, k] @ local[:, i]
            pos[:, i + 1] = pos[:, k + 1] + world[:, k] @ (rest[j] - rest[p])
    return pos, world


def _classify(bend: np.ndarray) -> np.ndarray:
    state = np.where(bend >= FLEXED_MIN_DEG, 2, np.where(bend < EXTENDED_MAX_DEG, 0, 1))
    return np.where(np.isfinite(bend), state, -1).astype(int)


def handshape_features(rotations, rest_joints, parents, side: str) -> HandshapeFeatures:
    """Bend, state, spread and thumb abduction of one hand over frames (see the module docstring)."""
    pos, world = _hand_pose(rotations, rest_joints, parents, side)
    axes = flexion_axes(rest_joints, side)
    bends, bones = [], {}
    for name in FINGERS:
        s0, s1, s2 = FINGER_SLOTS[name]
        w, j0, j1, j2 = pos[:, 0], pos[:, 1 + s0], pos[:, 1 + s1], pos[:, 1 + s2]
        b0, b1, b2 = j0 - w, j1 - j0, j2 - j1
        bones[name] = (b0, b1)
        hinge0 = np.broadcast_to(axes[s0], b0.shape)  # the MCP turns in the wrist frame
        hinge1 = world[:, s0] @ axes[s1]  # the PIP turns in the MCP's world frame
        bends.append(_signed_angle_deg(b0, b1, hinge0) + _signed_angle_deg(b1, b2, hinge1))
    bend = np.stack(bends, axis=1)
    spread = _angle_deg(bones["index"][1], bones["pinky"][1])
    thumb = _angle_deg(bones["thumb"][1], bones["index"][0])
    return HandshapeFeatures(bend, _classify(bend), spread, thumb)


def _median_state(state: np.ndarray) -> np.ndarray:
    """Lower median of the valid frames' states per finger (all -1 if no frame is valid)."""
    out = np.full(state.shape[1], -1, dtype=int)
    for f in range(state.shape[1]):
        valid = np.sort(state[state[:, f] >= 0, f])
        if len(valid):
            out[f] = valid[(len(valid) - 1) // 2]
    return out


def selected_fingers(features: HandshapeFeatures) -> Tuple[bool, ...]:
    """Fingers whose median state is extended or curved; the index finger if there are none."""
    median = _median_state(features.state)
    chosen = [bool(0 <= m <= 1) for m in median]
    if not any(chosen):
        chosen[INDEX] = True
    return tuple(chosen)


def feature_signature(features: HandshapeFeatures) -> Tuple:
    """Median finger states, spread class and thumb class: what "matching hands" compares."""
    spread, thumb = features.spread_deg, features.thumb_abduction_deg
    spread, thumb = spread[np.isfinite(spread)], thumb[np.isfinite(thumb)]
    wide = bool(len(spread) and np.median(spread) >= SPREAD_THRESHOLD_DEG)
    abducted = bool(len(thumb) and np.median(thumb) >= THUMB_THRESHOLD_DEG)
    return (tuple(int(m) for m in _median_state(features.state)), wide, abducted)


def _crossed(before, after, threshold: float, margin: float) -> bool:
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    ok = np.isfinite(before) & np.isfinite(after) & (np.abs(before - threshold) >= margin)
    flipped = (before >= threshold) != (after >= threshold)
    return bool(np.any(ok & flipped & (np.abs(after - threshold) > margin)))


def features_changed(
    before: HandshapeFeatures, after: HandshapeFeatures, margin_deg: float = 5.0
) -> bool:
    """
    Whether `after` changes the finger states, spread class or thumb class of `before`.

    Each threshold is tested on its own: a frame is skipped when `before` lies within `margin_deg`
    of the threshold, and a change counts only when `after` is on the other side by more than
    `margin_deg`. Frames that are not finite in either set are ignored.
    """
    for t in (EXTENDED_MAX_DEG, FLEXED_MIN_DEG):
        if _crossed(before.bend_deg, after.bend_deg, t, margin_deg):
            return True
    if _crossed(before.spread_deg, after.spread_deg, SPREAD_THRESHOLD_DEG, margin_deg):
        return True
    return _crossed(
        before.thumb_abduction_deg, after.thumb_abduction_deg, THUMB_THRESHOLD_DEG, margin_deg
    )


def _joint_bones(rest_joints, side: str) -> np.ndarray:
    """The bone each of a hand's 15 joints turns (its outgoing bone; the last joint's incoming)."""
    rest = np.asarray(rest_joints, dtype=np.float64)
    start = HAND_START[side]
    bones = np.zeros((15, 3))
    for name in FINGERS:
        s0, s1, s2 = FINGER_SLOTS[name]
        p0, p1, p2 = rest[start + s0], rest[start + s1], rest[start + s2]
        bones[s0], bones[s1], bones[s2] = p1 - p0, p2 - p1, p2 - p1
    return bones


def _palm_normal(rest_joints, side: str) -> np.ndarray:
    """Unit palm normal, out of the palm, from the wrist, index MCP and pinky MCP."""
    rest = np.asarray(rest_joints, dtype=np.float64)
    start = HAND_START[side]
    to_index = rest[start + FINGER_SLOTS["index"][0]] - rest[WRIST[side]]
    to_pinky = rest[start + FINGER_SLOTS["pinky"][0]] - rest[WRIST[side]]
    normal = np.cross(to_pinky, to_index) * _SIDE_SIGN[side]
    return normal / np.linalg.norm(normal)


def flexion_axes(rest_joints, side: str) -> np.ndarray:
    """
    Flexion axis (15, 3) of every hand joint: unit(bone x palm normal) in the parent frame at rest.

    The palm normal points out of the palm (it comes from the wrist, index MCP and pinky MCP, with
    its sign set so a positive turn about the axis curls the finger into the palm). A positive
    rotation about the returned axis therefore flexes the joint on both hands.
    """
    return finger_bases(rest_joints, side)[:, 0]


def finger_bases(rest_joints, side: str) -> np.ndarray:
    """Per-joint orthonormal bases (15, 3, 3), rows (flexion axis a, bone b, normal n = a x b)."""
    bones = _joint_bones(rest_joints, side)
    normal = _palm_normal(rest_joints, side)
    a = _unit(np.cross(bones, normal))
    b = _unit(bones)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def apply_finger_offsets(rotations, rest_joints, side: str, offsets_deg, frames=None) -> np.ndarray:
    """
    A copy of `rotations` (T, 55, 3) with per-joint offsets turned into one hand.

    `offsets_deg` is (15, 3): per hand joint the flexion, twist and abduction angles in degrees
    (see the module docstring for the basis, the mirroring and `R_new = R(axis, angle) @ R_old`).
    `frames` (bool mask or indices, default all) selects the frames to change; frames whose hand
    rotations are not finite are always left alone, as are joints with an all-zero offset.
    """
    out = np.array(rotations, dtype=np.float64, copy=True)
    offsets = np.asarray(offsets_deg, dtype=np.float64)
    joints = np.flatnonzero(np.any(offsets != 0.0, axis=1))
    if not len(joints):
        return out
    start = HAND_START[side]
    sel = np.arange(len(out)) if frames is None else np.arange(len(out))[np.asarray(frames)]
    block = out[sel][:, start : start + 15]
    sel = sel[np.isfinite(block).all(axis=(1, 2))]
    if not len(sel):
        return out
    basis = finger_bases(rest_joints, side)[joints]
    sign = _SIDE_SIGN[side]
    o = np.radians(offsets[joints])
    rotvec = o[:, 0:1] * basis[:, 0] + sign * (o[:, 1:2] * basis[:, 1] + o[:, 2:3] * basis[:, 2])
    old = out[sel][:, start + joints]  # (F, J, 3)
    shift = Rotation.from_rotvec(np.tile(rotvec, (len(sel), 1)))
    new = (shift * Rotation.from_rotvec(old.reshape(-1, 3))).as_rotvec()
    sel_idx = sel[:, None]
    out[sel_idx, (start + joints)[None, :]] = new.reshape(len(sel), len(joints), 3)
    return out
