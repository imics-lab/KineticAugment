"""
Week-4 body augmentations on SMPL-X arm joints.

`ArmCoupledNoise` draws one small rotation per arm joint (shoulder, elbow, wrist, both sides) for
the clip. Joints along a chain share a common component (`chain_coupling`), and the right side is
correlated with the mirrored left (`side_coupling`). `Tier1ArmJoints` applies the Tier 1 arm-joint
draw that the chain hands over in `ChainContext.tier1_draw`. `LimbLengthScaling` rescales the arm
bones in the rest skeleton and, optionally, re-aims each frame's arm so the wrists stay put.

Both store their draw as plain data: `{"joints": [ArmJointDraw as dict, ...], "drift": ...}`.
The draws apply to every frame whose body was fitted; unfitted (NaN) frames are left alone.
"""

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, freqz

from .arm_ik import two_bone_ik, world_pose
from .base import AugmentationRegistry, ClipAugmentation
from .clip import ChainContext, ClipData
from .ranges import parse_range, truncated_normal
from .tier1 import ARM_JOINTS, SIDES, ArmJointDraw, perturb_arm_joints

SIDE_MODES = ("mirror", "synchronous", "independent")
_MIRROR = np.array([1.0, -1.0, -1.0])
_ELBOW_AXIS = (0.0, 0.0, 1.0)  # ignored by the elbow, which bends in its own plane


def _clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v * (limit / norm) if norm > limit else v


def _draw_dict(side: str, joint: str, vector: Optional[np.ndarray], elbow: float = 0.0) -> dict:
    if joint == "elbow":
        return dataclasses.asdict(ArmJointDraw(side, joint, float(elbow), _ELBOW_AXIS))
    norm = float(np.linalg.norm(vector))
    axis = (1.0, 0.0, 0.0) if norm == 0.0 else tuple(float(a) for a in vector / norm)
    return dataclasses.asdict(ArmJointDraw(side, joint, norm, axis))


def _apply_joints(clip: ClipData, per_frame_joints, ctx: ChainContext) -> ClipData:
    """Apply `per_frame_joints(t)` (a list of ArmJointDraw) to every fitted frame."""
    rotations = np.array(clip.rotations, dtype=np.float64, copy=True)
    parents = np.asarray(ctx.parents, dtype=int)
    for t in range(rotations.shape[0]):
        if not clip.body_ok[t] or not np.isfinite(rotations[t]).all():
            continue
        joints = per_frame_joints(t)
        if joints:
            rotations[t] = perturb_arm_joints(rotations[t], clip.rest_joints, parents, joints)
    return clip.replace(rotations=rotations)


@AugmentationRegistry.register("body.joint_coupled_noise")
class ArmCoupledNoise(ClipAugmentation):
    """Coupled, constant-over-the-clip rotation noise on both arms' joints."""

    key = "joint_coupled_noise"
    regions = ("body",)
    DEFAULT_RANGES = {
        "sigma_deg": 2.5,
        "truncate_deg": 5.0,
        "elbow_sigma_deg": 4.75,
        "elbow_truncate_deg": 9.5,
        "chain_coupling": 0.5,
        "side_coupling": 0.7,
        "side_mode": "mirror",
        "drift_deg": 0.0,
    }

    def _side(self, rng: np.random.Generator):
        """One side's shoulder vector, elbow scalar and wrist vector, before side coupling."""
        r = self.ranges
        sigma, sigma_e = float(r["sigma_deg"]), float(r["elbow_sigma_deg"])
        c = float(r["chain_coupling"])
        common = rng.normal(0.0, sigma, size=3)
        shoulder = np.sqrt(c) * common + np.sqrt(1 - c) * rng.normal(0.0, sigma, size=3)
        wrist = np.sqrt(c) * common + np.sqrt(1 - c) * rng.normal(0.0, sigma, size=3)
        scale = sigma_e / sigma if sigma > 0 else 0.0
        elbow = np.sqrt(c) * common[0] * scale + np.sqrt(1 - c) * rng.normal(0.0, sigma_e)
        return shoulder, float(elbow), wrist

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: Any) -> Dict[str, Any]:
        r = self.ranges
        mode = r["side_mode"]
        if mode not in SIDE_MODES:
            raise ValueError(f"side_mode must be one of {SIDE_MODES}, got {mode!r}")
        k = float(r["side_coupling"]) if mode == "mirror" else 0.0
        left = self._side(rng)
        indep = self._side(rng)
        if mode == "synchronous":
            right = left
        else:
            right = (
                k * left[0] * _MIRROR + np.sqrt(1 - k**2) * indep[0],
                k * left[1] + np.sqrt(1 - k**2) * indep[1],
                k * left[2] * _MIRROR + np.sqrt(1 - k**2) * indep[2],
            )
        limit, limit_e = float(r["truncate_deg"]), float(r["elbow_truncate_deg"])
        joints: List[dict] = []
        for side, (shoulder, elbow, wrist) in zip(SIDES, (left, right)):
            joints.append(_draw_dict(side, "shoulder", _clamp_norm(shoulder, limit)))
            joints.append(_draw_dict(side, "elbow", None, float(np.clip(elbow, -limit_e, limit_e))))
            joints.append(_draw_dict(side, "wrist", _clamp_norm(wrist, limit)))
        return {"joints": joints, "drift": self._drift(rng, clip)}

    def _drift(self, rng: np.random.Generator, clip: ClipData) -> Optional[np.ndarray]:
        std = float(self.ranges["drift_deg"])
        if std <= 0:
            return None
        frames = clip.rotations.shape[0]
        b, a = butter(2, 1.0, fs=clip.fps)
        # Filter extra noise on both sides and keep the middle: filtfilt's edge padding would
        # otherwise leave the clip's first and last frames far larger than the rest.
        pad = int(np.ceil(3.0 * clip.fps))
        noise = rng.normal(size=(frames + 2 * pad, 2 * len(ARM_JOINTS)))
        # Scale by the filter's noise gain (|H|^4 averaged over frequency), so the std is exact.
        _, h = freqz(b, a, worN=4096)
        gain = float(np.sqrt(np.mean(np.abs(h) ** 4)))
        return filtfilt(b, a, noise, axis=0)[pad : pad + frames] / gain * std

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        base = [ArmJointDraw(**{**j, "axis": tuple(j["axis"])}) for j in draw["joints"]]
        drift = draw.get("drift")
        if drift is None:
            return _apply_joints(target, lambda t: base, ctx)
        drift = np.asarray(drift, dtype=np.float64)

        def joints(t: int):
            return [
                dataclasses.replace(j, angle_deg=j.angle_deg + float(drift[t, i]))
                for i, j in enumerate(base)
            ]

        return _apply_joints(target, joints, ctx)

    def identity_draw(self) -> Dict[str, Any]:
        return {"joints": [], "drift": None}


@AugmentationRegistry.register("body.tier1_arm_joints")
class Tier1ArmJoints(ClipAugmentation):
    """Apply the Tier 1 arm-joint draw carried in `ChainContext.tier1_draw` to every frame."""

    key = "tier1_arm_joints"
    regions = ("body",)
    DEFAULT_RANGES = {"p": 0.3, "shoulder_wrist_deg": 5.0, "elbow_deg": 9.5}

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: ChainContext):
        if ctx.tier1_draw is None:
            raise ValueError("Tier1ArmJoints needs ctx.tier1_draw, but it is None")
        return {"joints": [dataclasses.asdict(j) for j in ctx.tier1_draw.arm_joints], "drift": None}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        joints = [ArmJointDraw(**{**j, "axis": tuple(j["axis"])}) for j in draw["joints"]]
        return _apply_joints(target, lambda t: joints, ctx)

    def identity_draw(self) -> Dict[str, Any]:
        return {"joints": [], "drift": None}


# Bone name -> (SMPL-X joint per side). The hand scales its fingers about the wrist instead.
_BONE_JOINTS = {"upper_arm": {"left": 18, "right": 19}, "forearm": {"left": 20, "right": 21}}
_WRISTS = {"left": 20, "right": 21}
_BONES = ("upper_arm", "forearm", "hand")


def _descendants(parents: np.ndarray, joint: int) -> List[int]:
    """`joint` and every joint below it (parents index lower joints, so one pass suffices)."""
    found = {int(joint)}
    for j in range(len(parents)):
        if int(parents[j]) in found:
            found.add(j)
    return sorted(found)


def scale_bone(rest_joints, parents, joint: int, factor: float) -> np.ndarray:
    """
    Rest joints (55, 3) with the bone ending at `joint` scaled by `factor`.

    `joint` and all its descendants move by (factor - 1) * (rest[joint] - rest[parent[joint]]),
    so the bone's length is multiplied by `factor` and everything below it keeps its shape.
    """
    rest = np.array(rest_joints, dtype=np.float64, copy=True)
    parents = np.asarray(parents, dtype=int)
    delta = (factor - 1.0) * (rest[joint] - rest[parents[joint]])
    rest[_descendants(parents, joint)] += delta
    return rest


def _scale_hand(rest: np.ndarray, parents: np.ndarray, wrist: int, factor: float) -> np.ndarray:
    out = np.array(rest, dtype=np.float64, copy=True)
    fingers = [j for j in _descendants(parents, wrist) if j != wrist]
    out[fingers] = out[wrist] + factor * (out[fingers] - out[wrist])
    return out


@AugmentationRegistry.register("body.limb_length_scaling")
class LimbLengthScaling(ClipAugmentation):
    """Scale arm bone lengths in the rest skeleton, keeping the wrists in place if `retarget`."""

    key = "limb_length_scaling"
    regions = ("body",)
    DEFAULT_RANGES = {
        "factor": [0.93, 1.07],
        "bones": ["upper_arm", "forearm"],
        "retarget": True,
    }

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: Any) -> Dict[str, Any]:
        bones = list(self.ranges["bones"])
        for bone in bones:
            if bone not in _BONES:
                raise ValueError(f"bones must be drawn from {_BONES}, got {bone!r}")
        spec = parse_range(self.ranges["factor"], self.DISTRIBUTIONS.get("factor", "uniform"))
        return {"factors": {bone: float(spec.draw(rng)) for bone in bones}}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        factors = {b: float(f) for b, f in draw["factors"].items()}
        for bone in factors:
            if bone not in _BONES:
                raise ValueError(f"bones must be drawn from {_BONES}, got {bone!r}")
        if all(f == 1.0 for f in factors.values()):
            return target
        parents = np.asarray(ctx.parents, dtype=int)
        original = np.asarray(target.rest_joints, dtype=np.float64)
        rest = original
        for bone in _BONES:  # proximal to distal, so the order is fixed whatever the dict's
            if bone not in factors or factors[bone] == 1.0:
                continue
            if bone == "hand":
                for wrist in _WRISTS.values():
                    rest = _scale_hand(rest, parents, wrist, factors[bone])
            else:
                for joint in _BONE_JOINTS[bone].values():
                    rest = scale_bone(rest, parents, joint, factors[bone])
        if not self.ranges["retarget"]:
            return target.replace(rest_joints=rest)

        rotations = np.array(target.rotations, dtype=np.float64, copy=True)
        residual = 0.0
        for t in range(rotations.shape[0]):
            if not target.body_ok[t] or not np.isfinite(rotations[t]).all():
                continue
            _, pos = world_pose(rotations[t], original, parents)
            frame = rotations[t]
            for side, wrist in _WRISTS.items():
                frame, r = two_bone_ik(frame, rest, parents, side, pos[wrist])
                residual = max(residual, r)
            rotations[t] = frame
        draw["residual_m"] = float(residual)
        return target.replace(rest_joints=rest, rotations=rotations)

    def identity_draw(self) -> Dict[str, Any]:
        return {"factors": {}}
