"""
From a `ClipData` to MediaPipe-layout landmarks: pose arms, 21-point hands, 478 face points.

The body is posed by `pose_mesh_rest`, a copy of SMPL-X's linear blend skinning
(`smplx.lbs.lbs`) that takes the rest joints as an input instead of regressing them from the
shaped template. `assembly.pose_mesh` runs the stock SMPL-X forward, which re-regresses the
joints from `betas`, so an edited bone length in `rest_joints` would never reach the mesh;
here it does. With `rest_joints = skeleton.shaped(betas).rest_joints` the two agree.

Moving a rest joint alone would leave the template mesh where it was: the skinning maps each
vertex relative to its joints' rest positions, so a lengthened forearm would carry the wrist
and finger joints away from the hand's own vertices (the fingertips). The rest mesh is
therefore moved first by the skinning-weighted rest-joint edit, `W @ (rest_joints - regressed)`:
a hand vertex moves with its (moved) hand joints, a forearm vertex partly, so the forearm
stretches and the hand stays on the wrist. The offset is zero when nothing was edited.

The face is the observed face carried in the head joint's frame, put back on the posed head.
Then the clip's scene transform (and, when mirrored, the left/right relabelling) is applied to
every point set, and the points are projected through the clip's weak camera if it has one.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from ..body_model.assembly import HEAD, PosedMesh, global_rotations
from ..body_model.face_stream import from_head_frame
from ..body_model.hand_model import (
    POSE_SLOT_LANDMARKS,
    SMPLX_FIRST_FINGER_JOINT,
    SMPLX_TIP_VERTICES,
    SMPLX_WRIST_JOINT,
    TIPS,
)
from .clip import ClipData
from .tier1 import POSE_MIRROR

NUM_JOINTS = 55
SIDES = ("left", "right")
POSE_ARMS = slice(11, 17)  # MediaPipe pose: L/R shoulder, elbow, wrist
SMPLX_ARMS = slice(16, 22)  # SMPL-X: the same six joints


@dataclass(frozen=True, eq=False)
class ClipOutputs:
    """
    One clip's MediaPipe-layout outputs, after the scene transform. T frames.

    `pose_world` fills only rows 11-16 (the arms); the other 27 are NaN. Hands are the posed
    SMPL-X hands for every fitted frame, unfitted fingers included; `hand_valid` says which
    are fitted. `face_valid` and `hand_valid` are false wherever the body was not posed.
    """

    pose_world: np.ndarray  # (T, 33, 3)
    hands_world: np.ndarray  # (T, 2, 21, 3), index 0 = left
    face_world: np.ndarray  # (T, 478, 3)
    head_world: np.ndarray  # (T, 3) the SMPL-X head joint
    blendshapes: np.ndarray  # (T, 52)
    blendshape_names: Tuple[str, ...]
    pose_image: Optional[np.ndarray]  # (T, 33, 2)
    hands_image: Optional[np.ndarray]  # (T, 2, 21, 2)
    face_image: Optional[np.ndarray]  # (T, 478, 2)
    hand_valid: np.ndarray  # (T, 2)
    face_valid: np.ndarray  # (T,)
    body_ok: np.ndarray  # (T,)
    times_s: np.ndarray  # (T,)
    vertices: Optional[List[Optional[np.ndarray]]]  # per frame (V, 3), None where not posed


def blendshape_mirror(names: Sequence[str]) -> np.ndarray:
    """
    Column permutation that swaps every `...Left` blendshape with its `...Right` partner.

    `mirrored = blendshapes[:, blendshape_mirror(names)]`. Names without a side, or whose
    partner is absent, map to themselves.
    """
    names = list(names)
    where = {name: i for i, name in enumerate(names)}
    perm = np.arange(len(names))
    for i, name in enumerate(names):
        if name.endswith("Left"):
            partner = name[: -len("Left")] + "Right"
        elif name.endswith("Right"):
            partner = name[: -len("Right")] + "Left"
        else:
            continue
        if partner in where:
            perm[i] = where[partner]
    return perm


# --------------------------------------------------------------------------- skinning

_LBS_CACHE: Dict[int, Tuple[object, Dict[str, np.ndarray]]] = {}


def _lbs_arrays(model) -> Dict[str, np.ndarray]:
    """The SMPL-X model's skinning tensors as float64 arrays, converted once per model."""
    cached = _LBS_CACHE.get(id(model))
    if cached is not None and cached[0] is model:
        return cached[1]

    def array(tensor):
        return tensor.detach().cpu().numpy().astype(np.float64)

    template = array(model.v_template)
    arrays = {
        "v_template": template,  # (V, 3)
        "shapedirs": array(model.shapedirs),  # (V, 3, K)
        "posedirs": array(model.posedirs).reshape(-1, len(template), 3),  # (P, V, 3)
        "weights": array(model.lbs_weights)[:, :NUM_JOINTS],  # (V, 55)
        "parents": model.parents.detach().cpu().numpy().astype(int)[:NUM_JOINTS],
    }
    _LBS_CACHE[id(model)] = (model, arrays)
    return arrays


def _shaped_template(arrays: Dict[str, np.ndarray], betas: np.ndarray) -> np.ndarray:
    """(V, 3) template with the shape blend; expression zero."""
    betas = np.asarray(betas, dtype=np.float64).ravel()
    shapedirs = arrays["shapedirs"]
    if len(betas) > shapedirs.shape[-1]:
        raise ValueError(f"{len(betas)} betas given; the shape space holds {shapedirs.shape[-1]}")
    return arrays["v_template"] + shapedirs[..., : len(betas)] @ betas


def _regressed_joints(skeleton, betas) -> np.ndarray:
    """(55, 3) rest joints SMPL-X regresses from the shaped template, as `shaped(betas)`."""
    coefficients = np.zeros(skeleton.joint_shapedirs.shape[-1])
    betas = np.asarray(betas, dtype=np.float64).ravel()
    coefficients[: len(betas)] = betas
    return skeleton.joint_template + skeleton.joint_shapedirs @ coefficients


def _skin(arrays, v_shaped, rotations, transl, rest_joints, rest_offset, vertex_ids=None):
    """
    Linear blend skinning as `smplx.lbs.lbs`, with the given rest joints.

    `rest_offset` (55, 3) is `rest_joints` minus the regressed rest joints; the rest mesh is
    moved by its skinning-weighted blend first. Returns (vertices (V', 3), joints (55, 3)),
    both plus `transl`. `vertex_ids` limits the skinned vertices (the joints do not depend on
    them).
    """
    rotations = np.asarray(rotations, dtype=np.float64).reshape(NUM_JOINTS, 3)
    rest = np.asarray(rest_joints, dtype=np.float64)
    parents = arrays["parents"]
    local = Rotation.from_rotvec(rotations).as_matrix()  # (55, 3, 3)

    posedirs, weights = arrays["posedirs"], arrays["weights"]
    if vertex_ids is not None:
        ids = np.asarray(vertex_ids, dtype=int)
        v_shaped, posedirs, weights = v_shaped[ids], posedirs[:, ids], weights[ids]
    if np.any(rest_offset):
        v_shaped = v_shaped + weights @ rest_offset
    pose_feature = (local[1:] - np.eye(3)).reshape(-1)  # (54 * 9,)
    v_posed = v_shaped + np.tensordot(pose_feature, posedirs, axes=1)

    world_rot = np.empty_like(local)
    world_pos = np.empty((NUM_JOINTS, 3))
    for j in range(NUM_JOINTS):
        p = parents[j]
        if p < 0:
            world_rot[j], world_pos[j] = local[j], rest[j]
        else:
            world_rot[j] = world_rot[p] @ local[j]
            world_pos[j] = world_rot[p] @ (rest[j] - rest[p]) + world_pos[p]

    # Per-joint transform relative to the rest pose: x -> G x + (t - G rest).
    relative = np.concatenate(
        [world_rot, (world_pos - np.einsum("jab,jb->ja", world_rot, rest))[..., None]], axis=-1
    )  # (55, 3, 4)
    blended = (weights @ relative.reshape(NUM_JOINTS, 12)).reshape(-1, 3, 4)
    vertices = np.einsum("vab,vb->va", blended[..., :3], v_posed) + blended[..., 3]
    transl = np.asarray(transl, dtype=np.float64).reshape(3)
    return vertices + transl, world_pos + transl


def pose_mesh_rest(skeleton, rotations, betas, transl, rest_joints) -> PosedMesh:
    """
    Pose SMPL-X with every joint rotation, the body shape, the translation and `rest_joints`.

    A copy of `smplx.lbs.lbs`: the shape blend from `skeleton.model.shapedirs` with `betas`,
    the pose blend from `posedirs`, the rigid chain built on the **given** (55, 3) rest joints
    instead of `J_regressor @ v_shaped`, and `lbs_weights` skinning; expression zero. Equal to
    `assembly.pose_mesh` when `rest_joints == skeleton.shaped(betas).rest_joints`. Edited rest
    joints first move the rest mesh by their skinning-weighted offset (module docstring).
    """
    arrays = _lbs_arrays(skeleton.model)
    v_shaped = _shaped_template(arrays, betas)
    offset = np.asarray(rest_joints, dtype=np.float64) - _regressed_joints(skeleton, betas)
    vertices, joints = _skin(arrays, v_shaped, rotations, transl, rest_joints, offset)
    return PosedMesh(vertices=vertices, joints=joints)


# --------------------------------------------------------------------------- outputs


def _tip_ids() -> np.ndarray:
    return np.array([v for side in SIDES for v in SMPLX_TIP_VERTICES[side]], dtype=int)


def hand_points(joints: np.ndarray, tips: np.ndarray, side: str) -> np.ndarray:
    """
    MediaPipe's 21 hand points of one side from the posed SMPL-X joints and that side's five
    fingertip vertices (thumb, index, middle, ring, pinky), as the round-trip demo reads them.
    """
    out = np.full((21, 3), np.nan)
    out[0] = joints[SMPLX_WRIST_JOINT[side]]
    first = SMPLX_FIRST_FINGER_JOINT[side]
    out[list(POSE_SLOT_LANDMARKS)] = joints[first : first + len(POSE_SLOT_LANDMARKS)]
    out[list(TIPS)] = tips
    return out


def _face_mirror() -> np.ndarray:
    try:
        from .face_mirror import FACE_MIRROR
    except ImportError as error:
        raise RuntimeError(
            "scene.mirror is set but the face mirror table "
            "(kinetic_augment.augmentations.face_mirror.FACE_MIRROR) is not generated yet"
        ) from error
    return np.asarray(FACE_MIRROR, dtype=int)


def build_outputs(clip: ClipData, skeleton, keep_vertices: bool = False) -> ClipOutputs:
    """
    Pose every fitted frame of `clip` and read off MediaPipe-layout landmarks.

    Per `body_ok` frame (with finite rotations and translation): `pose_mesh_rest` on the clip's
    rest joints; pose 11-16 from SMPL-X joints 16-21; the hands from the mesh; the face from
    `face_local` on the posed head joint where `face_valid`. Unposed frames stay NaN. Then the
    clip's scene transform goes on every point set; when it mirrors, the pose rows, the hands
    (and `hand_valid`), the face rows (`face_mirror.FACE_MIRROR`) and the `*Left`/`*Right`
    blendshapes swap sides. Image arrays come from the clip's camera, or are None without one.
    """
    scene = clip.scene
    face_perm = _face_mirror() if scene.mirror else None
    T = clip.frames
    arrays = _lbs_arrays(skeleton.model)
    v_shaped = _shaped_template(arrays, clip.betas)
    parents = arrays["parents"]
    offset = np.asarray(clip.rest_joints, dtype=np.float64) - _regressed_joints(
        skeleton, clip.betas
    )

    posed = (
        np.asarray(clip.body_ok, dtype=bool)
        & np.isfinite(clip.rotations).all(axis=(1, 2))
        & np.isfinite(clip.transl).all(axis=1)
    )
    face_valid = posed & np.asarray(clip.face_valid, dtype=bool)
    hand_valid = posed[:, None] & np.asarray(clip.hand_valid, dtype=bool)

    pose = np.full((T, 33, 3), np.nan)
    hands = np.full((T, 2, 21, 3), np.nan)
    face = np.full((T, 478, 3), np.nan)
    head = np.full((T, 3), np.nan)
    vertices: Optional[List[Optional[np.ndarray]]] = [None] * T if keep_vertices else None
    tip_ids = _tip_ids()

    for t in np.flatnonzero(posed):
        if keep_vertices:
            verts, joints = _skin(
                arrays, v_shaped, clip.rotations[t], clip.transl[t], clip.rest_joints, offset
            )
            tips = verts[tip_ids]
            vertices[t] = scene.apply_points(verts)
        else:
            tips, joints = _skin(
                arrays,
                v_shaped,
                clip.rotations[t],
                clip.transl[t],
                clip.rest_joints,
                offset,
                vertex_ids=tip_ids,
            )
        pose[t, POSE_ARMS] = joints[SMPLX_ARMS]
        for s, side in enumerate(SIDES):
            hands[t, s] = hand_points(joints, tips[5 * s : 5 * s + 5], side)
        head[t] = joints[HEAD]
        if face_valid[t]:
            rotation = global_rotations(clip.rotations[t], parents)[HEAD]
            face[t] = from_head_frame(clip.face_local[t], rotation, joints[HEAD])

    pose, hands = scene.apply_points(pose), scene.apply_points(hands)
    face, head = scene.apply_points(face), scene.apply_points(head)
    blendshapes = np.array(clip.blendshapes, dtype=np.float64, copy=True)
    if scene.mirror:
        pose = pose[:, POSE_MIRROR]
        hands = hands[:, ::-1].copy()
        hand_valid = hand_valid[:, ::-1].copy()
        face = face[:, face_perm]
        blendshapes = blendshapes[:, blendshape_mirror(clip.blendshape_names)]

    camera = clip.camera
    return ClipOutputs(
        pose_world=pose,
        hands_world=hands,
        face_world=face,
        head_world=head,
        blendshapes=blendshapes,
        blendshape_names=tuple(clip.blendshape_names),
        pose_image=None if camera is None else camera.project(pose),
        hands_image=None if camera is None else camera.project(hands),
        face_image=None if camera is None else camera.project(face),
        hand_valid=hand_valid,
        face_valid=face_valid,
        body_ok=np.array(clip.body_ok, dtype=bool, copy=True),
        times_s=np.array(clip.times_s, dtype=np.float64, copy=True),
        vertices=vertices,
    )
