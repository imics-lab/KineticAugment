"""
One posed SMPL-X body from the separate fits.

The body fit gives global_orient, body_pose, betas and transl; each hand fit gives a finger pose
and, through the palm of its reconstruction, the wrist's world rotation; the head fit gives the
neck, head and jaw. SMPL-X takes all of them as one (55, 3) array of local axis-angle rotations
in its joint order -- the order `forward_kinematics` uses -- so assembly fills that array and
runs the model.

Hands are attached as the V1.2 viewer did: the SMPL-X hand at the fitted finger pose, with the
wrist unrotated, is turned so its palm (wrist and knuckles) lines up with the reconstruction's
palm, and that rotation is written into the wrist relative to its parent.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

NUM_JOINTS = 55
NECK, HEAD, JAW = 12, 15, 22
WRIST = {"left": 20, "right": 21}
HAND_SLICE = {"left": slice(25, 40), "right": slice(40, 55)}
PALM = (0, 5, 9, 13, 17)  # MediaPipe hand points: the wrist and the four knuckles


@dataclass
class PosedMesh:
    vertices: np.ndarray  # (V, 3) metres, MediaPipe world frame
    joints: np.ndarray    # (55, 3) model joints, same frame


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R source - target| over centred point sets."""
    u, _, vt = np.linalg.svd((source - source.mean(0)).T @ (target - target.mean(0)))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def global_rotations(rotations: np.ndarray, parents) -> np.ndarray:
    """(J, 3, 3) world rotation of every joint from (J, 3) local axis-angle rotations."""
    local = Rotation.from_rotvec(np.asarray(rotations, dtype=np.float64)).as_matrix()
    world = np.empty_like(local)
    for j, parent in enumerate(parents):
        world[j] = local[j] if parent < 0 else world[parent] @ local[j]
    return world


def body_rotations(body_fit) -> np.ndarray:
    """(55, 3) rotations holding the body fit; every other joint at rest."""
    rotations = np.zeros((NUM_JOINTS, 3))
    rotations[0] = body_fit.global_orient
    rotations[1:22] = np.asarray(body_fit.body_pose).reshape(21, 3)
    return rotations


def attach_hands(rotations, parents, hand_fits, hand_recon,
                 hand_models) -> Tuple[np.ndarray, Dict[str, bool]]:
    """
    Rotations with each fitted hand's finger pose and wrist rotation written in.

    Args:
        rotations: (55, 3); not modified.
        parents: (55,) kinematic parents.
        hand_fits: side -> HandFit for this frame.
        hand_recon: side -> (21, 3) reconstructed hand landmarks for this frame (NaN if none).
        hand_models: side -> SMPL-X `HandModel`, used for the unrotated SMPL-X palm.

    Returns:
        (new rotations, side -> whether that hand was posed). An unposed hand keeps a flat
        finger pose and the body fit's wrist.
    """
    rotations = np.array(rotations, dtype=np.float64)
    posed = {}
    for side in ("left", "right"):
        fit, target = hand_fits[side], np.asarray(hand_recon[side])
        posed[side] = fit.status == "ok" and bool(np.isfinite(target).all())
        if not posed[side]:
            rotations[HAND_SLICE[side]] = 0.0
            continue
        rotations[HAND_SLICE[side]] = np.asarray(fit.hand_pose).reshape(15, 3)
        with torch.no_grad():
            rest = hand_models[side].points(np.zeros(3), fit.hand_pose,
                                            np.zeros(10))[0].numpy()
        world_wrist = kabsch(rest[list(PALM)], target[list(PALM)])
        wrist = WRIST[side]
        parent_world = global_rotations(rotations, parents)[parents[wrist]]
        rotations[wrist] = Rotation.from_matrix(parent_world.T @ world_wrist).as_rotvec()
    return rotations, posed


def attach_head(rotations, head_fit) -> np.ndarray:
    """Rotations with the head fit's neck, head and jaw written in."""
    rotations = np.array(rotations, dtype=np.float64)
    rotations[NECK] = head_fit.neck_pose
    rotations[HEAD] = head_fit.head_pose
    rotations[JAW] = head_fit.jaw_pose
    return rotations


def pose_mesh(skeleton, rotations, betas, transl) -> PosedMesh:
    """
    Run SMPL-X with every joint rotation, the body shape and the translation; expression zero.

    `skeleton` is an `SMPLXSkeleton`; its model's shape space must hold `len(betas)` values.
    """
    r = np.asarray(rotations, dtype=np.float32)

    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)

    with torch.no_grad():
        out = skeleton.model(
            betas=tensor(betas), global_orient=tensor(r[0]), body_pose=tensor(r[1:22]),
            jaw_pose=tensor(r[22]), leye_pose=tensor(r[23]), reye_pose=tensor(r[24]),
            left_hand_pose=tensor(r[25:40]), right_hand_pose=tensor(r[40:55]),
            transl=tensor(transl),
        )
    return PosedMesh(vertices=out.vertices[0].numpy().astype(np.float64),
                     joints=out.joints[0, :NUM_JOINTS].numpy().astype(np.float64))
