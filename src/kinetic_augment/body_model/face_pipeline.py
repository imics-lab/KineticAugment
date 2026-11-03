"""
One clip end to end: body, hands and head fitted per frame, assembled into one SMPL-X pose,
and hand-to-face geometry measured on it (V1.3 design §3).

The body is fitted to the pose landmarks stored in the hand clip, so every stream shares the
same decoded frames. Hands are scaled to one palm size per track, fitted with MANO and attached
as the V1.2 viewer attaches them.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from .assembly import (
    HEAD,
    NECK,
    attach_hands,
    attach_head,
    body_rotations,
    global_rotations,
    pose_mesh,
)
from .converter import fit_body
from .face_correspondence import (
    LEFT_EYE_OUTER,
    RIGID,
    RIGHT_EYE_OUTER,
    load_face_correspondence,
)
from .face_regions import build_head_surface
from .face_stream import attach_face
from .face_features import save_face_features as _save_face_features, validate_face_inputs
from .hand_converter import fit_hand_shape, fit_hands, normalise_palm_size, reconstruct_hand
from .hand_depth import DEFAULT_VERTICAL_FOV_DEG, REACH_TOLERANCE, wrist_targets
from .hand_face_geometry import HAND_POINTS, HandFaceGeometry, hand_face_geometry
from .hand_model import HandModel
from .head_converter import OCCLUSION_PAD, HeadFit, fit_head, hand_outlines, head_template
from .head_converter import occluded, points_in_polygon
from .image_proximity import (
    ImageProximity,
    image_hand_points,
    image_proximity,
    project_head,
)
from .shape import fit_shape
from .skeleton import SMPLXSkeleton

VISIBILITY_THRESHOLD = 0.5  # matches measure_round_trip.py
EARS = (7, 8)  # MediaPipe pose: left ear, right ear
SIDES = ("left", "right")
NECK_MODES = ("rest", "swing", "split")
POSE_WRISTS = (15, 16)  # MediaPipe pose: left wrist, right wrist
SMPLX_WRISTS = (20, 21)  # SMPL-X joints: left wrist, right wrist
POSE_ARMS = ((11, 13), (12, 14))  # MediaPipe pose: (shoulder, elbow) of the left, right arm


@dataclass
class FaceModels:
    skeleton: SMPLXSkeleton
    corr: object
    mano: Dict[str, HandModel]  # fitting backend
    smplx_hands: Dict[str, HandModel]  # SMPL-X palms for attaching the wrists
    model_dir: str


def load_face_models(model_dir="models") -> FaceModels:
    return FaceModels(
        skeleton=SMPLXSkeleton(model_dir),
        corr=load_face_correspondence(model_dir),
        mano={s: HandModel("mano", s, model_dir=model_dir) for s in SIDES},
        smplx_hands={s: HandModel("smplx", s, model_dir=model_dir) for s in SIDES},
        model_dir=model_dir,
    )


def hand_near_face(face_px, outlines, pad: float) -> bool:
    """True if a hand outline comes within `pad` pixels of the face's bounding box."""
    points = np.asarray(face_px)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points) or not outlines:
        return False
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    corners = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    box_edges = list(zip(corners, np.roll(corners, -1, axis=0)))
    for outline in outlines:
        if ((outline >= lo) & (outline <= hi)).all(axis=1).any():
            return True
        if points_in_polygon(corners, outline).any():
            return True
        for a, b in zip(outline, np.roll(outline, -1, axis=0)):
            if _segments_intersect(a, b, box_edges):
                return True
    return False


def _segments_intersect(a, b, edges):
    """Whether segment a-b crosses any axis-aligned box edge."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for c, d in edges:
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if (
            (o1 == 0 and _on_segment(a, b, c))
            or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a))
            or (o4 == 0 and _on_segment(c, d, b))
            or (o1 * o2 < 0 and o3 * o4 < 0)
        ):
            return True
    return False


def _on_segment(a, b, p):
    return bool(np.all(p >= np.minimum(a, b)) and np.all(p <= np.maximum(a, b)))


@dataclass
class ClipFaceResult:
    head_fits: List[HeadFit]
    rotations: np.ndarray  # (T, 55, 3); NaN where the body could not be fitted
    transl: np.ndarray  # (T, 3)
    betas: np.ndarray  # (300,)
    geometry: List[Optional[HandFaceGeometry]]
    hand_near_face: np.ndarray  # (T,) bool
    body_ok: np.ndarray  # (T,) bool
    frames: np.ndarray  # (T,) source frame indices
    image_geometry: List[Optional[ImageProximity]] = field(default_factory=list)
    head_scale_px_per_m: np.ndarray = None  # (T,) px per metre at the head; NaN if none
    head_camera_rotation_deg: np.ndarray = None  # (T,) world->pixel rotation angle; NaN if none
    # Depth correction (task 14); filled in every run, empty unless depth_corrected is set.
    wrist_target: np.ndarray = None  # (T, 2, 3) image-derived world wrist; NaN if none
    wrist_corrected: np.ndarray = None  # (T, 2) bool: the re-fit used the target
    depth_ratio: np.ndarray = None  # (T, 2) Z_hand / Z_head; NaN where unavailable
    wrist_residual_m: np.ndarray = None  # (T, 2) |SMPL-X wrist - target| after the re-fit
    wrist_shift_m: np.ndarray = None  # (T, 2) |target - MediaPipe pose-world wrist|
    wrist_reached: np.ndarray = None  # (T, 2) bool: corrected and residual <= REACH_TOLERANCE
    face_landmarks: np.ndarray = None
    face_pixels: np.ndarray = None
    face_stream_local: np.ndarray = None
    head_transform: np.ndarray = None
    face_pixel_to_world: np.ndarray = None
    face_observed: np.ndarray = None
    face_attachment_valid: np.ndarray = None
    face_point_valid: np.ndarray = None
    face_point_visible: np.ndarray = None
    head_world_rotation: np.ndarray = None
    head_relative_rotation: np.ndarray = None
    pose_image: np.ndarray = None
    pose_world: np.ndarray = None
    pose_visibility: np.ndarray = None
    hand_image: np.ndarray = None
    hand_world: np.ndarray = None
    hand_present: np.ndarray = None
    geometry_valid: np.ndarray = None
    contact_valid: np.ndarray = None
    depth_corrected: bool = False
    fov_deg: float = DEFAULT_VERTICAL_FOV_DEG
    hands_meta: Dict = field(default_factory=dict)


def _fit_hands(hands, mano, frames):
    fits, recon = {}, {}
    for s, side in enumerate(SIDES):
        observed = hands.world[frames, s].astype(np.float64)
        valid = np.isfinite(observed).all(axis=-1)
        # MediaPipe's hand changes size frame to frame; the fit has one shape per track.
        observed, _ = normalise_palm_size(observed, valid)
        betas = fit_hand_shape(observed, valid, mano[side])
        fits[side] = fit_hands(observed, valid, mano[side], betas)
        recon[side] = np.stack([reconstruct_hand(f, mano[side]) for f in fits[side]])
    return fits, recon


def run_clip(
    hands,
    face,
    models: FaceModels,
    frames=None,
    depth_corrected=False,
    fov_deg=DEFAULT_VERTICAL_FOV_DEG,
    attach_face_stream=True,
) -> ClipFaceResult:
    """
    Fit and measure every frame (or `frames`, source indices) of one clip.

    With `depth_corrected`, each frame whose head projection exists and where at least one hand's
    wrist can be placed from the image (`hand_depth.wrist_targets`) gets a second pass: the pose
    wrist is replaced by that target, the body is re-fitted with the same shaped skeleton, and
    hands, head, mesh and both geometries are recomputed. Those frames then hold the second-pass
    values; every other frame, and every frame with `depth_corrected` False, is the first pass.
    A hand whose same-side shoulder or elbow is not valid in the pose is left uncorrected, since
    the body fit cannot place its wrist.

    The second pass keeps the first pass's body translation (so head and torso stay where the
    target was measured from) and the bone lengths; the shoulder and elbow directions come from
    MediaPipe, and only the forearm direction changes. The SMPL-X wrist may therefore miss its
    target; `wrist_residual_m` measures the miss and `wrist_reached` flags residuals
    within `REACH_TOLERANCE`. 3-D contact on a corrected hand that was not reached is not
    trustworthy: the arm cannot reach the target, so the SMPL-X hand ends up somewhere else. The
    affected hand's fitted geometry is masked and `geometry_valid` records whether each side is
    usable. `contact_valid` remains false until contact has independent calibration.

    Raises:
        ValueError: the hand and face clips do not share frames.
    """
    validate_face_inputs(hands, face)
    frames = np.arange(len(face.valid)) if frames is None else np.asarray(frames)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("frames must be a one-dimensional array of source frame indices")
    if np.any(frames < 0) or np.any(frames >= len(face.valid)):
        raise ValueError("frames contains an out-of-range source frame index")
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= VISIBILITY_THRESHOLD)
    betas = fit_shape(pose, ok, models.skeleton)
    body = models.skeleton.shaped(betas)
    template = head_template(body, models.corr)
    surface = build_head_surface(body, models.corr, models.model_dir)
    hand_fits, hand_recon = _fit_hands(hands, models.mano, frames)
    face_px = face.pixels()

    n = len(frames)
    result = ClipFaceResult(
        [],
        np.full((n, 55, 3), np.nan),
        np.full((n, 3), np.nan),
        betas,
        [],
        np.zeros(n, bool),
        np.zeros(n, bool),
        frames,
        [],
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full((n, 2, 3), np.nan),
        np.zeros((n, 2), bool),
        np.full((n, 2), np.nan),
        np.full((n, 2), np.nan),
        np.full((n, 2), np.nan),
        np.zeros((n, 2), bool),
    )
    result.depth_corrected, result.fov_deg = bool(depth_corrected), float(fov_deg)
    result.hands_meta = dict(getattr(hands, "meta", {}) or {})
    result.face_landmarks = np.asarray(face.landmarks[frames]).copy()
    result.face_pixels = np.asarray(face.pixels()[frames]).copy()
    result.face_observed = np.asarray(face.valid[frames], dtype=bool).copy()
    result.face_point_valid = (
        np.isfinite(result.face_pixels).all(axis=-1) & result.face_observed[:, None]
    )
    result.face_point_visible = np.zeros((n, 478), dtype=bool)
    result.face_stream_local = np.full((n, 478, 3), np.nan)
    result.head_transform = np.full((n, 4, 4), np.nan)
    result.face_pixel_to_world = np.full((n, 4, 4), np.nan)
    result.face_attachment_valid = np.zeros(n, dtype=bool)
    result.head_world_rotation = np.full((n, 3, 3), np.nan)
    result.head_relative_rotation = np.full((n, 3, 3), np.nan)
    result.geometry_valid = np.zeros((n, 2), dtype=bool)
    result.contact_valid = np.zeros((n, 2, len(HAND_POINTS)), dtype=bool)
    result.pose_image = np.asarray(hands.pose_image[frames]).copy()
    result.pose_world = np.asarray(hands.pose_world[frames]).copy()
    result.pose_visibility = np.asarray(hands.pose_visibility[frames]).copy()
    result.hand_image = np.asarray(hands.image[frames]).copy()
    result.hand_world = np.asarray(hands.world[frames]).copy()
    result.hand_present = np.asarray(hands.present[frames], dtype=bool).copy()
    for i, t in enumerate(frames):
        outlines = hand_outlines(hands.image[t], hands.width, hands.height)
        observed_face = face_px[t] if face.valid[t] else None
        if observed_face is not None:
            reference = np.linalg.norm(
                observed_face[RIGHT_EYE_OUTER] - observed_face[LEFT_EYE_OUTER]
            )
            result.hand_near_face[i] = hand_near_face(
                observed_face, outlines, OCCLUSION_PAD * reference
            )
        result.face_point_visible[i] = (
            _face_visibility(result.face_pixels[i], outlines) & result.face_point_valid[i]
        )
        try:
            body_fit = fit_body(pose[t], ok[t], body)
        except ValueError:
            result.head_fits.append(fit_head(None, [], np.zeros((55, 3)), np.zeros(3), template))
            result.geometry.append(None)
            result.image_geometry.append(None)
            continue
        result.body_ok[i] = True
        hand_fit_i = {side: hand_fits[side][i] for side in SIDES}
        hand_recon_i = {side: hand_recon[side][i] for side in SIDES}
        ears = pose[t, list(EARS)].mean(axis=0) if ok[t, list(EARS)].all() else None
        rotations, head, mesh, posed = _assemble(
            body,
            betas,
            models,
            body_fit,
            hand_fit_i,
            hand_recon_i,
            observed_face,
            outlines,
            template,
            ears,
        )
        geometry = hand_face_geometry(mesh, surface, posed)
        image_geometry = _image_geometry(
            result,
            i,
            mesh,
            surface,
            models.corr,
            head,
            observed_face,
            hands.image[t],
            hands.width,
            hands.height,
        )
        transl = body_fit.transl
        if depth_corrected and image_geometry is not None:
            head_px, head_world = _head_centroid(models.corr, mesh, observed_face, head)
            targets = wrist_targets(
                head_px,
                head_world,
                result.head_scale_px_per_m[i],
                hands.world[t],
                hands.image[t][..., :2] * np.array([hands.width, hands.height], np.float64),
                hands.width,
                hands.height,
                fov_deg,
                arm_valid=np.array([ok[t, list(arm)].all() for arm in POSE_ARMS]),
            )
            result.depth_ratio[i] = targets.depth_ratio
            if targets.corrected.any():
                second = _depth_pass(
                    result,
                    i,
                    t,
                    targets,
                    pose,
                    ok,
                    body,
                    betas,
                    models,
                    surface,
                    hand_fit_i,
                    hand_recon_i,
                    observed_face,
                    outlines,
                    template,
                    ears,
                    hands,
                    body_fit.transl,
                )
                if second is not None:
                    rotations, head, mesh, posed, geometry, image_geometry, transl = second
        global_rotation = global_rotations(rotations, body.parents)
        result.head_world_rotation[i] = global_rotation[HEAD]
        torso_rotation = global_rotation[body.parents[NECK]]
        result.head_relative_rotation[i] = torso_rotation.T @ global_rotation[HEAD]
        if (
            attach_face_stream
            and result.face_observed[i]
            and head.reliable
            and np.isfinite(mesh.joints[HEAD]).all()
        ):
            attachment = attach_face(
                result.face_pixels[i],
                head.rigid_weights,
                models.corr.points(mesh.vertices, RIGID),
                global_rotation[HEAD],
                mesh.joints[HEAD],
            )
            if attachment.valid:
                result.face_stream_local[i] = attachment.local
                result.head_transform[i] = attachment.world_from_head
                result.face_pixel_to_world[i] = attachment.pixel_to_world
                result.face_attachment_valid[i] = True
        hand_ok = np.array([bool(posed.get(side, False)) for side in SIDES])
        result.geometry_valid[i] = geometry_valid_mask(
            head.reliable,
            result.body_ok[i],
            hand_ok,
            result.wrist_corrected[i] if depth_corrected else np.zeros(2, bool),
            result.wrist_reached[i],
        )
        _mask_geometry(geometry, result.geometry_valid[i])
        result.head_fits.append(head)
        result.rotations[i], result.transl[i] = rotations, transl
        result.geometry.append(geometry)
        result.image_geometry.append(image_geometry)
    return result


def _head_centroid(corr, mesh, observed_face, head):
    """
    Weighted centroid of the head's rigid points: observed (pixels) and on the posed mesh (world).

    `project_head` fits its similarity with these weights, so its scale is the depth of this
    centroid; it is the reference that `wrist_targets` measures wrists from.
    """
    weights = np.asarray(head.rigid_weights, dtype=np.float64)
    weights = weights / weights.sum()
    used = weights > 0
    px = weights[used] @ observed_face[list(RIGID)][used][:, :2]
    world = weights[used] @ corr.points(mesh.vertices, RIGID)[used]
    return px, world


def _assemble(
    body, betas, models, body_fit, hand_fit, hand_recon, observed_face, outlines, template, ears
):
    """Hands, then head, attached to one body fit; returns rotations, head, mesh, hands posed."""
    rotations, posed = attach_hands(
        body_rotations(body_fit), body.parents, hand_fit, hand_recon, models.smplx_hands
    )
    head = fit_head(observed_face, outlines, rotations, body_fit.transl, template, ears)
    rotations = attach_head(rotations, head)
    return rotations, head, pose_mesh(body, rotations, betas, body_fit.transl), posed


def _depth_pass(
    result,
    i,
    t,
    targets,
    pose,
    ok,
    body,
    betas,
    models,
    surface,
    hand_fit,
    hand_recon,
    observed_face,
    outlines,
    template,
    ears,
    hands,
    first_transl,
):
    """
    Second pass of one frame with the corrected wrists (see `run_clip`).

    Fills the wrist target, corrected, residual and shift fields of `result` row `i` and
    returns (rotations, head, geometry, image geometry, transl), or None if the body cannot be
    re-fitted.
    """
    landmarks, valid = pose[t].copy(), ok[t].copy()
    for s, wrist in enumerate(POSE_WRISTS):
        if targets.corrected[s]:
            landmarks[wrist], valid[wrist] = targets.target[s], True
    try:
        body_fit = fit_body(landmarks, valid, body)
    except ValueError:
        return None
    # `fit_body` sets the translation to the mean arm residual, wrists included, so a wrist 0.1-
    # 0.35 m away would move the whole body by 1-3 cm. The target was measured against the first
    # pass's head, so the head and torso stay where they were.
    body_fit = replace(body_fit, transl=first_transl)
    rotations, head, mesh, posed = _assemble(
        body, betas, models, body_fit, hand_fit, hand_recon, observed_face, outlines, template, ears
    )
    geometry = hand_face_geometry(mesh, surface, posed)
    image_geometry = _image_geometry(
        result,
        i,
        mesh,
        surface,
        models.corr,
        head,
        observed_face,
        hands.image[t],
        hands.width,
        hands.height,
    )
    result.wrist_target[i] = targets.target
    result.wrist_corrected[i] = targets.corrected
    for s in np.flatnonzero(targets.corrected):
        result.wrist_residual_m[i, s] = np.linalg.norm(
            mesh.joints[SMPLX_WRISTS[s]] - targets.target[s]
        )
        result.wrist_shift_m[i, s] = np.linalg.norm(targets.target[s] - pose[t, POSE_WRISTS[s]])
        result.wrist_reached[i, s] = result.wrist_residual_m[i, s] <= REACH_TOLERANCE
    return rotations, head, mesh, posed, geometry, image_geometry, body_fit.transl


def _face_visibility(face_pixels, outlines):
    """Estimate point visibility from padded hand outlines; detector confidence is separate."""
    finite = np.isfinite(face_pixels).all(axis=1)
    visible = np.zeros(len(face_pixels), dtype=bool)
    if not finite.any():
        return visible
    eye_distance = np.linalg.norm(
        face_pixels[RIGHT_EYE_OUTER, :2] - face_pixels[LEFT_EYE_OUTER, :2]
    )
    pad = OCCLUSION_PAD * eye_distance if np.isfinite(eye_distance) else 0.0
    visible[finite] = ~occluded(face_pixels[finite, :2], outlines, pad)
    return visible


def geometry_valid_mask(head_reliable, body_valid, hands_posed, wrist_corrected, wrist_reached):
    """Per-hand fitted-geometry validity; a failed correction invalidates that hand only."""
    posed = np.asarray(hands_posed, dtype=bool).reshape(2)
    corrected = np.asarray(wrist_corrected, dtype=bool).reshape(2)
    reached = np.asarray(wrist_reached, dtype=bool).reshape(2)
    return bool(head_reliable and body_valid) & posed & (~corrected | reached)


def _mask_geometry(geometry, valid):
    """Hide unaccepted fitted geometry from the legacy distance and contact fields."""
    if geometry is None:
        return
    invalid = ~np.asarray(valid, dtype=bool).reshape(2)
    geometry.distance[invalid] = np.nan
    geometry.region[invalid] = -1
    geometry.subregion[invalid] = -1
    geometry.contact[invalid] = False


def _image_geometry(
    result, i, mesh, surface, corr, head, observed_face, hand_image, width, height
) -> Optional[ImageProximity]:
    """Image-space proximity for one frame, or None without a usable head fit."""
    if observed_face is None or (head.rigid_weights > 0).sum() < 3:
        return None
    projected = project_head(
        mesh.vertices[surface.vertex_ids],
        surface.faces,
        corr.points(mesh.vertices, RIGID),
        observed_face[list(RIGID)],
        head.rigid_weights,
    )
    result.head_scale_px_per_m[i] = projected.scale
    result.head_camera_rotation_deg[i] = projected.rotation_deg
    return image_proximity(image_hand_points(hand_image, width, height), projected, surface)


def save_face_features(path, result: ClipFaceResult, face) -> None:
    """Write the per-frame face features (design §3) to versioned .npz."""
    _save_face_features(path, result, face, result.hands_meta)


def load_face_features(path):
    """Compatibility re-export for existing consumers."""
    from .face_features import load_face_features as loader

    return loader(path)
