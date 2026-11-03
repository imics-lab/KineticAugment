"""
V1.2 MediaPipe hand landmarks <-> MANO / SMPL-X hand parameters.

The analytic start follows the V1 body converter. MANO's palm is rigid -- the
thumb CMC and the four finger MCPs are children of the wrist -- so a Kabsch
rotation about the wrist over those five points fixes the wrist rotation. Each
finger bone is then swung onto its observed direction with zero twist
(`solve_swing_ik`), parents before children. On a hand the model can represent
exactly this reproduces every joint landmark; only the fingertips, which are
skinned mesh vertices, are approximate.

Contract, per frame and hand:
    21 world landmarks (metres) + per-landmark validity
        -> genuine model parameters (wrist_orient, hand_pose, betas, transl)
        -> 21 landmarks rebuilt from those parameters alone

`fit_hands` refines the analytic start through the real model forward pass
(L-BFGS on the summed squared wrist-relative landmark error), which settles the
fingertips -- skinned mesh vertices the analytic start only approximates -- and
whatever the model cannot represent exactly. `reconstruct_hand` reads nothing
but a fit's parameters and the model.

Shape is per hand track (`fit_hand_shape`): one plausible betas vector fitted
jointly with every frame's pose. `fit_hands_per_frame_shape` lets each frame
take its own plausible shape; it is a diagnostic of how much error comes from
the detector's frame-to-frame size changes, not the pipeline's contract.

The pipeline's callers (`face_pipeline._fit_hands`, `scripts/view_body_hand_round_trip.py`)
fit tracks that `normalise_palm_size` has brought to one palm size, since the detector's size
changes would otherwise be absorbed as finger curl.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from kinetic_augment.body_model.hand_model import (
    HAND_PARENTS,
    NUM_HAND_LANDMARKS,
    POSE_SLOT_LANDMARKS,
    WRIST,
)
from kinetic_augment.body_model.swing_ik import solve_swing_ik

PALM_LANDMARKS = (1, 5, 9, 13, 17)
PALM_SIZE_LANDMARKS = (5, 9, 13, 17)  # the four finger MCPs; palm size is their mean wrist distance

# (driven node, child node, observed parent, observed child): each finger joint's
# rotation aims the bone to its child. Bones leaving the wrist form the palm,
# which `palm_rotation` places instead.
FINGER_BONES = [(int(p), c, int(p), c) for c, p in enumerate(HAND_PARENTS) if p > 0]

FIT_DTYPE = torch.float64
POSE_PRIOR = 1e-4  # mm^2 per rad^2: only settles directions the landmarks leave free
MAX_ITERATIONS = 1000  # dry run: 200 stops short of convergence (0.07 mm); 1000 reaches 0.02 mm
FRAMES_PER_BATCH = 32
FIT_STATUSES = ("ok", "no_wrist", "no_palm", "diverged")
SHAPE_PLAUSIBILITY = 0.95
SHAPE_ITERATIONS = 1000


@dataclass
class HandFit:
    """
    Genuine model parameters for one hand in one frame.

    MANO takes them as they are: global_orient=wrist_orient, hand_pose, betas,
    transl. In SMPL-X, wrist_orient is the wrist joint's local rotation in
    body_pose with every other body joint at rest, and hand_pose is
    left_hand_pose or right_hand_pose. Both use the flat hand mean and no PCA.

    `transl` puts the model's wrist on the observed wrist. A frame without a fit
    has its reason in `status` (one of FIT_STATUSES) and NaN parameters.

    Status "ok" only means the parameters are finite. It does not certify that the
    optimiser converged, because the optimiser stops at an iteration cap (see
    `fit_hands`). `rms_mm` is the quality signal: the frame's RMS wrist-relative
    landmark residual in millimetres, over its valid landmarks (the wrist excluded).
    """

    backend: str
    side: str
    wrist_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    status: str = "ok"
    rms_mm: float = float("nan")


def palm_rotation(rest: np.ndarray, observed: np.ndarray,
                  valid: np.ndarray) -> Optional[np.ndarray]:
    """
    Proper rotation about the wrist best carrying the rest palm onto the observed one.

    Args:
        rest, observed: (21, 3) positions relative to their own wrist.
        valid: (21,) bool.

    Returns:
        (3, 3) rotation, or None when the valid palm points cannot fix one:
        fewer than two of them, or all on one line through the wrist.
    """
    idx = [i for i in PALM_LANDMARKS if valid[i]]
    if len(idx) < 2:
        return None
    u, s, vt = np.linalg.svd(rest[idx].T @ observed[idx])
    if s[1] <= 1e-6 * s[0]:
        return None
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, flip]) @ u.T


def initial_pose(observed, valid, skeleton) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Analytic start for one frame: (wrist_orient (3,), hand_pose (45,)), axis-angle.

    Args:
        observed: (21, 3) landmark positions, any origin.
        valid: (21,) bool; invalid landmarks never influence the result.
        skeleton: rest skeleton from `HandModel.skeleton(betas)`.

    Returns None when the wrist is invalid or the palm cannot fix a rotation.
    """
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=1)
    if not valid[WRIST]:
        return None
    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    rotation = palm_rotation(rest - rest[WRIST], observed - observed[WRIST], valid)
    if rotation is None:
        return None
    # Swing IK reads only directions, and takes an all-zero point as missing, so it
    # gets the observations as given rather than wrist-relative.
    masked = np.where(valid[:, None], observed, np.nan)
    local = solve_swing_ik(masked, skeleton, FINGER_BONES,
                           global_orient=Rotation.from_matrix(rotation).as_rotvec())
    return local[WRIST], local[list(POSE_SLOT_LANDMARKS)].reshape(-1)


def node_rotations(wrist_orient, hand_pose) -> np.ndarray:
    """(21, 3) local rotation per landmark node -- the layout `forward_kinematics` takes."""
    nodes = np.zeros((NUM_HAND_LANDMARKS, 3))
    nodes[WRIST] = wrist_orient
    nodes[list(POSE_SLOT_LANDMARKS)] = np.asarray(hand_pose, dtype=np.float64).reshape(-1, 3)
    return nodes


def landmark_errors_mm(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """
    The plan's primary metric: distance in millimetres between predicted and
    observed wrist-relative positions of landmarks 1-20, with no rotation or scale
    registration. Shape (..., 20); NaN where either side is missing.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    gap = (predicted - predicted[..., :1, :]) - (observed - observed[..., :1, :])
    return 1000.0 * np.linalg.norm(gap, axis=-1)[..., 1:]


def _prepared(observed, valid):
    """float64 observations, and validity that also excludes non-finite landmarks."""
    observed = np.asarray(observed, dtype=np.float64)
    return observed, np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=-1)


def palm_sizes(observed) -> np.ndarray:
    """(T,) palm size per frame (mean wrist -> MCP distance, metres); NaN where any is missing."""
    observed = np.asarray(observed, dtype=np.float64)
    reach = observed[:, list(PALM_SIZE_LANDMARKS)] - observed[:, WRIST:WRIST + 1]
    return np.linalg.norm(reach, axis=-1).mean(axis=-1)


def normalise_palm_size(observed, valid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each frame scaled about its wrist so its palm size is the track's median palm size.

    MediaPipe's world hand changes metric size from frame to frame with the handshape, while
    the hand is fitted with one shape per track, so the pose fit would bend the fingers to
    absorb the size gap. The palm is rigid, so its size should be constant: scaling every
    frame to the median palm size removes the gap without touching the finger directions.

    Args:
        observed: (T, 21, 3) world landmarks of one hand, metres.
        valid: (T, 21) bool; invalid or non-finite landmarks do not count.

    Returns (scaled, scale): a (T, 21, 3) copy and the (T,) factor each frame was scaled by
    (median / its palm size). A frame whose wrist or any of PALM_SIZE_LANDMARKS is invalid or
    non-finite, or whose palm has no size, has scale NaN and is returned unchanged; with no
    measurable frame the track comes back unchanged and every scale is NaN.
    """
    observed, valid = _prepared(observed, valid)
    sizes = palm_sizes(observed)
    measurable = valid[:, [WRIST, *PALM_SIZE_LANDMARKS]].all(axis=1) & (sizes > 0)
    scale = np.full(len(observed), np.nan)
    scaled = observed.copy()
    if measurable.any():
        scale[measurable] = np.median(sizes[measurable]) / sizes[measurable]
        wrist = observed[measurable, WRIST:WRIST + 1]
        scaled[measurable] = wrist + (observed[measurable] - wrist) * scale[measurable, None, None]
    return scaled, scale


def _unfitted(model, status: str) -> HandFit:
    return HandFit(
        model.backend, model.side, np.full(3, np.nan), np.full(45, np.nan),
        np.full(model.num_betas, np.nan), np.full(3, np.nan), status,
    )


def _rms_mm(residual: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-frame RMS of the masked landmark residuals."""
    squared = (residual ** 2).sum(dim=-1) * mask
    return torch.sqrt(squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).numpy()


def plausible_radius(num_betas: int, plausibility: float = SHAPE_PLAUSIBILITY) -> float:
    """The |betas| bound holding `plausibility` of the shape prior N(0, I)."""
    return float(np.sqrt(chi2.ppf(plausibility, num_betas)))


def _optimise(model, observed, valid, start, betas=None, shape=None,
              plausibility=SHAPE_PLAUSIBILITY):
    """
    L-BFGS through the real forward pass on the wrist-relative landmark error (mm).

    The loss is a sum over frames, so it separates by frame unless the frames share a
    fitted shape. The iterates do not: one run shares a single line search (one step length
    per iteration) and one curvature history across all the frames it is given, so a
    frame's result depends on the frames it is batched with. In every run checked the
    optimiser stopped at its iteration cap, not at a tolerance, so a result is the iterate
    at the cap rather than a certified optimum: judge a fit by the returned `rms_mm`.
    (docs/V1_2_Findings.md section 4.1 and docs/V1_2_Implementation_Evidence.md section 2
    give the measurements.)

    Args:
        observed, valid: (F, 21, 3) and (F, 21) for the frames being fitted.
        start: (F, 48) starting wrist_orient and hand_pose.
        betas: (K,) fixed shape, used when `shape` is None.
        shape: None, "shared" (one fitted shape for all frames) or "per_frame" (one
            fitted shape per frame). Fitted shapes stay inside |betas| <= plausible_radius
            by construction: betas = r u / sqrt(1 + |u|^2) for unconstrained u.

    Returns (params (F, 48), betas (1 or F, K), rms_mm (F,)).
    """
    kept = np.where(valid[..., None], observed, 0.0)
    target = torch.as_tensor(1000.0 * (kept - kept[:, :1]), dtype=FIT_DTYPE)
    mask = torch.tensor(valid)
    mask[:, WRIST] = False  # zero by construction
    params = torch.tensor(start, dtype=FIT_DTYPE, requires_grad=True)

    if shape is None:
        fixed = torch.as_tensor(betas, dtype=FIT_DTYPE).reshape(1, -1)
        variables, iterations = [params], MAX_ITERATIONS

        def current_betas():
            return fixed
    else:
        radius = plausible_radius(model.num_betas, plausibility)
        rows = len(start) if shape == "per_frame" else 1
        free = torch.zeros(rows, model.num_betas, dtype=FIT_DTYPE, requires_grad=True)
        variables, iterations = [params, free], SHAPE_ITERATIONS

        def current_betas():
            return radius * free / torch.sqrt(1.0 + (free ** 2).sum(dim=1, keepdim=True))

    optimiser = torch.optim.LBFGS(
        variables, lr=1.0, max_iter=iterations, tolerance_grad=1e-12,
        tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe",
    )

    def residuals():
        points = 1000.0 * model.points(params[:, :3], params[:, 3:], current_betas())
        return (points - points[:, :1]) - target

    def closure():
        optimiser.zero_grad()
        loss = (residuals()[mask] ** 2).sum() + POSE_PRIOR * (params[:, 3:] ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        fitted = params.detach().numpy().copy()
        return fitted, current_betas().numpy().copy(), _rms_mm(residuals(), mask)


def fit_hands(observed, valid, model, betas=None) -> List[HandFit]:
    """
    Fit every frame of one hand track, each with its own pose, at one fixed shape.

    The frames are optimised in batches of FRAMES_PER_BATCH that share one L-BFGS run (see
    `_optimise`). Each frame has its own parameters and its own loss term, but a frame's
    result depends on the other frames of its batch: fitted alone, or in another batch, it
    comes out different (by up to 0.181 mm in a landmark's position on synthetic hands;
    docs/V1_2_Implementation_Evidence.md section 2).

    Status "ok" only means that the fit is finite. The optimiser stops at its iteration cap
    rather than at a tolerance, so "ok" does not certify convergence; each fit's `rms_mm` is
    the quality signal.

    Args:
        observed: (T, 21, 3) world landmarks of one hand, metres.
        valid: (T, 21) bool; invalid or non-finite landmarks never influence a fit.
        model: `HandModel` of the hand's side.
        betas: (K,) shape used for every frame; neutral when omitted.

    Returns one `HandFit` per frame, in order.
    """
    observed, valid = _prepared(observed, valid)
    betas = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
    skeleton = model.skeleton(betas)

    fits: List[Optional[HandFit]] = [None] * len(observed)
    starts = {}
    for t in range(len(observed)):
        start = initial_pose(observed[t], valid[t], skeleton)
        if start is None:
            fits[t] = _unfitted(model, "no_palm" if valid[t, WRIST] else "no_wrist")
        else:
            starts[t] = np.concatenate(start)

    frames = sorted(starts)
    for first in range(0, len(frames), FRAMES_PER_BATCH):
        batch = frames[first:first + FRAMES_PER_BATCH]
        params, _, rms = _optimise(
            model, observed[batch], valid[batch], np.stack([starts[t] for t in batch]), betas
        )
        for k, t in enumerate(batch):
            if not (np.isfinite(rms[k]) and np.isfinite(params[k]).all()):
                fits[t] = _unfitted(model, "diverged")
                continue
            fits[t] = HandFit(
                model.backend, model.side, params[k, :3], params[k, 3:], betas.copy(),
                observed[t, WRIST] - skeleton.rest_joints[WRIST], "ok", float(rms[k]),
            )
    return fits


def reconstruct_hand(fit: HandFit, model) -> np.ndarray:
    """
    Rebuild the 21 world landmarks from a fit's parameters alone.

    Returns (21, 3) metres; all NaN when the frame has no fit.

    Raises:
        ValueError: `model` is a different backend or side from the fit's.
    """
    if (fit.backend, fit.side) != (model.backend, model.side):
        raise ValueError(
            f"fit is {fit.backend}/{fit.side} but the model is {model.backend}/{model.side}"
        )
    if fit.status != "ok":
        return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
    with torch.no_grad():
        points = model.points(fit.wrist_orient, fit.hand_pose, fit.betas)[0].numpy()
    return points + fit.transl


def save_hand_fits(path, fits: Sequence[HandFit]) -> None:
    """Write one hand track's fits to an .npz; every fit must share backend and side."""
    tracks = {(f.backend, f.side) for f in fits}
    if len(tracks) != 1:
        raise ValueError(f"one hand track per file, got {sorted(tracks)}")
    (backend, side), = tracks
    np.savez_compressed(
        path,
        backend=np.array(backend),
        side=np.array(side),
        wrist_orient=np.stack([f.wrist_orient for f in fits]),
        hand_pose=np.stack([f.hand_pose for f in fits]),
        betas=np.stack([f.betas for f in fits]),
        transl=np.stack([f.transl for f in fits]),
        status=np.array([f.status for f in fits]),
        rms_mm=np.array([f.rms_mm for f in fits], dtype=np.float64),
    )


def load_hand_fits(path) -> List[HandFit]:
    """Read fits written by `save_hand_fits`."""
    # An NpzFile reads and decompresses a member afresh on every access, so each column is
    # read once here and the fits are views of it.
    with np.load(path) as data:
        columns = {key: data[key] for key in data.files}
    backend, side = str(columns["backend"]), str(columns["side"])
    return [
        HandFit(
            backend,
            side,
            columns["wrist_orient"][t],
            columns["hand_pose"][t],
            columns["betas"][t],
            columns["transl"][t],
            str(columns["status"][t]),
            float(columns["rms_mm"][t]),
        )
        for t in range(len(columns["status"]))
    ]


def _starts_from(fits, frames) -> np.ndarray:
    return np.stack([np.concatenate([fits[t].wrist_orient, fits[t].hand_pose]) for t in frames])


def fit_hand_shape(observed, valid, model, plausibility=SHAPE_PLAUSIBILITY) -> np.ndarray:
    """
    One shape for a whole hand track: the plausible betas minimising the summed
    landmark error of every frame, fitted jointly with the frames' poses (started
    from neutral-shape fits). Frames the neutral fit cannot place are left out.

    Returns (K,) betas; neutral when no frame can be fitted.
    """
    observed, valid = _prepared(observed, valid)
    neutral = fit_hands(observed, valid, model)
    frames = [t for t, fit in enumerate(neutral) if fit.status == "ok"]
    if not frames:
        return np.zeros(model.num_betas)
    _, betas, _ = _optimise(model, observed[frames], valid[frames], _starts_from(neutral, frames),
                            shape="shared", plausibility=plausibility)
    return betas[0]


def fit_hands_per_frame_shape(observed, valid, model,
                              plausibility=SHAPE_PLAUSIBILITY) -> List[HandFit]:
    """
    Diagnostic, not the pipeline's contract: like `fit_hands`, but every frame takes
    its own plausible shape. Comparing it with a per-track shape measures how much
    of the round-trip error is the detector's frame-to-frame change of hand size.

    Its shape stage runs one L-BFGS over every frame of the track that the neutral fit could
    place, not in batches of FRAMES_PER_BATCH, with a fixed iteration budget
    (SHAPE_ITERATIONS). The frames' losses are separate but their iterates are not (see
    `_optimise`), and the budget does not grow with the track, so the accuracy it reaches
    falls as the track gets longer: on synthetic MANO tracks the mean landmark error rose
    from 0.100 mm at 8 frames to 0.174 mm at 250 (docs/V1_2_Implementation_Evidence.md
    section 4). A gap between this and a per-track shape that is smaller than that floor
    says nothing about changes of hand size.
    """
    observed, valid = _prepared(observed, valid)
    fits = fit_hands(observed, valid, model)
    frames = [t for t, fit in enumerate(fits) if fit.status == "ok"]
    if not frames:
        return fits
    params, betas, rms = _optimise(model, observed[frames], valid[frames],
                                   _starts_from(fits, frames), shape="per_frame",
                                   plausibility=plausibility)
    for k, t in enumerate(frames):
        if not (np.isfinite(rms[k]) and np.isfinite(params[k]).all()):
            fits[t] = _unfitted(model, "diverged")
            continue
        wrist = model.rest_points(betas[k])[WRIST]
        fits[t] = HandFit(model.backend, model.side, params[k, :3], params[k, 3:], betas[k],
                          observed[t, WRIST] - wrist, "ok", float(rms[k]))
    return fits
