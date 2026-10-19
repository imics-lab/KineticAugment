"""
Per-clip SMPL-X body shape for the V1 converter.

The converter reproduces every bone direction exactly, so a skeleton enters the
round trip through five numbers only: shoulder width and the four arm bone
lengths. Laid out from the left shoulder, each modelled landmark is a sum of
those lengths along unit directions the frame itself supplies -- the left wrist
is upper arm x its direction + forearm x its direction -- and the converter
spends its translation on the mean residual. A landmark's error is therefore

    (A_m - mean A) (p - p_t)

linear in the skeleton's proportions p, where p_t are the frame's own observed
proportions. Summed over a clip, the squared round-trip error is an exact
quadratic in p, and choosing a body shape becomes choosing p.

`fit_shape` minimises that error over SMPL-X's shape space, subject to the body
staying plausible under SMPL-X's own shape distribution. The bound is a
chi-square quantile on |betas|: at the optimum, betas lie in the span the five
proportions can see, so |betas| is the Mahalanobis distance of the fitted
proportions from the population mean.

What the fit claims is bounded. Five proportions cannot identify hundreds of
coefficients: among bodies with equal proportions the most typical is returned,
which is a convention, not a measurement of the signer. And the observed
proportions carry MediaPipe's own biases, so the fitted body is the plausible
human closest to MediaPipe's skeleton -- not the signer's anatomy.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from kinetic_augment.body_model.converter import LEFT_SHOULDER_MP, RIGHT_SHOULDER_MP

# (SMPL-X joint pair, MediaPipe landmark pair) for each arm proportion.
ARM_PROPORTIONS = [
    ((16, 17), (11, 12)),  # shoulder width
    ((16, 18), (11, 13)),  # left upper arm
    ((18, 20), (13, 15)),  # left forearm
    ((17, 19), (12, 14)),  # right upper arm
    ((19, 21), (14, 16)),  # right forearm
]
WIDTH, LEFT_UPPER, LEFT_FORE, RIGHT_UPPER, RIGHT_FORE = range(5)

# The proportions summed along the path from the left shoulder to each landmark.
_CHAINS = {
    11: (),
    12: (WIDTH,),
    13: (LEFT_UPPER,),
    14: (WIDTH, RIGHT_UPPER),
    15: (LEFT_UPPER, LEFT_FORE),
    16: (WIDTH, RIGHT_UPPER, RIGHT_FORE),
}

# Accept a body whose arm proportions are no more unusual than 95% of SMPL-X's.
PLAUSIBILITY = 0.95

# Selects the most typical body among those with equal round-trip error; small
# enough not to trade any error for typicality.
_TIE_BREAK = 1e-9


@dataclass
class RoundTripQuadratic:
    """A clip's summed squared round-trip error: error(p) = p'Hp - 2b'p + c."""

    H: np.ndarray
    b: np.ndarray
    c: float
    frames: int

    def error(self, proportions: np.ndarray) -> float:
        p = np.asarray(proportions, dtype=np.float64)
        return float(p @ self.H @ p - 2.0 * self.b @ p + self.c)


def arm_proportions(rest_joints: np.ndarray) -> np.ndarray:
    """Shoulder width, then left upper arm, left forearm, right upper arm, right forearm."""
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    return np.array([np.linalg.norm(rest_joints[b] - rest_joints[a])
                     for (a, b), _ in ARM_PROPORTIONS])


def _frame_terms(frame: np.ndarray, usable: np.ndarray):
    """One frame's error matrix and observed proportions."""
    observed = np.zeros(len(ARM_PROPORTIONS))
    directions = np.zeros((len(ARM_PROPORTIONS), 3))
    seen = np.zeros(len(ARM_PROPORTIONS), dtype=bool)
    for i, (_, (a, b)) in enumerate(ARM_PROPORTIONS):
        if usable[a] and usable[b]:
            length = np.linalg.norm(frame[b] - frame[a])
            if length > 0:
                observed[i], directions[i], seen[i] = length, (frame[b] - frame[a]) / length, True

    # A landmark whose path from the shoulder is not fully observed is left out:
    # the converter cannot solve the bones that place it.
    rows = []
    for landmark, chain in _CHAINS.items():
        if usable[landmark] and all(seen[i] for i in chain):
            row = np.zeros((3, len(ARM_PROPORTIONS)))
            for i in chain:
                row[:, i] = directions[i]
            rows.append(row)

    centred = np.array(rows) - np.mean(rows, axis=0)
    return np.einsum("mci,mcj->ij", centred, centred), observed


def round_trip_quadratic(landmarks: np.ndarray, validity: np.ndarray) -> RoundTripQuadratic:
    """
    The converter's summed squared round-trip error over a clip, as an exact
    quadratic in the skeleton's arm proportions.

    Args:
        landmarks: (T, 33, 3) MediaPipe world landmarks, metres.
        validity: (T, 33) bool. Frames without both shoulders cannot be fitted and
            are skipped, as the converter refuses them.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    usable = np.asarray(validity, dtype=bool) & np.isfinite(landmarks).all(axis=-1)

    size = len(ARM_PROPORTIONS)
    H, b, c, frames = np.zeros((size, size)), np.zeros(size), 0.0, 0
    for frame, ok in zip(landmarks, usable):
        if not (ok[LEFT_SHOULDER_MP] and ok[RIGHT_SHOULDER_MP]):
            continue
        H_t, p_t = _frame_terms(frame, ok)
        H, b, c, frames = H + H_t, b + H_t @ p_t, c + p_t @ H_t @ p_t, frames + 1
    return RoundTripQuadratic(H=H, b=b, c=c, frames=frames)


def fit_shape(landmarks: np.ndarray, validity: np.ndarray, skeleton,
              plausibility: float = PLAUSIBILITY) -> np.ndarray:
    """
    One body shape for a clip: the betas with the lowest round-trip error whose
    arm proportions are no more unusual than `plausibility` of SMPL-X's bodies.

    Args:
        landmarks: (T, 33, 3) MediaPipe world landmarks, metres.
        validity: (T, 33) bool; invalid landmarks never influence the shape.
        skeleton: `SMPLXSkeleton`; its shape space sets how many betas are fitted.
        plausibility: quantile of SMPL-X's shape distribution the fitted body must
            fall within. 1.0 removes the bound, for diagnostics only: the fit then
            matches MediaPipe's proportions with bodies no human has.

    Returns:
        (skeleton.num_betas,) betas; zeros when no frame can be fitted.
    """
    quadratic = round_trip_quadratic(landmarks, validity)
    observable = int(np.count_nonzero(np.diag(quadratic.H) > 0))
    num_betas = skeleton.num_betas
    if quadratic.frames == 0 or observable == 0:
        return np.zeros(num_betas)

    pairs = [smplx_pair for smplx_pair, _ in ARM_PROPORTIONS]
    offsets = np.array([skeleton.joint_template[b] - skeleton.joint_template[a]
                        for a, b in pairs])
    bases = np.array([skeleton.joint_shapedirs[b] - skeleton.joint_shapedirs[a]
                      for a, b in pairs])
    scale = max(quadratic.error(arm_proportions(skeleton.joint_template)), 1e-300)

    def objective(betas):
        bones = offsets + bases @ betas
        lengths = np.linalg.norm(bones, axis=1)
        jacobian = np.einsum("pc,pck->pk", bones / lengths[:, None], bases)
        value = quadratic.error(lengths) / scale + _TIE_BREAK * betas @ betas
        gradient = (2.0 * jacobian.T @ (quadratic.H @ lengths - quadratic.b) / scale
                    + 2.0 * _TIE_BREAK * betas)
        return value, gradient

    # The fitted betas span at most as many directions as there are observed
    # proportions, so that is the chi-square's number of degrees of freedom.
    radius = np.sqrt(chi2.ppf(plausibility, df=min(observable, num_betas)))
    constraints = [] if not np.isfinite(radius) else [{
        "type": "ineq",
        "fun": lambda betas: radius ** 2 - betas @ betas,
        "jac": lambda betas: -2.0 * betas,
    }]

    result = minimize(objective, np.zeros(num_betas), jac=True, method="SLSQP",
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-15})
    betas = result.x
    norm = np.linalg.norm(betas)
    if np.isfinite(radius) and norm > radius:
        betas = betas * (radius / norm)
    return betas
