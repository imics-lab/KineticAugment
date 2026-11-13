#!/usr/bin/env python3
"""
Reproduce the Step 10 evidence: why the shape fit is built the way it is, and
what it can and cannot recover.

  S0  The quadratic the fit minimises is the converter's own squared error.
  S1  SMPL-X's shape prior: population spread under betas ~ N(0, I), and how far
      each clip's MediaPipe proportions sit from it.
  S2  Identifiability: singular values of the five arm proportions' Jacobian.
  S3  The ladder, fitted and held out: template -> fitted shape -> best symmetric
      skeleton -> rigid-skeleton floor.
  S4  Plausibility sweep: the price of insisting on a human body.
  S5  Shape-space size: 10 coefficients against SMPL-X's full 300.
  S6  Objective: the round-trip error itself against matching median proportions.
  S7  An alternative V1 rejects: HybrIK-style adaptive IK, which trades exact bone
      directions for position error.

Every error is the converter's own direct round-trip error, as a pooled median
fraction of shoulder width. "Held out" fits shape on one contiguous half of each
clip and scores the other. See docs/V1_Problem_Scope.md, Step 10.

Usage:
    python scripts/extract_world_landmarks.py   # once
    python scripts/study_shape_fit.py
"""

import glob
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from measure_round_trip import (  # noqa: E402
    L_SHOULDER, R_SHOULDER, clip_median_lengths, errors, load_clip, pooled, with_arm_geometry)

from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body  # noqa: E402
from kinetic_augment.body_model.kinematics import forward_kinematics  # noqa: E402
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, swing_rotation  # noqa: E402
from kinetic_augment.body_model.shape import (  # noqa: E402
    ARM_PROPORTIONS, arm_proportions, fit_shape, round_trip_quadratic)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

NAMES = ["width", "L upper", "L fore", "R upper", "R fore"]


def clips():
    out = []
    for path in sorted(glob.glob("data/test_output/*_world.npz")):
        pose, ok = load_clip(path)
        both = ok[:, L_SHOULDER] & ok[:, R_SHOULDER]
        width = float(np.median(np.linalg.norm(pose[both, L_SHOULDER] - pose[both, R_SHOULDER],
                                               axis=-1)))
        out.append((Path(path).stem[:5], pose, ok, width))
    return out


def median_width(pose, ok):
    both = ok[:, L_SHOULDER] & ok[:, R_SHOULDER]
    return float(np.median(np.linalg.norm(pose[both, L_SHOULDER] - pose[both, R_SHOULDER],
                                          axis=-1)))


def score(data, template, make_body, held_out):
    """Pooled median error. make_body(pose, ok) -> (fit skeleton, reconstruction skeleton)."""
    found = []
    for _, pose, ok, width in data:
        if not held_out:
            fit_with, rebuild_with = make_body(pose, ok)
            found.append(pooled(errors(pose, ok, fit_with, rebuild_with, width)[0]))
            continue
        half = len(pose) // 2
        for train, test in ((slice(0, half), slice(half, None)),
                            (slice(half, None), slice(0, half))):
            fit_with, rebuild_with = make_body(pose[train], ok[train])
            found.append(pooled(errors(pose[test], ok[test], fit_with, rebuild_with, width)[0]))
    return 100 * np.median(np.concatenate(found))


def betas_body(skeleton, **kwargs):
    def make(pose, ok):
        return skeleton.shaped(fit_shape(pose, ok, skeleton, **kwargs)), skeleton
    return make


def floor_body(template, symmetric=False):
    def make(pose, ok):
        lengths = np.array(clip_median_lengths(pose, ok))
        if symmetric:   # SMPLX_ARM_BONES order: L upper, L fore, R upper, R fore
            lengths = np.tile(0.5 * (lengths[:2] + lengths[2:]), 2)
        body = with_arm_geometry(template, lengths, median_width(pose, ok))
        return body, body
    return make


def proxy_body(skeleton, plausibility=0.95):
    """Matches median proportions instead of minimising round-trip error (S6)."""
    pairs = [smplx_pair for smplx_pair, _ in ARM_PROPORTIONS]
    offsets = np.array([skeleton.joint_template[b] - skeleton.joint_template[a]
                        for a, b in pairs])
    bases = np.array([skeleton.joint_shapedirs[b] - skeleton.joint_shapedirs[a]
                      for a, b in pairs])
    radius = np.sqrt(chi2.ppf(plausibility, 5))

    def make(pose, ok):
        lengths = clip_median_lengths(pose, ok)
        target = np.array([median_width(pose, ok)] + list(lengths))

        def objective(betas):
            bones = offsets + bases @ betas
            norms = np.linalg.norm(bones, axis=1)
            jac = np.einsum("pc,pck->pk", bones / norms[:, None], bases)
            return np.sum((norms - target) ** 2), 2 * jac.T @ (norms - target)

        betas = minimize(objective, np.zeros(skeleton.num_betas), jac=True, method="SLSQP",
                         constraints=[{"type": "ineq", "fun": lambda b: radius ** 2 - b @ b,
                                       "jac": lambda b: -2 * b}],
                         options={"maxiter": 1000, "ftol": 1e-15}).x
        return skeleton.shaped(betas), skeleton
    return make


def adaptive_ik(frame, ok, body, iterations):
    """
    HybrIK's adaptive IK on top of fit_body: each bone aims from its *reconstructed*
    parent at the observed child, then translation is refitted. Positions improve,
    but a bone no longer points along its observed direction.
    """
    fit = fit_body(frame, ok, body)
    rotations = np.zeros((body.num_joints, 3))
    rotations[0] = fit.global_orient
    rotations[1:22] = fit.body_pose.reshape(21, 3)
    local = Rotation.from_rotvec(rotations).as_matrix()
    transl, parents, rest = fit.transl, body.parents, body.rest_joints

    def world(local):
        out = np.empty_like(local)
        for j, p in enumerate(parents):
            out[j] = local[j] if p < 0 else out[p] @ local[j]
        return out

    def posed(local):
        return forward_kinematics(Rotation.from_matrix(local).as_rotvec(), body)

    for _ in range(iterations):
        for joint, child, _, mp_child in SMPLX_ARM_BONES:
            if not fit.placed[mp_child]:
                continue
            target = frame[mp_child] - (posed(local)[joint] + transl)
            d_rest = rest[child] - rest[joint]
            local[joint] = swing_rotation(d_rest / np.linalg.norm(d_rest),
                                          world(local)[parents[joint]].T
                                          @ (target / np.linalg.norm(target)))
        joints = posed(local)
        transl = np.mean([frame[m] - joints[s] for s, m in ARM_CORRESPONDENCE
                          if fit.placed[m]], axis=0)

    out = np.full((33, 3), np.nan)
    joints = posed(local) + transl
    for s, m in ARM_CORRESPONDENCE:
        if fit.placed[m]:
            out[m] = joints[s]
    return out


def adaptive_scores(data, template, iterations):
    """Pooled median position error (% SW) and bone-direction error (degrees)."""
    position, direction = [], []
    for _, pose, ok, width in data:
        body = template.shaped(fit_shape(pose, ok, template))
        for frame, valid in zip(pose, ok):
            if not (valid[L_SHOULDER] and valid[R_SHOULDER]):
                continue
            out = adaptive_ik(frame, valid, body, iterations)
            position += [np.linalg.norm(out[m] - frame[m]) / width for m in range(11, 17)
                         if valid[m] and np.isfinite(out[m]).all()]
            for _, _, a, b in SMPLX_ARM_BONES:
                if np.isfinite(out[[a, b]]).all() and valid[a] and valid[b]:
                    u, v = out[b] - out[a], frame[b] - frame[a]
                    cos = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
                    direction.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return 100 * np.median(position), np.median(direction), np.percentile(direction, 95)


def rest_height(skeleton, betas):
    v = (skeleton.model.v_template.detach().numpy().astype(np.float64)
         + skeleton.model.shapedirs.detach().numpy().astype(np.float64) @ betas)
    return v[:, 1].max() - v[:, 1].min()


def main():
    template = SMPLXSkeleton(model_path="models")
    data = clips()
    rng = np.random.default_rng(0)

    print("S0  quadratic vs the converter's summed squared error, template body")
    worst = 0.0
    for name, pose, ok, _ in data:
        found, _ = errors(pose, ok, template, template, 1.0)
        sse = float(np.sum(pooled(found) ** 2))
        model = round_trip_quadratic(pose, ok).error(arm_proportions(template.rest_joints))
        worst = max(worst, abs(model - sse) / sse)
    print(f"    worst relative difference over {len(data)} clips: {worst:.1e}\n")

    print("S1  SMPL-X's shape prior (betas ~ N(0, I), all 300 coefficients)")
    draws = rng.normal(size=(2000, template.num_betas))
    pop = np.array([arm_proportions(template.shaped(b).rest_joints) for b in draws])
    ratio = (pop[:, 2] + pop[:, 4]) / (pop[:, 1] + pop[:, 3])
    asym = pop[:, 1] - pop[:, 3]
    heights = [rest_height(template, b) for b in draws[:200]]
    print(f"    rest height {np.mean(heights):.3f} m, sd {100 * np.std(heights):.1f} cm "
          f"(a plausible adult population if betas are unit-variance)")
    print(f"    forearm/upper-arm ratio {ratio.mean():.3f} +- {ratio.std():.3f}; "
          f"upper-arm L-R difference {1000 * asym.mean():.1f} +- {1000 * asym.std():.1f} mm")
    clip_props = np.array([[median_width(pose, ok)] + list(clip_median_lengths(pose, ok))
                           for _, pose, ok, _ in data])
    print("    spread (sd / mean) of " + ", ".join(NAMES) + ":")
    print(f"      SMPL-X population {np.round(100 * pop.std(0) / pop.mean(0), 1)} %")
    print(f"      across the clips  "
          f"{np.round(100 * clip_props.std(0) / clip_props.mean(0), 1)} %")
    print(f"    {'clip':<7}{'fore/upper':>11}{'z':>7}{'upper L-R (mm)':>16}{'z':>7}")
    for name, pose, ok, _ in data:
        lu, lf, ru, rf = clip_median_lengths(pose, ok)
        r, d = (lf + rf) / (lu + ru), lu - ru
        print(f"    {name:<7}{r:>11.3f}{(r - ratio.mean()) / ratio.std():>7.1f}"
              f"{1000 * d:>16.1f}{(d - asym.mean()) / asym.std():>7.1f}")

    print("\nS2  identifiability: proportion Jacobian at the template (mm per unit beta)")
    for k in (10, 300):
        J = []
        for (a, b), _ in ARM_PROPORTIONS:
            bone = template.joint_template[b] - template.joint_template[a]
            J.append((template.joint_shapedirs[b, :, :k] - template.joint_shapedirs[a, :, :k]).T
                     @ bone / np.linalg.norm(bone))
        J = np.array(J)
        cos = [J[i] @ J[j] / np.linalg.norm(J[i]) / np.linalg.norm(J[j])
               for i, j in ((1, 3), (2, 4))]
        print(f"    {k:>3} betas: singular values {np.round(1000 * np.linalg.svd(J)[1], 3)}; "
              f"L-R row cosines upper {cos[0]:.4f}, fore {cos[1]:.4f}")

    def both(make):
        return score(data, template, make, False), score(data, template, make, True)

    print("\nS3  the ladder                          fitted   held out")
    small = SMPLXSkeleton(model_path="models", num_betas=10)
    rows = [("template body (betas = 0)", lambda p, o: (template, template)),
            ("shape fit, 10 betas", betas_body(small)),
            ("shape fit, 300 betas  <- V1", betas_body(template)),
            ("best symmetric skeleton (no betas)", floor_body(template, symmetric=True)),
            ("rigid-skeleton floor (no betas)", floor_body(template))]
    for label, make in rows:
        fitted, held = both(make)
        print(f"    {label:<36}{fitted:>6.2f}%{held:>9.2f}%")

    print("\nS4  plausibility sweep, 300 betas      radius   fitted   held out   heights (m)")
    for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0):
        radius = np.sqrt(chi2.ppf(q, 5))
        fitted, held = both(betas_body(template, plausibility=q))
        h = [rest_height(template, fit_shape(pose, ok, template, plausibility=q))
             for _, pose, ok, _ in data]
        print(f"    {q:<36}{radius:>6.2f}{fitted:>8.2f}%{held:>9.2f}%   "
              f"{min(h):.2f}-{max(h):.2f}")
    unbounded = [fit_shape(pose, ok, small, plausibility=1.0) for _, pose, ok, _ in data]
    h = [rest_height(small, b) for b in unbounded]
    print(f"    10 betas, unbounded: |betas| up to {max(map(np.linalg.norm, unbounded)):.0f}, "
          f"heights {min(h):.2f}-{max(h):.2f} m")

    print("\nS5  shape-space size at 0.95           fitted   held out")
    for k in (10, 20, 50, 100, 300):
        skeleton = small if k == 10 else SMPLXSkeleton(model_path="models", num_betas=k)
        fitted, held = both(betas_body(skeleton))
        print(f"    {k:<36}{fitted:>6.2f}%{held:>9.2f}%")

    print("\nS6  objective at 0.95, 300 betas       fitted   held out")
    for label, make in (("round-trip error  <- V1", betas_body(template)),
                        ("median proportions (proxy)", proxy_body(template))):
        fitted, held = both(make)
        print(f"    {label:<36}{fitted:>6.2f}%{held:>9.2f}%")

    print("\nS7  adaptive IK on the fitted shape      position   direction (median / p95)")
    for iterations in (0, 1, 3):
        label = "V1: exact directions" if iterations == 0 else f"adaptive, {iterations} pass(es)"
        pos, med, p95 = adaptive_scores(data, template, iterations)
        print(f"    {label:<36}{pos:>6.2f}%   {med:5.2f} deg / {p95:5.2f} deg")


if __name__ == "__main__":
    main()
