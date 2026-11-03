"""
Tests for the V1.2 hand converter (plan P1-P2).

Synthetic hands come from a skeleton or the real model with known parameters,
so every target is exactly representable and any error is the converter's.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# The converter imports hand_model, which needs torch and smplx: skip cleanly without them.
torch = pytest.importorskip("torch")
smplx = pytest.importorskip("smplx")

from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape,
    fit_hands,
    fit_hands_per_frame_shape,
    initial_pose,
    landmark_errors_mm,
    load_hand_fits,
    node_rotations,
    normalise_palm_size,
    palm_rotation,
    palm_sizes,
    plausible_radius,
    reconstruct_hand,
    save_hand_fits,
)
from kinetic_augment.body_model.hand_model import HAND_PARENTS, TIPS  # noqa: E402
from kinetic_augment.body_model.kinematics import forward_kinematics  # noqa: E402
from tests.hand_assets import (  # noqa: E402
    BACKENDS,
    MODEL_DIR,
    SIDES,
    hand,
    requires_mano,
    training_poses,
)

# A hand-shaped rest skeleton in metres -- right hand, fingers along +y -- so the
# analytic start can be tested without licensed model assets.
SYNTHETIC_REST = np.array([
    [0.000, 0.000, 0.000],
    [-0.025, 0.020, 0.005], [-0.045, 0.045, 0.008], [-0.058, 0.070, 0.010], [-0.068, 0.092, 0.012],
    [-0.020, 0.085, 0.000], [-0.022, 0.125, 0.000], [-0.023, 0.150, 0.000], [-0.024, 0.172, 0.000],
    [0.000, 0.090, 0.000], [0.000, 0.135, 0.000], [0.000, 0.163, 0.000], [0.000, 0.187, 0.000],
    [0.018, 0.085, 0.000], [0.020, 0.127, 0.000], [0.021, 0.153, 0.000], [0.022, 0.175, 0.000],
    [0.034, 0.075, 0.000], [0.037, 0.105, 0.000], [0.039, 0.125, 0.000], [0.040, 0.143, 0.000],
])
SYNTHETIC = SimpleNamespace(rest_joints=SYNTHETIC_REST, parents=HAND_PARENTS, num_joints=21)
ALL = np.ones(21, dtype=bool)


def posed_synthetic(seed):
    """The synthetic skeleton under random rotations -- twist included -- and a random offset."""
    rng = np.random.default_rng(seed)
    wrist = Rotation.random(random_state=seed).as_rotvec()
    pose = rng.normal(0.0, 0.5, 45)
    return forward_kinematics(node_rotations(wrist, pose), SYNTHETIC) + rng.normal(0.0, 0.1, 3)


class TestPalmRotation:
    def test_recovers_a_known_rotation(self):
        rest = SYNTHETIC_REST - SYNTHETIC_REST[0]
        truth = Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()
        np.testing.assert_allclose(palm_rotation(rest, rest @ truth.T, ALL), truth, atol=1e-12)

    def test_needs_two_palm_points_off_one_line(self):
        rest = SYNTHETIC_REST - SYNTHETIC_REST[0]
        one = np.zeros(21, dtype=bool)
        one[[0, 5]] = True
        assert palm_rotation(rest, rest, one) is None
        line = rest.copy()
        line[[1, 5, 9, 13, 17]] = np.outer([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.02, 0.0])
        assert palm_rotation(line, line, ALL) is None


class TestInitialPose:
    def test_is_exact_on_a_rigid_skeleton(self):
        for seed in range(5):
            observed = posed_synthetic(seed)
            wrist, pose = initial_pose(observed, ALL, SYNTHETIC)
            rebuilt = forward_kinematics(node_rotations(wrist, pose), SYNTHETIC)
            np.testing.assert_allclose(rebuilt - rebuilt[0], observed - observed[0], atol=1e-12)

    def test_invalid_landmarks_never_influence_it(self):
        observed = posed_synthetic(7)
        valid = ALL.copy()
        valid[[8, 14]] = False
        corrupted = observed.copy()
        corrupted[[8, 14]] += 0.05
        a, b = initial_pose(observed, valid, SYNTHETIC), initial_pose(corrupted, valid, SYNTHETIC)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_without_a_wrist_or_a_palm_there_is_no_start(self):
        observed = posed_synthetic(8)
        no_wrist = ALL.copy()
        no_wrist[0] = False
        assert initial_pose(observed, no_wrist, SYNTHETIC) is None
        no_palm = ALL.copy()
        no_palm[[1, 5, 9, 13, 17]] = False
        assert initial_pose(observed, no_palm, SYNTHETIC) is None


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_on_the_real_model_joints_are_exact_and_tips_close(backend, side):
    """
    Joints are pure forward kinematics, so the swing start reproduces them exactly. Tips
    are mesh vertices moved by pose correctives the start ignores (MANO probe: up to
    11.8 mm over its 1554 training poses), so they are only close.
    """
    model = hand(backend, side)
    rng = np.random.default_rng(11)
    wrist = Rotation.random(4, random_state=11).as_rotvec()
    pose = rng.normal(0.0, 0.3, (4, 45))
    observed = model.points(wrist, pose, np.zeros(10)).detach().numpy()
    skeleton = model.skeleton()
    joints = [i for i in range(21) if i not in TIPS]
    for frame in observed:
        start = initial_pose(frame, ALL, skeleton)
        rebuilt = model.points(start[0], start[1], np.zeros(10))[0].detach().numpy()
        gap_mm = 1000.0 * np.linalg.norm(rebuilt - frame, axis=1)
        assert gap_mm[joints].max() < 1e-4  # exact; smplx's Rodrigues adds 1e-8 rad
        assert gap_mm[list(TIPS)].max() < 10.0


SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


def synthetic_track(model, count, seed, spread=0.3, betas=None, poses=None):
    """
    Model-generated landmarks with known parameters, offset the way hand-world origins
    are. Finger poses are random (N(0, spread) per value) unless `poses` is given.
    """
    rng = np.random.default_rng(seed)
    wrist = Rotation.random(count, random_state=seed).as_rotvec()
    pose = rng.normal(0.0, spread, (count, 45)) if poses is None else np.asarray(poses)
    shape = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=float)
    points = model.points(wrist, pose, shape).detach().numpy()
    return points + rng.normal(0.0, 0.05, (count, 1, 3))


def rebuild(fits, model):
    return np.stack([reconstruct_hand(f, model) for f in fits])


def all_valid(observed):
    return np.ones(observed.shape[:2], dtype=bool)


class TestLandmarkErrors:
    def test_scores_wrist_relative_distances_in_millimetres(self):
        observed = np.zeros((1, 21, 3))
        predicted = observed.copy()
        predicted[0, :, 0] += 0.5             # a pure translation is not an error
        predicted[0, 8, 1] += 0.003           # 3 mm at the index tip
        errors = landmark_errors_mm(predicted, observed)
        assert errors.shape == (1, 20)
        np.testing.assert_allclose(errors[0, 7], 3.0)
        np.testing.assert_allclose(np.delete(errors[0], 7), 0.0, atol=1e-12)


@pytest.mark.parametrize("poses", ["random", pytest.param("training", marks=requires_mano)])
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_recovers_hands_the_model_can_represent(backend, side, poses):
    """Plan Section 6 synthetic gate: mean <= 0.5 mm, p95 <= 1 mm."""
    model = hand(backend, side)
    real = training_poses(side, 12, seed=20) if poses == "training" else None
    observed = synthetic_track(model, 12, seed=21, poses=real)
    fits = fit_hands(observed, all_valid(observed), model)
    assert [f.status for f in fits] == ["ok"] * 12
    rebuilt = rebuild(fits, model)
    errors = landmark_errors_mm(rebuilt, observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0
    np.testing.assert_allclose(rebuilt[:, 0], observed[:, 0], atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rotating_the_observations_rotates_the_reconstruction(backend):
    model = hand(backend, "right")
    observed = synthetic_track(model, 3, seed=22)
    turn = Rotation.from_rotvec([0.4, -0.9, 1.3]).as_matrix()
    plain = rebuild(fit_hands(observed, all_valid(observed), model), model)
    turned = rebuild(fit_hands(observed @ turn.T, all_valid(observed), model), model)
    # Both runs converge to the same hand to well under 0.1 mm; a frame-convention bug
    # would differ by centimetres.
    np.testing.assert_allclose(turned, plain @ turn.T, atol=1e-4)


@requires_mano
def test_invalid_landmarks_never_influence_the_fit():
    model = hand("mano", "right")
    observed = synthetic_track(model, 2, seed=23)
    valid = all_valid(observed)
    valid[:, [8, 16]] = False
    corrupted = observed.copy()
    corrupted[:, [8, 16]] += 0.03
    for a, b in zip(fit_hands(observed, valid, model), fit_hands(corrupted, valid, model)):
        np.testing.assert_array_equal(a.wrist_orient, b.wrist_orient)
        np.testing.assert_array_equal(a.hand_pose, b.hand_pose)


@requires_mano
def test_frames_it_cannot_fit_are_reported_not_invented():
    model = hand("mano", "right")
    observed = synthetic_track(model, 3, seed=24)
    valid = all_valid(observed)
    valid[1, [1, 5, 9, 13, 17]] = False
    valid[2, 0] = False
    fits = fit_hands(observed, valid, model)
    assert [f.status for f in fits] == ["ok", "no_palm", "no_wrist"]
    assert np.isnan(reconstruct_hand(fits[1], model)).all()
    assert np.isnan(fits[2].hand_pose).all()


@requires_mano
def test_reconstruction_refuses_a_model_of_the_other_side():
    model = hand("mano", "right")
    observed = synthetic_track(model, 1, seed=25)
    fit = fit_hands(observed, all_valid(observed), model)[0]
    with pytest.raises(ValueError):
        reconstruct_hand(fit, hand("mano", "left"))


@requires_mano
def test_a_fresh_process_rebuilds_the_hands_from_saved_parameters(tmp_path):
    """Plan A008: the decoder needs only parameters and the model -- no landmarks, no cache."""
    model = hand("mano", "right")
    observed = synthetic_track(model, 3, seed=26)
    fits = fit_hands(observed, all_valid(observed), model)
    saved, out = tmp_path / "fits.npz", tmp_path / "rebuilt.npy"
    save_hand_fits(saved, fits)
    code = (
        "import numpy as np\n"
        "from kinetic_augment.body_model.hand_converter import load_hand_fits, reconstruct_hand\n"
        "from kinetic_augment.body_model.hand_model import HandModel\n"
        f"model = HandModel('mano', 'right', model_dir={MODEL_DIR!r})\n"
        f"fits = load_hand_fits({str(saved)!r})\n"
        f"np.save({str(out)!r}, np.stack([reconstruct_hand(f, model) for f in fits]))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC_DIR, os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    np.testing.assert_allclose(np.load(out), rebuild(fits, model), atol=1e-12)
    assert [f.status for f in load_hand_fits(saved)] == [f.status for f in fits]


@requires_mano
def test_a_track_with_gaps_round_trips_and_loads_into_shared_columns(tmp_path):
    """
    Unfitted frames are saved as NaN rows with their status and come back as they were.
    Loading reads each column once, so the fits are views of one array per column rather
    than T private copies of every column.
    """
    model = hand("mano", "right")
    observed = synthetic_track(model, 4, seed=27)
    valid = all_valid(observed)
    valid[1, [1, 5, 9, 13, 17]] = False
    valid[3, 0] = False
    fits = fit_hands(observed, valid, model)
    assert [f.status for f in fits] == ["ok", "no_palm", "ok", "no_wrist"]
    save_hand_fits(tmp_path / "gaps.npz", fits)
    loaded = load_hand_fits(tmp_path / "gaps.npz")
    assert [f.status for f in loaded] == [f.status for f in fits]
    assert [bool(np.isfinite(f.hand_pose).all()) for f in loaded] == [True, False, True, False]
    for before, after in zip(fits, loaded):
        assert (after.backend, after.side) == (before.backend, before.side)
        for name in ("wrist_orient", "hand_pose", "betas", "transl", "rms_mm"):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name), name)
    for name in ("wrist_orient", "hand_pose", "betas", "transl"):
        bases = [getattr(f, name).base for f in loaded]
        assert bases[0] is not None and bases[0].shape[0] == len(loaded), name
        assert all(base is bases[0] for base in bases), name


CURL_TRIPLES = [
    (a, b, c)
    for finger in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    for a, b, c in zip((0,) + finger[:-1], finger, finger[1:])
]


def curl_error_deg(fitted, observed):
    """Mean over the 15 finger joints of |bend angle (fitted) - bend angle (observed)|, degrees."""

    def bends(points):
        a, b, c = (points[list(i)] for i in zip(*CURL_TRIPLES))
        u, v = b - a, c - b
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return np.abs(bends(fitted) - bends(observed)).mean()


class TestNormalisePalmSize:
    def test_every_frame_gets_the_median_palm_size_about_its_wrist(self):
        factors = np.array([1.0, 0.75, 1.2, 1.0])
        reference = posed_synthetic(3)
        track = reference[0] + (reference - reference[0]) * factors[:, None, None]
        track = track + np.arange(4)[:, None, None] * 0.1  # each frame has its own wrist
        sizes = palm_sizes(track)
        np.testing.assert_allclose(sizes, factors * palm_sizes(reference[None])[0], atol=1e-12)
        frozen = track.copy()

        scaled, scale = normalise_palm_size(track, np.ones((4, 21), dtype=bool))

        np.testing.assert_array_equal(track, frozen)
        median = np.median(sizes)
        np.testing.assert_allclose(scale, median / sizes, atol=1e-12)
        np.testing.assert_allclose(palm_sizes(scaled), median, atol=1e-12)
        np.testing.assert_array_equal(scaled[:, 0], track[:, 0])
        common = median / palm_sizes(reference[None])[0]
        expected = (reference - reference[0]) * common
        for t in range(4):
            np.testing.assert_allclose(scaled[t] - scaled[t, 0], expected, atol=1e-12)

    def test_unmeasurable_frames_are_unchanged_and_out_of_the_median(self):
        reference = posed_synthetic(4)
        track = np.stack([reference, reference[0] + (reference - reference[0]) * 0.8,
                          reference[0] + (reference - reference[0]) * 1.3])
        valid = np.ones((3, 21), dtype=bool)
        valid[2, 9] = False
        scaled, scale = normalise_palm_size(track, valid)
        assert np.isnan(scale[2])
        np.testing.assert_array_equal(scaled[2], track[2])
        sizes = palm_sizes(track)
        np.testing.assert_allclose(scale[:2], np.mean(sizes[:2]) / sizes[:2], atol=1e-12)
        # A non-finite landmark counts as invalid, whatever `valid` says.
        broken = track.copy()
        broken[1, 17] = np.nan
        scaled, scale = normalise_palm_size(broken, np.ones((3, 21), dtype=bool))
        assert np.isnan(scale[1])
        np.testing.assert_array_equal(scaled[1], broken[1])

    def test_a_track_with_no_measurable_palm_comes_back_unchanged(self):
        track = np.stack([posed_synthetic(5), posed_synthetic(6)])
        valid = np.ones((2, 21), dtype=bool)
        valid[:, 0] = False
        scaled, scale = normalise_palm_size(track, valid)
        np.testing.assert_array_equal(scaled, track)
        assert np.isnan(scale).all()
        assert scaled is not track


# Measured on this track: 40.3 deg raw against 0.38 deg normalised. Half that gap leaves wide
# room for optimiser noise while still failing if the normalisation stops working.
MARGIN_DEG = 20.0


@requires_mano
def test_normalising_palm_size_stops_the_fit_curling_fingers_to_absorb_a_size_gap():
    """
    MediaPipe's world hand changes metric size with the handshape, but one shape is fitted
    per track, so the pose fit bends the fingers to close the gap. Every other frame of a
    near-open MANO hand is shrunk 0.75x about the wrist; its true curl is the unshrunk hand's.
    """
    model = hand("mano", "right")
    truth = synthetic_track(model, 6, seed=60, spread=0.05)
    observed = truth.copy()
    observed[1::2] = truth[1::2, :1] + (truth[1::2] - truth[1::2, :1]) * 0.75
    valid = all_valid(observed)

    def mean_curl_error(track):
        betas = fit_hand_shape(track, valid, model)
        fits = fit_hands(track, valid, model, betas)
        assert [f.status for f in fits] == ["ok"] * len(fits)
        rebuilt = rebuild(fits, model)
        return np.mean([curl_error_deg(r, t) for r, t in zip(rebuilt, truth)])

    scaled, _ = normalise_palm_size(observed, valid)
    with_normalising = mean_curl_error(scaled)
    without = mean_curl_error(observed)
    print(f"curl error: normalised {with_normalising:.2f} deg, raw {without:.2f} deg")
    assert with_normalising < 5.0
    assert without > with_normalising + MARGIN_DEG


def test_plausible_radius_is_the_chi_square_quantile():
    assert plausible_radius(10) == pytest.approx(4.2786732, abs=1e-6)
    assert plausible_radius(2) == pytest.approx(2.4477468, abs=1e-6)


def unit_shape(model, seed, norm=2.0):
    betas = np.random.default_rng(seed).normal(size=model.num_betas)
    return betas * norm / np.linalg.norm(betas)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fitted_shape_reproduces_a_plausible_hand(backend):
    model = hand(backend, "right")
    observed = synthetic_track(model, 8, seed=32, betas=unit_shape(model, 31))
    betas = fit_hand_shape(observed, all_valid(observed), model)
    rebuilt = rebuild(fit_hands(observed, all_valid(observed), model, betas), model)
    errors = landmark_errors_mm(rebuilt, observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_shape_never_leaves_the_plausible_region(backend):
    model = hand(backend, "right")
    radius = plausible_radius(model.num_betas)
    extreme = np.zeros(model.num_betas)
    extreme[0] = 3.0 * radius
    observed = synthetic_track(model, 6, seed=33, betas=extreme)
    assert np.linalg.norm(fit_hand_shape(observed, all_valid(observed), model)) <= radius


@requires_mano
def test_invalid_landmarks_never_influence_the_shape():
    model = hand("mano", "right")
    observed = synthetic_track(model, 4, seed=34)
    valid = all_valid(observed)
    valid[:, [4, 12]] = False
    corrupted = observed.copy()
    corrupted[:, [4, 12]] -= 0.02
    np.testing.assert_array_equal(
        fit_hand_shape(observed, valid, model), fit_hand_shape(corrupted, valid, model)
    )


@requires_mano
def test_with_nothing_to_fit_the_shape_stays_neutral():
    model = hand("mano", "right")
    observed = synthetic_track(model, 2, seed=35)
    no_landmarks = np.zeros(observed.shape[:2], dtype=bool)
    np.testing.assert_array_equal(fit_hand_shape(observed, no_landmarks, model), np.zeros(10))


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_frame_shapes_follow_a_hand_that_changes_size(backend):
    model = hand(backend, "right")
    observed = np.concatenate(
        [synthetic_track(model, 1, seed=40 + k, betas=unit_shape(model, 50 + k)) for k in range(4)]
    )
    fits = fit_hands_per_frame_shape(observed, all_valid(observed), model)
    assert [f.status for f in fits] == ["ok"] * 4
    errors = landmark_errors_mm(rebuild(fits, model), observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0
    radius = plausible_radius(model.num_betas)
    assert all(np.linalg.norm(f.betas) <= radius for f in fits)
    assert len({tuple(np.round(f.betas, 6)) for f in fits}) == 4
