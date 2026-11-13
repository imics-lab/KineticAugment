"""
Tests for the 21-point hand observation operator (V1.2 plan, P1).

Mapping checks compare against independent sources -- the full smplx forward
pass with index lists written out below, the raw model arrays, or anatomy that
holds for any human hand -- so a wrong table in hand_model.py cannot pass.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.hand_assets import BACKENDS, MODEL_DIR, SIDES, hand, requires_mano

torch = pytest.importorskip("torch")
smplx = pytest.importorskip("smplx")

from kinetic_augment.body_model.hand_model import FINGERS, TIPS  # noqa: E402

# MediaPipe landmark -> model joint, written out independently of hand_model.py.
MANO_JOINTS = {0: 0, 1: 13, 2: 14, 3: 15, 5: 1, 6: 2, 7: 3, 9: 4, 10: 5, 11: 6,
               13: 10, 14: 11, 15: 12, 17: 7, 18: 8, 19: 9}
SMPLX_JOINTS = {
    "left": {0: 20, 1: 37, 2: 38, 3: 39, 5: 25, 6: 26, 7: 27, 9: 28, 10: 29, 11: 30,
             13: 34, 14: 35, 15: 36, 17: 31, 18: 32, 19: 33},
    "right": {0: 21, 1: 52, 2: 53, 3: 54, 5: 40, 6: 41, 7: 42, 9: 43, 10: 44, 11: 45,
              13: 49, 14: 50, 15: 51, 17: 46, 18: 47, 19: 48},
}
TIP_VERTICES = {
    ("mano", "left"): (744, 320, 443, 554, 671),
    ("mano", "right"): (744, 320, 443, 554, 671),
    ("smplx", "left"): (5361, 4933, 5058, 5169, 5286),
    ("smplx", "right"): (8079, 7669, 7794, 7905, 8022),
}
MIRROR = np.array([-1.0, 1.0, 1.0])  # left and right hands reflect across the x = 0 plane
MIRROR_TOLERANCE_M = {"mano": 1e-6, "smplx": 0.015}


def reference_points(backend, side, wrist_orient, hand_pose, betas):
    """The same 21 points read from the full smplx forward pass."""
    batch = len(wrist_orient)

    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float64)

    if backend == "mano":
        model = smplx.create(MODEL_DIR, model_type="mano", is_rhand=side == "right",
                             use_pca=False, flat_hand_mean=True, num_betas=10, ext="npz",
                             batch_size=batch, dtype=torch.float64)
        out = model(betas=t(betas), global_orient=t(wrist_orient), hand_pose=t(hand_pose))
        joints = MANO_JOINTS
    else:
        model = smplx.create(MODEL_DIR, model_type="smplx", gender="neutral", use_pca=False,
                             flat_hand_mean=True, num_betas=10, batch_size=batch,
                             dtype=torch.float64)
        joints = SMPLX_JOINTS[side]
        body_pose = torch.zeros(batch, 63, dtype=torch.float64)
        wrist = joints[0]
        body_pose[:, 3 * (wrist - 1):3 * wrist] = t(wrist_orient)
        out = model(betas=t(betas), body_pose=body_pose, **{f"{side}_hand_pose": t(hand_pose)})
    expected = np.zeros((batch, 21, 3))
    for landmark, joint in joints.items():
        expected[:, landmark] = out.joints[:, joint].detach().numpy()
    tips = list(TIP_VERTICES[(backend, side)])
    expected[:, list(TIPS)] = out.vertices[:, tips].detach().numpy()
    return expected


def mirrored(rotations):
    """Axis-angle rotations reflected across x = 0: (x, y, z) -> (x, -y, -z)."""
    r = np.asarray(rotations, dtype=float)
    return (r.reshape(-1, 3) * [1.0, -1.0, -1.0]).reshape(r.shape)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_points_equal_the_full_model_forward(backend, side):
    rng = np.random.default_rng(0)
    wrist_orient = rng.normal(0.0, 1.0, (4, 3))
    hand_pose = rng.normal(0.0, 0.4, (4, 45))
    betas = rng.normal(0.0, 1.0, (4, 10))
    got = hand(backend, side).points(wrist_orient, hand_pose, betas).detach().numpy()
    want = reference_points(backend, side, wrist_orient, hand_pose, betas)
    assert np.abs(got - want).max() < 1e-9


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_pose_is_the_flat_template_hand(backend, side):
    """flat_hand_mean=True: zero pose is the raw template, not MANO's curled mean."""
    if backend == "mano":
        raw = np.load(Path(MODEL_DIR) / "mano" / f"MANO_{side.upper()}.npz", allow_pickle=True)
        joints = MANO_JOINTS
    else:
        raw = np.load(Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
        joints = SMPLX_JOINTS[side]
    template = np.asarray(raw["v_template"], dtype=np.float64)
    regressed = np.asarray(raw["J_regressor"], dtype=np.float64) @ template
    expected = np.zeros((21, 3))
    for landmark, joint in joints.items():
        expected[landmark] = regressed[joint]
    expected[list(TIPS)] = template[list(TIP_VERTICES[(backend, side)])]
    np.testing.assert_allclose(hand(backend, side).rest_points(), expected, atol=1e-9)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_landmarks_follow_hand_anatomy(backend, side):
    """At rest, for any hand: each chain runs outward and the knuckles run thumb to pinky."""
    rest = hand(backend, side).rest_points()
    from_wrist = np.linalg.norm(rest - rest[0], axis=1)
    for chain in FINGERS.values():
        assert np.all(np.diff(from_wrist[list(chain)]) > 0)
    knuckles = [5, 9, 13, 17]
    assert np.all(np.diff(np.linalg.norm(rest[knuckles] - rest[5], axis=1)) > 0)
    assert np.linalg.norm(rest[1] - rest[5]) < np.linalg.norm(rest[1] - rest[17])
    assert int(np.argmax(from_wrist[list(TIPS)])) == 2  # the middle fingertip reaches farthest


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_tips_extend_their_distal_bone(backend, side):
    rest = hand(backend, side).rest_points()
    for chain in FINGERS.values():
        pip, dip, tip = rest[chain[1]], rest[chain[2]], rest[chain[3]]
        length = np.linalg.norm(tip - dip)
        cosine = np.dot(dip - pip, tip - dip) / (np.linalg.norm(dip - pip) * length)
        assert 0.015 < length < 0.040  # probe: 20-35 mm (MANO), 21-34 mm (SMPL-X)
        assert cosine > np.cos(np.deg2rad(25.0))  # probe: at most 12.6 deg (MANO), 20.3 (SMPL-X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_left_hand_is_the_mirror_of_the_right(backend):
    """
    Same shape and mirrored rotations give mirrored hands -- the side conventions agree.
    MANO's left model is an exact mirror of its right; SMPL-X's body is only nearly
    symmetric (about 6 mm at rest), far below what a side or rotation-sign error causes.
    """
    tolerance = MIRROR_TOLERANCE_M[backend]
    rng = np.random.default_rng(1)
    wrist_orient = rng.normal(0.0, 1.0, (3, 3))
    hand_pose = rng.normal(0.0, 0.4, (3, 45))
    for betas in (np.zeros(10), 2.0 * np.eye(10)[0], 2.0 * np.eye(10)[1]):
        right = hand(backend, "right").points(wrist_orient, hand_pose, betas).detach().numpy()
        left = hand(backend, "left").points(
            mirrored(wrist_orient), mirrored(hand_pose), betas).detach().numpy()
        assert np.abs(left - right * MIRROR).max() < tolerance


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_joint_moves_only_its_own_finger(backend):
    model = hand(backend, "right")
    pose = np.zeros(45)
    pose[0:3] = [0.0, 0.0, np.deg2rad(30.0)]  # slot 0: the index MCP (landmark 5)
    moved = model.points(np.zeros(3), pose, np.zeros(10))[0].detach().numpy()
    shift = np.linalg.norm(moved - model.rest_points(), axis=1)
    for landmark in range(21):
        if landmark in (6, 7, 8):
            assert shift[landmark] > 1e-3
        elif landmark in TIPS:
            assert shift[landmark] < 1e-3  # pose correctives nudge other tips (MANO probe: 0.01 mm)
        else:
            assert shift[landmark] < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradients_are_exact(backend):
    model = hand(backend, "right")
    rng = np.random.default_rng(2)
    inputs = tuple(
        torch.tensor(rng.normal(0.0, scale, size), dtype=torch.float64, requires_grad=True)
        for scale, size in ((0.5, 3), (0.3, 45), (1.0, 10))
    )
    assert torch.autograd.gradcheck(lambda w, p, b: model.points(w, p, b), inputs,
                                    eps=1e-6, atol=1e-7)


@requires_mano
def test_rejects_an_unknown_side_or_backend():
    from kinetic_augment.body_model.hand_model import HandModel

    with pytest.raises(ValueError):
        HandModel("mano", "middle", model_dir=MODEL_DIR)
    with pytest.raises(ValueError):
        HandModel("flame", "left", model_dir=MODEL_DIR)
