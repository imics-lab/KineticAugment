"""Tests for the hand clip record and the geometric left/right assignment (V1.2 plan, P0)."""

import json

import numpy as np
import pytest

from kinetic_augment.body_model.hand_observations import (
    HAND_CLIP_SCHEMA,
    LEFT,
    RIGHT,
    HandClip,
    assign_sides,
    load_hand_clip,
    save_hand_clip,
)

# The signer faces the camera, so their left wrist is on the image's right.
POSE = np.array([[0.70, 0.60], [0.30, 0.60]])


def sides(hands, pose=POSE):
    return assign_sides(np.array(hands, dtype=float), pose, max_gap=0.10, margin=0.03).tolist()


class TestAssignSides:
    def test_a_hand_takes_the_side_of_the_nearest_pose_wrist(self):
        assert sides([[0.71, 0.61]]) == [LEFT]
        assert sides([[0.29, 0.58]]) == [RIGHT]

    def test_detection_order_carries_no_side(self):
        assert sides([[0.31, 0.60], [0.69, 0.60]]) == [RIGHT, LEFT]
        assert sides([[0.69, 0.60], [0.31, 0.60]]) == [LEFT, RIGHT]

    def test_a_second_detection_of_one_hand_is_not_assigned(self):
        assert sides([[0.72, 0.60], [0.705, 0.60]]) == [-1, LEFT]

    def test_hands_too_close_to_tell_apart_are_not_assigned(self):
        together = np.array([[0.52, 0.60], [0.48, 0.60]])
        assert sides([[0.505, 0.60]], together) == [-1]

    def test_a_hand_far_from_every_pose_wrist_is_not_assigned(self):
        assert sides([[0.70, 0.10]]) == [-1]

    def test_only_a_visible_pose_wrist_can_take_a_hand(self):
        one_wrist = np.array([[0.70, 0.60], [np.nan, np.nan]])
        assert sides([[0.70, 0.60], [0.30, 0.60]], one_wrist) == [LEFT, -1]
        assert sides([[0.70, 0.60]], np.full((2, 2), np.nan)) == [-1]

    def test_no_detections(self):
        assert sides(np.zeros((0, 2))) == []


def tiny_clip(frames=3):
    rng = np.random.default_rng(0)
    world = np.full((frames, 2, 21, 3), np.nan, dtype=np.float32)
    world[:, RIGHT] = rng.normal(0.0, 0.05, (frames, 21, 3))
    image = np.full_like(world, np.nan)
    image[:, RIGHT] = rng.uniform(0.0, 1.0, (frames, 21, 3))
    present = np.zeros((frames, 2), dtype=bool)
    present[:, RIGHT] = True
    score = np.where(present, 0.9, np.nan).astype(np.float32)
    return HandClip(
        world=world, image=image, present=present, handedness_score=score,
        label_agrees=present.copy(), unassigned=np.zeros(frames, dtype=np.int8),
        pose_world=rng.normal(0.0, 0.3, (frames, 33, 3)).astype(np.float32),
        pose_image=rng.uniform(0.0, 1.0, (frames, 33, 3)).astype(np.float32),
        pose_visibility=np.ones((frames, 33), dtype=np.float32),
        timestamp_ms=np.arange(frames) * 40.0, width=320, height=240, fps=25.0,
        meta={"source": "test"},
    )


class TestHandClipFile:
    ARRAYS = ("world", "image", "present", "handedness_score", "label_agrees", "unassigned",
              "pose_world", "pose_image", "pose_visibility", "timestamp_ms")

    def test_a_saved_clip_loads_back_unchanged(self, tmp_path):
        clip = tiny_clip()
        save_hand_clip(tmp_path / "x_hands.npz", clip)
        back = load_hand_clip(tmp_path / "x_hands.npz")
        for name in self.ARRAYS:
            np.testing.assert_array_equal(getattr(back, name), getattr(clip, name))
        assert (back.width, back.height, back.fps) == (320, 240, 25.0)
        assert back.meta == {"source": "test", "schema": HAND_CLIP_SCHEMA}

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = tmp_path / "x_hands.npz"
        save_hand_clip(path, tiny_clip())
        raw = dict(np.load(path))
        raw["meta"] = np.array(json.dumps({"schema": HAND_CLIP_SCHEMA + 1}))
        np.savez(path, **raw)
        with pytest.raises(ValueError):
            load_hand_clip(path)
