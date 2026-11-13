"""Behavior tests for the masked temporal face classifier and existing trainer."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset, TensorDataset

from kinetic_augment.evaluation.datasets.face_features import (
    FaceTrainingSample,
    collate_face_samples,
)
from kinetic_augment.evaluation.models import FaceStreamClassifier, LSTMClassifier, SLRTrainer


def _sample(streams, masks):
    length = len(next(iter(streams.values())))
    times = np.arange(length, dtype=np.float64)
    return FaceTrainingSample(
        streams=streams,
        masks=masks,
        timestamp_ms=times,
        source_frames=np.column_stack([np.arange(length)] * 2),
        source_time_ms=np.column_stack([times] * 2),
        interpolated=np.zeros(length, dtype=bool),
    )


def _batch(streams, masks, lengths=None):
    x = {
        "streams": {
            key: torch.tensor(value, dtype=torch.float32) for key, value in streams.items()
        },
        "masks": {key: torch.tensor(value, dtype=torch.bool) for key, value in masks.items()},
    }
    return x, None if lengths is None else torch.tensor(lengths, dtype=torch.long)


def test_padding_and_masked_garbage_do_not_change_predictions():
    torch.manual_seed(3)
    model = FaceStreamClassifier(3, {"body_hands": 2, "face": 2}, hidden_dim=5).eval()
    body = np.array([[[0.2, -0.1], [0.4, 0.3]]], dtype=np.float32)
    face = np.array([[[0.1, 0.8], [0.2, 0.7]]], dtype=np.float32)
    body_mask = np.ones_like(body, dtype=bool)
    face_mask = np.array([[[True, False], [True, True]]])
    short, short_lengths = _batch(
        {"body_hands": body, "face": face},
        {"body_hands": body_mask, "face": face_mask},
        [2],
    )
    padded, padded_lengths = _batch(
        {
            "body_hands": np.concatenate([body, np.full((1, 2, 2), 100)], axis=1),
            "face": np.concatenate([face, np.full((1, 2, 2), np.nan)], axis=1),
        },
        {
            "body_hands": np.ones((1, 4, 2), dtype=bool),
            "face": np.concatenate([face_mask, np.ones((1, 2, 2), dtype=bool)], axis=1),
        },
        [2],
    )
    with torch.no_grad():
        expected = model(short, short_lengths)
        actual = model(padded, padded_lengths)
    torch.testing.assert_close(actual, expected)

    # Masked NaN/Inf and arbitrary finite garbage must be sanitized before recurrence.
    face_garbage = face.copy()
    face_garbage[0, 0, 1] = np.nan
    changed, _ = _batch(
        {"body_hands": body, "face": face_garbage},
        {"body_hands": body_mask, "face": face_mask},
        [2],
    )
    with torch.no_grad():
        torch.testing.assert_close(model(changed, short_lengths), expected)


def test_all_missing_stream_is_gated_and_backward_update_stays_finite():
    torch.manual_seed(4)
    model = FaceStreamClassifier(2, {"body_hands": 2, "face": 2}, hidden_dim=4).eval()
    body = np.ones((2, 3, 2), dtype=np.float32)
    face = np.full((2, 3, 2), np.nan, dtype=np.float32)
    streams, masks = _batch(
        {"body_hands": body, "face": face},
        {"body_hands": np.ones_like(body, bool), "face": np.zeros_like(body, bool)},
        [3, 2],
    )
    logits = model(streams, masks)
    face_mutated, _ = _batch(
        {"body_hands": body, "face": np.full_like(face, 1e30)},
        {"body_hands": np.ones_like(body, bool), "face": np.zeros_like(body, bool)},
        [3, 2],
    )
    torch.testing.assert_close(logits, model(face_mutated, masks))
    loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert torch.isfinite(model(streams, masks)).all()

    no_modalities, _ = _batch(
        {"body_hands": np.full_like(body, np.nan), "face": face},
        {"body_hands": np.zeros_like(body, bool), "face": np.zeros_like(body, bool)},
        [3, 2],
    )
    all_missing_logits = model(no_modalities, masks)
    assert torch.isfinite(all_missing_logits).all()
    model.zero_grad(set_to_none=True)
    all_missing_logits.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


class _SyntheticFaceSet(Dataset):
    def __init__(self, count, seed):
        rng = np.random.default_rng(seed)
        self.rows = []
        for index in range(count):
            label = index % 2
            body = rng.normal(0, 1, size=(5, 2)).astype(np.float32)
            face = np.full((5, 1), 1.0 if label else -1.0, dtype=np.float32)
            self.rows.append(
                (
                    _sample(
                        {"body_hands": body, "face": face},
                        {"body_hands": np.ones_like(body, bool), "face": np.ones_like(face, bool)},
                    ),
                    label,
                )
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_task3_collate_trainer_learns_face_signal_over_body_only_baseline():
    torch.manual_seed(11)
    train_loader = DataLoader(
        _SyntheticFaceSet(48, 1), batch_size=8, shuffle=True, collate_fn=collate_face_samples
    )
    val_loader = DataLoader(_SyntheticFaceSet(24, 2), batch_size=8, collate_fn=collate_face_samples)

    full = FaceStreamClassifier(2, {"body_hands": 2, "face": 1}, hidden_dim=8, dropout=0)
    full_trainer = SLRTrainer(full, train_loader, val_loader)
    full_result = full_trainer.train(epochs=18, patience=18, verbose=False)
    full_accuracy = full_trainer.evaluate(val_loader)["accuracy"]

    body_only = FaceStreamClassifier(2, {"body_hands": 2}, hidden_dim=8, dropout=0)
    body_trainer = SLRTrainer(body_only, train_loader, val_loader)
    body_trainer.train(epochs=18, patience=18, verbose=False)
    body_accuracy = body_trainer.evaluate(val_loader)["accuracy"]
    assert full_result["epochs_trained"] > 0
    assert full_accuracy >= 0.95
    assert full_accuracy > body_accuracy + 0.25


def test_legacy_lstm_batches_still_train_evaluate_and_predict():
    torch.manual_seed(8)
    inputs = torch.randn(8, 4, 3)
    labels = torch.arange(8) % 2
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    model = LSTMClassifier(2, input_dim=3, hidden_dim=4, num_layers=1, dropout=0)
    trainer = SLRTrainer(model, loader, loader)
    result = trainer.train(epochs=1, patience=1, verbose=False)
    metrics = trainer.evaluate(loader)
    predictions, predicted_labels = trainer.predict(loader)
    assert result["epochs_trained"] == 1
    assert 0 <= metrics["accuracy"] <= 1
    assert predictions.shape == predicted_labels.shape == (8,)
