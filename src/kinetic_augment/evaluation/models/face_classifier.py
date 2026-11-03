"""Masked temporal classifier for the aligned body and facial feature streams."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from kinetic_augment.evaluation.models.base import (
    BaseSLRClassifier,
    TORCH_AVAILABLE,
    check_torch_available,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence


class FaceStreamClassifier(BaseSLRClassifier):
    """A separate masked GRU encoder per feature stream, followed by fusion.

    ``stream_dims`` selects the modalities and their feature dimensions. For example,
    ``{"body_hands": 96}`` is a body-only ablation, while adding ``face``, ``head``,
    or ``blendshapes`` enables those streams without changing the model architecture.
    """

    name = "FaceStreamGRU"

    def __init__(
        self,
        num_classes: int,
        stream_dims: Mapping[str, int],
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        check_torch_available()
        super().__init__(num_classes)
        if not stream_dims or any(not name or int(dim) <= 0 for name, dim in stream_dims.items()):
            raise ValueError("stream_dims must map stream names to positive dimensions")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.stream_dims = {name: int(dim) for name, dim in stream_dims.items()}
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict(
            {
                name: nn.GRU(input_size=dim * 2, hidden_size=hidden_dim, batch_first=True)
                for name, dim in self.stream_dims.items()
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(len(self.stream_dims) * (hidden_dim + 1)),
            nn.Dropout(dropout),
            nn.Linear(len(self.stream_dims) * (hidden_dim + 1), num_classes),
        )

    def forward(self, x: Dict[str, Dict[str, "torch.Tensor"]], lengths=None):
        """Return logits for a collated face feature batch."""
        if not isinstance(x, Mapping) or "streams" not in x or "masks" not in x:
            raise TypeError("x must contain 'streams' and 'masks' dictionaries")
        streams, masks = x["streams"], x["masks"]
        if set(streams) != set(masks) or not set(self.stream_dims).issubset(streams):
            raise ValueError("input must contain matching masks and all configured streams")

        embeddings = []
        batch_size = None
        max_length = None
        for name, dim in self.stream_dims.items():
            values, feature_mask = streams[name], masks[name]
            if values.ndim != 3 or values.shape != feature_mask.shape or values.shape[-1] != dim:
                raise ValueError(
                    "{} values and masks must have matching (B, T, {}) shapes".format(name, dim)
                )
            if batch_size is None:
                batch_size, max_length = values.shape[:2]
            elif values.shape[:2] != (batch_size, max_length):
                raise ValueError("all streams must share batch and time dimensions")

            valid = feature_mask.to(dtype=torch.bool) & torch.isfinite(values)
            # where (rather than multiplication) prevents masked NaN/Inf from entering projections.
            clean_values = torch.where(valid, values, torch.zeros_like(values))
            encoder_input = torch.cat([clean_values, valid.to(dtype=values.dtype)], dim=-1)
            if lengths is None:
                _, hidden = self.encoders[name](encoder_input)
            else:
                lengths_device = lengths.to(device=values.device, dtype=torch.long)
                if lengths_device.shape != (batch_size,):
                    raise ValueError("lengths must have shape (B,)")
                if torch.any(lengths_device <= 0) or torch.any(lengths_device > max_length):
                    raise ValueError("lengths must be between 1 and the padded sequence length")
                packed = pack_padded_sequence(
                    encoder_input,
                    lengths_device.detach().cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, hidden = self.encoders[name](packed)
            encoded = hidden[-1]
            valid_steps = torch.arange(max_length, device=values.device)[None, :] < (
                lengths.to(values.device)[:, None] if lengths is not None else max_length
            )
            present = (valid.any(dim=-1) & valid_steps).any(dim=1)
            encoded = torch.where(present[:, None], encoded, torch.zeros_like(encoded))
            embeddings.extend([encoded, present.to(encoded.dtype).unsqueeze(1)])

        return self.classifier(torch.cat(embeddings, dim=-1))
