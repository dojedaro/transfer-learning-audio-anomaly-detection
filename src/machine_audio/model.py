"""Log-mel features and a small CNN fault detector."""

from __future__ import annotations

import librosa
import numpy as np
import torch
from torch import nn

from machine_audio.synth import SAMPLE_RATE

N_MELS = 64


def log_mel(clips: list[np.ndarray]) -> np.ndarray:
    """(clips, 1, mels, frames) log-mel spectrograms in dB."""
    features = [
        librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=c, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS
            ),
            ref=1.0,
        )
        for c in clips
    ]
    return np.stack(features)[:, None].astype(np.float32)


class Standardizer:
    """Per-mel-band mean/std fitted on training features only."""

    def __init__(self, features: np.ndarray):
        self.mean = features.mean(axis=(0, 3), keepdims=True)
        self.std = features.std(axis=(0, 3), keepdims=True) + 1e-6

    def __call__(self, features: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((features - self.mean) / self.std)


class FaultCNN(nn.Module):
    """Three conv blocks (conv, GroupNorm, ReLU, pool), global average pooling, one logit."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        layers, previous = [], 1
        for width in channels:
            layers += [
                nn.Conv2d(previous, width, 3, padding=1),
                nn.GroupNorm(4, width),  # batch-size independent: target sets are tiny
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            previous = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(1)


def train(
    model: FaultCNN,
    x: torch.Tensor,
    y: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int = 16,
) -> FaultCNN:
    """Plain Adam + binary cross-entropy; fixed epochs, no early stopping on test data."""
    generator = torch.Generator().manual_seed(seed)
    target = torch.from_numpy(y.astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(x[batch]), target[batch]).backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: FaultCNN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return (torch.sigmoid(model(x)) > 0.5).numpy().astype(int)
