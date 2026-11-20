import numpy as np
import torch

from machine_audio.experiment import _balanced_subset
from machine_audio.model import FaultCNN, Standardizer, log_mel
from machine_audio.synth import SOURCE, TARGET, dataset


def test_dataset_is_balanced_and_reproducible():
    clips_a, labels_a = dataset(TARGET, per_class=3, seed=1)
    clips_b, _ = dataset(TARGET, per_class=3, seed=1)
    assert labels_a.tolist() == [0, 0, 0, 1, 1, 1]
    assert all(np.array_equal(a, b) for a, b in zip(clips_a, clips_b, strict=True))


def _band_energy(clip: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(clip)) ** 2
    freqs = np.fft.rfftfreq(len(clip), 1 / 16_000)
    return float(spectrum[(freqs >= low) & (freqs <= high)].sum())


def test_faults_add_energy_in_each_machines_own_band():
    rng_seed = 3
    for machine in (*SOURCE, TARGET):
        clips, labels = dataset(machine, per_class=4, seed=rng_seed)
        low, high = machine.burst_band_hz
        normal = np.mean(
            [_band_energy(c, low, high) for c, y in zip(clips, labels, strict=True) if y == 0]
        )
        faulty = np.mean(
            [_band_energy(c, low, high) for c, y in zip(clips, labels, strict=True) if y == 1]
        )
        assert faulty > 2 * normal, machine.name


def test_balanced_subset_takes_k_over_2_per_class_without_repeats():
    y = np.array([0] * 10 + [1] * 10)
    chosen = _balanced_subset(y, 8, np.random.default_rng(0))
    assert len(set(chosen)) == 8
    assert (y[chosen] == 0).sum() == (y[chosen] == 1).sum() == 4


def test_model_outputs_one_logit_per_clip():
    clips, _ = dataset(TARGET, per_class=2, seed=0)
    features = log_mel(clips)
    x = Standardizer(features)(features)
    assert FaultCNN()(x).shape == torch.Size([4])
