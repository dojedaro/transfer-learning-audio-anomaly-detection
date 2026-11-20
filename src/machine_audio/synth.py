"""Synthetic machine sounds with a deliberate domain shift between machines.

Every clip is a machine hum (fundamental + harmonics) with background noise. Faulty clips add
short impulsive bursts, the usual signature of a mechanical fault such as a chipped gear tooth or
a bearing defect. The shift that makes transfer non-trivial:

* the source machines (ToyCar, ToyTrain) fault with loud bursts at 2-3 kHz;
* the target machine (ToyConveyor) hums at a different pitch, is noisier, and faults with quieter
  bursts at 0.8-1.4 kHz.

So a detector trained only on the source machines has learned roughly the right *kind* of
evidence (bursts against a steady hum) but at the wrong place in the spectrum.

The names follow the ToyADMOS dataset's machine types; no real ToyADMOS audio is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Machine:
    name: str
    hum_hz: float
    noise: float
    burst_band_hz: tuple[float, float]
    burst_gain: float


SOURCE = (
    Machine("ToyCar", hum_hz=120.0, noise=0.05, burst_band_hz=(2000, 3000), burst_gain=0.5),
    Machine("ToyTrain", hum_hz=160.0, noise=0.05, burst_band_hz=(2000, 3000), burst_gain=0.5),
)
TARGET = Machine("ToyConveyor", hum_hz=210.0, noise=0.09, burst_band_hz=(800, 1400), burst_gain=0.3)


def clip(machine: Machine, faulty: bool, rng: np.random.Generator, seconds: float = 4.0):
    """One clip as float32 samples in [-1, 1]."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    hum_hz = machine.hum_hz + rng.uniform(-3, 3)  # unit-to-unit variation
    signal = sum(
        amplitude * np.sin(2 * np.pi * hum_hz * harmonic * t + rng.uniform(0, 2 * np.pi))
        for harmonic, amplitude in ((1, 0.4), (2, 0.15), (3, 0.08))
    )
    signal = signal * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t))  # load drift
    signal = signal + rng.normal(0, machine.noise, t.shape)

    if faulty:
        burst_len = SAMPLE_RATE // 20  # 50 ms
        envelope = np.hanning(burst_len)
        for _ in range(rng.integers(3, 7)):
            start = rng.integers(0, len(t) - burst_len)
            freq = rng.uniform(*machine.burst_band_hz)
            burst = np.sin(2 * np.pi * freq * t[:burst_len]) * envelope
            signal[start : start + burst_len] += machine.burst_gain * burst
    return np.clip(signal, -1, 1).astype(np.float32)


def dataset(machine: Machine, per_class: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Balanced clips for one machine: ``per_class`` normal and ``per_class`` faulty."""
    rng = np.random.default_rng(seed)
    labels = np.array([0] * per_class + [1] * per_class)
    clips = [clip(machine, bool(label), rng) for label in labels]
    return clips, labels
