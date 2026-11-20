"""Does a fault detector trained on two machines help on a third, new machine?

    python -m machine_audio.experiment --out results   # ~10 minutes on a laptop CPU

For each seed:

1. Train a CNN on the source machines (ToyCar + ToyTrain).
2. Split the target machine (ToyConveyor) into a fine-tuning pool and a **disjoint test set**
   that no model ever trains on.
3. With k labeled target clips (k = 4, 8, 16, 32, balanced), compare on the target test set:
   * **no transfer**: the source model as is;
   * **from scratch**: a new CNN trained on the k target clips only;
   * **fine-tuned**: the source model trained further on the same k target clips.

All arms share one preprocessing (log-mel, standardized with statistics from the source training
data) and fixed training schedules, so nothing is tuned on the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from sklearn.metrics import accuracy_score, f1_score

from machine_audio.model import FaultCNN, Standardizer, log_mel, predict, train
from machine_audio.synth import SOURCE, TARGET, dataset

SHOTS = (4, 8, 16, 32)
SOURCE_EPOCHS, SOURCE_LR = 20, 1e-3
# Fine-tuning uses a 3x smaller step than training from scratch, a standard default.
TARGET_EPOCHS, SCRATCH_LR, FINETUNE_LR = 60, 1e-3, 3e-4


def _score(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def _balanced_subset(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    per_class = k // 2
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == label), per_class, replace=False) for label in (0, 1)]
    )


def split_target(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stratified half/half split into a fine-tuning pool and a test set (disjoint, each with
    equal numbers of normal and faulty clips, so chance accuracy is exactly 50%)."""
    pool, test = [], []
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        pool.append(idx[: len(idx) // 2])
        test.append(idx[len(idx) // 2 :])
    return np.concatenate(pool), np.concatenate(test)


def run_seed(seed: int, per_class: int = 60) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Source machines: 80% train, 20% held out to confirm the source model works at all.
    source_clips, source_y = [], []
    for i, machine in enumerate(SOURCE):
        clips, y = dataset(machine, per_class, seed=seed * 100 + i)
        source_clips += clips
        source_y.append(y)
    source_x, source_y = log_mel(source_clips), np.concatenate(source_y)
    order = rng.permutation(len(source_y))
    cut = int(0.8 * len(order))
    source_train, source_test = order[:cut], order[cut:]
    standardize = Standardizer(source_x[source_train])

    # Target machine: half the clips form the fine-tuning pool, the other half the test set.
    target_clips, target_y = dataset(TARGET, per_class, seed=seed * 100 + 99)
    target_x = standardize(log_mel(target_clips))
    pool, test = split_target(target_y, rng)
    assert not set(pool) & set(test)

    x_source = standardize(source_x)
    source_model = train(
        FaultCNN(), x_source[source_train], source_y[source_train], SOURCE_EPOCHS, SOURCE_LR, seed
    )
    source_state = {k: v.clone() for k, v in source_model.state_dict().items()}
    result = {
        "source_heldout": _score(
            source_y[source_test], predict(source_model, x_source[source_test])
        ),
        "no_transfer": _score(target_y[test], predict(source_model, target_x[test])),
        "shots": {},
    }

    for k in SHOTS:
        chosen = pool[_balanced_subset(target_y[pool], k, rng)]
        scratch = train(
            FaultCNN(), target_x[chosen], target_y[chosen], TARGET_EPOCHS, SCRATCH_LR, seed
        )
        tuned = FaultCNN()
        tuned.load_state_dict(source_state)
        tuned = train(tuned, target_x[chosen], target_y[chosen], TARGET_EPOCHS, FINETUNE_LR, seed)
        result["shots"][k] = {
            "from_scratch": _score(target_y[test], predict(scratch, target_x[test])),
            "fine_tuned": _score(target_y[test], predict(tuned, target_x[test])),
        }
    result["target_test_clips"] = len(test)
    result["seed"] = seed
    return result


def paired_gain(runs: list[dict], k: int) -> dict[str, float]:
    """Fine-tuned minus from-scratch accuracy on the same seed: mean, 95% t-interval, wins."""
    diffs = np.array(
        [
            r["shots"][k]["fine_tuned"]["accuracy"] - r["shots"][k]["from_scratch"]["accuracy"]
            for r in runs
        ]
    )
    half = t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs))
    return {
        "mean": round(float(diffs.mean()), 3),
        "ci95": [round(float(diffs.mean() - half), 3), round(float(diffs.mean() + half), 3)],
        "wins": int((diffs > 0).sum()),
        "ties": int((diffs == 0).sum()),
        "losses": int((diffs < 0).sum()),
    }


def summarize(runs: list[dict]) -> dict:
    """Mean and sample standard deviation (ddof=1) across seeds for every arm, plus the paired
    fine-tuning gain per k."""

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(values)), 3),
            "std": round(float(np.std(values, ddof=1)), 3),
        }

    summary = {
        arm: {m: stats([r[arm][m] for r in runs]) for m in ("accuracy", "f1")}
        for arm in ("source_heldout", "no_transfer")
    }
    summary["shots"] = {
        k: {
            arm: {m: stats([r["shots"][k][arm][m] for r in runs]) for m in ("accuracy", "f1")}
            for arm in ("from_scratch", "fine_tuned")
        }
        for k in SHOTS
    }
    summary["fine_tuning_gain"] = {k: paired_gain(runs, k) for k in SHOTS}
    summary["seeds"] = len(runs)
    summary["seed_ids"] = [r["seed"] for r in runs]
    summary["target_test_clips"] = runs[0]["target_test_clips"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--first-seed",
        type=int,
        default=10,
        help="seeds 0-9 were used while developing the protocol; 10+ are confirmatory",
    )
    args = parser.parse_args()
    runs = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        runs.append(run_seed(seed))
        print(f"seed {seed}: no transfer {runs[-1]['no_transfer']['accuracy']:.2f}", flush=True)
    summary = summarize(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
