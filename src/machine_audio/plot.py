"""Learning-curve chart from ``results/metrics.json``.

python -m machine_audio.plot --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def learning_curve(metrics: dict, path: Path) -> None:
    summary, runs = metrics["summary"], metrics["runs"]
    shots = [int(k) for k in summary["shots"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color, label in (
        ("fine_tuned", "#2563eb", "Fine-tuned from the car/train model"),
        ("from_scratch", "#9ca3af", "Trained from scratch on the conveyor clips"),
    ):
        per_seed = np.array([[r["shots"][str(k)][arm]["accuracy"] for k in shots] for r in runs])
        mean, std = per_seed.mean(0), per_seed.std(0)
        ax.plot(shots, mean, marker="o", color=color, label=label)
        ax.fill_between(shots, mean - std, mean + std, color=color, alpha=0.15)
    ax.axhline(
        summary["no_transfer"]["accuracy"]["mean"],
        color="#dc2626",
        linestyle="--",
        label="No transfer (car/train model as is)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots, [str(k) for k in shots])
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Labeled conveyor clips used for training")
    ax.set_ylabel("Accuracy on held-out conveyor clips")
    ax.set_title(
        f"Mean ± 1 std over {summary['seeds']} seeds, {summary['target_test_clips']} test clips",
        fontsize=10,
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text())
    learning_curve(metrics, args.results / "learning_curve.png")


if __name__ == "__main__":
    main()
