from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MCU benchmark results from a JSON file")
    parser.add_argument("--input", type=Path, required=True, help="Input result json")
    parser.add_argument("--output", type=Path, required=True, help="Output png path")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metric(payload: dict, metric: str) -> tuple[list[str], list[float], list[float]]:
    labels = []
    means = []
    stds = []

    if "aggregate" in payload:
        source = payload["aggregate"]
        for label, metrics in source.items():
            labels.append(label)
            means.append(metrics[metric]["mean"])
            stds.append(metrics[metric]["std"])
        return labels, means, stds

    for label, metrics in payload.items():
        labels.append(label)
        means.append(metrics[metric])
        stds.append(0.0)
    return labels, means, stds


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_specs = [
        ("action_mse", "Action MSE", "#b45309"),
        ("success_rate", "Success Rate", "#0f766e"),
    ]

    for axis, (metric, title, color) in zip(axes, metric_specs):
        labels, means, stds = extract_metric(payload, metric)
        positions = list(range(len(labels)))
        bars = axis.bar(positions, means, yerr=stds, capsize=6, color=color, alpha=0.82)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("MCU Benchmark Summary")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
