from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig
from mcu_humanoid_colab.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCU benchmark across multiple seeds")
    parser.add_argument("--config", type=Path, required=True, help="Path to base config json")
    parser.add_argument("--output", type=Path, required=True, help="Path to aggregate output json")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset path")
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,11,19,23,31",
        help="Comma-separated seed list",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def main() -> None:
    args = parse_args()
    base_config = ExperimentConfig.from_json(args.config)
    if args.dataset_path:
        base_config.dataset = "npz"
        base_config.dataset_path = args.dataset_path

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    per_seed: dict[str, dict[str, dict[str, float]]] = {}

    for seed in seeds:
        config = replace(base_config, seed=seed)
        per_seed[str(seed)] = run_experiment(config, force_cpu=args.cpu)

    model_names = list(next(iter(per_seed.values())).keys())
    metric_names = list(next(iter(next(iter(per_seed.values())).values())).keys())

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in model_names:
        aggregate[model_name] = {}
        for metric_name in metric_names:
            metric_values = [
                per_seed[str(seed)][model_name][metric_name]
                for seed in seeds
            ]
            aggregate[model_name][metric_name] = mean_std(metric_values)

    payload = {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
