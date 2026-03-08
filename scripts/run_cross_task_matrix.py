from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig
from mcu_humanoid_colab.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise cross-task evaluation from per-task configs")
    parser.add_argument(
        "--task-config",
        action="append",
        default=[],
        help="Task config in the form name=path/to/config.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output json path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


def parse_task_specs(task_specs: list[str]) -> dict[str, Path]:
    if len(task_specs) < 2:
        raise SystemExit("Provide at least two --task-config entries")
    parsed: dict[str, Path] = {}
    for item in task_specs:
        if "=" not in item:
            raise SystemExit(f"Invalid task config '{item}'. Expected name=path.json")
        name, raw_path = item.split("=", 1)
        parsed[name] = Path(raw_path)
    return parsed


def summarize(matrix: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, dict[str, float]]:
    records: dict[str, list[float]] = {}
    for train_payload in matrix.values():
        for test_payload in train_payload.values():
            for controller, metrics in test_payload.items():
                for metric, value in metrics.items():
                    records.setdefault(f"{controller}:{metric}", []).append(value)

    summary: dict[str, dict[str, float]] = {}
    for key, values in records.items():
        controller, metric = key.split(":", 1)
        summary.setdefault(controller, {})
        summary[controller][f"{metric}_mean"] = float(np.mean(values))
        summary[controller][f"{metric}_std"] = float(np.std(values))
    return summary


def main() -> None:
    args = parse_args()
    task_configs = parse_task_specs(args.task_config)

    loaded = {
        name: ExperimentConfig.from_json(path)
        for name, path in task_configs.items()
    }

    matrix: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for train_name, train_config in loaded.items():
        matrix[train_name] = {}
        for test_name, test_config in loaded.items():
            run_config = replace(
                train_config,
                dataset_path=train_config.dataset_path,
                test_dataset_path=test_config.dataset_path,
                holdout_skill=None,
            )
            matrix[train_name][test_name] = run_experiment(run_config, force_cpu=args.cpu)

    payload = {
        "tasks": list(task_configs.keys()),
        "matrix": matrix,
        "aggregate": summarize(matrix),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
