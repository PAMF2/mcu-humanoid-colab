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
    parser = argparse.ArgumentParser(description="Run leave-one-task-out evaluation on an npz dataset")
    parser.add_argument("--config", type=Path, required=True, help="Base experiment config")
    parser.add_argument("--output", type=Path, required=True, help="Output json path")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional dataset override")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


def summarize(controller_results: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    controllers = next(iter(controller_results.values())).keys()
    summary: dict[str, dict[str, float]] = {}
    for controller in controllers:
        metrics = {}
        for metric in next(iter(controller_results.values()))[controller].keys():
            values = [payload[controller][metric] for payload in controller_results.values()]
            metrics[f"{metric}_mean"] = float(np.mean(values))
            metrics[f"{metric}_std"] = float(np.std(values))
        summary[controller] = metrics
    return summary


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.dataset_path is not None:
        config = replace(config, dataset_path=str(args.dataset_path).replace("\\", "/"))
    if not config.dataset_path:
        raise SystemExit("dataset_path is required for leave-one-task-out evaluation")

    payload = np.load(config.dataset_path, allow_pickle=False)
    if "skill" not in payload:
        raise SystemExit("Dataset must include a 'skill' array for leave-one-task-out evaluation")

    skill_ids = sorted(int(skill) for skill in np.unique(payload["skill"]))
    task_names = config.task_names or [f"task_{skill}" for skill in skill_ids]
    if len(task_names) <= max(skill_ids):
        task_names = [f"task_{skill}" for skill in skill_ids]

    results_by_task: dict[str, dict[str, dict[str, float]]] = {}
    for skill in skill_ids:
        held_out_name = task_names[skill]
        run_config = replace(config, holdout_skill=skill, test_dataset_path=None)
        results_by_task[held_out_name] = run_experiment(run_config, force_cpu=args.cpu)

    final_payload = {
        "dataset_path": config.dataset_path,
        "task_names": task_names,
        "results_by_task": results_by_task,
        "aggregate": summarize(results_by_task),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(json.dumps(final_payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
