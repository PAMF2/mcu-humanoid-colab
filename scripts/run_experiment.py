from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig
from mcu_humanoid_colab.experiment import run_experiment, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCU humanoid Colab experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output json")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.dataset_path:
        config.dataset = "npz"
        config.dataset_path = args.dataset_path

    results = run_experiment(config, force_cpu=args.cpu)
    save_results(results, args.output)
    print(args.output.resolve())
    print(results)


if __name__ == "__main__":
    main()
