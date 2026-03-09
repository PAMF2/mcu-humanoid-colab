from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig


WORKSPACE_DIR = ROOT / "workspace"
ACTIVE_CONFIG_PATH = WORKSPACE_DIR / "active_config.json"
RESULTS_PATH = ROOT / "results.tsv"
PROGRESS_PATH = WORKSPACE_DIR / "progress.log"
FIXED_CONFIG_PATH = ROOT / "sample_data" / "libero_real_config.json"
FIXED_DATASET_PATH = ROOT / "sample_data" / "libero_real_sample.npz"


def write_results_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "commit\tprimary_metric\taction_mse\tsuccess_rate\tfall_rate\tmemory_gb\tstatus\tdescription\n",
        encoding="utf-8",
    )


def write_progress_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text("iteration\tstatus\tprimary_metric\tnote\n", encoding="utf-8")


def main() -> None:
    config = ExperimentConfig.from_json(FIXED_CONFIG_PATH)
    config.dataset_path = str(FIXED_DATASET_PATH.resolve()).replace("\\", "/")

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    write_results_header(RESULTS_PATH)
    write_progress_header(PROGRESS_PATH)

    payload = {
        "active_config": str(ACTIVE_CONFIG_PATH.resolve()),
        "results_tsv": str(RESULTS_PATH.resolve()),
        "progress_log": str(PROGRESS_PATH.resolve()),
        "dataset": config.dataset,
        "dataset_path": config.dataset_path,
        "config_path": str(FIXED_CONFIG_PATH.resolve()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
