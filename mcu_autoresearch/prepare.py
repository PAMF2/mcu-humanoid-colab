from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig


WORKSPACE_DIR = ROOT / "mcu_autoresearch" / "workspace"
ACTIVE_CONFIG_PATH = WORKSPACE_DIR / "active_config.json"
RESULTS_PATH = ROOT / "mcu_autoresearch" / "results.tsv"

PRESETS = {
    "synthetic-smoke": ROOT / "configs" / "smoke.json",
    "synthetic-default": ROOT / "configs" / "default.json",
    "real-sample": ROOT / "sample_data" / "libero_real_config.json",
    "real-medium": ROOT / "sample_data" / "libero_real_medium_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MCU autoresearch workspace")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="real-medium")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path override")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional dataset path override")
    return parser.parse_args()


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config.resolve()
    return PRESETS[args.preset].resolve()


def _resolve_dataset_path(config_path: Path, config: ExperimentConfig, dataset_override: Path | None) -> str | None:
    if dataset_override is not None:
        return str(dataset_override.resolve()).replace("\\", "/")
    if config.dataset_path is None:
        return None
    raw = Path(config.dataset_path)
    if raw.is_absolute():
        return str(raw).replace("\\", "/")
    if (ROOT / raw).exists():
        return str((ROOT / raw).resolve()).replace("\\", "/")
    return str((config_path.parent / raw).resolve()).replace("\\", "/")


def _write_results_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "commit\tprimary_metric\taction_mse\tsuccess_rate\tfall_rate\tmemory_gb\tstatus\tdescription\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args)
    config = ExperimentConfig.from_json(config_path)
    config.dataset_path = _resolve_dataset_path(config_path, config, args.dataset_path)

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    _write_results_header(RESULTS_PATH)

    payload = {
        "active_config": str(ACTIVE_CONFIG_PATH.resolve()),
        "results_tsv": str(RESULTS_PATH.resolve()),
        "dataset": config.dataset,
        "dataset_path": config.dataset_path,
        "preset": args.preset if args.config is None else "custom",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
