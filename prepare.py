from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.config import ExperimentConfig
from mcu_humanoid_colab.experiment import load_episodes, normalize_phase, set_seed
from mcu_humanoid_colab.memory import MemoryBank, flatten_history
from mcu_humanoid_colab.models import (
    build_chunk_decoder_data,
    build_world_model_data,
    train_chunk_decoder,
    train_world_model,
)
from mcu_humanoid_colab.schema import EpisodeBatch, PredictiveBundle


WORKSPACE_DIR = ROOT / "workspace"
ACTIVE_CONFIG_PATH = WORKSPACE_DIR / "active_config.json"
RESULTS_PATH = ROOT / "results.tsv"
PROGRESS_PATH = WORKSPACE_DIR / "progress.log"
FIXED_CONFIG_PATH = ROOT / "sample_data" / "libero_real_config.json"
FIXED_DATASET_PATH = ROOT / "sample_data" / "libero_real_sample.npz"
BASELINE_DESCRIPTION = "baseline"


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


def load_active_config() -> ExperimentConfig:
    if not ACTIVE_CONFIG_PATH.exists():
        raise SystemExit(f"Missing {ACTIVE_CONFIG_PATH}. Run `python prepare.py` first.")
    return ExperimentConfig.from_json(ACTIVE_CONFIG_PATH)


def prepare_workspace() -> ExperimentConfig:
    config = ExperimentConfig.from_json(FIXED_CONFIG_PATH)
    config.dataset_path = str(FIXED_DATASET_PATH.resolve()).replace("\\", "/")

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    write_results_header(RESULTS_PATH)
    write_progress_header(PROGRESS_PATH)
    return config


def results_header() -> str:
    return "commit\tprimary_metric\taction_mse\tsuccess_rate\tfall_rate\tmemory_gb\tstatus\tdescription\n"


def append_progress(iteration: int, status: str, primary_metric: float, note: str) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{iteration}\t{status}\t{primary_metric:.6f}\t{note}\n")


def load_runtime(config: ExperimentConfig) -> tuple[object | None, List[EpisodeBatch], List[EpisodeBatch]]:
    set_seed(config.seed)
    return load_episodes(config)


def build_predictive_bundle(
    train_episodes: List[EpisodeBatch],
    config: ExperimentConfig,
    world_model_epochs: int,
    batch_size: int,
    learning_rate: float,
    device,
) -> PredictiveBundle:
    wm_states, wm_actions, wm_next, wm_instability, wm_progress = build_world_model_data(
        train_episodes, instability_threshold=config.instability_threshold
    )
    chunk_states, chunk_inputs, chunk_targets = build_chunk_decoder_data(
        train_episodes,
        chunk_len=config.chunk_len,
        instability_threshold=config.instability_threshold,
    )
    return PredictiveBundle(
        dynamics_model=train_world_model(
            states=wm_states,
            actions=wm_actions,
            next_states=wm_next,
            instability=wm_instability,
            progress=wm_progress,
            epochs=world_model_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            device=device,
        ),
        chunk_decoder=train_chunk_decoder(
            state_inputs=chunk_states,
            chunk_inputs=chunk_inputs,
            targets=chunk_targets,
            epochs=max(6, world_model_epochs),
            batch_size=batch_size,
            lr=learning_rate,
            device=device,
        ),
    )


def main() -> None:
    config = prepare_workspace()

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
