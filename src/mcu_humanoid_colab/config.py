from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"
    dataset_path: str | None = None
    test_dataset_path: str | None = None
    train_split: float = 0.8
    holdout_skill: int | None = None
    seed: int = 7
    train_episodes: int = 96
    test_episodes: int = 24
    horizon: int = 48
    history: int = 4
    chunk_len: int = 6
    top_k: int = 8
    num_skills: int = 4
    state_dim: int = 6
    vision_dim: int = 10
    context_dim: int = 3
    action_dim: int = 6
    world_model_epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 1e-3
    instability_threshold: float = 4.25
    success_threshold: float = 0.18
    rerank_margin: float = 0.08
    task_names: list[str] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)
