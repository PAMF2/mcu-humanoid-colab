from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .schema import EpisodeBatch


def flatten_history(window: np.ndarray) -> np.ndarray:
    return window.reshape(-1).astype(np.float32)


class MemoryBank:
    def __init__(self, keys: np.ndarray, values: np.ndarray) -> None:
        key_tensor = torch.as_tensor(keys, dtype=torch.float32)
        self.keys = key_tensor / (key_tensor.norm(dim=1, keepdim=True) + 1e-6)
        self.values = torch.as_tensor(values, dtype=torch.float32)

    def topk(self, query: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        query_tensor = torch.as_tensor(query, dtype=torch.float32)
        query_tensor = query_tensor / (query_tensor.norm() + 1e-6)
        similarity = self.keys @ query_tensor
        scores, indices = torch.topk(similarity, k=min(k, len(self.keys)))
        return scores, indices


def build_memories(
    episodes: Sequence[EpisodeBatch],
    history: int,
    chunk_len: int,
) -> Dict[str, np.ndarray]:
    vision_keys = []
    multimodal_keys = []
    instant_actions = []
    chunk_keys = []
    chunk_actions = []

    for episode in episodes:
        horizon = len(episode.action)
        full = np.concatenate(
            [
                episode.vision,
                episode.proprio,
                episode.contact,
                episode.phase,
                episode.command,
            ],
            axis=-1,
        )
        for step in range(history - 1, horizon - chunk_len):
            vision_keys.append(episode.vision[step])
            multimodal_keys.append(full[step])
            chunk_keys.append(flatten_history(full[step - history + 1 : step + 1]))
            instant_actions.append(episode.action[step])
            chunk_actions.append(episode.action[step : step + chunk_len])

    return {
        "vision_keys": np.stack(vision_keys),
        "multimodal_keys": np.stack(multimodal_keys),
        "instant_actions": np.stack(instant_actions),
        "chunk_keys": np.stack(chunk_keys),
        "chunk_actions": np.stack(chunk_actions),
    }


def build_banks(memories: Dict[str, np.ndarray]) -> Dict[str, MemoryBank]:
    return {
        "vision": MemoryBank(memories["vision_keys"], memories["instant_actions"]),
        "multimodal": MemoryBank(memories["multimodal_keys"], memories["instant_actions"]),
        "chunk": MemoryBank(memories["chunk_keys"], memories["chunk_actions"]),
    }
