from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .schema import EpisodeBatch


def _default_context(num_steps: int, context_dim: int) -> np.ndarray:
    return np.zeros((num_steps, context_dim), dtype=np.float32)


def _default_phase_scalar(phase: np.ndarray) -> np.ndarray:
    return np.arctan2(phase[:, 0], phase[:, 1]).astype(np.float32)


def _validate_episode_arrays(
    vision: np.ndarray,
    proprio: np.ndarray,
    contact: np.ndarray,
    phase: np.ndarray,
    command: np.ndarray,
    action: np.ndarray,
    state: np.ndarray,
) -> None:
    lengths = {
        "vision": vision.shape[0],
        "proprio": proprio.shape[0],
        "contact": contact.shape[0],
        "phase": phase.shape[0],
        "command": command.shape[0],
        "action": action.shape[0],
        "state": state.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent episode lengths: {lengths}")

    if phase.shape[-1] != 2:
        raise ValueError(f"Expected phase shape [T, 2], got {phase.shape}")


def load_npz_episodes(path: str | Path, context_dim: int) -> List[EpisodeBatch]:
    payload = np.load(Path(path), allow_pickle=False)
    required = ["vision", "proprio", "contact", "phase", "command", "action", "state"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing arrays in npz dataset: {missing}")

    num_episodes = payload["vision"].shape[0]
    episodes: List[EpisodeBatch] = []
    for index in range(num_episodes):
        vision = payload["vision"][index].astype(np.float32)
        proprio = payload["proprio"][index].astype(np.float32)
        contact = payload["contact"][index].astype(np.float32)
        phase = payload["phase"][index].astype(np.float32)
        command = payload["command"][index].astype(np.float32)
        action = payload["action"][index].astype(np.float32)
        state = payload["state"][index].astype(np.float32)
        _validate_episode_arrays(vision, proprio, contact, phase, command, action, state)

        steps = vision.shape[0]
        skill = (
            payload["skill"][index].astype(np.int64)
            if "skill" in payload
            else np.zeros((steps,), dtype=np.int64)
        )
        context = (
            payload["context"][index].astype(np.float32)
            if "context" in payload
            else _default_context(steps, context_dim)
        )
        phase_scalar = (
            payload["phase_scalar"][index].astype(np.float32)
            if "phase_scalar" in payload
            else _default_phase_scalar(phase)
        )

        episodes.append(
            EpisodeBatch(
                vision=vision,
                proprio=proprio,
                contact=contact,
                phase=phase,
                command=command,
                action=action,
                state=state,
                skill=skill,
                context=context,
                phase_scalar=phase_scalar,
            )
        )

    return episodes


def split_episodes(
    episodes: List[EpisodeBatch], train_split: float
) -> Tuple[List[EpisodeBatch], List[EpisodeBatch]]:
    if len(episodes) < 2:
        raise ValueError("At least 2 episodes are required to create train/test splits.")
    cut = max(1, int(len(episodes) * train_split))
    cut = min(cut, len(episodes) - 1)
    return episodes[:cut], episodes[cut:]
