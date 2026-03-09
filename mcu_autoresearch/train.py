from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
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


ACTIVE_CONFIG_PATH = ROOT / "mcu_autoresearch" / "workspace" / "active_config.json"


@dataclass
class ResearchConfig:
    retrieval_mode: str = "multimodal_instant"
    top_k: int = 8
    chunk_len: int = 6
    history: int = 4
    vision_weight: float = 1.0
    proprio_weight: float = 1.0
    contact_weight: float = 1.0
    phase_weight: float = 1.0
    command_weight: float = 1.0
    similarity_weight: float = 0.15
    rollout_weight: float = 0.2
    decoder_weight: float = 1.0
    rerank_margin: float = 0.08
    chunk_aggregation: str = "first"
    use_world_model: bool = False
    use_chunk_decoder: bool = False
    world_model_epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 1e-3


RESEARCH = ResearchConfig()


def load_active_config() -> ExperimentConfig:
    if not ACTIVE_CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing {ACTIVE_CONFIG_PATH}. Run `python mcu_autoresearch/prepare.py --preset real-medium` first."
        )
    return ExperimentConfig.from_json(ACTIVE_CONFIG_PATH)


def weighted_features(episode: EpisodeBatch, knobs: ResearchConfig) -> np.ndarray:
    return np.concatenate(
        [
            episode.vision * knobs.vision_weight,
            episode.proprio * knobs.proprio_weight,
            episode.contact * knobs.contact_weight,
            episode.phase * knobs.phase_weight,
            episode.command * knobs.command_weight,
        ],
        axis=-1,
    )


def build_custom_memories(
    episodes: List[EpisodeBatch],
    knobs: ResearchConfig,
) -> Dict[str, np.ndarray]:
    vision_keys = []
    multimodal_keys = []
    instant_actions = []
    chunk_keys = []
    chunk_actions = []

    for episode in episodes:
        full = weighted_features(episode, knobs)
        horizon = len(episode.action)
        for step in range(knobs.history - 1, horizon - knobs.chunk_len):
            vision_keys.append(episode.vision[step] * knobs.vision_weight)
            multimodal_keys.append(full[step])
            chunk_keys.append(flatten_history(full[step - knobs.history + 1 : step + 1]))
            instant_actions.append(episode.action[step])
            chunk_actions.append(episode.action[step : step + knobs.chunk_len])

    return {
        "vision_keys": np.stack(vision_keys),
        "multimodal_keys": np.stack(multimodal_keys),
        "instant_actions": np.stack(instant_actions),
        "chunk_keys": np.stack(chunk_keys),
        "chunk_actions": np.stack(chunk_actions),
    }


def build_custom_banks(memories: Dict[str, np.ndarray]) -> Dict[str, MemoryBank]:
    return {
        "vision": MemoryBank(memories["vision_keys"], memories["instant_actions"]),
        "multimodal": MemoryBank(memories["multimodal_keys"], memories["instant_actions"]),
        "chunk": MemoryBank(memories["chunk_keys"], memories["chunk_actions"]),
    }


def rollout_score(
    world_model,
    state_vec: np.ndarray,
    chunk: torch.Tensor,
    goal_command: np.ndarray,
    device: torch.device,
    proprio_dim: int,
    contact_dim: int,
    phase_dim: int,
) -> float:
    with torch.no_grad():
        current = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        command = torch.as_tensor(goal_command, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0.0
        for action in chunk.to(device):
            pred_next, instability, pred_value = world_model(current, action.unsqueeze(0))
            pred_proprio = pred_next[:, :proprio_dim]
            pred_contact = torch.sigmoid(pred_next[:, proprio_dim : proprio_dim + contact_dim])
            pred_phase = normalize_phase(
                pred_next[:, proprio_dim + contact_dim : proprio_dim + contact_dim + phase_dim]
            )
            current = torch.cat([pred_proprio, pred_contact, pred_phase, command], dim=-1)
            score += float((pred_value - 0.6 * torch.sigmoid(instability)).item())
        return score


def aggregate_chunk(chunk: torch.Tensor, similarities: torch.Tensor, mode: str) -> np.ndarray:
    if mode == "first":
        return chunk[0].cpu().numpy()
    if mode == "mean":
        return chunk.mean(dim=0)[0].cpu().numpy()
    if mode == "sim_weighted":
        weights = torch.softmax(similarities, dim=0).view(-1, 1, 1)
        return (chunk * weights).sum(dim=0)[0].cpu().numpy()
    raise ValueError(f"Unsupported chunk aggregation: {mode}")


def select_action(
    knobs: ResearchConfig,
    history_buffer: List[np.ndarray],
    current_obs: Dict[str, np.ndarray],
    banks: Dict[str, MemoryBank],
    predictive_bundle: PredictiveBundle | None,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    if knobs.retrieval_mode == "vision_only":
        query = current_obs["vision"] * knobs.vision_weight
        _, indices = banks["vision"].topk(query, k=1)
        return banks["vision"].values[indices[0]].cpu().numpy()

    if knobs.retrieval_mode == "multimodal_instant":
        query = np.concatenate(
            [
                current_obs["vision"] * knobs.vision_weight,
                current_obs["proprio"] * knobs.proprio_weight,
                current_obs["contact"] * knobs.contact_weight,
                current_obs["phase"] * knobs.phase_weight,
                current_obs["command"] * knobs.command_weight,
            ],
            axis=0,
        )
        _, indices = banks["multimodal"].topk(query, k=1)
        return banks["multimodal"].values[indices[0]].cpu().numpy()

    chunk_query = flatten_history(np.stack(history_buffer[-knobs.history :], axis=0))
    similarities, indices = banks["chunk"].topk(chunk_query, k=knobs.top_k)
    chunks = banks["chunk"].values[indices]

    if knobs.retrieval_mode == "chunk_only" or predictive_bundle is None:
        return aggregate_chunk(chunks, similarities, knobs.chunk_aggregation)

    current_state = np.concatenate(
        [
            current_obs["latent_state"],
            current_obs["contact"],
            current_obs["phase"],
            current_obs["command"],
        ],
        axis=0,
    )

    scores = []
    for similarity, chunk in zip(similarities, chunks):
        score = knobs.similarity_weight * float(similarity.item())
        if knobs.use_world_model:
            score += knobs.rollout_weight * rollout_score(
                world_model=predictive_bundle.dynamics_model,
                state_vec=current_state,
                chunk=chunk,
                goal_command=current_obs["command"],
                device=device,
                proprio_dim=config.state_dim,
                contact_dim=2,
                phase_dim=2,
            )
        if knobs.use_chunk_decoder:
            decoder_state = torch.as_tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
            decoder_input = chunk.reshape(1, -1).to(device)
            with torch.no_grad():
                decoder_score = float(
                    predictive_bundle.chunk_decoder(decoder_state, decoder_input).item()
                )
            score += knobs.decoder_weight * decoder_score
        scores.append(score)

    best_index = int(np.argmax(scores))
    if best_index != 0 and scores[best_index] < scores[0] + knobs.rerank_margin:
        best_index = 0
    return aggregate_chunk(chunks[best_index : best_index + 1], similarities[best_index : best_index + 1], "first")


def evaluate_offline(
    episodes: List[EpisodeBatch],
    banks: Dict[str, MemoryBank],
    knobs: ResearchConfig,
    config: ExperimentConfig,
    device: torch.device,
    predictive_bundle: PredictiveBundle | None,
) -> Dict[str, float]:
    action_errors = []

    for episode in episodes:
        history_buffer: List[np.ndarray] = []
        weighted = weighted_features(episode, knobs)
        for step in range(knobs.history - 1, len(episode.action)):
            temporal_obs = weighted[step]
            while len(history_buffer) < knobs.history:
                history_buffer.append(temporal_obs.copy())
            history_buffer.append(temporal_obs.copy())
            history_buffer = history_buffer[-knobs.history :]

            current_obs = {
                "vision": episode.vision[step],
                "proprio": episode.proprio[step],
                "contact": episode.contact[step],
                "phase": episode.phase[step],
                "command": episode.command[step],
                "latent_state": episode.state[step],
            }
            action = select_action(knobs, history_buffer, current_obs, banks, predictive_bundle, config, device)
            action_errors.append(float(np.mean((action - episode.action[step]) ** 2)))

    return {
        "action_mse": float(np.mean(action_errors)),
        "success_rate": 0.0,
        "fall_rate": 0.0,
    }


def evaluate_synthetic(
    env,
    episodes: List[EpisodeBatch],
    banks: Dict[str, MemoryBank],
    knobs: ResearchConfig,
    config: ExperimentConfig,
    device: torch.device,
    predictive_bundle: PredictiveBundle | None,
) -> Dict[str, float]:
    action_errors = []
    successes = 0
    falls = 0

    for episode in episodes:
        state = episode.state[0].copy()
        phase = float(episode.phase_scalar[0])
        context = episode.context[0].copy()
        skill = int(episode.skill[0])
        history_buffer: List[np.ndarray] = []
        failed = False

        for step in range(len(episode.action)):
            vision, proprio, contact, phase_vec, command = env.observe(state, skill, context, phase)
            weighted_obs = np.concatenate(
                [
                    vision * knobs.vision_weight,
                    proprio * knobs.proprio_weight,
                    contact * knobs.contact_weight,
                    phase_vec * knobs.phase_weight,
                    command * knobs.command_weight,
                ],
                axis=0,
            )
            while len(history_buffer) < knobs.history:
                history_buffer.append(weighted_obs.copy())
            history_buffer.append(weighted_obs.copy())
            history_buffer = history_buffer[-knobs.history :]

            current_obs = {
                "vision": vision,
                "proprio": proprio,
                "contact": contact,
                "phase": phase_vec,
                "command": command,
                "latent_state": state.copy(),
            }
            action = select_action(knobs, history_buffer, current_obs, banks, predictive_bundle, config, device)
            action_errors.append(float(np.mean((action - episode.action[step]) ** 2)))
            state = env.transition(state, action, skill, context)
            phase = env.step_phase(phase, skill, context)

            if np.linalg.norm(state) > config.instability_threshold:
                failed = True
                break

        if failed:
            falls += 1
        else:
            final_error = float(np.mean((state - episode.state[-1]) ** 2))
            if final_error < config.success_threshold:
                successes += 1

    total = len(episodes)
    return {
        "action_mse": float(np.mean(action_errors)),
        "success_rate": successes / total,
        "fall_rate": falls / total,
    }


def primary_metric(metrics: Dict[str, float]) -> float:
    if metrics["success_rate"] == 0.0 and metrics["fall_rate"] == 0.0:
        return metrics["action_mse"]
    return metrics["action_mse"] + 0.5 * metrics["fall_rate"] - 0.25 * metrics["success_rate"]


def build_predictive_bundle(
    train_episodes: List[EpisodeBatch],
    config: ExperimentConfig,
    knobs: ResearchConfig,
    device: torch.device,
) -> PredictiveBundle | None:
    if knobs.retrieval_mode != "chunk_world_model" and not knobs.use_world_model and not knobs.use_chunk_decoder:
        return None

    wm_states, wm_actions, wm_next, wm_instability, wm_progress = build_world_model_data(
        train_episodes, instability_threshold=config.instability_threshold
    )
    chunk_states, chunk_inputs, chunk_targets = build_chunk_decoder_data(
        train_episodes,
        chunk_len=knobs.chunk_len,
        instability_threshold=config.instability_threshold,
    )
    return PredictiveBundle(
        dynamics_model=train_world_model(
            states=wm_states,
            actions=wm_actions,
            next_states=wm_next,
            instability=wm_instability,
            progress=wm_progress,
            epochs=knobs.world_model_epochs,
            batch_size=knobs.batch_size,
            lr=knobs.learning_rate,
            device=device,
        ),
        chunk_decoder=train_chunk_decoder(
            state_inputs=chunk_states,
            chunk_inputs=chunk_inputs,
            targets=chunk_targets,
            epochs=max(6, knobs.world_model_epochs),
            batch_size=knobs.batch_size,
            lr=knobs.learning_rate,
            device=device,
        ),
    )


def main() -> None:
    t0 = time.time()
    config = load_active_config()
    config.history = RESEARCH.history
    config.chunk_len = RESEARCH.chunk_len
    config.top_k = RESEARCH.top_k
    config.world_model_epochs = RESEARCH.world_model_epochs
    config.batch_size = RESEARCH.batch_size
    config.learning_rate = RESEARCH.learning_rate
    config.rerank_margin = RESEARCH.rerank_margin

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, train_episodes, test_episodes = load_episodes(config)

    memories = build_custom_memories(train_episodes, RESEARCH)
    banks = build_custom_banks(memories)
    predictive_bundle = build_predictive_bundle(train_episodes, config, RESEARCH, device)

    if env is None:
        metrics = evaluate_offline(test_episodes, banks, RESEARCH, config, device, predictive_bundle)
    else:
        metrics = evaluate_synthetic(env, test_episodes, banks, RESEARCH, config, device, predictive_bundle)

    elapsed = time.time() - t0
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    )
    score = primary_metric(metrics)

    print("---")
    print(f"primary_metric:    {score:.6f}")
    print(f"action_mse:        {metrics['action_mse']:.6f}")
    print(f"success_rate:      {metrics['success_rate']:.6f}")
    print(f"fall_rate:         {metrics['fall_rate']:.6f}")
    print(f"training_seconds:  {elapsed:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_train_episodes: {len(train_episodes)}")
    print(f"num_test_episodes:  {len(test_episodes)}")
    print("config_json:")
    print(json.dumps(RESEARCH.__dict__, indent=2))


if __name__ == "__main__":
    main()
