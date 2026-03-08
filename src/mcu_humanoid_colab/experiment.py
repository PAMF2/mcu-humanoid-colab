from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_npz_episodes, split_episodes
from .memory import build_banks, build_memories, flatten_history
from .models import (
    build_chunk_decoder_data,
    build_world_model_data,
    train_chunk_decoder,
    train_world_model,
)
from .schema import EpisodeBatch, PredictiveBundle
from .synthetic import SyntheticHumanoidBenchmark, build_episode_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_phase(phase_vec: torch.Tensor) -> torch.Tensor:
    return phase_vec / (phase_vec.norm(dim=-1, keepdim=True) + 1e-6)


def load_episodes(
    config: ExperimentConfig,
) -> tuple[SyntheticHumanoidBenchmark | None, List[EpisodeBatch], List[EpisodeBatch]]:
    if config.dataset == "synthetic":
        env = SyntheticHumanoidBenchmark(
            seed=config.seed,
            num_skills=config.num_skills,
            state_dim=config.state_dim,
            vision_dim=config.vision_dim,
            context_dim=config.context_dim,
            action_dim=config.action_dim,
        )
        train = build_episode_split(env, num_episodes=config.train_episodes, horizon=config.horizon)
        test = build_episode_split(env, num_episodes=config.test_episodes, horizon=config.horizon)
        return env, train, test

    if config.dataset == "npz" and config.dataset_path:
        episodes = load_npz_episodes(config.dataset_path, context_dim=config.context_dim)
        train, test = split_episodes(episodes, train_split=config.train_split)
        return None, train, test

    raise ValueError("Unsupported dataset setup. Use synthetic or dataset=npz with dataset_path.")


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
            pred_contact = torch.sigmoid(
                pred_next[:, proprio_dim : proprio_dim + contact_dim]
            )
            pred_phase = normalize_phase(
                pred_next[:, proprio_dim + contact_dim : proprio_dim + contact_dim + phase_dim]
            )
            current = torch.cat([pred_proprio, pred_contact, pred_phase, command], dim=-1)
            score += float((pred_value - 0.6 * torch.sigmoid(instability)).item())
        return score


def select_action(
    mode: str,
    history_buffer: List[np.ndarray],
    current_obs: Dict[str, np.ndarray],
    banks,
    predictive_bundle: PredictiveBundle | None,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    if mode == "vision_only":
        _, indices = banks["vision"].topk(current_obs["vision"], k=1)
        return banks["vision"].values[indices[0]].cpu().numpy()

    if mode == "multimodal_instant":
        query = np.concatenate(
            [
                current_obs["vision"],
                current_obs["proprio"],
                current_obs["contact"],
                current_obs["phase"],
                current_obs["command"],
            ],
            axis=0,
        )
        _, indices = banks["multimodal"].topk(query, k=1)
        return banks["multimodal"].values[indices[0]].cpu().numpy()

    chunk_query = flatten_history(np.stack(history_buffer[-config.history :], axis=0))
    similarities, indices = banks["chunk"].topk(chunk_query, k=config.top_k)
    chunks = banks["chunk"].values[indices]

    if mode == "chunk_only" or predictive_bundle is None:
        return chunks[0, 0].cpu().numpy()

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
        rollout = rollout_score(
            world_model=predictive_bundle.dynamics_model,
            state_vec=current_state,
            chunk=chunk,
            goal_command=current_obs["command"],
            device=device,
            proprio_dim=config.state_dim,
            contact_dim=2,
            phase_dim=2,
        )
        decoder_state = torch.as_tensor(
            current_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        decoder_input = chunk.reshape(1, -1).to(device)
        with torch.no_grad():
            decoder_score = float(
                predictive_bundle.chunk_decoder(decoder_state, decoder_input).item()
            )
        scores.append(decoder_score + 0.2 * rollout + 0.15 * float(similarity.item()))

    best_index = int(np.argmax(scores))
    if best_index != 0 and scores[best_index] < scores[0] + config.rerank_margin:
        best_index = 0
    return chunks[best_index, 0].cpu().numpy()


def evaluate_synthetic_controller(
    mode: str,
    env: SyntheticHumanoidBenchmark,
    episodes: List[EpisodeBatch],
    banks,
    config: ExperimentConfig,
    device: torch.device,
    predictive_bundle: PredictiveBundle | None = None,
) -> Dict[str, float]:
    action_errors = []
    state_errors = []
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
            temporal_obs = np.concatenate([vision, proprio, contact, phase_vec, command], axis=0)
            while len(history_buffer) < config.history:
                history_buffer.append(temporal_obs.copy())
            history_buffer.append(temporal_obs.copy())
            history_buffer = history_buffer[-config.history :]

            current_obs = {
                "vision": vision,
                "proprio": proprio,
                "contact": contact,
                "phase": phase_vec,
                "command": command,
                "latent_state": state.copy(),
            }
            action = select_action(
                mode=mode,
                history_buffer=history_buffer,
                current_obs=current_obs,
                banks=banks,
                predictive_bundle=predictive_bundle,
                config=config,
                device=device,
            )

            action_errors.append(float(np.mean((action - episode.action[step]) ** 2)))
            state = env.transition(state, action, skill, context)
            phase = env.step_phase(phase, skill, context)

            ref_index = min(step + 1, len(episode.state) - 1)
            state_errors.append(float(np.mean((state - episode.state[ref_index]) ** 2)))

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
        "state_mse": float(np.mean(state_errors)),
        "success_rate": successes / total,
        "fall_rate": falls / total,
    }


def evaluate_offline_controller(
    mode: str,
    episodes: List[EpisodeBatch],
    banks,
    config: ExperimentConfig,
    device: torch.device,
    predictive_bundle: PredictiveBundle | None = None,
) -> Dict[str, float]:
    action_errors = []

    for episode in episodes:
        history_buffer: List[np.ndarray] = []
        for step in range(config.history - 1, len(episode.action)):
            temporal_obs = np.concatenate(
                [
                    episode.vision[step],
                    episode.proprio[step],
                    episode.contact[step],
                    episode.phase[step],
                    episode.command[step],
                ],
                axis=0,
            )
            while len(history_buffer) < config.history:
                history_buffer.append(temporal_obs.copy())
            history_buffer.append(temporal_obs.copy())
            history_buffer = history_buffer[-config.history :]

            current_obs = {
                "vision": episode.vision[step],
                "proprio": episode.proprio[step],
                "contact": episode.contact[step],
                "phase": episode.phase[step],
                "command": episode.command[step],
                "latent_state": episode.state[step],
            }
            action = select_action(
                mode=mode,
                history_buffer=history_buffer,
                current_obs=current_obs,
                banks=banks,
                predictive_bundle=predictive_bundle,
                config=config,
                device=device,
            )
            action_errors.append(float(np.mean((action - episode.action[step]) ** 2)))

    return {
        "action_mse": float(np.mean(action_errors)),
        "state_mse": 0.0,
        "success_rate": 0.0,
        "fall_rate": 0.0,
    }


def run_experiment(config: ExperimentConfig, force_cpu: bool = False) -> Dict[str, Dict[str, float]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    env, train_episodes, test_episodes = load_episodes(config)
    memories = build_memories(train_episodes, history=config.history, chunk_len=config.chunk_len)
    banks = build_banks(memories)

    wm_states, wm_actions, wm_next, wm_instability, wm_progress = build_world_model_data(
        train_episodes, instability_threshold=config.instability_threshold
    )
    chunk_states, chunk_inputs, chunk_targets = build_chunk_decoder_data(
        train_episodes,
        chunk_len=config.chunk_len,
        instability_threshold=config.instability_threshold,
    )
    predictive_bundle = PredictiveBundle(
        dynamics_model=train_world_model(
            states=wm_states,
            actions=wm_actions,
            next_states=wm_next,
            instability=wm_instability,
            progress=wm_progress,
            epochs=config.world_model_epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            device=device,
        ),
        chunk_decoder=train_chunk_decoder(
            state_inputs=chunk_states,
            chunk_inputs=chunk_inputs,
            targets=chunk_targets,
            epochs=max(6, config.world_model_epochs),
            batch_size=config.batch_size,
            lr=config.learning_rate,
            device=device,
        ),
    )

    if env is not None:
        return {
            "vision_only": evaluate_synthetic_controller(
                "vision_only", env, test_episodes, banks, config, device
            ),
            "multimodal_instant": evaluate_synthetic_controller(
                "multimodal_instant", env, test_episodes, banks, config, device
            ),
            "chunk_only": evaluate_synthetic_controller(
                "chunk_only", env, test_episodes, banks, config, device
            ),
            "chunk_world_model": evaluate_synthetic_controller(
                "chunk_world_model",
                env,
                test_episodes,
                banks,
                config,
                device,
                predictive_bundle,
            ),
        }

    return {
        "vision_only": evaluate_offline_controller(
            "vision_only", test_episodes, banks, config, device
        ),
        "multimodal_instant": evaluate_offline_controller(
            "multimodal_instant", test_episodes, banks, config, device
        ),
        "chunk_only": evaluate_offline_controller(
            "chunk_only", test_episodes, banks, config, device
        ),
        "chunk_world_model": evaluate_offline_controller(
            "chunk_world_model",
            test_episodes,
            banks,
            config,
            device,
            predictive_bundle,
        ),
    }


def save_results(results: Dict[str, Dict[str, float]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
