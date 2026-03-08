from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .schema import EpisodeBatch


def _goal_state(episode: EpisodeBatch, step: int) -> np.ndarray:
    if episode.command.shape[-1] == episode.state.shape[-1]:
        return episode.command[step]
    return episode.state[-1]


class TinyWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.instability_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state_vec: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(torch.cat([state_vec, action], dim=-1))
        return (
            self.next_state_head(hidden),
            self.instability_head(hidden).squeeze(-1),
            self.value_head(hidden).squeeze(-1),
        )


class TinyChunkDecoder(nn.Module):
    def __init__(self, state_dim: int, chunk_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + chunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vec: torch.Tensor, chunk_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_vec, chunk_vec], dim=-1)).squeeze(-1)


def build_world_model_data(
    episodes: Sequence[EpisodeBatch],
    instability_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_inputs = []
    actions = []
    next_states = []
    instability = []
    progress = []

    for episode in episodes:
        current = np.concatenate(
            [episode.state, episode.contact, episode.phase, episode.command], axis=-1
        )
        nxt = np.concatenate(
            [episode.state[1:], episode.contact[1:], episode.phase[1:], episode.command[1:]],
            axis=-1,
        )
        for step in range(len(episode.action) - 1):
            state_inputs.append(current[step])
            actions.append(episode.action[step])
            next_states.append(nxt[step])
            goal_state = _goal_state(episode, step)
            current_goal = np.mean((episode.state[step] - goal_state) ** 2)
            next_goal = np.mean((episode.state[step + 1] - goal_state) ** 2)
            progress.append(current_goal - next_goal)
            instability.append(
                1.0 if np.linalg.norm(episode.state[step + 1]) > instability_threshold else 0.0
            )

    return (
        np.stack(state_inputs),
        np.stack(actions),
        np.stack(next_states),
        np.asarray(instability, dtype=np.float32),
        np.asarray(progress, dtype=np.float32),
    )


def build_chunk_decoder_data(
    episodes: Sequence[EpisodeBatch],
    chunk_len: int,
    instability_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_inputs = []
    chunk_inputs = []
    targets = []

    for episode in episodes:
        for step in range(len(episode.action) - chunk_len):
            goal_state = _goal_state(episode, step)
            state_inputs.append(
                np.concatenate(
                    [
                        episode.state[step],
                        episode.contact[step],
                        episode.phase[step],
                        episode.command[step],
                    ],
                    axis=0,
                )
            )
            chunk_inputs.append(episode.action[step : step + chunk_len].reshape(-1))

            score = 0.0
            for offset in range(1, chunk_len + 1):
                current_error = np.mean(
                    (episode.state[step + offset - 1] - goal_state) ** 2
                )
                next_error = np.mean((episode.state[step + offset] - goal_state) ** 2)
                score += current_error - next_error
                if np.linalg.norm(episode.state[step + offset]) > instability_threshold:
                    score -= 0.6
            targets.append(score)

    return (
        np.stack(state_inputs),
        np.stack(chunk_inputs),
        np.asarray(targets, dtype=np.float32),
    )


def train_world_model(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    instability: np.ndarray,
    progress: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> TinyWorldModel:
    model = TinyWorldModel(state_dim=states.shape[1], action_dim=actions.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
        torch.as_tensor(next_states, dtype=torch.float32),
        torch.as_tensor(instability, dtype=torch.float32),
        torch.as_tensor(progress, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_state, batch_action, batch_next, batch_instability, batch_progress in loader:
            batch_state = batch_state.to(device)
            batch_action = batch_action.to(device)
            batch_next = batch_next.to(device)
            batch_instability = batch_instability.to(device)
            batch_progress = batch_progress.to(device)

            pred_next, pred_instability, pred_value = model(batch_state, batch_action)
            loss = (
                mse_loss(pred_next, batch_next)
                + 0.3 * bce_loss(pred_instability, batch_instability)
                + 0.4 * mse_loss(pred_value, batch_progress)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_chunk_decoder(
    state_inputs: np.ndarray,
    chunk_inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> TinyChunkDecoder:
    model = TinyChunkDecoder(
        state_dim=state_inputs.shape[1], chunk_dim=chunk_inputs.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    dataset = TensorDataset(
        torch.as_tensor(state_inputs, dtype=torch.float32),
        torch.as_tensor(chunk_inputs, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_state, batch_chunk, batch_target in loader:
            batch_state = batch_state.to(device)
            batch_chunk = batch_chunk.to(device)
            batch_target = batch_target.to(device)

            pred = model(batch_state, batch_chunk)
            loss = mse_loss(pred, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model
