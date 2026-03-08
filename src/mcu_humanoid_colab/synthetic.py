from __future__ import annotations

import math
from typing import List

import numpy as np

from .schema import EpisodeBatch


class SyntheticHumanoidBenchmark:
    def __init__(
        self,
        seed: int,
        num_skills: int = 4,
        state_dim: int = 6,
        vision_dim: int = 10,
        context_dim: int = 3,
        action_dim: int = 6,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_skills = num_skills
        self.state_dim = state_dim
        self.vision_dim = vision_dim
        self.context_dim = context_dim
        self.action_dim = action_dim

        feature_dim = state_dim + context_dim + 7
        self.skill_bias = self.rng.normal(0.0, 0.35, size=(num_skills, state_dim))
        self.goal_states = self.rng.normal(0.0, 1.2, size=(num_skills, state_dim))
        self.action_weights = self.rng.normal(
            0.0, 0.45, size=(num_skills, action_dim, feature_dim)
        )
        self.vision_weights = self.rng.normal(0.0, 0.55, size=(vision_dim, state_dim))
        self.texture_weights = self.rng.normal(
            0.0, 0.45, size=(vision_dim, context_dim)
        )
        self.phase_offsets = np.linspace(0.0, math.pi, num_skills, endpoint=False)
        self.command_basis = (
            self.goal_states + self.rng.normal(0.0, 0.12, size=(num_skills, state_dim))
        )
        self.command_vision_weights = self.rng.normal(
            0.0, 0.35, size=(vision_dim, state_dim)
        )

    def command_for_skill(self, skill: int) -> np.ndarray:
        return self.command_basis[skill].copy()

    def contact_from_phase(self, skill: int, phase: float) -> np.ndarray:
        offset = self.phase_offsets[skill]
        left = 1.0 if math.sin(phase + offset) >= 0.0 else 0.0
        right = 1.0 if math.sin(phase + offset + math.pi) >= 0.0 else 0.0
        return np.array([left, right], dtype=np.float32)

    def phase_rate(self, skill: int, context: np.ndarray) -> float:
        return 0.22 + 0.015 * skill + 0.01 * context[0] + 0.01 * context[1]

    def expert_action(
        self,
        state: np.ndarray,
        skill: int,
        context: np.ndarray,
        phase: float,
        velocity: np.ndarray,
        phase_rate: float,
    ) -> np.ndarray:
        contact = self.contact_from_phase(skill, phase)
        phase_vec = np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)
        hidden_dynamics = np.concatenate(
            [velocity[:2], np.array([phase_rate], dtype=np.float32)], axis=0
        )
        features = np.concatenate(
            [state, context, phase_vec, contact, hidden_dynamics], axis=0
        )
        raw = self.action_weights[skill] @ features
        goal_pull = 0.45 * (self.goal_states[skill] - state)
        return np.tanh(raw + goal_pull).astype(np.float32)

    def transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        skill: int,
        context: np.ndarray,
    ) -> np.ndarray:
        context_pad = np.pad(
            context, (0, self.state_dim - self.context_dim), mode="constant"
        )
        next_state = (
            0.82 * state
            + 0.34 * np.tanh(action)
            + 0.11 * self.skill_bias[skill]
            + 0.07 * context_pad
            + 0.08 * (self.goal_states[skill] - state)
            + self.rng.normal(0.0, 0.015, size=self.state_dim)
        )
        return next_state.astype(np.float32)

    def observe(
        self,
        state: np.ndarray,
        skill: int,
        context: np.ndarray,
        phase: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        command = self.command_for_skill(skill)
        phase_vec = np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)
        contact = self.contact_from_phase(skill, phase)
        vision = (
            self.vision_weights @ np.abs(state)
            + 0.55 * (self.texture_weights @ context)
            + 0.15 * (self.command_vision_weights @ command)
            + self.rng.normal(0.0, 0.03, size=self.vision_dim)
        )
        vision = np.tanh(vision).astype(np.float32)
        proprio = (state + self.rng.normal(0.0, 0.02, size=self.state_dim)).astype(
            np.float32
        )
        return vision, proprio, contact, phase_vec, command.astype(np.float32)

    def step_phase(self, phase: float, skill: int, context: np.ndarray) -> float:
        return float((phase + self.phase_rate(skill, context)) % (2.0 * math.pi))

    def generate_episode(self, horizon: int) -> EpisodeBatch:
        skill = int(self.rng.integers(0, self.num_skills))
        context = self.rng.normal(0.0, 1.0, size=self.context_dim).astype(np.float32)
        state = self.rng.normal(0.0, 0.25, size=self.state_dim).astype(np.float32)
        prev_state = state.copy()
        phase = float(self.rng.uniform(0.0, 2.0 * math.pi))

        vision_seq = []
        proprio_seq = []
        contact_seq = []
        phase_seq = []
        command_seq = []
        action_seq = []
        state_seq = []
        skill_seq = []
        context_seq = []
        phase_scalar_seq = []

        for _ in range(horizon):
            vision, proprio, contact, phase_vec, command = self.observe(
                state, skill, context, phase
            )
            velocity = state - prev_state
            phase_rate = self.phase_rate(skill, context)
            action = self.expert_action(state, skill, context, phase, velocity, phase_rate)

            vision_seq.append(vision)
            proprio_seq.append(proprio)
            contact_seq.append(contact)
            phase_seq.append(phase_vec)
            command_seq.append(command)
            action_seq.append(action)
            state_seq.append(state.copy())
            skill_seq.append(skill)
            context_seq.append(context.copy())
            phase_scalar_seq.append(phase)

            prev_state = state.copy()
            state = self.transition(state, action, skill, context)
            phase = self.step_phase(phase, skill, context)

        return EpisodeBatch(
            vision=np.stack(vision_seq),
            proprio=np.stack(proprio_seq),
            contact=np.stack(contact_seq),
            phase=np.stack(phase_seq),
            command=np.stack(command_seq),
            action=np.stack(action_seq),
            state=np.stack(state_seq),
            skill=np.asarray(skill_seq, dtype=np.int64),
            context=np.stack(context_seq),
            phase_scalar=np.asarray(phase_scalar_seq, dtype=np.float32),
        )


def build_episode_split(
    env: SyntheticHumanoidBenchmark,
    num_episodes: int,
    horizon: int,
) -> List[EpisodeBatch]:
    return [env.generate_episode(horizon=horizon) for _ in range(num_episodes)]
