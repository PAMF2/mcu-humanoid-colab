from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeBatch:
    vision: np.ndarray
    proprio: np.ndarray
    contact: np.ndarray
    phase: np.ndarray
    command: np.ndarray
    action: np.ndarray
    state: np.ndarray
    skill: np.ndarray
    context: np.ndarray
    phase_scalar: np.ndarray


@dataclass
class PredictiveBundle:
    dynamics_model: object
    chunk_decoder: object
