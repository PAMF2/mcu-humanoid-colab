from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.synthetic import SyntheticHumanoidBenchmark, build_episode_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the synthetic benchmark to a real .npz schema")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-skills", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--vision-dim", type=int, default=10)
    parser.add_argument("--context-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SyntheticHumanoidBenchmark(
        seed=args.seed,
        num_skills=args.num_skills,
        state_dim=args.state_dim,
        vision_dim=args.vision_dim,
        context_dim=args.context_dim,
        action_dim=args.action_dim,
    )
    episodes = build_episode_split(env, num_episodes=args.episodes, horizon=args.horizon)
    payload = {
        "vision": np.stack([episode.vision for episode in episodes]),
        "proprio": np.stack([episode.proprio for episode in episodes]),
        "contact": np.stack([episode.contact for episode in episodes]),
        "phase": np.stack([episode.phase for episode in episodes]),
        "command": np.stack([episode.command for episode in episodes]),
        "action": np.stack([episode.action for episode in episodes]),
        "state": np.stack([episode.state for episode in episodes]),
        "skill": np.stack([episode.skill for episode in episodes]),
        "context": np.stack([episode.context for episode in episodes]),
        "phase_scalar": np.stack([episode.phase_scalar for episode in episodes]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
