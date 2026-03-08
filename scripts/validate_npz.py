from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcu_humanoid_colab.data import load_npz_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an .npz dataset against the MCU humanoid schema")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .npz dataset")
    parser.add_argument("--context-dim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_npz_episodes(args.dataset, context_dim=args.context_dim)
    first = episodes[0]
    summary = {
        "episodes": len(episodes),
        "vision_shape": list(first.vision.shape),
        "proprio_shape": list(first.proprio.shape),
        "contact_shape": list(first.contact.shape),
        "phase_shape": list(first.phase.shape),
        "command_shape": list(first.command.shape),
        "action_shape": list(first.action.shape),
        "state_shape": list(first.state.shape),
        "mean_action_abs": float(np.mean(np.abs(first.action))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
