from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


IMAGE_COLUMNS = [
    "observation.images.image",
    "observation.images.top",
    "observation.images.image2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MCU .npz dataset from real parquet episodes")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing parquet files")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--window-len", type=int, default=48, help="Fixed sequence length")
    parser.add_argument("--stride", type=int, default=24, help="Sliding window stride")
    parser.add_argument("--sample-every", type=int, default=1, help="Frame subsampling factor")
    parser.add_argument("--limit-files", type=int, default=0, help="Optional max number of parquet files")
    parser.add_argument("--max-windows", type=int, default=0, help="Optional max number of exported windows")
    parser.add_argument("--command-horizon", type=int, default=8, help="Future horizon for command proxy")
    parser.add_argument("--vision-dim", type=int, default=128, help="Output visual feature dimension")
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Optional path to write a companion config json",
    )
    return parser.parse_args()


def find_image_column(df: pd.DataFrame) -> str | None:
    for column in IMAGE_COLUMNS:
        if column in df.columns:
            return column
    return None


def decode_image(cell: object) -> np.ndarray:
    if isinstance(cell, dict) and "bytes" in cell:
        image = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        return np.asarray(image)
    raise ValueError("Unsupported image cell format")


def resize_image(image: np.ndarray, size: int = 64) -> np.ndarray:
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize((size, size), Image.BILINEAR))


def extract_visual_feature(image: np.ndarray, vision_dim: int) -> np.ndarray:
    resized = resize_image(image, size=64).astype(np.float32) / 255.0
    gray = resized.mean(axis=2)

    pooled = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
    hist_gray, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
    rgb_mean = resized.mean(axis=(0, 1))
    rgb_std = resized.std(axis=(0, 1))
    grad_y, grad_x = np.gradient(gray)
    grad_stats = np.array(
        [
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate(
        [
            pooled.astype(np.float32),
            hist_gray.astype(np.float32),
            rgb_mean.astype(np.float32),
            rgb_std.astype(np.float32),
            grad_stats,
        ]
    )
    if len(feat) < vision_dim:
        feat = np.pad(feat, (0, vision_dim - len(feat)))
    else:
        feat = feat[:vision_dim]
    norm = np.linalg.norm(feat) + 1e-6
    return (feat / norm).astype(np.float32)


def compute_phase(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 1:
        scalar = np.zeros((length,), dtype=np.float32)
    else:
        scalar = np.linspace(0.0, 2.0 * np.pi, num=length, endpoint=False, dtype=np.float32)
    phase = np.stack([np.sin(scalar), np.cos(scalar)], axis=-1).astype(np.float32)
    return phase, scalar


def compute_command_proxy(state_seq: np.ndarray, horizon: int) -> np.ndarray:
    commands = []
    total = len(state_seq)
    for index in range(total):
        end = min(total, index + horizon + 1)
        future = state_seq[index + 1 : end]
        if len(future) == 0:
            future = state_seq[index : index + 1]
        commands.append(future.mean(axis=0))
    return np.stack(commands).astype(np.float32)


def compute_contact_proxy(state_seq: np.ndarray, effort_seq: np.ndarray | None) -> np.ndarray:
    if effort_seq is not None:
        half = max(1, effort_seq.shape[1] // 2)
        left = np.mean(np.abs(effort_seq[:, :half]), axis=1)
        right = np.mean(np.abs(effort_seq[:, half:]), axis=1) if effort_seq.shape[1] > half else left
    else:
        delta = np.diff(state_seq, axis=0, prepend=state_seq[:1])
        half = max(1, delta.shape[1] // 2)
        left = 1.0 / (1.0 + np.mean(np.abs(delta[:, :half]), axis=1))
        right = 1.0 / (1.0 + np.mean(np.abs(delta[:, half:]), axis=1)) if delta.shape[1] > half else left

    left = left / (np.max(left) + 1e-6)
    right = right / (np.max(right) + 1e-6)
    return np.stack([left, right], axis=-1).astype(np.float32)


def window_sequence(sequence: np.ndarray, window_len: int, stride: int) -> list[np.ndarray]:
    windows = []
    total = len(sequence)
    if total < window_len:
        return windows
    for start in range(0, total - window_len + 1, stride):
        windows.append(sequence[start : start + window_len])
    return windows


def build_windows_for_episode(
    episode_df: pd.DataFrame,
    image_column: str,
    args: argparse.Namespace,
) -> list[dict[str, np.ndarray]]:
    episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)
    if args.sample_every > 1:
        episode_df = episode_df.iloc[:: args.sample_every].reset_index(drop=True)
    if len(episode_df) < args.window_len:
        return []

    state_seq = np.stack(episode_df["observation.state"].to_list()).astype(np.float32)
    action_seq = np.stack(episode_df["action"].to_list()).astype(np.float32)
    effort_seq = (
        np.stack(episode_df["observation.effort"].to_list()).astype(np.float32)
        if "observation.effort" in episode_df.columns
        else None
    )
    vision_seq = np.stack(
        [
            extract_visual_feature(decode_image(cell), vision_dim=args.vision_dim)
            for cell in episode_df[image_column].to_list()
        ]
    ).astype(np.float32)

    phase_seq, phase_scalar = compute_phase(len(episode_df))
    command_seq = compute_command_proxy(state_seq, horizon=args.command_horizon)
    contact_seq = compute_contact_proxy(state_seq, effort_seq)
    context_seq = np.zeros((len(episode_df), 3), dtype=np.float32)
    skill_seq = np.zeros((len(episode_df),), dtype=np.int64)

    fields = {
        "vision": vision_seq,
        "proprio": state_seq.copy(),
        "contact": contact_seq,
        "phase": phase_seq,
        "command": command_seq,
        "action": action_seq,
        "state": state_seq,
        "skill": skill_seq,
        "context": context_seq,
        "phase_scalar": phase_scalar,
    }

    total_windows = []
    for start in range(0, len(episode_df) - args.window_len + 1, args.stride):
        total_windows.append(
            {
                key: value[start : start + args.window_len]
                for key, value in fields.items()
            }
        )
    return total_windows


def write_companion_config(
    output_path: Path,
    npz_path: Path,
    sample_window: dict[str, np.ndarray],
) -> None:
    payload = {
        "dataset": "npz",
        "dataset_path": str(npz_path).replace("\\", "/"),
        "train_split": 0.8,
        "seed": 7,
        "train_episodes": 0,
        "test_episodes": 0,
        "horizon": int(sample_window["vision"].shape[0]),
        "history": 4,
        "chunk_len": 6,
        "top_k": 8,
        "num_skills": 1,
        "state_dim": int(sample_window["state"].shape[1]),
        "vision_dim": int(sample_window["vision"].shape[1]),
        "context_dim": int(sample_window["context"].shape[1]),
        "action_dim": int(sample_window["action"].shape[1]),
        "world_model_epochs": 18,
        "batch_size": 256,
        "learning_rate": 0.001,
        "instability_threshold": 4.25,
        "success_threshold": 0.18,
        "rerank_margin": 0.08,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    parquet_files = sorted(args.dataset_dir.glob("*.parquet"))
    if args.limit_files > 0:
        parquet_files = parquet_files[: args.limit_files]
    if not parquet_files:
        raise SystemExit(f"No parquet files found in {args.dataset_dir}")

    windows: list[dict[str, np.ndarray]] = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        image_column = find_image_column(df)
        if image_column is None:
            print(f"Skipping {parquet_path.name}: no supported image column")
            continue

        for _, episode_df in df.groupby("episode_index"):
            episode_windows = build_windows_for_episode(episode_df, image_column, args)
            windows.extend(episode_windows)
            if args.max_windows > 0 and len(windows) >= args.max_windows:
                windows = windows[: args.max_windows]
                break
        if args.max_windows > 0 and len(windows) >= args.max_windows:
            break

    if not windows:
        raise SystemExit("No windows were exported. Check dataset path and image columns.")

    payload = {
        key: np.stack([window[key] for window in windows]).astype(
            np.int64 if key == "skill" else np.float32
        )
        for key in windows[0].keys()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output.resolve())
    print(
        json.dumps(
            {
                "windows": len(windows),
                "vision_shape": list(payload["vision"].shape),
                "state_shape": list(payload["state"].shape),
                "action_shape": list(payload["action"].shape),
            },
            indent=2,
        )
    )

    if args.config_output is not None:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        write_companion_config(args.config_output, args.output, windows[0])
        print(args.config_output.resolve())


if __name__ == "__main__":
    main()
