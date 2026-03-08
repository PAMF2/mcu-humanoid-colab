from __future__ import annotations

import argparse
import hashlib
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


CAMERA_KEY = "observation.images.ego_view"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MCU .npz dataset from NVIDIA GR00T Teleop-G1 task folders"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Root of the downloaded GR00T dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Optional path to write a companion config json",
    )
    parser.add_argument("--window-len", type=int, default=48, help="Fixed sequence length")
    parser.add_argument("--stride", type=int, default=24, help="Sliding window stride")
    parser.add_argument("--sample-every", type=int, default=1, help="Frame subsampling factor")
    parser.add_argument("--limit-tasks", type=int, default=0, help="Optional max number of task folders")
    parser.add_argument("--max-episodes", type=int, default=0, help="Optional max number of episodes")
    parser.add_argument("--max-windows", type=int, default=0, help="Optional max number of exported windows")
    parser.add_argument("--vision-dim", type=int, default=128, help="Output visual feature dimension")
    parser.add_argument("--command-dim", type=int, default=8, help="Hashed task embedding dimension")
    parser.add_argument(
        "--camera-key",
        type=str,
        default=CAMERA_KEY,
        help="Video key under each task's videos directory",
    )
    return parser.parse_args()


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


def hashed_text_embedding(text: str, dim: int) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    payload = text.encode("utf-8")
    for index in range(dim):
        digest = hashlib.sha256(payload + index.to_bytes(2, "little")).digest()
        value = int.from_bytes(digest[:4], "little") / np.float32(2**32 - 1)
        vector[index] = np.float32(value * 2.0 - 1.0)
    norm = np.linalg.norm(vector) + 1e-6
    return (vector / norm).astype(np.float32)


def discover_task_dirs(dataset_dir: Path) -> list[Path]:
    if dataset_dir.is_dir() and dataset_dir.name.startswith("g1-") and (dataset_dir / "data").exists():
        return [dataset_dir]
    return sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("g1-"))


def load_task_texts(task_dir: Path) -> dict[int, str]:
    episodes_path = task_dir / "meta" / "episodes.jsonl"
    payload: dict[int, str] = {}
    if not episodes_path.exists():
        return payload
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        task_text = " ".join(item.get("tasks", []))
        payload[int(item["episode_index"])] = task_text
    return payload


def compute_phase_from_timestamp(timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scalar = timestamp.astype(np.float32)
    scalar = scalar - scalar.min()
    duration = float(scalar.max())
    if duration > 1e-6:
        scalar = scalar / duration
    scalar = (scalar * 2.0 * np.pi).astype(np.float32)
    phase = np.stack([np.sin(scalar), np.cos(scalar)], axis=-1).astype(np.float32)
    return phase, scalar


def compute_contact_proxy(state_seq: np.ndarray) -> np.ndarray:
    delta = np.diff(state_seq, axis=0, prepend=state_seq[:1])
    left_speed = np.mean(np.abs(delta[:, :6]), axis=1)
    right_speed = np.mean(np.abs(delta[:, 6:12]), axis=1)
    left_contact = 1.0 / (1.0 + left_speed)
    right_contact = 1.0 / (1.0 + right_speed)
    left_contact = left_contact / (left_contact.max() + 1e-6)
    right_contact = right_contact / (right_contact.max() + 1e-6)
    return np.stack([left_contact, right_contact], axis=-1).astype(np.float32)


def video_path_for_episode(task_dir: Path, episode_index: int, camera_key: str) -> Path:
    return task_dir / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"


def fallback_vision_features(df: pd.DataFrame, vision_dim: int) -> np.ndarray:
    if "observation.img_state_delta" in df.columns:
        source = np.asarray(df["observation.img_state_delta"].to_list(), dtype=np.float32).reshape(len(df), -1)
    else:
        source = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
    repeats = int(np.ceil(vision_dim / source.shape[1]))
    tiled = np.tile(source, (1, repeats))[:, :vision_dim]
    norms = np.linalg.norm(tiled, axis=1, keepdims=True) + 1e-6
    return (tiled / norms).astype(np.float32)


def load_video_features(
    video_path: Path,
    num_steps: int,
    vision_dim: int,
) -> np.ndarray | None:
    if not video_path.exists():
        return None

    try:
        import imageio.v3 as iio

        frames = list(iio.imiter(video_path, plugin="ffmpeg"))
    except Exception:
        return None

    if not frames:
        return None

    indices = np.linspace(0, len(frames) - 1, num=num_steps).round().astype(np.int64)
    features = [extract_visual_feature(np.asarray(frames[index]), vision_dim=vision_dim) for index in indices]
    return np.stack(features).astype(np.float32)


def write_companion_config(
    output_path: Path,
    npz_path: Path,
    sample_window: dict[str, np.ndarray],
    num_skills: int,
    task_names: list[str],
) -> None:
    payload = {
        "dataset": "npz",
        "dataset_path": str(npz_path).replace("\\", "/"),
        "test_dataset_path": None,
        "train_split": 0.8,
        "holdout_skill": None,
        "seed": 7,
        "train_episodes": 0,
        "test_episodes": 0,
        "horizon": int(sample_window["vision"].shape[0]),
        "history": 4,
        "chunk_len": 6,
        "top_k": 8,
        "num_skills": int(max(1, num_skills)),
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
        "task_names": task_names,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_windows_for_episode(
    df: pd.DataFrame,
    task_index: int,
    task_text: str,
    task_count: int,
    task_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, np.ndarray]]:
    df = df.sort_values("frame_index").reset_index(drop=True)
    if args.sample_every > 1:
        df = df.iloc[:: args.sample_every].reset_index(drop=True)
    if len(df) < args.window_len:
        return []

    episode_index = int(df["episode_index"].iloc[0])
    state_seq = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
    action_seq = np.asarray(df["action"].to_list(), dtype=np.float32)
    timestamp = np.asarray(df["timestamp"].to_list(), dtype=np.float32).reshape(-1)

    vision_seq = load_video_features(
        video_path=video_path_for_episode(task_dir, episode_index, args.camera_key),
        num_steps=len(df),
        vision_dim=args.vision_dim,
    )
    if vision_seq is None:
        vision_seq = fallback_vision_features(df, vision_dim=args.vision_dim)

    phase_seq, phase_scalar = compute_phase_from_timestamp(timestamp)
    command_embedding = hashed_text_embedding(task_text or task_dir.name, dim=args.command_dim)
    command_seq = np.repeat(command_embedding[None, :], len(df), axis=0).astype(np.float32)
    contact_seq = compute_contact_proxy(state_seq)
    context_seq = np.zeros((len(df), task_count), dtype=np.float32)
    context_seq[:, task_index] = 1.0
    skill_seq = np.full((len(df),), task_index, dtype=np.int64)

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

    windows = []
    for start in range(0, len(df) - args.window_len + 1, args.stride):
        windows.append({key: value[start : start + args.window_len] for key, value in fields.items()})
    return windows


def main() -> None:
    args = parse_args()
    task_dirs = discover_task_dirs(args.dataset_dir)
    if args.limit_tasks > 0:
        task_dirs = task_dirs[: args.limit_tasks]
    if not task_dirs:
        raise SystemExit(f"No GR00T task folders found in {args.dataset_dir}")

    windows: list[dict[str, np.ndarray]] = []
    exported_episodes = 0

    for task_index, task_dir in enumerate(task_dirs):
        task_texts = load_task_texts(task_dir)
        parquet_files = sorted((task_dir / "data").rglob("*.parquet"))
        for parquet_path in parquet_files:
            df = pd.read_parquet(parquet_path)
            task_text = task_texts.get(int(df["episode_index"].iloc[0]), task_dir.name)
            episode_windows = build_windows_for_episode(
                df=df,
                task_index=task_index,
                task_text=task_text,
                task_count=len(task_dirs),
                task_dir=task_dir,
                args=args,
            )
            if not episode_windows:
                continue
            windows.extend(episode_windows)
            exported_episodes += 1
            if args.max_episodes > 0 and exported_episodes >= args.max_episodes:
                break
            if args.max_windows > 0 and len(windows) >= args.max_windows:
                windows = windows[: args.max_windows]
                break
        if args.max_episodes > 0 and exported_episodes >= args.max_episodes:
            break
        if args.max_windows > 0 and len(windows) >= args.max_windows:
            break

    if not windows:
        raise SystemExit("No windows were exported. Check dataset path, episode limits, and video files.")

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
                "episodes": exported_episodes,
                "vision_shape": list(payload["vision"].shape),
                "state_shape": list(payload["state"].shape),
                "action_shape": list(payload["action"].shape),
                "context_shape": list(payload["context"].shape),
            },
            indent=2,
        )
    )

    if args.config_output is not None:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        write_companion_config(
            args.config_output,
            args.output,
            windows[0],
            num_skills=len(task_dirs),
            task_names=[task_dir.name for task_dir in task_dirs],
        )
        print(args.config_output.resolve())


if __name__ == "__main__":
    main()
