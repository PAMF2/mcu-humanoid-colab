from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the local structure of a GR00T Teleop-G1 dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Root of the downloaded GR00T dataset")
    parser.add_argument("--task", type=str, default="", help="Optional task folder name to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.dataset_dir.is_dir()
        and args.dataset_dir.name.startswith("g1-")
        and (args.dataset_dir / "data").exists()
    ):
        task_dirs = [args.dataset_dir]
    else:
        task_dirs = sorted(
            path for path in args.dataset_dir.iterdir() if path.is_dir() and path.name.startswith("g1-")
        )
    if args.task:
        task_dirs = [path for path in task_dirs if path.name == args.task]
    if not task_dirs:
        raise SystemExit("No matching task folders found")

    summaries = []
    for task_dir in task_dirs:
        parquet_files = sorted((task_dir / "data").rglob("*.parquet"))
        if not parquet_files:
            continue
        sample_df = pd.read_parquet(parquet_files[0])
        summaries.append(
            {
                "task": task_dir.name,
                "parquet_files": len(parquet_files),
                "columns": list(sample_df.columns),
                "rows_in_first_parquet": int(len(sample_df)),
                "videos_dir_exists": (task_dir / "videos").exists(),
                "meta_dir_exists": (task_dir / "meta").exists(),
            }
        )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
