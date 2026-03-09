from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "mcu_autoresearch" / "results.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append an MCU autoresearch run to results.tsv")
    parser.add_argument("--log", type=Path, default=ROOT / "mcu_autoresearch" / "run.log")
    parser.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    parser.add_argument("--description", type=str, required=True)
    return parser.parse_args()


def git_short_hash() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_metric(log_text: str, key: str, default: float) -> float:
    prefix = f"{key}:"
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return float(line.split(":", 1)[1].strip())
    return default


def require_metric(log_text: str, key: str) -> float:
    prefix = f"{key}:"
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return float(line.split(":", 1)[1].strip())
    raise SystemExit(f"Missing `{key}` in log. Refusing to append an invalid result.")


def main() -> None:
    args = parse_args()
    log_text = read_log_text(args.log)
    commit = git_short_hash()

    if args.status == "crash":
        primary_metric = 0.0
        action_mse = 0.0
        success_rate = 0.0
        fall_rate = 0.0
        memory_gb = 0.0
    else:
        primary_metric = require_metric(log_text, "primary_metric")
        action_mse = require_metric(log_text, "action_mse")
        success_rate = require_metric(log_text, "success_rate")
        fall_rate = require_metric(log_text, "fall_rate")
        peak_vram_mb = parse_metric(log_text, "peak_vram_mb", 0.0)
        memory_gb = peak_vram_mb / 1024.0

    row = (
        f"{commit}\t"
        f"{primary_metric:.6f}\t"
        f"{action_mse:.6f}\t"
        f"{success_rate:.6f}\t"
        f"{fall_rate:.6f}\t"
        f"{memory_gb:.1f}\t"
        f"{args.status}\t"
        f"{args.description}\n"
    )
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(row)
    print(RESULTS_PATH.resolve())
    print(row.strip())


if __name__ == "__main__":
    main()
