from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH = ROOT / "mcu_autoresearch"
BOOTSTRAP = AUTORESEARCH / "bootstrap.py"
LAST_MESSAGE = AUTORESEARCH / "workspace" / "claude_last_message.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCU autoresearch end-to-end with Claude Code")
    parser.add_argument("--preset", default="real-medium", help="Benchmark preset passed to bootstrap.py")
    parser.add_argument(
        "--branch-tag",
        default="",
        help="Optional suffix for autoresearch branch. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional Claude model override, for example sonnet or opus.",
    )
    return parser.parse_args()


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, **kwargs)


def ensure_branch(tag: str) -> str:
    branch = f"autoresearch/{tag or datetime.now().strftime('%Y%m%d-%H%M%S')}"
    current = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True).stdout.strip()
    if current != branch:
        existing = run(["git", "branch", "--list", branch], capture_output=True).stdout.strip()
        if existing:
            run(["git", "checkout", branch])
        else:
            run(["git", "checkout", "-b", branch])
    return branch


def build_prompt(branch: str) -> str:
    return (
        "Read mcu_autoresearch/program.md. "
        "Work only in this repository. "
        f"Work on branch {branch}. "
        "The baseline has already been executed by bootstrap.py. "
        "Only edit mcu_autoresearch/train.py. "
        "Do not change benchmark code, datasets, or evaluation harness. "
        "Run an autonomous research loop with multiple iterations. "
        "After each edit, run python mcu_autoresearch/train.py > mcu_autoresearch/run.log 2>&1. "
        "Then run python mcu_autoresearch/log_result.py with keep or discard based on primary_metric. "
        "Optimize primary_metric downward. "
        "At the end, print a short summary with the best metric found."
    )


def main() -> None:
    args = parse_args()
    branch = ensure_branch(args.branch_tag)

    run([sys.executable, str(BOOTSTRAP), "--preset", args.preset])

    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "auto",
        "--allowedTools",
        "Read,Edit,Bash(git:*),Bash(python:*),Bash(ls:*),Bash(cat:*),Bash(tail:*)",
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    cmd.append(build_prompt(branch))

    try:
        result = run(cmd, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        message = (
            "Claude Code execution failed.\n"
            "--- stdout ---\n"
            f"{stdout}\n"
            "--- stderr ---\n"
            f"{stderr}\n"
        )
        LAST_MESSAGE.write_text(message, encoding="utf-8")
        print(message)
        raise

    LAST_MESSAGE.write_text(result.stdout, encoding="utf-8")

    print((AUTORESEARCH / "results.tsv").resolve())
    print(LAST_MESSAGE.resolve())


if __name__ == "__main__":
    main()
