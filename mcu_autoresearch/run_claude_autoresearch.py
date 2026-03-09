from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH = ROOT / "mcu_autoresearch"
BOOTSTRAP = AUTORESEARCH / "bootstrap.py"
PROMPT_FILE = AUTORESEARCH / "agent_prompt.md"
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
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    kickoff = (
        "\n\nAutonomous kickoff:\n"
        f"- Work on branch `{branch}`\n"
        "- Start from the existing baseline already produced by bootstrap.py\n"
        "- Continue autonomous experimentation immediately\n"
        "- Keep the loop narrow and only edit `mcu_autoresearch/train.py`\n"
    )
    return prompt + kickoff


def main() -> None:
    args = parse_args()
    branch = ensure_branch(args.branch_tag)

    run([sys.executable, str(BOOTSTRAP), "--preset", args.preset])

    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
    ]
    if args.model:
        cmd.extend(["--model", args.model])

    result = run(cmd, input=build_prompt(branch), capture_output=True)
    LAST_MESSAGE.write_text(result.stdout, encoding="utf-8")

    print((AUTORESEARCH / "results.tsv").resolve())
    print(LAST_MESSAGE.resolve())


if __name__ == "__main__":
    main()
