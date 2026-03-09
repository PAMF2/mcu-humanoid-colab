from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "mcu_autoresearch" / "prepare.py"
TRAIN = ROOT / "mcu_autoresearch" / "train.py"
LOGGER = ROOT / "mcu_autoresearch" / "log_result.py"
RUN_LOG = ROOT / "mcu_autoresearch" / "run.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap MCU autoresearch with setup + baseline")
    parser.add_argument("--preset", default="real-medium", help="Prepare preset")
    parser.add_argument(
        "--description",
        default="baseline multimodal instant",
        help="Description to log for the baseline run",
    )
    return parser.parse_args()


def run(cmd: list[str], capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def main() -> None:
    args = parse_args()

    run([sys.executable, str(PREPARE), "--preset", args.preset])

    with RUN_LOG.open("w", encoding="utf-16") as handle:
        subprocess.run(
            [sys.executable, str(TRAIN)],
            cwd=ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    run(
        [
            sys.executable,
            str(LOGGER),
            "--status",
            "keep",
            "--description",
            args.description,
        ]
    )

    print(RUN_LOG.resolve())
    print((ROOT / "mcu_autoresearch" / "results.tsv").resolve())


if __name__ == "__main__":
    main()
