from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "mcu_autoresearch" / "prepare.py"
TRAIN = ROOT / "mcu_autoresearch" / "train.py"
LOGGER = ROOT / "mcu_autoresearch" / "log_result.py"
RUN_LOG = ROOT / "mcu_autoresearch" / "run.log"


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
    description = "baseline"

    run([sys.executable, str(PREPARE)])

    with RUN_LOG.open("w", encoding="utf-8") as handle:
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
            description,
        ]
    )

    print(RUN_LOG.resolve())
    print((ROOT / "mcu_autoresearch" / "results.tsv").resolve())


if __name__ == "__main__":
    main()
