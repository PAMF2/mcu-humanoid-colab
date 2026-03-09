from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREPARE = ROOT / "prepare.py"
TRAIN = ROOT / "train.py"
RUN_LOG = ROOT / "run.log"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
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

    print(RUN_LOG.resolve())
    print((ROOT / "results.tsv").resolve())


if __name__ == "__main__":
    main()
