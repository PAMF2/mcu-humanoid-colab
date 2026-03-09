# MCU Autoresearch Prompt

Use this prompt in your coding agent after cloning the repository and entering the project root.

```text
Read `mcu_autoresearch/program.md` and follow it exactly.

Important constraints:
- Only edit `mcu_autoresearch/train.py`
- Do not change the benchmark, datasets, or evaluation harness
- Log every run to `mcu_autoresearch/results.tsv`
- Keep or discard based on `primary_metric`

Start by doing setup if needed, then run the baseline, log it, and continue autonomously.
```

This file is also consumed automatically by:

```bash
python mcu_autoresearch/run_codex_autoresearch.py --preset real-medium
```

If you want a more explicit kickoff prompt, use:

```text
Open `mcu_autoresearch/program.md`, `mcu_autoresearch/README.md`, and `mcu_autoresearch/train.py`.
Set up the MCU autoresearch workspace on branch `autoresearch/<today-tag>`.
Run the baseline with the `real-medium` preset, log it, and then continue autonomous experimentation indefinitely.
Only edit `mcu_autoresearch/train.py`.
```
