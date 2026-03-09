# MCU Autoresearch

`mcu_autoresearch` adapts the `autoresearch` pattern to the MCU benchmark in this repository.

The contract is intentionally narrow:

- `prepare.py` is fixed and should not be modified by the agent.
- `train.py` is the single editable file. It defines the candidate MCU controller and its search knobs.
- `program.md` is the agent instruction file for autonomous experimentation.

This setup uses the existing benchmark implementation in `../src/mcu_humanoid_colab/` and keeps the experiment loop lightweight:

- choose a benchmark preset or config
- run one candidate controller
- report a compact summary
- log the result in `results.tsv`

## Quick start

From the repository root:

```bash
python mcu_autoresearch/prepare.py
python mcu_autoresearch/train.py
```

That creates:

- `mcu_autoresearch/workspace/active_config.json`
- `mcu_autoresearch/results.tsv`

## Available presets

- `synthetic-smoke`
- `synthetic-default`
- `real-sample`
- `real-medium`

You can also point directly at an existing config:

```bash
python mcu_autoresearch/prepare.py
```

## Autoresearch loop

This folder is meant to be driven exactly like `autoresearch`:

```bash
python mcu_autoresearch/prepare.py
python mcu_autoresearch/train.py > mcu_autoresearch/run.log 2>&1
python mcu_autoresearch/log_result.py --status keep --description "baseline multimodal instant"
```

Or do setup + baseline in one shot:

```bash
python mcu_autoresearch/bootstrap.py
```

Or run the full Codex-driven loop in one command:

```bash
python mcu_autoresearch/run_codex_autoresearch.py
```

Or run the full Claude Code-driven loop in one command:

```bash
python mcu_autoresearch/run_claude_autoresearch.py
```

Only `mcu_autoresearch/train.py` should be edited by the agent.
`mcu_autoresearch/workspace/` is ignored by git so setup does not dirty the branch.

## Output

`train.py` prints a summary block like:

```text
---
primary_metric:    0.123456
action_mse:        0.123456
success_rate:      0.000000
fall_rate:         0.000000
training_seconds:  12.3
peak_vram_mb:      512.0
num_train_episodes: 76
num_test_episodes: 20
```

Lower `primary_metric` is better.

For offline datasets this is just `action_mse`.
For synthetic runs it adds a small penalty for falls and a small bonus for success:

`primary_metric = action_mse + 0.5 * fall_rate - 0.25 * success_rate`

## Search space

The intended search space in `train.py` includes:

- query weights for `vision`, `proprio`, `contact`, `phase`, `command`
- instant vs chunk retrieval
- `top_k`
- chunk aggregation strategy
- rerank weights
- use of the tiny world model and chunk decoder

This keeps the autoresearch loop aligned with the actual MCU claim instead of drifting into unrelated code changes.

## Agent handoff

Files intended for direct agent use:

- `mcu_autoresearch/program.md`
- `mcu_autoresearch/agent_prompt.md`

The shortest path is:

1. run `python mcu_autoresearch/bootstrap.py`
2. paste the prompt from `mcu_autoresearch/agent_prompt.md` into the agent

If you already have Codex CLI configured locally, skip the manual prompt handoff and run:

```bash
python mcu_autoresearch/run_codex_autoresearch.py
```

If you want Claude Code instead, use:

```bash
python mcu_autoresearch/run_claude_autoresearch.py
```
