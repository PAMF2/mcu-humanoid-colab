# MCU Autoresearch

This folder adapts the `autoresearch` loop to the MCU benchmark.

## Setup

To set up a new run:

1. Choose a short tag, e.g. `mar8-mcu`.
2. Create a branch: `git checkout -b autoresearch/<tag>`.
3. Read these files for context:
   - `mcu_autoresearch/README.md`
   - `mcu_autoresearch/prepare.py`
   - `mcu_autoresearch/train.py`
   - `src/mcu_humanoid_colab/experiment.py`
   - `src/mcu_humanoid_colab/memory.py`
   - `src/mcu_humanoid_colab/models.py`
4. Run setup:
   - `python mcu_autoresearch/prepare.py --preset real-medium`
5. Confirm that:
   - `mcu_autoresearch/workspace/active_config.json` exists
   - `mcu_autoresearch/results.tsv` exists

## Constraints

The benchmark is fixed.

What you may change:

- `mcu_autoresearch/train.py`

What you must not change during the autonomous loop:

- `mcu_autoresearch/prepare.py`
- `src/mcu_humanoid_colab/data.py`
- `src/mcu_humanoid_colab/experiment.py`
- dataset files
- evaluation harness

## Goal

Minimize `primary_metric`.

- On offline runs, this is `action_mse`.
- On synthetic runs, it also reflects `success_rate` and `fall_rate`.

Lower is better.

## Search directions

Good experiment ideas:

- rebalance multimodal query weights
- compare instant retrieval against chunk retrieval
- vary `top_k`
- change chunk aggregation from `first` to weighted average
- tune rerank weights for similarity, rollout score, decoder score
- enable or disable the tiny world model
- simplify the controller if performance stays the same

Bad experiment ideas:

- editing many files outside `train.py`
- changing the benchmark or metric
- adding new dependencies
- rewriting data loading

## Logging

After each run, append to `mcu_autoresearch/results.tsv`:

```text
commit	primary_metric	action_mse	success_rate	fall_rate	memory_gb	status	description
```

Rules:

- `status=keep` if `primary_metric` improved
- `status=discard` if it got worse or stayed equal
- `status=crash` if the run failed

## Loop

1. Edit `mcu_autoresearch/train.py`
2. Commit
3. Run: `python mcu_autoresearch/train.py > run.log 2>&1`
4. Extract metrics from `mcu_autoresearch/run.log`
5. Append to the log with:
   - `python mcu_autoresearch/log_result.py --status keep --description "<short note>"`
   - `python mcu_autoresearch/log_result.py --status discard --description "<short note>"`
   - `python mcu_autoresearch/log_result.py --status crash --description "<short note>"`
6. Keep or discard the commit based on `primary_metric`

Keep the loop narrow and data-driven.
