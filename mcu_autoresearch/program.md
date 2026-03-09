# autoresearch

This is an experiment to let the agent do autonomous research on the MCU controller.

This run is for real model training, not cheap CPU-only retrieval tuning.
Use the GPU.

## Setup

To set up a new experiment:

1. Create a fresh branch: `git checkout -b autoresearch/<tag>`
2. Read the in-scope files for full context:
   - `mcu_autoresearch/program.md`
   - `mcu_autoresearch/prepare.py`
   - `mcu_autoresearch/train.py`
3. Run setup:
   - `python mcu_autoresearch/prepare.py`
4. Verify that:
   - `mcu_autoresearch/workspace/active_config.json` exists
   - `mcu_autoresearch/results.tsv` exists
5. Confirm setup looks good and start experimentation immediately.

The prepared benchmark is fixed:

- dataset: `sample_data/libero_real_sample.npz`
- config: `sample_data/libero_real_config.json`

Do not change the benchmark.

## Experimentation

Each experiment is a single run of:

`python mcu_autoresearch/train.py`

The output metric is `primary_metric`.

For this benchmark, `primary_metric == action_mse`.
Lower is better.

The initial baseline is approximately:

- `multimodal_instant ~= 0.354766`

## What you CAN do

- Modify `mcu_autoresearch/train.py`

This is the only file you edit.

## What you CANNOT do

- Modify `mcu_autoresearch/prepare.py`
- Modify files under `src/mcu_humanoid_colab/`
- Modify dataset files
- Add dependencies
- Change the evaluation harness

## Goal

Get the lowest `primary_metric`.

This time, prioritize experiments that actually train the predictive parts of the controller:

- tiny world model
- chunk decoder
- GPU-backed training

Do not spend the whole run only changing cheap retrieval knobs.

Good search directions:

- instant retrieval vs chunk retrieval
- `top_k`
- `chunk_len`
- `history`
- query weights for `vision`, `proprio`, `contact`, `phase`, `command`
- chunk aggregation strategy
- use of the tiny world model
- use of the chunk decoder
- world model epochs
- batch size
- predictive reranking weights

Bad directions:

- editing anything outside `mcu_autoresearch/train.py`
- changing the benchmark
- changing the metric
- rewriting data loading

## Output format

`train.py` prints a block like:

```text
---
primary_metric:    0.123456
action_mse:        0.123456
success_rate:      0.000000
fall_rate:         0.000000
training_seconds:  1.2
peak_vram_mb:      0.0
num_train_episodes: 9
num_test_episodes:  3
```

You can extract the key metric with:

```bash
grep "^primary_metric:\|^peak_vram_mb:" mcu_autoresearch/run.log
```

## Logging results

Log every experiment to `mcu_autoresearch/results.tsv`.

The TSV has 8 columns:

```text
commit	primary_metric	action_mse	success_rate	fall_rate	memory_gb	status	description
```

Rules:

- `keep` if `primary_metric` improved
- `discard` if it got worse or stayed equal
- `crash` if the run failed

## The experiment loop

LOOP FOREVER:

1. Look at the current git state.
2. Modify `mcu_autoresearch/train.py` with one experimental idea.
3. Commit.
4. Run:
   - `python mcu_autoresearch/train.py > mcu_autoresearch/run.log 2>&1`
5. Read out the result:
   - `grep "^primary_metric:\|^peak_vram_mb:" mcu_autoresearch/run.log`
6. If the grep output is empty, the run crashed:
   - read `tail -n 50 mcu_autoresearch/run.log`
   - log `crash`
   - move on
7. Append the result with:
   - `python mcu_autoresearch/log_result.py --status keep --description "<short note>"`
   - `python mcu_autoresearch/log_result.py --status discard --description "<short note>"`
   - `python mcu_autoresearch/log_result.py --status crash --description "<short note>"`
8. Also append one plain-text progress line to `mcu_autoresearch/progress.log`:
   - format: `iteration<TAB>status<TAB>primary_metric<TAB>note`
   - example: `3\tkeep\t0.301234\ttop_k=16 improved`
   - also print a short line like `ITERATION 3 keep 0.301234 top_k=16 improved`
9. Keep the commit only if `primary_metric` improved.
10. Otherwise discard and continue.

Do not stop to summarize the codebase.
Do not stop after one experiment.
Do not ask whether to continue.
Continue autonomously until manually interrupted.
