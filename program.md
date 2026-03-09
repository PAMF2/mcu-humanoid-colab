# autoresearch

This is an experiment to have the LLM do its own research on the MCU controller.

The repo is intentionally small. Only three root files matter:

- `prepare.py` — fixed benchmark prep and runtime utilities. Do not modify.
- `train.py` — the single file you edit.
- `program.md` — these instructions.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag and create a fresh branch: `git checkout -b autoresearch/<tag>`
2. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
3. Verify data exists:
   - `sample_data/libero_real_sample.npz`
   - `sample_data/libero_real_config.json`
4. Run:
   - `python prepare.py`
5. Confirm:
   - `workspace/active_config.json` exists
   - `results.tsv` exists
   - `workspace/progress.log` exists
6. Kick off experimentation immediately.

## Experimentation

Each experiment runs on a single GPU.

The training script is:

`python train.py`

The goal is simple: get the lowest `primary_metric`.

For this benchmark, `primary_metric` is `action_mse`.
Lower is better.

The baseline is approximately:

- `0.322180`

This run is for real model training, not cheap CPU-only retrieval tuning.
Use the GPU.

## What you CAN do

- Modify `train.py`

This is the only file you edit.

## What you CANNOT do

- Modify `prepare.py`
- Modify datasets
- Add dependencies
- Modify the evaluation harness

## Good experiment ideas

- chunk retrieval vs instant retrieval
- `top_k`
- `chunk_len`
- `history`
- query weights
- chunk aggregation
- world model epochs
- chunk decoder usage
- predictive reranking weights
- batch size and learning rate

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated values:

```text
commit	primary_metric	action_mse	success_rate	fall_rate	memory_gb	status	description
```

Rules:

- `keep` if `primary_metric` improved
- `discard` if it got worse or stayed equal
- `crash` if the run failed

Also append one progress line to `workspace/progress.log`:

```text
iteration	status	primary_metric	note
```

Print a line like:

```text
ITERATION 3 keep 0.301234 top_k=16 improved
```

## The experiment loop

LOOP FOREVER:

1. Look at the current git state.
2. Tune `train.py` with one experimental idea.
3. Commit.
4. Run: `python train.py > run.log 2>&1`
5. Read out the result: `grep "^primary_metric:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Read `tail -n 50 run.log`, log a crash, and move on.
7. Record the result in `results.tsv`
8. Append one line to `workspace/progress.log`
9. If `primary_metric` improved, keep the commit.
10. If it did not improve, discard and continue.

Never stop after one run.
Never stop to ask whether to continue.
Continue until manually interrupted.
