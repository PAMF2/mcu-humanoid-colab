# MCU Humanoid Colab

Clean Colab-first project to validate this claim:

`vision-only retrieval < multimodal retrieval < temporal skill-chunk retrieval < predictive reranking`

The project is intentionally split into small parts so you can swap synthetic data for real retargeted humanoid trajectories later without rewriting the pipeline.

## Layout

```text
mcu_humanoid_colab/
  configs/
  notebooks/
  results/
  scripts/
  src/mcu_humanoid_colab/
  requirements.txt
  README.md
```

## What is improved here

- Config-driven runs instead of a single monolithic script
- Separate modules for data, synthetic benchmark, memory bank, models, and experiment loop
- A dedicated `scripts/run_experiment.py` entrypoint for Colab
- A notebook scaffold ready to upload and run
- An `npz` dataset adapter so you can plug real data into the same schema later
- Synthetic dataset export to `.npz`
- Dataset validation before running long Colab jobs

## Quickstart

From the project root:

```bash
python scripts/run_experiment.py --config configs/smoke.json --output results/smoke.json --cpu
```

Full synthetic run:

```bash
python scripts/run_experiment.py --config configs/default.json --output results/default.json --cpu
```

Local install with PyTorch:

```bash
pip install -r requirements-local.txt
```

Export a schema-correct synthetic `.npz`:

```bash
python scripts/export_synthetic_npz.py --output data/synthetic_humanoid.npz
```

Validate a real `.npz` before training:

```bash
python scripts/validate_npz.py --dataset data/your_dataset.npz
```

Run the benchmark across multiple seeds:

```bash
python scripts/run_multiseed.py --config configs/default.json --output results/multiseed_default.json --cpu
```

Build a real `.npz` from local parquet episodes with embedded images:

```bash
python scripts/build_real_npz.py --dataset-dir path/to/libero_train --output results/libero_real.npz --config-output results/libero_real_config.json
```

Build a real `.npz` directly from the downloaded NVIDIA `GR00T-Teleop-G1` dataset:

```bash
python scripts/inspect_gr00t_dataset.py --dataset-dir /content/nvidia_g1
python scripts/build_gr00t_npz.py --dataset-dir /content/nvidia_g1 --output results/groot_real.npz --config-output results/groot_real_config.json --max-episodes 64
python scripts/validate_npz.py --dataset results/groot_real.npz
python scripts/run_experiment.py --config results/groot_real_config.json --dataset-path results/groot_real.npz --output results/groot_real_run.json
python scripts/plot_results.py --input results/groot_real_run.json --output results/groot_real_run_plot.png
```

Build a single-task GR00T dataset:

```bash
python scripts/inspect_gr00t_dataset.py --dataset-dir /content/nvidia_g1/g1-pick-apple
python scripts/build_gr00t_npz.py --dataset-dir /content/nvidia_g1/g1-pick-apple --output results/groot_apple.npz --config-output results/groot_apple_config.json --max-episodes 128
python scripts/run_experiment.py --config results/groot_apple_config.json --dataset-path results/groot_apple.npz --output results/groot_apple_run.json
```

Run the bundled real-data sample from GitHub:

```bash
python scripts/validate_npz.py --dataset sample_data/libero_real_sample.npz
python scripts/run_experiment.py --config sample_data/libero_real_config.json --dataset-path sample_data/libero_real_sample.npz --output results/libero_real_run.json
python scripts/run_multiseed.py --config sample_data/libero_real_config.json --dataset-path sample_data/libero_real_sample.npz --output results/libero_real_multiseed.json
```

Run the bundled medium real-data sample from GitHub:

```bash
python scripts/validate_npz.py --dataset sample_data/libero_real_medium.npz
python scripts/run_experiment.py --config sample_data/libero_real_medium_config.json --dataset-path sample_data/libero_real_medium.npz --output results/libero_real_medium_run.json
python scripts/run_multiseed.py --config sample_data/libero_real_medium_config.json --dataset-path sample_data/libero_real_medium.npz --output results/libero_real_medium_multiseed.json
python scripts/plot_results.py --input results/libero_real_medium_multiseed.json --output results/libero_real_medium_plot.png
```

## Colab workflow

1. Upload this folder to `/content/mcu_humanoid_colab`.
2. Open `notebooks/MCU_Humanoid_Colab.ipynb`.
3. Run the setup cell.
4. Run the smoke config first.
5. If the baseline table looks sane, run the default config.

Note for Colab:

- `requirements.txt` intentionally avoids reinstalling `torch`, so Colab keeps its preinstalled CUDA build.
- If you see `torch ... +cpu`, your environment is using a CPU-only wheel even if the runtime type is GPU.

## Current benchmark stages

The synthetic benchmark gives you four controllers:

1. `vision_only`
2. `multimodal_instant`
3. `chunk_only`
4. `chunk_world_model`

Interpretation:

- `vision_only` should be clearly worst
- `multimodal_instant` should improve by using proprio, contact, and phase
- `chunk_only` should improve whenever hidden dynamics make single-step retrieval ambiguous
- `chunk_world_model` is the experimental predictive reranker. If it underperforms, the retrieval prior is still stronger than the predictive model.

## Real dataset schema

When you move from synthetic to real humanoid data, prepare an `.npz` with:

- `vision`: `[N, T, Dv]`
- `proprio`: `[N, T, Dp]`
- `contact`: `[N, T, Dc]`
- `phase`: `[N, T, 2]`
- `command`: `[N, T, Dcmd]`
- `action`: `[N, T, Da]`
- `state`: `[N, T, Ds]`

Optional:

- `skill`: `[N, T]`
- `context`: `[N, T, Dcxt]`
- `phase_scalar`: `[N, T]`

You can start from `configs/npz_template.json` and override `--dataset-path`.

## Practical real-data path

If you already have LeRobot-style parquet files with embedded images, the shortest path is:

1. convert parquet to `.npz` with `scripts/build_real_npz.py`
2. validate it with `scripts/validate_npz.py`
3. run `scripts/run_experiment.py` or `scripts/run_multiseed.py` using the generated config

The easiest local starting point in this workspace is `REAL/libero_train`, because it already contains image bytes inside the parquet rows.

For quick tests directly from GitHub, use the bundled `sample_data/libero_real_sample.npz`.

## Safe paper claim

Use this claim first:

`Retrieval-based control improves when memory keys include multimodal temporal context and when actions are retrieved as skill chunks rather than isolated motor commands.`

Only upgrade the claim on the world model after its ablation is stable.
