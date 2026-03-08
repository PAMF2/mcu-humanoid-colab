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

## Safe paper claim

Use this claim first:

`Retrieval-based control improves when memory keys include multimodal temporal context and when actions are retrieved as skill chunks rather than isolated motor commands.`

Only upgrade the claim on the world model after its ablation is stable.
