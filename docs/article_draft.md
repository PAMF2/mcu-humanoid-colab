# Memory-Constrained Multimodal Retrieval for Humanoid Control

## Abstract

Physical control cannot rely on visual similarity alone. In embodied settings, states that look similar may require different actions because of hidden dynamics, contact conditions, or motion phase. We evaluate a retrieval-based control pipeline built around a Memory Consolidation Unit (MCU) and test whether adding multimodal state and temporal skill chunks improves performance over vision-only retrieval. On a multi-seed benchmark, vision-only retrieval achieves only `24.2% +/- 14.8` success, while instantaneous multimodal retrieval reaches `74.2% +/- 13.0`. Retrieving temporally extended skill chunks further improves mean success to `75.0% +/- 5.9`, indicating better robustness and lower variance across seeds. A small predictive reranker based on a short-horizon world model does not yet outperform chunk retrieval alone, suggesting that the main gain comes from memory representation and temporal grounding rather than from the current predictive module.

## 1. Introduction

Humanoid and embodied control require decisions that remain consistent under partial observability, contact changes, and temporal ambiguity. Pure visual retrieval is weak in this setting because similar images may correspond to different latent physical states. This work tests a narrower claim: retrieval quality improves substantially when memory keys include multimodal state and temporal context, and control becomes more robust when the system retrieves short skill chunks instead of isolated actions.

Rather than training a large end-to-end policy, we study a memory-centric alternative. The MCU stores previous experience and retrieves relevant trajectories at inference time. This makes the method suitable for fast iteration in constrained environments such as Colab and provides clean ablations for identifying where the gain actually appears.

## 2. Methodology

### 2.1 Benchmark Design

We use a compact embodied benchmark with four ingredients:

1. visual observations with aliasing
2. proprioceptive state
3. contact state
4. motion phase and command conditioning

Each episode contains latent dynamics, so the current image alone is insufficient to recover the correct action. This forces the model to benefit from multimodal information and temporal context.

### 2.2 MCU Memory Structure

The MCU stores three families of retrieval keys:

1. `vision_only`: current visual embedding
2. `multimodal_instant`: current visual, proprioceptive, contact, phase, and command state
3. `chunk_only`: temporal history over the same multimodal state, used to retrieve a skill chunk rather than a single action

The chunk setup uses:

- `history = 4`
- `chunk_len = 6`
- `top_k = 8`

### 2.3 Compared Controllers

We evaluate four controllers:

1. `vision_only`
   Retrieves an instantaneous action from the current visual state only.

2. `multimodal_instant`
   Retrieves an instantaneous action using visual state, proprioception, contact, phase, and command.

3. `chunk_only`
   Retrieves a temporally extended skill chunk using multimodal history.

4. `chunk_world_model`
   Applies a tiny predictive reranker over candidate chunks using a short-horizon dynamics model and a chunk feasibility decoder.

### 2.4 Metrics

We report:

1. `action_mse`
2. `state_mse`
3. `success_rate`
4. `fall_rate`

All main results are aggregated across multiple random seeds.

## 3. Experimental Setup

The default benchmark configuration uses:

- `train_episodes = 96`
- `test_episodes = 24`
- `state_dim = 6`
- `vision_dim = 10`
- `action_dim = 6`
- seeds: `7, 11, 19, 23, 31`

The experiments are implemented in the `mcu_humanoid_colab` repository and can be executed on Colab.

## 4. Results

### 4.1 Multi-Seed Results

| Controller | Action MSE | State MSE | Success Rate | Fall Rate |
| :--- | :--- | :--- | :--- | :--- |
| `vision_only` | `0.5384 +/- 0.1133` | `0.4751 +/- 0.1184` | `24.2% +/- 14.8` | `0.0% +/- 0.0` |
| `multimodal_instant` | `0.1418 +/- 0.0260` | `0.1368 +/- 0.0326` | `74.2% +/- 13.0` | `0.0% +/- 0.0` |
| `chunk_only` | `0.1443 +/- 0.0330` | `0.1423 +/- 0.0384` | `75.0% +/- 5.9` | `0.0% +/- 0.0` |
| `chunk_world_model` | `0.1916 +/- 0.0378` | `0.1809 +/- 0.0456` | `64.2% +/- 10.5` | `0.0% +/- 0.0` |

### 4.2 Interpretation

The strongest improvement comes from multimodal retrieval. Moving from `vision_only` to `multimodal_instant` increases success from `24.2%` to `74.2%`, showing that embodied retrieval must be grounded in proprioception, contact, and phase.

The `chunk_only` controller produces the best overall result. Its mean success is slightly higher than `multimodal_instant`, but the more important difference is variance: `chunk_only` reduces success-rate variability from `13.0` to `5.9` percentage points. This suggests that temporal skill retrieval improves robustness and consistency even when per-step prediction error changes only slightly.

The current `chunk_world_model` variant does not outperform `chunk_only`. This indicates that the present predictive reranker is weaker than the retrieval prior. Therefore, the current evidence supports a memory-and-temporal-grounding claim more strongly than a world-model claim.

## 5. Discussion

These experiments support three conclusions.

First, vision-only retrieval is inadequate for embodied control. The gap between `24.2%` and `74.2%` success is too large to ignore and shows that the state representation must include physical context.

Second, retrieving skill chunks is more effective than retrieving instantaneous actions when the environment contains hidden temporal structure. The main advantage is robustness, not just raw pointwise accuracy.

Third, a tiny predictive module is not automatically useful. If the predictive model is weaker than the memory prior, reranking can make the policy worse rather than better. This is an important negative result because it prevents overstating the contribution of the world model.

## 6. Limitations

This benchmark is still synthetic, so the results should be interpreted as controlled empirical evidence rather than a universal proof for real humanoids. The world model is intentionally small and may be underpowered for reranking under stronger dynamics. Future work should replace synthetic observations with real retargeted trajectories and pretrained visual embeddings such as DINOv2 or VideoMimic-style representations.

## 7. Conclusion

The main gain in the current MCU pipeline comes from multimodal temporal retrieval, not from the predictive reranker. A safe conclusion is:

`Temporal skill-chunk retrieval improves robustness and success consistency over instantaneous multimodal retrieval, while vision-only retrieval remains substantially weaker.`

This is the claim best supported by the current evidence.

## Recommended Tests

Run the experiments in this order.

1. Sanity check:
   `python scripts/run_experiment.py --config configs/smoke.json --output results/smoke.json`

2. Single full run:
   `python scripts/run_experiment.py --config configs/default.json --output results/default.json`

3. Main paper result:
   `python scripts/run_multiseed.py --config configs/default.json --output results/multiseed_default.json`

4. Synthetic dataset export:
   `python scripts/export_synthetic_npz.py --output results/synthetic_dataset.npz`

5. Dataset schema validation:
   `python scripts/validate_npz.py --dataset results/synthetic_dataset.npz`

6. Real-data run after you prepare an `.npz`:
   `python scripts/run_experiment.py --config configs/npz_template.json --dataset-path data/your_dataset.npz --output results/real_run.json`

## Which Test Matters Most

If you only run one test for the article, run:

`python scripts/run_multiseed.py --config configs/default.json --output results/multiseed_default.json`

That is the result you should cite in the paper.
