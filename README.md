[English](README.md) | [Simplified Chinese](README_ZH.md)

# MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration

This repository accompanies the paper **MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration** ([arXiv:2603.28254](https://arxiv.org/abs/2603.28254)) and includes the implementations, training scripts, and reproduction entry points for `MuonEq` across both GPU and Ascend NPU experiments.

The repository currently has three main parts:

- `llm-opt-baseline-gpu/cifar10`: quick optimizer comparisons on CIFAR-10
- `llm-opt-baseline-gpu/llm-baselines`: GPU-side Transformer / LLM baselines with `MuonEq`
- `llm-opt-baseline-npu/llama2_pretrain`: Ascend NPU pretraining experiments

This is not a new training framework built from scratch. `llm-opt-baseline-gpu/llm-baselines` and `llm-opt-baseline-npu/llama2_pretrain` are experimental directories derived from public codebases and extended locally. We keep most of their original structure because it is the most practical setup for reproduction and follow-up work.

## Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Using MuonEq Directly](#using-muoneq-directly)
- [Results](#results)
- [Environment and Dependencies](#environment-and-dependencies)
- [License](#license)
- [Citation](#citation)

## Overview

`MuonEq` adds a lightweight equilibration / normalization step before `Muon` orthogonalization to improve the numerical condition of the update matrix before it enters the zeropower / Newton-Schulz iteration. This repository is the experimental codebase used for the paper rather than a single unified training framework.

If you just want to decide where to start, use the table below:

| Goal | Recommended Entry Point | Notes |
| --- | --- | --- |
| Run the smallest possible experiment | `llm-opt-baseline-gpu/cifar10` | Single-machine setup, good for checking `MuonEq` behavior and command flow |
| Run GPU LLM baselines | `llm-opt-baseline-gpu/llm-baselines` | Includes `MuonEq` and related optimizer variants |
| Reproduce Ascend NPU pretraining experiments | `llm-opt-baseline-npu/llama2_pretrain` | Includes multi-node scripts and cluster-specific arguments |

## Repository Layout

```text
llm-opt-baseline-gpu/
  cifar10/
  llm-baselines/

llm-opt-baseline-npu/
  llama2_pretrain/
```

## Quick Start

### 1. CIFAR-10

If you want the fastest way to sanity-check `MuonEq`, start with CIFAR-10:

```bash
cd llm-opt-baseline-gpu
python cifar10/compare_resnet_optimizers.py --epochs 1 --num-runs 1 --num-workers 0
```

In this branch, the `Muon` learning-rate scaling follows Keller Jordan's implementation in [`cifar10-airbench`](https://github.com/KellerJordan/cifar10-airbench).

For more commands and argument details, see [llm-opt-baseline-gpu/cifar10/README.md](llm-opt-baseline-gpu/cifar10/README.md).

### 2. GPU LLM Baselines

`llm-opt-baseline-gpu/llm-baselines` is used for GPU-side Transformer / LLM baseline runs, including `MuonEq` and related optimizer variants.

In this branch, the `Muon` learning-rate scaling follows the RMS-matched strategy used in MoonshotAI's [`Moonlight`](https://github.com/MoonshotAI/Moonlight).

Install dependencies:

```bash
cd llm-opt-baseline-gpu/llm-baselines
pip install -r requirements.txt
```

Run a basic training job:

```bash
python ./src/main.py --config_format base
```

For `MuonEq` sweep scripts, see [llm-opt-baseline-gpu/llm-baselines/scripts/optimizers_compare/readme.md](llm-opt-baseline-gpu/llm-baselines/scripts/optimizers_compare/readme.md).

### 3. NPU Pretraining

`llm-opt-baseline-npu/llama2_pretrain` is used for Ascend NPU pretraining experiments. The main paper results use the `cosine` scheduler, while the scripts also retain `wsd` support for follow-up runs.

Install dependencies:

```bash
cd llm-opt-baseline-npu/llama2_pretrain
pip install -r requirements.txt
```

Before running, you need at least:

- `C4_DATA_DIR`
- `TOKENIZER_PATH`
- `--nodes` or `MULTI_NODE_HOSTS` for multi-node runs
- If you pass short node IDs instead of full IPs in `--nodes`, you also need to set `MULTI_NODE_HOST_PREFIX` or `--host-prefix` to match your network. The default prefix in the scripts is `10.0.0.`

The `2026compare` experiment entry points are under `llm-opt-baseline-npu/llama2_pretrain/experiments/2026compare/`, mainly through `multi_node_sweep_*.sh` and `multi_node_main_*.sh`. A minimal example:

```bash
cd llm-opt-baseline-npu/llama2_pretrain/experiments/2026compare
C4_DATA_DIR=/path/to/c4 \
TOKENIZER_PATH=t5-base \
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_sweep_350m.sh --nodes 10.0.1.131,10.0.1.132 adamw
```

For more details on multi-node arguments, see [llm-opt-baseline-npu/llama2_pretrain/experiments/2026compare/multi_node_usage.md](llm-opt-baseline-npu/llama2_pretrain/experiments/2026compare/multi_node_usage.md).

## Using MuonEq Directly

If you want to call `MuonEq` directly in your own code, the two implementations live at:

- NPU: `llm-opt-baseline-npu/llama2_pretrain/optimizers/muon_variants/muoneq.py`
- GPU: `llm-opt-baseline-gpu/llm-baselines/src/optim/muoneq.py`

Both classes are named `MuonEq`, and the core API is aligned. The main difference is the import path:

```python
# NPU
from optimizers.muon_variants.muoneq import MuonEq

# GPU
from optim.muoneq import MuonEq

optimizer = MuonEq(
    lr=1e-3,
    wd=0.1,
    muon_params=muon_params,
    adamw_params=adamw_params,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    adamw_betas=(0.95, 0.95),
    normalize_mode="row",   # also supports "rowcol" and "col"
    phase=None,             # can switch from row/col to row after phase
    zeropower_mode="native" # or "spc"
)
```

If you want to reuse the pretraining script in this repository directly, you can also call:

- `llm-opt-baseline-npu/llama2_pretrain/scripts/pretrain_c4_dist.py`

The corresponding optimizer names are:

- `muoneq-row`
- `muoneq-rowcol`
- `muoneq-col`

## Results

Selected results from `LLaMA2-1B` training up to `21B tokens`:

| Train Loss vs Tokens | Val Loss vs Tokens | Val Loss vs Train Time |
| --- | --- | --- |
| ![Train Loss vs Tokens](assert/train_loss_vs_tokens.png) | ![Val Loss vs Tokens](assert/val_loss_vs_tokens.png) | ![Val Loss vs Train Time](assert/val_loss_vs_train_time.png) |

## Environment and Dependencies

The GPU baseline and NPU pretraining parts maintain separate dependency sets:

- `llm-opt-baseline-gpu/llm-baselines/requirements.txt`
- `llm-opt-baseline-npu/llama2_pretrain/requirements.txt`

In practice, it is safer to use separate environments for these two parts instead of forcing them into one shared environment.

## License

The root-level code is released under MIT; see `LICENSE`.

The third-party experiment directories preserved in this repository keep their own license files:

- `llm-opt-baseline-gpu/llm-baselines/LICENSE`
- `llm-opt-baseline-npu/llama2_pretrain/LICENSE`

## Citation

If this repository or the paper is useful for your work, please cite:

```bibtex
@article{chang2026muoneq,
  title={MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration},
  author={Chang, Da and Shi, Qiankun and Zhang, Lvgang and Li, Yu and Zhang, Ruijie and Lu, Yao and Liu, Yongxiang and Yuan, Ganzhao},
  journal={arXiv preprint arXiv:2603.28254},
  year={2026}
}
```
