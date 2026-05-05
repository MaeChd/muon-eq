# CIFAR-10 Optimizer Benchmark

The main entry point in this directory is [`compare_resnet_optimizers.py`](./compare_resnet_optimizers.py). It compares `ResNet18 + SGD / AdamW / Muon / Muon-MVR / MuonEq-rc / MuonEq-r / MuonEq-c` on CIFAR-10.

The multi-GPU DDP entry point is [`compare_resnet_optimizers_ddp.py`](./compare_resnet_optimizers_ddp.py).

## Prerequisites

Enter the project root first:

```bash
cd /path/to/llm-opt-baseline-gpu
```

Make sure the current Python environment has at least these dependencies installed:

```bash
python -c "import torch, torchvision"
```

If you want to enable `wandb` tracking, also verify:

```bash
python -c "import wandb"
```

To verify that `tqdm` is installed:

```bash
python -c "import tqdm"
```

## Ready-to-Run Commands

Compare all optimizers by default:

```bash
python cifar10/compare_resnet_optimizers.py
```

By default, this also generates:

- `results/*.json` result summaries
- `results/*.log` local training logs with the same base name as the JSON file

Training continues even if `wandb` or `swanlab` fails because of network issues. The local `.log` file keeps complete epoch-level records.

Run only 1 epoch for a quick smoke test:

```bash
python cifar10/compare_resnet_optimizers.py --epochs 1 --num-runs 1 --num-workers 0
```

Enable learning-rate grid search:

```bash
python cifar10/compare_resnet_optimizers.py --enable-lr-grid-search --grid-search-epochs 20
```

Disable the `tqdm` progress bar:

```bash
python cifar10/compare_resnet_optimizers.py --disable-tqdm
```

Make `muoneq-rc` switch from row/column normalization to row normalization after a specific step:

```bash
python cifar10/compare_resnet_optimizers.py --optimizers muoneq-rc --muoneq-rc-phase 1000
```

Compare only the three normalization variants `muoneq-rc / muoneq-r / muoneq-c`:

```bash
python cifar10/compare_resnet_optimizers.py --optimizers muoneq-rc muoneq-r muoneq-c
```

Specify the output JSON path:

```bash
python cifar10/compare_resnet_optimizers.py --output-json cifar10/results/run.json
```

Enable SwanLab logging:

```bash
python cifar10/compare_resnet_optimizers.py --use-swanlab
```

Enable Weights & Biases logging:

```bash
python cifar10/compare_resnet_optimizers.py --use-wandb
```

Specify the wandb project:

```bash
python cifar10/compare_resnet_optimizers.py --use-wandb --wandb-project cifar10-benchmark
```

## Excluding muon_mvr

The easiest option is to exclude it directly:

```bash
python cifar10/compare_resnet_optimizers.py --exclude-optimizers muon_mvr
```

If you want to run a specific optimizer subset, pass that subset explicitly:

```bash
python cifar10/compare_resnet_optimizers.py --optimizers sgd adamw muon muoneq-rc muoneq-r muoneq-c
```

Neither option requires changes to the Python code.

## Common Combinations

Exclude `muon_mvr` and enable LR grid search:

```bash
python cifar10/compare_resnet_optimizers.py --exclude-optimizers muon_mvr --enable-lr-grid-search
```

Exclude `muon_mvr` and enable wandb:

```bash
python cifar10/compare_resnet_optimizers.py --exclude-optimizers muon_mvr --use-wandb
```

Compare only `sgd` and `muon`:

```bash
python cifar10/compare_resnet_optimizers.py --optimizers sgd muon
```

Use GPU 0:

```bash
python cifar10/compare_resnet_optimizers.py --device cuda:0
```

## Multi-GPU DDP Runs

Run 4-GPU DDP and compare all optimizers by default:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda
```

The DDP version also generates local `.log` files. Rank 0 writes logs, JSON files, SwanLab records, and wandb records.

Run 4-GPU DDP and exclude `muon_mvr`:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda --exclude-optimizers muon_mvr
```

Run 4-GPU DDP and enable LR grid search:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda --enable-lr-grid-search
```

Run 4-GPU DDP and enable wandb:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda --use-wandb
```

Run 4-GPU DDP and disable `tqdm`:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda --disable-tqdm
```

Compare only `sgd` and `muon`:

```bash
torchrun --standalone --nproc_per_node=4 cifar10/compare_resnet_optimizers_ddp.py --device cuda --optimizers sgd muon
```

Notes:

- In the DDP script, `--batch-size` is the per-GPU batch size.
- Global batch size = `--batch-size * nproc_per_node`.
- JSON output, SwanLab, and wandb are written only on rank 0.

## Arguments

- `--optimizers`: run only the specified optimizer subset.
- `--exclude-optimizers`: exclude selected optimizers from the default set.
- `--enable-lr-grid-search`: run LR grid search for each optimizer before the main experiment.
- `--grid-search-epochs`: number of epochs in the LR grid-search phase.
- `--epochs`: number of epochs in the main experiment.
- `--num-runs`: number of repeated runs per optimizer.
- `--use-swanlab`: enable SwanLab tracking.
- `--use-wandb`: enable Weights & Biases tracking.
- `--wandb-project`: specify the wandb project.
- `--wandb-entity`: specify the wandb entity.
- `--disable-tqdm`: disable batch-level training progress bars.
- `--muoneq-rc-phase`: phase-switch step for `MuonEq-rc`; `None` keeps row/column normalization for the whole run.
- `--backend`: DDP backend; defaults to `nccl` on CUDA.
- `--output-json`: output path for the result JSON.
