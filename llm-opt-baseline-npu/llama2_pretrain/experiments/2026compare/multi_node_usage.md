# 2026compare Multi-Node Usage

This document only covers how to use the multi-node scripts under `llama2_pretrain/experiments/2026compare`.

Prerequisite: **all nodes share the same filesystem**. The code directory, data path, checkpoint path, and Python environment must be identical and directly usable on every machine.

Supported scripts:

- `multi_node_sweep_130m.sh`
- `multi_node_sweep_350m.sh`
- `multi_node_sweep_1b.sh`
- `multi_node_main_130m.sh`
- `multi_node_main_350m.sh`
- `multi_node_main_1b.sh`

## 1. How It Works

The script runs locally on the master node and starts workers on other nodes through SSH:

1. Iterate over the `--nodes` list. Rank 0 stays local, and ranks 1..N-1 are started through SSH.
2. Each node runs `cd` into the same `PROJECT_ROOT` and executes the same `torchrun` command, differing only in `--node_rank`.
3. The master node runs rank 0 in the foreground and waits for training to finish. If interrupted, it automatically kills the remote workers.

Node abbreviations are expanded automatically: `131` -> `10.0.0.131`. The prefix is controlled by `MULTI_NODE_HOST_PREFIX` and defaults to `10.0.0.`.

## 2. Minimal Usage

Run the command on the first node; it automatically becomes the master.

### 2.1 sweep

350M, all optimizers, 2 nodes, 8 cards per node:

```bash
CACHE_EVAL_DATA=1 \
EVAL_BATCH_SIZE=64 \
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_sweep_350m.sh --nodes 131,132
```

Run only `adamw`:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_sweep_350m.sh --nodes 131,132 adamw
```

### 2.2 main

Run the 350M trunk:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_main_350m.sh --nodes 131,132 trunk
```

Run the full 350M trunk + decay setup:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_main_350m.sh --nodes 131,132
```

## 3. Environment Variable Form

You can use an environment variable instead of `--nodes`:

```bash
MULTI_NODE_HOSTS=131,132 \
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_sweep_1b.sh adamw
```

## 4. Supported Multi-Node Arguments

### 4.1 `--nodes`

Specify the node list participating in training.

```bash
--nodes 131,132
--nodes 131,132,133,134
--nodes 10.0.0.131,10.0.0.132
```

### 4.2 `--master-addr`

Specify the rendezvous master address. By default, the first node in `--nodes` is used.

```bash
bash multi_node_sweep_350m.sh --nodes 131,132 --master-addr 132 adamw
```

### 4.3 `--ssh-user`

Remote SSH user. If omitted, the current user is used.

```bash
bash multi_node_sweep_350m.sh --nodes 131,132 --ssh-user your-user adamw
```

### 4.4 `--host-prefix`

Override the default prefix `10.0.0.`.

```bash
bash multi_node_sweep_350m.sh --host-prefix 10.0.1. --nodes 131,132 adamw
```

## 5. Python Environment

All nodes share the same filesystem, but SSH non-interactive shells often enter the base environment by default. The script does not force a specific environment. If needed, use `MULTI_NODE_PRE_CMD` to prepare each node:

```bash
MULTI_NODE_PRE_CMD='export PATH=/path/to/conda/env/bin:$PATH'
```

If the current login environment can already call `python` / `torchrun` directly, you can leave this variable unset:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_main_130m.sh --nodes 131,132 trunk
```

If you need a different environment, override this variable:

```bash
MULTI_NODE_PRE_CMD='export PATH=/other/env/bin:$PATH' \
bash multi_node_main_130m.sh --nodes 131,132 trunk
```

## 6. Batch Size Constraint

Multi-node runs must satisfy:

```text
total_batch_size % (batch_size * world_size) == 0
```

Here `world_size = node_count * NUM_NPUS_PER_JOB`. The script errors if this condition is not met.

### Common 130M Case

130M defaults to `batch_size=16` and `total_batch_size=128`. With 2 nodes x 8 cards = world_size 16, `16 x 16 = 256 > 128`, so you need to adjust the setup:

Option 1: use 4 cards per node:

```bash
NUM_NPUS_PER_JOB=4 \
bash multi_node_sweep_130m.sh --nodes 131,132 muon
```

Option 2: increase the total batch size:

```bash
NUM_NPUS_PER_JOB=8 \
MULTI_NODE_TOTAL_BATCH_SIZE=256 \
bash multi_node_sweep_130m.sh --nodes 131,132 muon
```

## 7. Four-Node Examples

1B, 4 nodes, 8 cards per node:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_sweep_1b.sh --nodes 131,132,133,134
```

1B, 4 nodes, trunk only:

```bash
VISIBLE_NPUS=0,1,2,3,4,5,6,7 \
NUM_NPUS_PER_JOB=8 \
bash multi_node_main_1b.sh --nodes 131,132,133,134 trunk
```

## 8. Recommendations

- Start the script on the first node; it automatically becomes `master_addr`.
- Make sure passwordless SSH works between nodes.
- `VISIBLE_NPUS` is the locally visible card list on each node, not global card IDs.
- `main_*` depends on `BEST_CONFIGS` in the scripts. Fill it in before formal runs.
