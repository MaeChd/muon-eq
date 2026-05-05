# Optimizer Baseline Script Guide

The `train-baselines.sh` script in this directory runs optimizer baseline comparisons for `fineweb10b` in `llm-baselines`. It is currently tuned for a 4-GPU `RTX PRO 6000 Blackwell 96GB` machine, with defaults ready for large-batch, `bf16`, `seq_len=4096` baseline sweeps.

**Default Configuration**

- Dataset: `fineweb10b`
- Data path: specified with `DATASETS_DIR`; the script default placeholder is `/path/to/fineweb10b`
- GPU: `0,1,2,3`
- `nproc_per_node=4`
- Model: `base`
- Architecture: `n_layer=12`, `n_embd=768`, `n_head=12`
- Precision: `bfloat16`
- `batch_size=16`
- `acc_steps=2`
- `sequence_length=4096`
- `iterations=20000`
- `warmup_steps=1000`
- `eval_interval=500`
- `muoneq_phase=1000`
- `muon_ns_steps=5`
- `zeropower_mode=native`

**Default Experiment Matrix**

By default, the script expands the following experiments over the learning-rate range `5e-4, 1e-3, 2e-3, 5e-3`:

- `adamw`
- `muon`
  Here `muon` now maps to `muon_kimi.Muon` by default, which is the RMS-aligned variant.
- `muonplus`
- `mousse`
- `muoneq-rc` with `phase=None`
- `muoneq-rc` with `phase=1000`
- `muoneq-r`
- `muoneq-c`

The legacy dual-learning-rate Muon is not run by default. Enable the `muon-splitlr` sweep if you need to keep the `--muon_lr_factor` search.

Details:

- `muonplus` uses `--opt muonplus` directly.
- `mousse` uses `--opt mousse` directly.
- `muoneq-rc / muoneq-r / muoneq-c` are implemented through `MuonEq(..., normalize_mode=...)`.
- `zeropower_mode` defaults to `native` and can be switched to `spc` with an environment variable.
- `muon_ns_steps` defaults to `5`.

**Basic Usage**

Run from the repository root:

```bash
DATASETS_DIR=/path/to/fineweb10b \
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

Print commands without executing them:

```bash
DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

Customize the learning-rate range with comma-separated values:

```bash
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh \
  --lr-range 5e-4,1e-3,2e-3,5e-3
```

Space-separated values are also supported:

```bash
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh \
  --lr-range 5e-4 1e-3 2e-3 5e-3
```

Customize `zeropower_mode` and `muon_ns_steps` per experiment:

```bash
EXPERIMENT_SPECS="muoneq-r@ns=4;muon@zp=spc@ns=4;muon@zp=spc@ns=5" \
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

This command expands 3 experiment groups over the current `LR_RANGE`:

- `muoneq-r` with `ns=4` and default `zp=native`
- `muon` with `zp=spc, ns=4`
- `muon` with `zp=spc, ns=5`

**Optional: Enable the Legacy Muon Split-LR Search**

Enable this if you want to keep the legacy Muon variant where the main Muon branch learning rate and the AdamW backup learning rate are separated:

```bash
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh \
  --include-muon-splitlr-sweep \
  --muon-factor-range 5e-4,1e-3,2e-3,5e-3
```

This additionally generates:

- `muon-splitlr`

Details:

- `--lr` controls the AdamW backup learning rate.
- `--muon_lr_factor` controls the Muon branch learning rate.
- The default is `MUON_ADAMW_BACKUP_LR=1e-3`.

**Run a Subset of Experiments**

The script supports stricter comma-separated filtering through `ONLY_EXPERIMENTS`, replacing the previous fuzzy substring matching.

Matching rules:

- Exact full experiment ID match, such as `adamw_lr1e-3`.
- Prefix match at an underscore boundary, such as `muon` matching `muon_lr...`.
- Prefix match before a numeric suffix, such as `muoneq-rc_phase` matching `muoneq-rc_phase1000_lr...`.

Therefore:

- `muon` no longer accidentally matches `muoneq-rc`.
- `muon` no longer accidentally matches `muonplus`.
- `muon` no longer accidentally matches `muonsplit`.
- `muoneq-rc_phase` does not match `muoneq-rc_phaseNone`.

Example:

```bash
ONLY_EXPERIMENTS=muon,muoneq-rc_phase DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

This command keeps only:

- `muon_lr...`
- `muoneq-rc_phase1000_lr...`

This is also commonly used:

```bash
ONLY_EXPERIMENTS=muonplus,muoneq-rc DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

This command keeps only:

- `muonplus_lr...`
- `muoneq-rc_phaseNone_lr...`
- `muoneq-rc_phase1000_lr...`

The following forms are also supported:

```bash
ONLY_EXPERIMENTS=muon DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

```bash
ONLY_EXPERIMENTS=muoneq-rc_phase DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

```bash
ONLY_EXPERIMENTS=muoneq-r,muoneq-c DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

```bash
ONLY_EXPERIMENTS=muonsplit DRY_RUN=1 bash llm-baselines/scripts/optimizers_compare/train-baselines.sh \
  --include-muon-splitlr-sweep
```

**Common Environment Variables**

Most training parameters in the script can be overridden with environment variables. Common ones include:

- `GPU_IDS`
- `NPROC_PER_NODE`
- `BATCH_SIZE`
- `SEQUENCE_LENGTH`
- `ITERATIONS`
- `WARMUP_STEPS`
- `EVAL_INTERVAL`
- `MUONEQ_PHASE_SWITCH`
- `ZEROPOWER_MODE`
- `MUON_NS_STEPS`
- `WANDB_PROJECT`
- `RESULTS_BASE_FOLDER`
- `LOG_DIR`
- `RUN_NAME_PREFIX`
- `EXPERIMENT_SPECS`
- `EXTRA_COMMON_ARGS`
- `SLEEP_BETWEEN_RUNS`
- `DRY_RUN`

Example:

```bash
BATCH_SIZE=24 ITERATIONS=10000 WARMUP_STEPS=500 \
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

Append extra arguments to all experiments:

```bash
EXTRA_COMMON_ARGS="--compile --save_every 0" \
bash llm-baselines/scripts/optimizers_compare/train-baselines.sh
```

**WandB and Logs**

Each experiment run name automatically includes:

- Optimizer name
- Learning rate or factor
- `adamw` appends `beta1_<BETA1>_beta2_<BETA2>`
- `muon` variants append `mom_<MOMENTUM>`
- `zp_<ZEROPOWER_MODE>`
- `ns<MUON_NS_STEPS>`
- batch size
- acc steps
- GPU count
- seed

If `RUN_NAME_PREFIX` is set, it is prepended to the run name.

Logs are written to:

- `llm-baselines/logs/optimizer_comparison_fineweb10b/`

Experiment results are written to:

- `llm-baselines/exps/optimizer_comparison_fineweb10b/`

During training, the script records:

- Peak memory
- Training time
- Step time
- `tokens/s`
- WandB summary

**Dataset Notes**

The current code only loads data locally and does not download automatically from Hugging Face. `fineweb10b` supports two file layouts:

- Single files: `train.bin` and `val.bin`
- Sharded files: `fineweb_train_*.bin` and `fineweb_val_*.bin`

Sharded files additionally support two internal formats:

- Plain `uint16` token streams
- Shard format with a 256-entry `int32` header
  This format automatically skips the header and reads only the following `uint16` token region.

If the corresponding local files are not found, the program errors directly and lists the directories it searched.

**Extending the Script**

To add more optimizers later, mainly update `build_experiment_matrix()` in `train-baselines.sh` and add another `register_experiment` entry. The run name, log file name, and `torchrun` command expand automatically.
