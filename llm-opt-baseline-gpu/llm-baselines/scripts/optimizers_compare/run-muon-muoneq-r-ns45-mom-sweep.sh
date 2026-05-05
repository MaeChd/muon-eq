#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train-baselines.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "train-baselines.sh not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-gpt2-base-fineweb10b-muon-muoneq-r-ns45-mom-sweep}"
LR_RANGE="${LR_RANGE:-1e-3,2e-3,5e-3}"
ZEROPOWER_MODE="${ZEROPOWER_MODE:-native}"
EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-muon@ns=4;muon@ns=5;muoneq-r@ns=4;muoneq-r@ns=5}"
MOMENTUM_VALUES_RAW="${MOMENTUM_VALUES_RAW:-0.9 0.95 0.98}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${REPO_ROOT}/logs/optimizer_comparison_fineweb10b}"

read -r -a MOMENTUM_VALUES <<< "${MOMENTUM_VALUES_RAW}"

mkdir -p "${SWEEP_LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
sweep_log="${SWEEP_LOG_DIR}/muon_muoneq-r_ns45_mom_sweep_${timestamp}.log"

{
    echo "Sweep started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  repo_root=${REPO_ROOT}"
    echo "  train_script=${TRAIN_SCRIPT}"
    echo "  wandb_project=${WANDB_PROJECT}"
    echo "  lr_range=${LR_RANGE}"
    echo "  zeropower_mode=${ZEROPOWER_MODE}"
    echo "  experiment_specs=${EXPERIMENT_SPECS}"
    echo "  momentums=${MOMENTUM_VALUES[*]}"
    echo "  sweep_log=${sweep_log}"
    echo ""

    for mom in "${MOMENTUM_VALUES[@]}"; do
        echo "=== MOMENTUM=${mom} ==="
        WANDB_PROJECT="${WANDB_PROJECT}" \
        LR_RANGE="${LR_RANGE}" \
        ZEROPOWER_MODE="${ZEROPOWER_MODE}" \
        EXPERIMENT_SPECS="${EXPERIMENT_SPECS}" \
        MOMENTUM="${mom}" \
        bash "${TRAIN_SCRIPT}"
        echo ""
    done
} 2>&1 | tee "${sweep_log}"
