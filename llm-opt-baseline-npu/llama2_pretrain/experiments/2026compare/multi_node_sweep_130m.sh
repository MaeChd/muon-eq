#!/bin/bash
# ============================================================================
# 2026 Optimizer Comparison - 130M Model LR Sweep (Multi-Node, 1X Budget)
# ============================================================================
# Usage:
#   NUM_NPUS_PER_JOB=4 bash multi_node_sweep_130m.sh --nodes 131,132
#
#   NUM_NPUS_PER_JOB=8 \
#   MULTI_NODE_TOTAL_BATCH_SIZE=256 \
#   bash multi_node_sweep_130m.sh --nodes 131,132 adamw
#
#   NUM_NPUS_PER_JOB=4 \
#   bash multi_node_sweep_130m.sh --nodes 131,132 --tracker muon
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/multi_node_common.sh"

MODEL_SIZE="130m"
ITERS=${ITERS_1X[$MODEL_SIZE]}
WARMUP_STEPS=${WARMUP[$MODEL_SIZE]}
DECAY_STEPS=$(( ITERS / 10 ))

parse_multi_node_cli_args "$@" || exit $?
parse_sweep_cli_args "$MODEL_SIZE" "${MULTI_NODE_SCRIPT_ARGS[@]}" || exit $?
if (( SWEEP_SHOW_HELP )); then
    print_sweep_cli_usage "$0" "$MODEL_SIZE"
    exit 0
fi
validate_multi_node_batch_layout "$MODEL_SIZE" || exit $?

OPTIMIZERS="$SWEEP_OPTIMIZERS"
LR_VALUES="$SWEEP_LR_VALUES"
EFFECTIVE_BATCH_SIZE="$(get_effective_batch_size "$MODEL_SIZE")"
EFFECTIVE_TOTAL_BATCH_SIZE="$(get_effective_total_batch_size "$MODEL_SIZE")"

echo "============================================================"
echo " 130M LR Sweep (Multi-Node, 1X budget: ${ITERS} steps)"
echo " Optimizers: ${OPTIMIZERS}"
echo " LR range: ${LR_VALUES}"
echo " LR scheduler: ${LR_SCHEDULER}"
echo " Tracker: ${TRACKER}"
echo " Hosts: ${MULTI_NODE_HOSTS}"
echo " Master: ${MULTI_NODE_MASTER_ADDR}"
echo " Nodes: ${MULTI_NODE_NNODES}"
echo " Visible NPUs per node: ${VISIBLE_NPUS}"
echo " NPUs per node/job: ${NUM_NPUS_PER_JOB}"
echo " Global world size: ${MULTI_NODE_WORLD_SIZE}"
echo " Micro batch per rank: ${EFFECTIVE_BATCH_SIZE}"
echo " Total batch size: ${EFFECTIVE_TOTAL_BATCH_SIZE}"
echo " Concurrent jobs per batch: ${MAX_CONCURRENT_JOBS}"
echo "============================================================"

JOB_SLOT=0
MAX_JOBS=$MAX_CONCURRENT_JOBS
PIDS=()

launch_job() {
    local opt="$1"
    local lr="$2"
    local gamma="$3"
    local fold="$4"
    local device_group
    device_group=$(get_device_group "$JOB_SLOT")

    local run_name
    run_name=$(gen_sweep_name "$opt" "$lr" "$gamma" "$fold")
    local ckpt_dir
    ckpt_dir=$(gen_ckpt_dir "$MODEL_SIZE" "$opt" "$lr" "$gamma" "$fold")

    local extra_args=""
    [[ -n "$gamma" ]] && extra_args+=" --gamma $gamma"
    [[ -n "$fold" ]] && extra_args+=" --fold $fold"

    run_training "$device_group" "$MODEL_SIZE" "$opt" "$lr" "$run_name" \
        "$ITERS" "$WARMUP_STEPS" "$DECAY_STEPS" "none" "$ckpt_dir" \
        --log_group sweep \
        $extra_args
    PIDS+=("$RUN_TRAINING_PID")

    JOB_SLOT=$(( JOB_SLOT + 1 ))

    if [[ ${#PIDS[@]} -ge $MAX_JOBS ]]; then
        echo ""
        echo "All multi-node device groups are occupied. Waiting for current batch..."
        wait_for_pids "${PIDS[@]}"
        PIDS=()
        JOB_SLOT=0
        echo ""
    fi

    sleep 3
}

for opt in $OPTIMIZERS; do
    echo ""
    echo "--- Optimizer: $opt ---"

    if needs_gamma "$opt"; then
        for lr in ${LR_VALUES}; do
            for gamma in $GAMMA_RANGE; do
                launch_job "$opt" "$lr" "$gamma" ""
            done
        done
    elif needs_fold "$opt"; then
        for lr in ${LR_VALUES}; do
            for fold in $FOLD_RANGE; do
                launch_job "$opt" "$lr" "" "$fold"
            done
        done
    else
        for lr in ${LR_VALUES}; do
            launch_job "$opt" "$lr" "" ""
        done
    fi
done

if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    wait_for_pids "${PIDS[@]}"
fi

echo ""
echo "============================================================"
echo " 130M LR Sweep Complete!"
echo " Check logs: ${LOG_BASE}/sweep/<optimizer>/"
echo " Check swanlab project: ${SWANLAB_PROJECT[$MODEL_SIZE]}"
echo "============================================================"
