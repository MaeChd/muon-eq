#!/bin/bash
# ============================================================================
# 2026 Optimizer Comparison - 1B Main Training (Multi-Node)
# ============================================================================
# Usage:
#   NUM_NPUS_PER_JOB=8 bash multi_node_main_1b.sh --nodes 131,132
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/multi_node_common.sh"

MODEL_SIZE="1b"

BEST_CONFIGS=(
    "muon 5e-4"
    "muoneq-row 5e-4"
    "muon-nesterov 5e-4"
    "muon-mvr1 5e-4 0.05"
    "adamw 5e-4"
    "mars 5e-4 0.025"
    "foam 5e-4 _ 1"
)

WARMUP_STEPS=${WARMUP[$MODEL_SIZE]}

parse_multi_node_cli_args "$@" || exit $?
parse_experiment_cli_args "${MULTI_NODE_SCRIPT_ARGS[@]}" || exit $?
PHASE="${EXPERIMENT_SCRIPT_ARGS[0]:-all}"
SELECTED_BUDGETS="$(get_main_selected_budgets)"

declare -A BUDGET_TOTAL_STEPS
declare -A BUDGET_SAVE_STEP
declare -A BUDGET_DECAY_STEPS
BUDGET_SAVE_STEP_LIST=()
BUDGET_TOTAL_SUMMARY=""
BUDGET_DECAY_SUMMARY=""
MAX_BUDGET=""
USES_SHARED_TRUNK=0

budget_label() {
    local budget="$1"
    echo "${budget%x}X"
}

for budget in $SELECTED_BUDGETS; do
    total_steps="$(get_budget_total_steps "$MODEL_SIZE" "$budget")" || exit $?
    BUDGET_TOTAL_STEPS[$budget]="$total_steps"
    BUDGET_SAVE_STEP[$budget]=$(( total_steps * 9 / 10 ))
    BUDGET_DECAY_STEPS[$budget]=$(( total_steps / 10 ))
    BUDGET_SAVE_STEP_LIST+=("${BUDGET_SAVE_STEP[$budget]}")

    if [[ -n "$BUDGET_TOTAL_SUMMARY" ]]; then
        BUDGET_TOTAL_SUMMARY+=", "
    fi
    BUDGET_TOTAL_SUMMARY+="$(budget_label "$budget")=${BUDGET_TOTAL_STEPS[$budget]}"

    if [[ -n "$BUDGET_DECAY_SUMMARY" ]]; then
        BUDGET_DECAY_SUMMARY+=", "
    fi
    BUDGET_DECAY_SUMMARY+="$(budget_label "$budget")=${BUDGET_DECAY_STEPS[$budget]}"
    MAX_BUDGET="$budget"
done

if lr_scheduler_is_wsd; then
    USES_SHARED_TRUNK=1
fi

TRUNK_STEPS="${BUDGET_SAVE_STEP[$MAX_BUDGET]}"
SAVE_STEPS="$(IFS=,; echo "${BUDGET_SAVE_STEP_LIST[*]}")"

validate_multi_node_batch_layout "$MODEL_SIZE" || exit $?
EFFECTIVE_BATCH_SIZE="$(get_effective_batch_size "$MODEL_SIZE")"
EFFECTIVE_TOTAL_BATCH_SIZE="$(get_effective_total_batch_size "$MODEL_SIZE")"

echo "============================================================"
echo " 1B Main Training (Multi-Node)"
echo " Budgets: ${SELECTED_BUDGETS}"
if (( USES_SHARED_TRUNK )); then
    echo " Mode: shared trunk + resume decay"
    echo " Trunk: ${TRUNK_STEPS} steps (warmup=${WARMUP_STEPS})"
    echo " Checkpoints at: ${SAVE_STEPS}"
    echo " Decay: ${BUDGET_DECAY_SUMMARY}"
else
    echo " Mode: direct full-budget training"
    echo " Full runs: ${BUDGET_TOTAL_SUMMARY} (warmup=${WARMUP_STEPS})"
fi
echo " LR scheduler: ${LR_SCHEDULER}"
echo " Phase: ${PHASE}"
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

parse_config() {
    local config="$1"
    local parts=($config)
    OPT="${parts[0]}"
    LR="${parts[1]}"
    GAMMA="${parts[2]:-}"
    FOLD="${parts[3]:-}"
    [[ "$GAMMA" == "_" ]] && GAMMA=""
    [[ "$FOLD" == "_" ]] && FOLD=""
    return 0
}

run_trunk() {
    echo ""
    echo "================ Phase 1: Trunk Training ================"
    JOB_SLOT=0
    PIDS=()

    for config_line in "${BEST_CONFIGS[@]}"; do
        parse_config "$config_line"
        local device_group
        device_group=$(get_device_group "$JOB_SLOT")

        local run_name
        run_name=$(gen_main_name "trunk" "$OPT" "$LR" "$GAMMA" "$FOLD")
        local ckpt_dir
        ckpt_dir=$(gen_ckpt_dir "$MODEL_SIZE" "$OPT" "$LR" "$GAMMA" "$FOLD")

        local extra_args=""
        [[ -n "$GAMMA" ]] && extra_args+=" --gamma $GAMMA"
        [[ -n "$FOLD" ]] && extra_args+=" --fold $FOLD"

        echo ""
        echo "Launching trunk: $OPT (lr=$LR)"

        run_training "$device_group" "$MODEL_SIZE" "$OPT" "$LR" "$run_name" \
            "$TRUNK_STEPS" "$WARMUP_STEPS" "0" "$SAVE_STEPS" "$ckpt_dir" \
            --log_group main \
            $extra_args
        PIDS+=("$RUN_TRAINING_PID")

        JOB_SLOT=$(( JOB_SLOT + 1 ))
        if [[ ${#PIDS[@]} -ge $MAX_CONCURRENT_JOBS ]]; then
            echo ""
            echo "All multi-node device groups are occupied. Waiting for current batch..."
            wait_for_pids "${PIDS[@]}"
            PIDS=()
            JOB_SLOT=0
            echo ""
        fi
        sleep 3
    done

    if [[ ${#PIDS[@]} -gt 0 ]]; then
        wait_for_pids "${PIDS[@]}"
    fi
    echo "Trunk training complete!"
}

run_direct() {
    echo ""
    echo "================ Direct Training ================"

    for budget in $SELECTED_BUDGETS; do
        local total_steps="${BUDGET_TOTAL_STEPS[$budget]}"

        echo ""
        echo "--- Direct full run for ${budget} (${total_steps} total steps, warmup=${WARMUP_STEPS}) ---"

        JOB_SLOT=0
        PIDS=()

        for config_line in "${BEST_CONFIGS[@]}"; do
            parse_config "$config_line"
            local device_group
            device_group=$(get_device_group "$JOB_SLOT")

            local run_name
            run_name=$(gen_main_name "$budget" "$OPT" "$LR" "$GAMMA" "$FOLD")
            local ckpt_dir
            ckpt_dir=$(gen_ckpt_dir "$MODEL_SIZE" "$OPT" "$LR" "$GAMMA" "$FOLD")

            local extra_args=""
            [[ -n "$GAMMA" ]] && extra_args+=" --gamma $GAMMA"
            [[ -n "$FOLD" ]] && extra_args+=" --fold $FOLD"

            echo "  Launching direct run: $OPT ${budget}"

            run_training "$device_group" "$MODEL_SIZE" "$OPT" "$LR" "$run_name" \
                "$total_steps" "$WARMUP_STEPS" "0" "none" "$ckpt_dir" \
                --log_group main \
                $extra_args
            PIDS+=("$RUN_TRAINING_PID")

            JOB_SLOT=$(( JOB_SLOT + 1 ))
            if [[ ${#PIDS[@]} -ge $MAX_CONCURRENT_JOBS ]]; then
                echo ""
                echo "All multi-node device groups are occupied. Waiting for current batch..."
                wait_for_pids "${PIDS[@]}"
                PIDS=()
                JOB_SLOT=0
                echo ""
            fi
            sleep 3
        done

        if [[ ${#PIDS[@]} -gt 0 ]]; then
            wait_for_pids "${PIDS[@]}"
        fi
    done

    echo "Direct training complete!"
}

run_decay() {
    echo ""
    echo "================ Phase 2: Decay Training ================"

    for budget in $SELECTED_BUDGETS; do
        local decay_steps="${BUDGET_DECAY_STEPS[$budget]}"
        local total_steps="${BUDGET_TOTAL_STEPS[$budget]}"
        local save_step="${BUDGET_SAVE_STEP[$budget]}"

        echo ""
        echo "--- Decay for ${budget} (${decay_steps} decay steps from step ${save_step} to total step ${total_steps}) ---"

        JOB_SLOT=0
        PIDS=()

        for config_line in "${BEST_CONFIGS[@]}"; do
            parse_config "$config_line"

            local ckpt_dir
            ckpt_dir=$(gen_ckpt_dir "$MODEL_SIZE" "$OPT" "$LR" "$GAMMA" "$FOLD")
            local resume_path="${ckpt_dir}/resume/resume_step${save_step}"
            local run_name
            run_name=$(gen_main_name "$budget" "$OPT" "$LR" "$GAMMA" "$FOLD")

            if [[ ! -d "$resume_path" ]]; then
                echo "  WARNING: Checkpoint not found: $resume_path (skipping $OPT $budget)"
                continue
            fi

            local extra_args="--continue_from $resume_path"
            [[ -n "$GAMMA" ]] && extra_args+=" --gamma $GAMMA"
            [[ -n "$FOLD" ]] && extra_args+=" --fold $FOLD"

            echo "  Launching decay: $OPT ${budget}"
            local device_group
            device_group=$(get_device_group "$JOB_SLOT")

            run_training "$device_group" "$MODEL_SIZE" "$OPT" "$LR" "$run_name" \
                "$total_steps" "0" "$decay_steps" "none" "$ckpt_dir" \
                --log_group main \
                $extra_args
            PIDS+=("$RUN_TRAINING_PID")

            JOB_SLOT=$(( JOB_SLOT + 1 ))
            if [[ ${#PIDS[@]} -ge $MAX_CONCURRENT_JOBS ]]; then
                echo ""
                echo "All multi-node device groups are occupied. Waiting for current batch..."
                wait_for_pids "${PIDS[@]}"
                PIDS=()
                JOB_SLOT=0
                echo ""
            fi
            sleep 3
        done

        if [[ ${#PIDS[@]} -gt 0 ]]; then
            wait_for_pids "${PIDS[@]}"
        fi
    done

    echo "Decay training complete!"
}

case "$PHASE" in
    trunk)
        if (( USES_SHARED_TRUNK )); then
            run_trunk
        else
            run_direct
        fi
        ;;
    decay)
        if (( USES_SHARED_TRUNK )); then
            run_decay
        else
            echo "Phase 'decay' requires a WSD scheduler with shared trunk checkpoints." >&2
            echo "Use phase 'all' or 'trunk' to run direct full-budget training for scheduler '${LR_SCHEDULER}'." >&2
            exit 1
        fi
        ;;
    all)
        if (( USES_SHARED_TRUNK )); then
            run_trunk
            run_decay
        else
            run_direct
        fi
        ;;
    *)
        echo "Unknown phase: $PHASE. Use: trunk, decay, or all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " 1B Main Training Complete!"
echo " Checkpoints: ${CKPT_BASE}/${MODEL_SIZE}/"
echo " Logs: ${LOG_BASE}/main/<optimizer>/"
echo "============================================================"