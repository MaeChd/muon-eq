#!/bin/bash
# ============================================================================
# 2026 Optimizer Comparison - Multi-Node Common Helpers (Simplified)
# ============================================================================
# Source this after config.sh. It overrides run_training() so the existing
# sweep/main scheduling logic can launch one distributed job across multiple
# SSH-reachable nodes.
#
# Assumption: All nodes share the same filesystem (shared directory), so
# PROJECT_ROOT, data paths, checkpoint paths are identical everywhere.
# No path remapping needed.
#
# Set MULTI_NODE_PRE_CMD to configure env on all nodes when needed.
# Example:
#   export PATH=/path/to/conda/env/bin:$PATH
# ============================================================================

MULTI_NODE_HOSTS="${MULTI_NODE_HOSTS:-}"
MULTI_NODE_HOSTS="${MULTI_NODE_HOSTS// /}"
MULTI_NODE_HOST_PREFIX="${MULTI_NODE_HOST_PREFIX:-10.0.0.}"
MULTI_NODE_SSH_USER="${MULTI_NODE_SSH_USER:-}"
MULTI_NODE_PRE_CMD="${MULTI_NODE_PRE_CMD:-}"
MULTI_NODE_MASTER_PORT_BASE="${MULTI_NODE_MASTER_PORT_BASE:-29600}"
MULTI_NODE_MASTER_PORT_STRIDE="${MULTI_NODE_MASTER_PORT_STRIDE:-10}"
MULTI_NODE_BATCH_SIZE="${MULTI_NODE_BATCH_SIZE:-}"
MULTI_NODE_TOTAL_BATCH_SIZE="${MULTI_NODE_TOTAL_BATCH_SIZE:-}"
MULTI_NODE_IS_CONFIGURED=0
MULTI_NODE_NNODES=0
MULTI_NODE_WORLD_SIZE=0
MULTI_NODE_MASTER_ADDR="${MULTI_NODE_MASTER_ADDR:-}"
MULTI_NODE_HOST_ARRAY=()
MULTI_NODE_SCRIPT_ARGS=()

# ======================== Host Expansion ========================

expand_multi_node_host_entry() {
    local entry="${1:-}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && { echo ""; return 0; }
    if [[ "$entry" == *@* ]]; then
        local user_part="${entry%@*}"
        local host_part="${entry##*@}"
        echo "${user_part}@$(expand_multi_node_host_entry "$host_part")"
        return 0
    fi
    if [[ "$entry" =~ ^[0-9]{1,3}$ ]]; then
        echo "${MULTI_NODE_HOST_PREFIX}${entry}"
        return 0
    fi
    echo "$entry"
}

expand_multi_node_hosts_csv() {
    local raw="${1:-}"
    raw="${raw// /}"
    [[ -z "$raw" ]] && { echo ""; return 0; }
    local expanded=()
    IFS=',' read -r -a _items <<< "$raw"
    for item in "${_items[@]}"; do
        expanded+=("$(expand_multi_node_host_entry "$item")")
    done
    local IFS=,
    echo "${expanded[*]}"
}

# ======================== Configuration ========================

configure_multi_node_env() {
    local configured_master_addr="${MULTI_NODE_MASTER_ADDR:-}"
    MULTI_NODE_HOSTS="${MULTI_NODE_HOSTS// /}"
    MULTI_NODE_IS_CONFIGURED=0
    MULTI_NODE_NNODES=0
    MULTI_NODE_WORLD_SIZE=0
    MULTI_NODE_MASTER_ADDR="$configured_master_addr"
    MULTI_NODE_HOST_ARRAY=()

    MULTI_NODE_HOSTS="$(expand_multi_node_hosts_csv "$MULTI_NODE_HOSTS")"
    MULTI_NODE_MASTER_ADDR="$(expand_multi_node_host_entry "$MULTI_NODE_MASTER_ADDR")"

    [[ -z "$MULTI_NODE_HOSTS" ]] && return 0

    IFS=',' read -r -a MULTI_NODE_HOST_ARRAY <<< "$MULTI_NODE_HOSTS"
    if (( ${#MULTI_NODE_HOST_ARRAY[@]} < 2 )); then
        echo "ERROR: MULTI_NODE_HOSTS must contain at least 2 hosts, got '$MULTI_NODE_HOSTS'" >&2
        return 1
    fi

    MULTI_NODE_NNODES=${#MULTI_NODE_HOST_ARRAY[@]}
    MULTI_NODE_MASTER_ADDR="${MULTI_NODE_MASTER_ADDR:-${MULTI_NODE_HOST_ARRAY[0]##*@}}"
    MULTI_NODE_WORLD_SIZE=$(( MULTI_NODE_NNODES * NUM_NPUS_PER_JOB ))
    MULTI_NODE_IS_CONFIGURED=1
}

# ======================== CLI Parsing ========================

parse_multi_node_cli_args() {
    MULTI_NODE_SCRIPT_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --nodes|--hosts)
                MULTI_NODE_HOSTS="$2"; shift 2 ;;
            --nodes=*|--hosts=*)
                MULTI_NODE_HOSTS="${1#*=}"; shift ;;
            --master-addr)
                MULTI_NODE_MASTER_ADDR="$2"; shift 2 ;;
            --master-addr=*)
                MULTI_NODE_MASTER_ADDR="${1#*=}"; shift ;;
            --ssh-user)
                MULTI_NODE_SSH_USER="$2"; shift 2 ;;
            --ssh-user=*)
                MULTI_NODE_SSH_USER="${1#*=}"; shift ;;
            --host-prefix)
                MULTI_NODE_HOST_PREFIX="$2"; shift 2 ;;
            --host-prefix=*)
                MULTI_NODE_HOST_PREFIX="${1#*=}"; shift ;;
            --)
                shift
                while [[ $# -gt 0 ]]; do
                    MULTI_NODE_SCRIPT_ARGS+=("$1"); shift
                done ;;
            *)
                MULTI_NODE_SCRIPT_ARGS+=("$1"); shift ;;
        esac
    done
    configure_multi_node_env
}

ensure_multi_node_ready() {
    if (( MULTI_NODE_IS_CONFIGURED )); then return 0; fi
    echo "ERROR: MULTI_NODE_HOSTS is required. Example: --nodes 131,132" >&2
    return 1
}

# ======================== Batch Size Helpers ========================

get_effective_batch_size() {
    local model_size="$1"
    if [[ -n "$MULTI_NODE_BATCH_SIZE" ]]; then
        echo "$MULTI_NODE_BATCH_SIZE"
    else
        echo "${MICRO_BATCH_SIZE[$model_size]}"
    fi
}

get_effective_total_batch_size() {
    local model_size="$1"
    if [[ -n "$MULTI_NODE_TOTAL_BATCH_SIZE" ]]; then
        echo "$MULTI_NODE_TOTAL_BATCH_SIZE"
    else
        echo "${TOTAL_BATCH_SIZE[$model_size]}"
    fi
}

validate_multi_node_batch_layout() {
    local model_size="$1"
    ensure_multi_node_ready || return 1

    local batch_size total_batch_size
    batch_size="$(get_effective_batch_size "$model_size")"
    total_batch_size="$(get_effective_total_batch_size "$model_size")"

    local denom=$(( batch_size * MULTI_NODE_WORLD_SIZE ))
    if (( total_batch_size % denom != 0 )); then
        echo "ERROR: total_batch_size=${total_batch_size} incompatible with batch_size=${batch_size} * world_size=${MULTI_NODE_WORLD_SIZE}" >&2
        return 1
    fi
}

get_job_master_port() {
    local slot="${1:-0}"
    echo $(( MULTI_NODE_MASTER_PORT_BASE + (($$ % 100) * MULTI_NODE_MASTER_PORT_STRIDE) + (slot * MULTI_NODE_MASTER_PORT_STRIDE) ))
}

# ======================== SSH Helper ========================

get_ssh_target() {
    local host="$1"
    if [[ "$host" == *@* ]]; then
        echo "$host"
    elif [[ -n "$MULTI_NODE_SSH_USER" ]]; then
        echo "${MULTI_NODE_SSH_USER}@${host}"
    else
        echo "$host"
    fi
}

# ======================== run_training (overrides config.sh) ========================
# All nodes share the same filesystem, so we use the same paths everywhere.
# For each worker node: ssh -> cd PROJECT_ROOT -> torchrun with node_rank.
# Master node (rank 0) runs locally.

run_training() {
    ensure_multi_node_ready || return 1

    local device_group="$1"
    local model_size="$2"
    local optimizer="$3"
    local lr="$4"
    local run_name="$5"
    local num_steps="$6"
    local warmup="$7"
    local decay_steps="$8"
    local save_steps="$9"
    local ckpt_dir="${10}"
    shift 10

    local continue_from="" gamma="" fold="" log_group="misc" seed="$DEFAULT_SEED"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --continue_from) continue_from="$2"; shift 2 ;;
            --gamma)         gamma="$2"; shift 2 ;;
            --fold)          fold="$2"; shift 2 ;;
            --log_group)     log_group="$2"; shift 2 ;;
            --seed)          seed="$2"; shift 2 ;;
            *)               echo "Unknown arg: $1"; shift ;;
        esac
    done

    validate_multi_node_batch_layout "$model_size" || return 1
    validate_project_layout "$PROJECT_ROOT" || return 1

    local config="${MODEL_CONFIG[$model_size]}"
    local batch project
    batch="$(get_effective_batch_size "$model_size")"
    local total_batch
    total_batch="$(get_effective_total_batch_size "$model_size")"
    project="${SWANLAB_PROJECT[$model_size]}"

    local log_dir="${LOG_BASE}/${log_group}/${optimizer}"
    local log_ts
    log_ts="$(date '+%Y%m%d-%H%M%S')"
    local log_file="${log_dir}/${log_ts}-${model_size}-${run_name}-pid$$.log"
    mkdir -p "$log_dir" "$ckpt_dir"

    local slot="${JOB_SLOT:-0}"
    local master_port
    master_port="$(get_job_master_port "$slot")"

    echo "  [${MULTI_NODE_HOSTS}] $optimizer lr=$lr scheduler=$LR_SCHEDULER name=$run_name steps=$num_steps decay=$decay_steps port=$master_port"

    # Build train_args (same for all nodes since paths are shared)
    local -a adamw_betas
    read -r -a adamw_betas <<< "$ADAMW_BETAS"

    local -a train_args=(
        "${TRAIN_SCRIPT}"
        --model_config "$config"
        --data_dir "$data_dir"
        --tokenizer_path "$tokenizer_dir"
        --lr "$lr"
        --batch_size "$batch"
        --total_batch_size "$total_batch"
        --num_training_steps "$num_steps"
        --warmup_steps "$warmup"
        --weight_decay "$WEIGHT_DECAY"
        --grad_clipping "$GRAD_CLIP"
        --min_lr_ratio "$MIN_LR_RATIO"
        --dtype "$DTYPE"
        --eval_every "$EVAL_EVERY"
        --max_length "$MAX_LENGTH"
        --workers "$WORKERS"
        --seed "$seed"
        --optimizer "$optimizer"
        --scheduler "$LR_SCHEDULER"
        --name "$run_name"
        --save_dir "${ckpt_dir}/${run_name}"
        --beta1 "$MUON_MOMENTUM"
        --betas "${adamw_betas[@]}"
    )
    if [[ "$TRACKER" == "1" ]]; then
        train_args+=(--tracker)
    fi
    if lr_scheduler_is_wsd; then
        train_args+=(--wsd_decay_steps "$decay_steps")
    fi

    [[ -n "$EVAL_BATCH_SIZE" ]] && train_args+=(--eval_batch_size "$EVAL_BATCH_SIZE")
    [[ "$CACHE_EVAL_DATA" == "1" ]] && train_args+=(--cache_eval_data)
    [[ "$ENABLE_ACTIVATION_CHECKPOINTING" == "1" ]] && train_args+=(--activation_checkpointing)
    [[ -n "${TRAIN_DEBUG_STEP_START:-}" ]] && train_args+=(--debug_update_step_start "$TRAIN_DEBUG_STEP_START")
    [[ -n "${TRAIN_DEBUG_STEP_END:-}" ]] && train_args+=(--debug_update_step_end "$TRAIN_DEBUG_STEP_END")
    [[ -n "${TRAIN_DEBUG_RANKS:-}" ]] && train_args+=(--debug_ranks "$TRAIN_DEBUG_RANKS")
    [[ "${TRAIN_DEBUG_FORCE_SYNC:-0}" == "1" ]] && train_args+=(--debug_force_sync)
    [[ -n "${TRAIN_DEBUG_DUMP_BATCH_DIR:-}" ]] && train_args+=(--debug_dump_batch_dir "$TRAIN_DEBUG_DUMP_BATCH_DIR")
    (( MULTI_NODE_WORLD_SIZE == 1 )) && train_args+=(--single_gpu)

    case "$REPORT_TO" in
        wandb)   train_args+=(--report_to wandb --wandb_proj "$project") ;;
        swanlab) train_args+=(--report_to swanlab --swanlab_proj "$project") ;;
        none)    train_args+=(--report_to none) ;;
    esac

    [[ -n "$save_steps" && "$save_steps" != "none" ]] && train_args+=(--save_steps "$save_steps" --checkpoint_dir "${ckpt_dir}/resume")
    [[ -n "$continue_from" ]] && train_args+=(--continue_from "$continue_from")
    [[ -n "$gamma" ]] && train_args+=(--mars_gamma "$gamma")
    [[ -n "$fold" ]] && train_args+=(--fold_level "$fold")

    # Common torchrun args for multi-node
    local -a torchrun_args=(
        --nnodes "$MULTI_NODE_NNODES"
        --nproc_per_node "$NUM_NPUS_PER_JOB"
        --master_addr "$MULTI_NODE_MASTER_ADDR"
        --master_port "$master_port"
    )

    (
        set -euo pipefail

        # Track remote worker PIDs for cleanup
        remote_workers=()
        cleanup_remote_workers() {
            for worker in "${remote_workers[@]}"; do
                local host="${worker%%:*}" pid="${worker##*:}"
                ssh -n -o StrictHostKeyChecking=no "$(get_ssh_target "$host")" \
                    "kill ${pid} 2>/dev/null || true" 2>/dev/null || true
            done
        }
        trap cleanup_remote_workers EXIT INT TERM

        # Build the pre-command (conda activate etc.) for SSH
        local pre_cmd=""
        if [[ -n "$MULTI_NODE_PRE_CMD" ]]; then
            pre_cmd="${MULTI_NODE_PRE_CMD} && "
        fi

        # Build the training command string once (same for all nodes)
        local ascend_launch_blocking="${ASCEND_LAUNCH_BLOCKING:-1}"
        local ascend_slog_print_to_stdout="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
        local ascend_global_log_level="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
        local hccl_entry_log_enable="${HCCL_ENTRY_LOG_ENABLE:-1}"
        local train_cmd="env ASCEND_RT_VISIBLE_DEVICES=${device_group} ASCEND_LAUNCH_BLOCKING=${ascend_launch_blocking} ASCEND_SLOG_PRINT_TO_STDOUT=${ascend_slog_print_to_stdout} ASCEND_GLOBAL_LOG_LEVEL=${ascend_global_log_level} HCCL_ENTRY_LOG_ENABLE=${hccl_entry_log_enable} TOKENIZERS_PARALLELISM=false "
        train_cmd+="${TORCHRUN_BIN} ${torchrun_args[*]}"

        # Launch worker nodes (rank 1..N-1) via SSH
        # Use ssh -f + PID file on shared filesystem to avoid SSH hanging
        local pid_dir="${log_dir}/.pids_$$"
        mkdir -p "$pid_dir"

        for (( rank=1; rank<MULTI_NODE_NNODES; rank++ )); do
            local host="${MULTI_NODE_HOST_ARRAY[$rank]}"
            local ssh_target
            ssh_target="$(get_ssh_target "$host")"
            local worker_log="${log_file%.log}-node${rank}.log"
            local pid_file="${pid_dir}/rank${rank}.pid"

            local worker_cmd="${pre_cmd}cd ${PROJECT_ROOT} && nohup ${train_cmd} --node_rank ${rank} ${train_args[*]} > ${worker_log} 2>&1 </dev/null & echo \$! > ${pid_file}"

            echo "    launching worker rank ${rank} on ${host}..."
            ssh -n -f -o StrictHostKeyChecking=no "$ssh_target" "${worker_cmd}"
        done

        # Wait for all PID files and collect PIDs
        for (( rank=1; rank<MULTI_NODE_NNODES; rank++ )); do
            local host="${MULTI_NODE_HOST_ARRAY[$rank]}"
            local pid_file="${pid_dir}/rank${rank}.pid"
            local worker_log="${log_file%.log}-node${rank}.log"

            # Wait up to 10s for PID file to appear on shared filesystem
            local wait_count=0
            while [[ ! -s "${pid_file}" ]] && (( wait_count < 20 )); do
                sleep 0.5
                (( wait_count++ )) || true
            done

            if [[ -s "${pid_file}" ]]; then
                remote_pid="$(cat "${pid_file}" | tr -d '[:space:]')"
            else
                echo "Timeout waiting for PID file from ${host} (rank ${rank})" >&2
                exit 1
            fi

            if ! [[ "$remote_pid" =~ ^[0-9]+$ ]]; then
                echo "Failed to launch worker on ${host}: ${remote_pid}" >&2
                exit 1
            fi
            remote_workers+=("${host}:${remote_pid}")
            echo "    worker rank ${rank} on ${host} pid=${remote_pid} log=${worker_log}"
        done

        # Clean up PID files
        rm -rf "$pid_dir"

        # Launch master node (rank 0) locally
        if [[ -n "$MULTI_NODE_PRE_CMD" ]]; then
            eval "$MULTI_NODE_PRE_CMD"
        fi
        cd "$PROJECT_ROOT"
        env \
            ASCEND_RT_VISIBLE_DEVICES="$device_group" \
            ASCEND_LAUNCH_BLOCKING="$ascend_launch_blocking" \
            ASCEND_SLOG_PRINT_TO_STDOUT="$ascend_slog_print_to_stdout" \
            ASCEND_GLOBAL_LOG_LEVEL="$ascend_global_log_level" \
            HCCL_ENTRY_LOG_ENABLE="$hccl_entry_log_enable" \
            TOKENIZERS_PARALLELISM=false \
            "$TORCHRUN_BIN" \
            "${torchrun_args[@]}" \
            --node_rank 0 \
            "${train_args[@]}"
    ) > "$log_file" 2>&1 &

    RUN_TRAINING_PID=$!
    echo "    PID: ${RUN_TRAINING_PID} | Log: ${log_file}"
}

# ======================== Init ========================

configure_multi_node_env || return 1 2>/dev/null || exit 1

echo "[multi_node_common.sh] Multi-node launcher ready."
if (( MULTI_NODE_IS_CONFIGURED )); then
    echo "  Hosts: ${MULTI_NODE_HOSTS}"
    echo "  Master: ${MULTI_NODE_MASTER_ADDR}"
    echo "  Nodes: ${MULTI_NODE_NNODES}  |  World size: ${MULTI_NODE_WORLD_SIZE}"
    [[ -n "$MULTI_NODE_PRE_CMD" ]] && echo "  Pre-cmd: ${MULTI_NODE_PRE_CMD}"
fi
