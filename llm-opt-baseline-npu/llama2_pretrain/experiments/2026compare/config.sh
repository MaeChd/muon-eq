#!/bin/bash
# ============================================================================
# 2026 Optimizer Comparison - Common Configuration
# ============================================================================
# Shared variables and functions used by all sweep/trunk/decay scripts.
# Source this file: source "$(dirname "$0")/config.sh"
# ============================================================================

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="scripts/pretrain_c4_dist.py"
# TRAIN_SCRIPT="scripts/pretrain_muon_mvr_dist_compile.py"


data_dir="${C4_DATA_DIR:-/path/to/c4}"
tokenizer_dir="${TOKENIZER_PATH:-t5-base}"
# Log and checkpoint directories
LOG_BASE="${SCRIPT_DIR}/logs"
CKPT_BASE="${PROJECT_ROOT}/wsd_checkpoints"

# ======================== Common Hyperparameters ========================
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
MIN_LR_RATIO=0.1        # min lr = 10% of peak lr
LR_SCHEDULER="${LR_SCHEDULER:-wsd-l}"
REQUESTED_BUDGETS="${REQUESTED_BUDGETS:-}"
ADAMW_BETAS="0.9 0.95"
MUON_MOMENTUM=0.95
DTYPE="bfloat16"
DEFAULT_SEED=42
MAX_LENGTH=4096
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"      # default: 2x training batch_size
CACHE_EVAL_DATA="${CACHE_EVAL_DATA:-0}"     # set to 1 to cache validation dataset in memory
WORKERS="${WORKERS:-8}"
ENABLE_ACTIVATION_CHECKPOINTING="${ENABLE_ACTIVATION_CHECKPOINTING:-0}"
REPORT_TO="${REPORT_TO:-wandb}"
TRACKER="${TRACKER:-0}"

if [[ "$REPORT_TO" != "wandb" && "$REPORT_TO" != "swanlab" && "$REPORT_TO" != "none" ]]; then
    echo "ERROR: REPORT_TO must be one of: wandb, swanlab, none (got '$REPORT_TO')" >&2
    return 1 2>/dev/null || exit 1
fi

# ======================== Launch / Distributed Settings ========================
# Comma-separated physical NPU ids available to these experiment scripts.
# Example: VISIBLE_NPUS=0,1,2,3
VISIBLE_NPUS="${VISIBLE_NPUS:-0,1,2,3,4,5,6,7}"
VISIBLE_NPUS="${VISIBLE_NPUS// /}"

# Number of NPUs used by a single training job launched from sweep/main scripts.
# Default keeps the original behavior: one job per card.
NUM_NPUS_PER_JOB="${NUM_NPUS_PER_JOB:-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TRAIN_LAUNCH_PRE_CMD="${TRAIN_LAUNCH_PRE_CMD:-}"

if ! [[ "$NUM_NPUS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_NPUS_PER_JOB must be a positive integer, got '$NUM_NPUS_PER_JOB'" >&2
    return 1 2>/dev/null || exit 1
fi

IFS=',' read -r -a AVAILABLE_NPUS <<< "$VISIBLE_NPUS"
TOTAL_AVAILABLE_NPUS=${#AVAILABLE_NPUS[@]}

if [[ $TOTAL_AVAILABLE_NPUS -eq 0 ]]; then
    echo "ERROR: VISIBLE_NPUS must contain at least one device id" >&2
    return 1 2>/dev/null || exit 1
fi

if (( NUM_NPUS_PER_JOB > TOTAL_AVAILABLE_NPUS )); then
    echo "ERROR: NUM_NPUS_PER_JOB=$NUM_NPUS_PER_JOB exceeds available NPUs ($VISIBLE_NPUS)" >&2
    return 1 2>/dev/null || exit 1
fi

MAX_CONCURRENT_JOBS=$(( TOTAL_AVAILABLE_NPUS / NUM_NPUS_PER_JOB ))

if (( MAX_CONCURRENT_JOBS == 0 )); then
    echo "ERROR: Unable to derive any runnable job slots from VISIBLE_NPUS=$VISIBLE_NPUS" >&2
    return 1 2>/dev/null || exit 1
fi

if (( TOTAL_AVAILABLE_NPUS % NUM_NPUS_PER_JOB != 0 )); then
    echo "WARNING: TOTAL_AVAILABLE_NPUS=$TOTAL_AVAILABLE_NPUS is not divisible by NUM_NPUS_PER_JOB=$NUM_NPUS_PER_JOB" >&2
    echo "         Trailing devices will be left unused in each scheduling batch." >&2
fi

# ======================== Model Configurations ========================
# Model config files
declare -A MODEL_CONFIG
MODEL_CONFIG[130m]="configs/llama_130m.json"
MODEL_CONFIG[350m]="configs/llama_350m.json"
MODEL_CONFIG[1b]="configs/llama_1b.json"

# Total batch sizes (global)
declare -A TOTAL_BATCH_SIZE
TOTAL_BATCH_SIZE[130m]=128
TOTAL_BATCH_SIZE[350m]=256
TOTAL_BATCH_SIZE[1b]=512

# Per-GPU micro batch sizes (adjust based on NPU memory)
declare -A MICRO_BATCH_SIZE
MICRO_BATCH_SIZE[130m]=2
MICRO_BATCH_SIZE[350m]=4
MICRO_BATCH_SIZE[1b]=4

# ======================== Token Budgets (based on Chinchilla) ========================
# 1X iterations
declare -A ITERS_1X
ITERS_1X[130m]=5000
ITERS_1X[350m]=7500
ITERS_1X[1b]=10000

# 2X iterations
declare -A ITERS_2X
ITERS_2X[130m]=10000
ITERS_2X[350m]=15000
ITERS_2X[1b]=20000

# 4X iterations
declare -A ITERS_4X
ITERS_4X[130m]=20000
ITERS_4X[350m]=30000
ITERS_4X[1b]=40000

# 8X iterations
declare -A ITERS_8X
ITERS_8X[130m]=40000
ITERS_8X[350m]=60000
ITERS_8X[1b]=80000

# Warmup steps
declare -A WARMUP
WARMUP[130m]=500
WARMUP[350m]=500
WARMUP[1b]=1000

# ======================== Tracking Project Names ========================
# These names are passed to wandb or SwanLab according to REPORT_TO.
declare -A SWANLAB_PROJECT
SWANLAB_PROJECT[130m]="llama2-130m-opt-wsd"
SWANLAB_PROJECT[350m]="llama2-350m-opt-wsd"
SWANLAB_PROJECT[1b]="llama2-1b-opt-wsd"

# ======================== LR Search Ranges ========================
declare -A LR_RANGE
LR_RANGE[130m]="5e-4 1e-3 2e-3 3e-3 5e-3"
LR_RANGE[350m]="5e-4 1e-3 1.5e-3 2e-3"
LR_RANGE[1b]="2e-4 3e-4 5e-4 8e-4 1e-3"

# MARS / Muon-MVR1 gamma search range
# GAMMA_RANGE="0.1 0.05 0.025"
GAMMA_RANGE="0.025"


# FOAM fold level search range
FOLD_RANGE="1"

# ======================== Optimizer Definitions ========================
# Each optimizer defines: name, has_gamma, has_fold
# Optimizers to compare
ALL_OPTIMIZERS="muon muon-nesterov muon-mvr1 adamw mars foam"

# Check if optimizer needs gamma sweep
needs_gamma() {
    local opt="$1"
    [[ "$opt" == "mars" || "$opt" == "muon-mvr1" ]]
}

# Check if optimizer needs fold_level sweep
needs_fold() {
    local opt="$1"
    [[ "$opt" == "foam" ]]
}

# ======================== Helper Functions ========================

quote_cmd() {
    local quoted=()
    local arg
    for arg in "$@"; do
        quoted+=("$(printf '%q' "$arg")")
    done
    printf '%s' "${quoted[*]}"
}

run_launch_setup_cmd() {
    local setup_cmd="${1:-}"
    if [[ -z "$setup_cmd" ]]; then
        return 0
    fi

    eval "$setup_cmd"
}

remap_path_prefix() {
    local path="$1"
    local from_prefix="$2"
    local to_prefix="$3"

    if [[ "$path" == "$from_prefix" ]]; then
        echo "$to_prefix"
        return 0
    fi

    case "$path" in
        "$from_prefix"/*)
            echo "${to_prefix}${path#$from_prefix}"
            ;;
        *)
            echo "$path"
            ;;
    esac
}

resolve_train_script_path() {
    local root="$1"
    if [[ "$TRAIN_SCRIPT" = /* ]]; then
        echo "$TRAIN_SCRIPT"
    else
        echo "${root}/${TRAIN_SCRIPT}"
    fi
}

validate_project_layout() {
    local root="$1"

    if [[ ! -d "$root" ]]; then
        echo "ERROR: project root not found: $root" >&2
        return 1
    fi

    local train_script_path
    train_script_path="$(resolve_train_script_path "$root")"
    if [[ ! -f "$train_script_path" ]]; then
        echo "ERROR: training script not found: $train_script_path" >&2
        echo "       Check PROJECT_ROOT/MULTI_NODE_PROJECT_ROOT or TRAIN_SCRIPT." >&2
        return 1
    fi
}

SWEEP_OPTIMIZERS=""
SWEEP_LR_VALUES=""
SWEEP_SHOW_HELP=0
EXPERIMENT_SCRIPT_ARGS=()

normalize_sweep_list() {
    local items=()
    local token
    local item
    for token in "$@"; do
        token="${token//,/ }"
        for item in $token; do
            items+=("$item")
        done
    done
    echo "${items[*]}"
}

normalize_lr_scheduler() {
    local scheduler="${1,,}"
    case "$scheduler" in
        wsd|wsd-l|wsd-linear|wsd_linear)
            echo "wsd-l"
            ;;
        wsd-c|wsd-cosine|wsd_cosine)
            echo "wsd-c"
            ;;
        linear|cosine|cosine_restarts)
            echo "$scheduler"
            ;;
        *)
            return 1
            ;;
    esac
}

set_lr_scheduler() {
    local normalized
    if ! normalized="$(normalize_lr_scheduler "$1")"; then
        echo "ERROR: Unsupported LR scheduler '$1'. Use one of: wsd-l, wsd-c, linear, cosine, cosine_restarts" >&2
        return 1
    fi
    LR_SCHEDULER="$normalized"
}

set_eval_every() {
    local value="$1"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: eval_every must be a positive integer, got '$value'" >&2
        return 1
    fi
    EVAL_EVERY="$value"
}

lr_scheduler_is_wsd() {
    local normalized
    normalized="$(normalize_lr_scheduler "${1:-$LR_SCHEDULER}")" || return 1
    [[ "$normalized" == "wsd-l" || "$normalized" == "wsd-c" ]]
}

lr_scheduler_tag() {
    local normalized
    normalized="$(normalize_lr_scheduler "${1:-$LR_SCHEDULER}")" || return 1
    if [[ "$normalized" == "wsd-l" ]]; then
        return 0
    fi
    echo "sched-${normalized}"
}

normalize_budget() {
    local budget="${1,,}"
    case "$budget" in
        1|1x)
            echo "1x"
            ;;
        2|2x)
            echo "2x"
            ;;
        4|4x)
            echo "4x"
            ;;
        8|8x)
            echo "8x"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_budget_list() {
    local token
    local item
    local normalized
    local want_1x=0
    local want_2x=0
    local want_4x=0
    local want_8x=0
    local budgets=()

    for token in "$@"; do
        token="${token//,/ }"
        for item in $token; do
            if ! normalized="$(normalize_budget "$item")"; then
                echo "ERROR: Unsupported budget '$item'. Use one of: 1x, 2x, 4x, 8x" >&2
                return 1
            fi

            case "$normalized" in
                1x) want_1x=1 ;;
                2x) want_2x=1 ;;
                4x) want_4x=1 ;;
                8x) want_8x=1 ;;
            esac
        done
    done

    (( want_1x )) && budgets+=("1x")
    (( want_2x )) && budgets+=("2x")
    (( want_4x )) && budgets+=("4x")
    (( want_8x )) && budgets+=("8x")

    if [[ ${#budgets[@]} -eq 0 ]]; then
        echo "ERROR: --budget/--budgets requires at least one value" >&2
        return 1
    fi

    echo "${budgets[*]}"
}

set_requested_budgets() {
    local normalized
    normalized="$(normalize_budget_list "$@")" || return 1
    REQUESTED_BUDGETS="$normalized"
}

get_main_selected_budgets() {
    echo "${REQUESTED_BUDGETS:-1x 2x 4x}"
}

get_budget_total_steps() {
    local model_size="$1"
    local budget
    budget="$(normalize_budget "$2")" || {
        echo "ERROR: Unsupported budget '$2' for model size '$model_size'" >&2
        return 1
    }

    case "$budget" in
        1x)
            echo "${ITERS_1X[$model_size]}"
            ;;
        2x)
            echo "${ITERS_2X[$model_size]}"
            ;;
        4x)
            echo "${ITERS_4X[$model_size]}"
            ;;
        8x)
            echo "${ITERS_8X[$model_size]}"
            ;;
    esac
}

parse_experiment_cli_args() {
    EXPERIMENT_SCRIPT_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --budget|--budgets)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 1
                fi
                set_requested_budgets "$2" || return 1
                shift 2
                ;;
            --budget=*|--budgets=*)
                set_requested_budgets "${1#*=}" || return 1
                shift
                ;;
            --lr-scheduler|--scheduler)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 1
                fi
                set_lr_scheduler "$2" || return 1
                shift 2
                ;;
            --lr-scheduler=*|--scheduler=*)
                set_lr_scheduler "${1#*=}" || return 1
                shift
                ;;
            --eval-every|--eval_every)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 1
                fi
                set_eval_every "$2" || return 1
                shift 2
                ;;
            --eval-every=*|--eval_every=*)
                set_eval_every "${1#*=}" || return 1
                shift
                ;;
            --tracker)
                TRACKER=1
                shift
                ;;
            *)
                EXPERIMENT_SCRIPT_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

set_lr_scheduler "$LR_SCHEDULER" || return 1 2>/dev/null || exit 1

print_sweep_cli_usage() {
    local script_name="$1"
    local model_size="$2"
    local default_lrs="${LR_RANGE[$model_size]}"
    cat <<EOF
Usage:
  bash ${script_name}
  bash ${script_name} muon adamw
  bash ${script_name} --optimizers muon adamw
  bash ${script_name} --optimizers muon adamw --lr-range 1e-3 2e-3
  bash ${script_name} --optimizers=muon,adamw --lrs=1e-3,2e-3
  bash ${script_name} --lr-scheduler wsd-c
  bash ${script_name} --eval-every 500
  bash ${script_name} --tracker muon

Defaults:
  Optimizers: ${ALL_OPTIMIZERS}
  LR range (${model_size}): ${default_lrs}
  LR scheduler: ${LR_SCHEDULER}
  Eval every: ${EVAL_EVERY}
  Tracker: ${TRACKER}
EOF
}

parse_sweep_cli_args() {
    local model_size="$1"
    shift

    parse_experiment_cli_args "$@" || return 1
    set -- "${EXPERIMENT_SCRIPT_ARGS[@]}"

    local default_lrs="${LR_RANGE[$model_size]}"
    local positional_optimizers=()
    local optimizer_flag_used=0

    SWEEP_OPTIMIZERS="$ALL_OPTIMIZERS"
    SWEEP_LR_VALUES="$default_lrs"
    SWEEP_SHOW_HELP=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                SWEEP_SHOW_HELP=1
                shift
                ;;
            --optimizers)
                optimizer_flag_used=1
                shift
                local values=()
                while [[ $# -gt 0 && "$1" != --* ]]; do
                    values+=("$1")
                    shift
                done
                if [[ ${#values[@]} -eq 0 ]]; then
                    echo "ERROR: --optimizers requires at least one optimizer" >&2
                    return 1
                fi
                SWEEP_OPTIMIZERS="$(normalize_sweep_list "${values[@]}")"
                ;;
            --optimizers=*)
                optimizer_flag_used=1
                SWEEP_OPTIMIZERS="$(normalize_sweep_list "${1#*=}")"
                if [[ -z "$SWEEP_OPTIMIZERS" ]]; then
                    echo "ERROR: --optimizers requires at least one optimizer" >&2
                    return 1
                fi
                shift
                ;;
            --lrs|--lr-range)
                shift
                local values=()
                while [[ $# -gt 0 && "$1" != --* ]]; do
                    values+=("$1")
                    shift
                done
                if [[ ${#values[@]} -eq 0 ]]; then
                    echo "ERROR: --lr-range/--lrs requires at least one LR value" >&2
                    return 1
                fi
                SWEEP_LR_VALUES="$(normalize_sweep_list "${values[@]}")"
                ;;
            --lrs=*|--lr-range=*)
                SWEEP_LR_VALUES="$(normalize_sweep_list "${1#*=}")"
                if [[ -z "$SWEEP_LR_VALUES" ]]; then
                    echo "ERROR: --lr-range/--lrs requires at least one LR value" >&2
                    return 1
                fi
                shift
                ;;
            --)
                shift
                while [[ $# -gt 0 ]]; do
                    positional_optimizers+=("$1")
                    shift
                done
                ;;
            -*)
                echo "ERROR: Unknown option '$1'" >&2
                return 1
                ;;
            *)
                positional_optimizers+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#positional_optimizers[@]} -gt 0 ]]; then
        if (( optimizer_flag_used )); then
            echo "ERROR: Do not mix positional optimizers with --optimizers" >&2
            return 1
        fi
        SWEEP_OPTIMIZERS="$(normalize_sweep_list "${positional_optimizers[@]}")"
    fi
}

get_device_group() {
    local slot="$1"
    local start=$(( slot * NUM_NPUS_PER_JOB ))

    if (( start + NUM_NPUS_PER_JOB > TOTAL_AVAILABLE_NPUS )); then
        echo "ERROR: Slot $slot exceeds available device groups" >&2
        return 1
    fi

    local devices=("${AVAILABLE_NPUS[@]:start:NUM_NPUS_PER_JOB}")
    local IFS=,
    echo "${devices[*]}"
}

# Format LR for naming (e.g., 5e-4 -> 5e-4, 0.003 -> 3e-3)
format_lr() {
    local lr="$1"
    echo "$lr" | sed 's/\.//'
}

# Generate run name for sweep
# Usage: gen_sweep_name <optimizer> <lr> [gamma] [fold_level] [seed]
gen_sweep_name() {
    local opt="$1"
    local lr="$2"
    local gamma="${3:-}"
    local fold="${4:-}"
    local seed="${5:-$DEFAULT_SEED}"
    local scheduler_tag=""

    local name="1x-sweep-${opt}-lr${lr}"
    [[ -n "$gamma" ]] && name="${name}-g$(echo $gamma | sed 's/\.//')"
    [[ -n "$fold" ]] && name="${name}-f${fold}"
    scheduler_tag="$(lr_scheduler_tag || true)"
    [[ -n "$scheduler_tag" ]] && name="${name}-${scheduler_tag}"
    name="${name}-s${seed}"
    echo "$name"
}

# Generate run name for main training
# Usage: gen_main_name <budget> <optimizer> <lr> [gamma] [fold_level] [seed]
gen_main_name() {
    local budget="$1"
    local opt="$2"
    local lr="$3"
    local gamma="${4:-}"
    local fold="${5:-}"
    local seed="${6:-$DEFAULT_SEED}"
    local scheduler_tag=""

    local name="${budget}-${opt}-lr${lr}"
    [[ -n "$gamma" ]] && name="${name}-g$(echo $gamma | sed 's/\.//')"
    [[ -n "$fold" ]] && name="${name}-f${fold}"
    scheduler_tag="$(lr_scheduler_tag || true)"
    [[ -n "$scheduler_tag" ]] && name="${name}-${scheduler_tag}"
    name="${name}-s${seed}"
    echo "$name"
}

# Generate checkpoint directory path
# Usage: gen_ckpt_dir <model_size> <optimizer> <lr> [gamma] [fold_level] [seed]
gen_ckpt_dir() {
    local size="$1"
    local opt="$2"
    local lr="$3"
    local gamma="${4:-}"
    local fold="${5:-}"
    local seed="${6:-$DEFAULT_SEED}"
    local scheduler_tag=""

    local subdir="lr${lr}"
    [[ -n "$gamma" ]] && subdir="${subdir}_g$(echo $gamma | sed 's/\.//')"
    [[ -n "$fold" ]] && subdir="${subdir}_f${fold}"
    scheduler_tag="$(lr_scheduler_tag || true)"
    [[ -n "$scheduler_tag" ]] && subdir="${subdir}_${scheduler_tag}"
    subdir="${subdir}_s${seed}"

    echo "${CKPT_BASE}/${size}/${opt}/${subdir}"
}

# PID of the most recently launched training job.
RUN_TRAINING_PID=""

# Run a single training job
# Usage: run_training <device_group> <model_size> <optimizer> <lr> <run_name> \
#          <num_steps> <warmup> <decay_steps> <save_steps> <ckpt_dir> \
#          [--continue_from <path>] [--gamma <val>] [--fold <val>] [--seed <val>] \
#          [--log_group <group>]
# Sets RUN_TRAINING_PID in the current shell so callers can wait on it.
run_training() {
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

    # Parse optional args
    local continue_from=""
    local gamma=""
    local fold=""
    local log_group="misc"
    local seed="$DEFAULT_SEED"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --continue_from) continue_from="$2"; shift 2 ;;
            --gamma) gamma="$2"; shift 2 ;;
            --fold) fold="$2"; shift 2 ;;
            --log_group) log_group="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; shift ;;
        esac
    done

    local config="${MODEL_CONFIG[$model_size]}"
    local batch="${MICRO_BATCH_SIZE[$model_size]}"
    local total_batch="${TOTAL_BATCH_SIZE[$model_size]}"
    local project="${SWANLAB_PROJECT[$model_size]}"

    # Group logs by optimizer and add a launch timestamp to avoid overwrites.
    local log_dir="${LOG_BASE}/${log_group}/${optimizer}"
    local log_ts
    log_ts="$(date '+%Y%m%d-%H%M%S')"
    local log_file="${log_dir}/${log_ts}-${model_size}-${run_name}-pid$$.log"
    mkdir -p "$log_dir" "$ckpt_dir"

    echo "  [NPUs $device_group] $optimizer lr=$lr scheduler=$LR_SCHEDULER name=$run_name steps=$num_steps decay=$decay_steps"

    local -a adamw_betas
    read -r -a adamw_betas <<< "$ADAMW_BETAS"

    local -a train_args
    train_args=(
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
    )
    if [[ -n "$EVAL_BATCH_SIZE" ]]; then
        train_args+=(--eval_batch_size "$EVAL_BATCH_SIZE")
    fi
    if [[ "$CACHE_EVAL_DATA" == "1" ]]; then
        train_args+=(--cache_eval_data)
    fi
    if [[ "$ENABLE_ACTIVATION_CHECKPOINTING" == "1" ]]; then
        train_args+=(--activation_checkpointing)
    fi
    if (( NUM_NPUS_PER_JOB == 1 )); then
        train_args+=(--single_gpu)
    fi
    train_args+=(
        --optimizer "$optimizer"
        --scheduler "$LR_SCHEDULER"
    )
    if lr_scheduler_is_wsd; then
        train_args+=(--wsd_decay_steps "$decay_steps")
    fi
    case "$REPORT_TO" in
        wandb)
            train_args+=(--report_to wandb --wandb_proj "$project")
            ;;
        swanlab)
            train_args+=(--report_to swanlab --swanlab_proj "$project")
            ;;
        none)
            train_args+=(--report_to none)
            ;;
    esac
    train_args+=(
        --name "$run_name"
        --save_dir "${ckpt_dir}/${run_name}"
        --beta1 "$MUON_MOMENTUM"
        --betas "${adamw_betas[@]}"
    )
    if [[ "$TRACKER" == "1" ]]; then
        train_args+=(--tracker)
    fi

    # Optional checkpoint dir and save steps
    if [[ -n "$save_steps" && "$save_steps" != "none" ]]; then
        train_args+=(--save_steps "$save_steps" --checkpoint_dir "${ckpt_dir}/resume")
    fi

    # Optional continue_from
    if [[ -n "$continue_from" ]]; then
        train_args+=(--continue_from "$continue_from")
    fi

    # Optional gamma for mars/muon-mvr1
    if [[ -n "$gamma" ]]; then
        train_args+=(--mars_gamma "$gamma")
    fi

    # Optional fold_level for foam
    if [[ -n "$fold" ]]; then
        train_args+=(--fold_level "$fold")
    fi

    # Launch in background
    (
        set -euo pipefail
        validate_project_layout "$PROJECT_ROOT"
        run_launch_setup_cmd "$TRAIN_LAUNCH_PRE_CMD"
        cd "$PROJECT_ROOT"
        env \
            ASCEND_RT_VISIBLE_DEVICES="$device_group" \
            TOKENIZERS_PARALLELISM=false \
            "$TORCHRUN_BIN" \
            --standalone \
            --nproc_per_node "$NUM_NPUS_PER_JOB" \
            "${train_args[@]}"
    ) > "$log_file" 2>&1 &
    RUN_TRAINING_PID=$!
    echo "    PID: ${RUN_TRAINING_PID} | Log: ${log_file}"
}

# Wait for a set of PIDs
wait_for_pids() {
    local pids=("$@")
    echo "Waiting for ${#pids[@]} jobs to complete..."
    for pid in "${pids[@]}"; do
        if wait "$pid" 2>/dev/null; then
            echo "  PID $pid completed successfully"
        else
            local status=$?
            echo "  PID $pid failed with status $status"
        fi
    done
    echo "All jobs completed."
}

echo "[config.sh] Loaded 2026compare configuration."
echo "  Project root: ${PROJECT_ROOT}"
echo "  Checkpoint base: ${CKPT_BASE}"
echo "  Log base: ${LOG_BASE}"
echo "  Visible NPUs: ${VISIBLE_NPUS}"
echo "  NPUs per job: ${NUM_NPUS_PER_JOB}"
echo "  Max concurrent jobs: ${MAX_CONCURRENT_JOBS}"
echo "  Activation checkpointing: ${ENABLE_ACTIVATION_CHECKPOINTING}"
echo "  Tracking backend: ${REPORT_TO}"
echo "  Torchrun bin: ${TORCHRUN_BIN}"
if [[ -n "$TRAIN_LAUNCH_PRE_CMD" ]]; then
    echo "  Launch pre-command: configured"
fi
