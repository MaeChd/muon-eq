#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/optimizers_compare/train-baselines.sh
#   ONLY_EXPERIMENTS=muon,muoneq-rc_phase DRY_RUN=1 bash scripts/optimizers_compare/train-baselines.sh
#   ONLY_EXPERIMENTS=muonplus,muoneq-rc DRY_RUN=1 bash scripts/optimizers_compare/train-baselines.sh
#   bash scripts/optimizers_compare/train-baselines.sh --lr-range 5e-4,1e-3,2e-3,5e-3
#   bash scripts/optimizers_compare/train-baselines.sh --lr-range 5e-4 1e-3 2e-3 5e-3
#   bash scripts/optimizers_compare/train-baselines.sh --include-muon-splitlr-sweep --muon-factor-range 5e-4,1e-3,2e-3,5e-3
#   EXPERIMENT_SPECS="muoneq-r@ns=4;muon@zp=spc@ns=4;muon@zp=spc@ns=5" bash scripts/optimizers_compare/train-baselines.sh
#   EXPERIMENT_SPECS="adamuon@lr=2e-3;fismo@lr=1e-3;muonplus@lr=5e-4@ns=4" bash scripts/optimizers_compare/train-baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_PY="${REPO_ROOT}/src/main.py"

if [[ ! -f "${MAIN_PY}" ]]; then
    echo "main.py not found: ${MAIN_PY}" >&2
    exit 1
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "NPROC_PER_NODE must be >= 1, got ${NPROC_PER_NODE}" >&2
    exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-gpt2-base-fineweb10b-optcmp}"
RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-${REPO_ROOT}/exps/optimizer_comparison_fineweb10b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/optimizer_comparison_fineweb10b}"
DATASET_NAME="${DATASET_NAME:-fineweb10b}"
DATASET_LABEL="${DATASET_LABEL:-fineweb-10b}"
DATASETS_DIR="${DATASETS_DIR:-/path/to/fineweb10b}"

MODEL="${MODEL:-base}"
N_LAYER="${N_LAYER:-12}"
N_EMBD="${N_EMBD:-768}"
N_HEAD="${N_HEAD:-12}"
DROPOUT="${DROPOUT:-0.05}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACC_STEPS="${ACC_STEPS:-2}"
DTYPE="${DTYPE:-bfloat16}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SCHEDULER="${SCHEDULER:-cos_inf}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
MOMENTUM="${MOMENTUM:-0.95}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"

LR_RANGE_RAW="${LR_RANGE:-5e-4,1e-3,2e-3,5e-3}"
MUON_FACTOR_RANGE_RAW="${MUON_FACTOR_RANGE:-5e-4,1e-3,2e-3,5e-3}"
MUON_ADAMW_BACKUP_LR="${MUON_ADAMW_BACKUP_LR:-1e-3}"
MUONEQ_PHASE_SWITCH="${MUONEQ_PHASE_SWITCH:-${MUONRC_PHASE_SWITCH:-1000}}"
ZEROPOWER_MODE="${ZEROPOWER_MODE:-native}"
INCLUDE_MUON_SPLITLR_SWEEP="${INCLUDE_MUON_SPLITLR_SWEEP:-0}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
ONLY_EXPERIMENTS="${ONLY_EXPERIMENTS:-}"
EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-}"
EXTRA_COMMON_ARGS="${EXTRA_COMMON_ARGS:-}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-3}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/optimizers_compare/train-baselines.sh [--lr-range 5e-4,1e-3,2e-3,5e-3]
  bash scripts/optimizers_compare/train-baselines.sh [--lr-range 5e-4 1e-3 2e-3 5e-3]
  bash scripts/optimizers_compare/train-baselines.sh [--include-muon-splitlr-sweep --muon-factor-range 5e-4,1e-3]
  EXPERIMENT_SPECS="muoneq-r@ns=4;muon@zp=spc@ns=4;muon@zp=spc@ns=5" \
    bash scripts/optimizers_compare/train-baselines.sh
  EXPERIMENT_SPECS="adamuon@lr=2e-3;fismo@lr=1e-3;muonplus@lr=5e-4@ns=4" \
    bash scripts/optimizers_compare/train-baselines.sh

Options:
  --lr-range                    LR sweep values. Supports comma-separated or repeated values.
  --include-muon-splitlr-sweep  Add the legacy split-LR Muon sweep that uses --muon_lr_factor.
  --muon-factor-range           Sweep values for legacy split-LR Muon's --muon_lr_factor.
  EXPERIMENT_SPECS              Per-optimizer overrides support @lr=..., @ns=..., @zp=..., etc.
  --help                        Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr-range)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--lr-range requires at least one value" >&2
                exit 1
            fi

            lr_range_parts=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                lr_range_parts+=("$1")
                shift
            done
            LR_RANGE_RAW="${lr_range_parts[*]}"
            ;;
        --lr-range=*)
            LR_RANGE_RAW="${1#*=}"
            shift
            ;;
        --include-muon-splitlr-sweep)
            INCLUDE_MUON_SPLITLR_SWEEP=1
            shift
            ;;
        --muon-factor-range)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--muon-factor-range requires at least one value" >&2
                exit 1
            fi

            muon_factor_parts=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                muon_factor_parts+=("$1")
                shift
            done
            MUON_FACTOR_RANGE_RAW="${muon_factor_parts[*]}"
            INCLUDE_MUON_SPLITLR_SWEEP=1
            ;;
        --muon-factor-range=*)
            MUON_FACTOR_RANGE_RAW="${1#*=}"
            INCLUDE_MUON_SPLITLR_SWEEP=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}" "${RESULTS_BASE_FOLDER}"

declare -a EXPERIMENTS=()
declare -a LR_VALUES=()
declare -a MUON_FACTOR_VALUES=()

register_experiment() {
    local exp_id="$1"
    local opt_name="$2"
    local lr="$3"
    local run_tag="$4"
    local extra_args="${5:-}"
    local zeropower_mode="${6:-${ZEROPOWER_MODE}}"
    local ns_steps="${7:-${MUON_NS_STEPS}}"

    EXPERIMENTS+=("${exp_id}|${opt_name}|${lr}|${run_tag}|${extra_args}|${zeropower_mode}|${ns_steps}")
}

parse_lr_range() {
    local normalized="${LR_RANGE_RAW//,/ }"
    read -r -a LR_VALUES <<< "${normalized}"
    if [[ ${#LR_VALUES[@]} -eq 0 ]]; then
        echo "No LR values parsed from --lr-range='${LR_RANGE_RAW}'" >&2
        exit 1
    fi
}

parse_muon_factor_range() {
    local normalized="${MUON_FACTOR_RANGE_RAW//,/ }"
    read -r -a MUON_FACTOR_VALUES <<< "${normalized}"
    if [[ ${#MUON_FACTOR_VALUES[@]} -eq 0 ]]; then
        echo "No factor values parsed from --muon-factor-range='${MUON_FACTOR_RANGE_RAW}'" >&2
        exit 1
    fi
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf "%s" "${value}"
}

default_extra_args_for_opt() {
    local opt_name="$1"
    local phase="${2:-__unset__}"
    local factor="${3:-}"

    case "${opt_name}" in
        adamw)
            printf ""
            ;;
        muon|muonplus|muoneq-r|muoneq-c|adamuon)
            printf "%s" "--nesterov True"
            ;;
        muoneq-rc)
            if [[ "${phase}" == "__unset__" ]]; then
                printf "%s" "--nesterov True"
            else
                printf "%s" "--nesterov True --muoneq_phase ${phase}"
            fi
            ;;
        mousse)
            printf ""
            ;;
        fismo)
            printf ""
            ;;
        muon-splitlr)
            if [[ -z "${factor}" ]]; then
                echo "muon-splitlr spec requires factor=..." >&2
                exit 1
            fi
            printf "%s" "--nesterov True --muon_lr_factor ${factor}"
            ;;
        *)
            echo "Unsupported optimizer in EXPERIMENT_SPECS: ${opt_name}" >&2
            exit 1
            ;;
    esac
}

build_run_tag_base() {
    local opt_name="$1"
    local phase="${2:-__unset__}"
    local factor="${3:-}"

    case "${opt_name}" in
        muoneq-rc)
            if [[ "${phase}" == "__unset__" ]]; then
                printf "muoneq-rc"
            elif [[ "${phase}" == "None" ]]; then
                printf "muoneq-rc_phaseNone"
            else
                printf "muoneq-rc_phase%s" "${phase}"
            fi
            ;;
        muon-splitlr)
            if [[ -z "${factor}" ]]; then
                echo "muon-splitlr spec requires factor=..." >&2
                exit 1
            fi
            printf "muonsplit_adamlr%s_mulr%s" "${MUON_ADAMW_BACKUP_LR}" "${factor}"
            ;;
        *)
            printf "%s" "${opt_name}"
            ;;
    esac
}

register_custom_experiment_spec() {
    local raw_spec="$1"
    local spec
    local -a spec_parts
    local part
    local key
    local value
    local opt_name
    local zeropower_mode="${ZEROPOWER_MODE}"
    local ns_steps="${MUON_NS_STEPS}"
    local phase="__unset__"
    local factor=""
    local tag=""
    local lr_override=""
    local has_zp_override=0
    local has_ns_override=0
    local has_factor_override=0
    local has_lr_override=0
    local run_tag_base
    local exp_base
    local extra_args
    local lr
    local normalized_lrs
    local -a selected_lrs

    spec="$(trim_whitespace "${raw_spec}")"
    if [[ -z "${spec}" ]]; then
        return
    fi

    IFS='@' read -r -a spec_parts <<< "${spec}"
    opt_name="$(trim_whitespace "${spec_parts[0]}")"
    if [[ -z "${opt_name}" ]]; then
        echo "Invalid EXPERIMENT_SPECS entry: '${raw_spec}'" >&2
        exit 1
    fi

    for part in "${spec_parts[@]:1}"; do
        part="$(trim_whitespace "${part}")"
        if [[ -z "${part}" ]]; then
            continue
        fi
        if [[ "${part}" != *=* ]]; then
            echo "Invalid EXPERIMENT_SPECS token '${part}' in '${raw_spec}'. Expected key=value." >&2
            exit 1
        fi

        key="$(trim_whitespace "${part%%=*}")"
        value="$(trim_whitespace "${part#*=}")"
        case "${key}" in
            zp|zeropower|zeropower_mode)
                zeropower_mode="${value}"
                has_zp_override=1
                ;;
            ns|muon_ns_steps)
                ns_steps="${value}"
                has_ns_override=1
                ;;
            phase|muoneq_phase)
                phase="${value}"
                ;;
            factor|muon_lr_factor)
                factor="${value}"
                has_factor_override=1
                ;;
            lr|learning_rate)
                lr_override="${value}"
                has_lr_override=1
                ;;
            tag|label)
                tag="${value}"
                ;;
            *)
                echo "Unsupported EXPERIMENT_SPECS key '${key}' in '${raw_spec}'" >&2
                exit 1
                ;;
        esac
    done

    run_tag_base="$(build_run_tag_base "${opt_name}" "${phase}" "${factor}")"
    exp_base="${run_tag_base}"
    if [[ -n "${tag}" ]]; then
        exp_base="${tag}"
    else
        if [[ "${has_zp_override}" == "1" ]]; then
            exp_base="${exp_base}_zp_${zeropower_mode}"
        fi
        if [[ "${has_ns_override}" == "1" ]]; then
            exp_base="${exp_base}_ns${ns_steps}"
        fi
        if [[ "${has_factor_override}" == "1" ]]; then
            exp_base="${exp_base}_mulr${factor}"
        fi
    fi

    extra_args="$(default_extra_args_for_opt "${opt_name}" "${phase}" "${factor}")"

    if [[ "${has_lr_override}" == "1" ]]; then
        normalized_lrs="${lr_override//,/ }"
        read -r -a selected_lrs <<< "${normalized_lrs}"
        if [[ ${#selected_lrs[@]} -eq 0 ]]; then
            echo "No LR values parsed from lr='${lr_override}' in '${raw_spec}'" >&2
            exit 1
        fi
    else
        selected_lrs=("${LR_VALUES[@]}")
    fi

    for lr in "${selected_lrs[@]}"; do
        register_experiment \
            "${exp_base}_lr${lr}" \
            "${opt_name}" \
            "${lr}" \
            "${run_tag_base}_lr${lr}" \
            "${extra_args}" \
            "${zeropower_mode}" \
            "${ns_steps}"
    done
}

build_experiment_matrix() {
    local lr
    local factor
    local spec

    if [[ -n "${EXPERIMENT_SPECS}" ]]; then
        IFS=';' read -r -a CUSTOM_SPECS_ARRAY <<< "${EXPERIMENT_SPECS}"
        for spec in "${CUSTOM_SPECS_ARRAY[@]}"; do
            register_custom_experiment_spec "${spec}"
        done
        return
    fi

    for lr in "${LR_VALUES[@]}"; do
        register_experiment \
            "adamw_lr${lr}" \
            "adamw" \
            "${lr}" \
            "adamw_lr${lr}"

        register_experiment \
            "muon_lr${lr}" \
            "muon" \
            "${lr}" \
            "muon_lr${lr}" \
            "--nesterov True"

        register_experiment \
            "muonplus_lr${lr}" \
            "muonplus" \
            "${lr}" \
            "muonplus_lr${lr}" \
            "--nesterov True"

        register_experiment \
            "adamuon_lr${lr}" \
            "adamuon" \
            "${lr}" \
            "adamuon_lr${lr}" \
            "--nesterov True"

        register_experiment \
            "mousse_lr${lr}" \
            "mousse" \
            "${lr}" \
            "mousse_lr${lr}"

        register_experiment \
            "fismo_lr${lr}" \
            "fismo" \
            "${lr}" \
            "fismo_lr${lr}"

        register_experiment \
            "muoneq-rc_phaseNone_lr${lr}" \
            "muoneq-rc" \
            "${lr}" \
            "muoneq-rc_phaseNone_lr${lr}" \
            "--nesterov True --muoneq_phase None"

        register_experiment \
            "muoneq-rc_phase${MUONEQ_PHASE_SWITCH}_lr${lr}" \
            "muoneq-rc" \
            "${lr}" \
            "muoneq-rc_phase${MUONEQ_PHASE_SWITCH}_lr${lr}" \
            "--nesterov True --muoneq_phase ${MUONEQ_PHASE_SWITCH}"

        register_experiment \
            "muoneq-r_lr${lr}" \
            "muoneq-r" \
            "${lr}" \
            "muoneq-r_lr${lr}" \
            "--nesterov True"

        register_experiment \
            "muoneq-c_lr${lr}" \
            "muoneq-c" \
            "${lr}" \
            "muoneq-c_lr${lr}" \
            "--nesterov True"
    done

    if [[ "${INCLUDE_MUON_SPLITLR_SWEEP}" == "1" ]]; then
        for factor in "${MUON_FACTOR_VALUES[@]}"; do
            register_experiment \
                "muonsplit_mulr${factor}" \
                "muon-splitlr" \
                "${MUON_ADAMW_BACKUP_LR}" \
                "muonsplit_adamlr${MUON_ADAMW_BACKUP_LR}_mulr${factor}" \
                "--nesterov True --muon_lr_factor ${factor}"
        done
    fi
}

parse_lr_range
parse_muon_factor_range
build_experiment_matrix

COMMON_ARGS=(
    --config_format base
    --wandb
    --wandb_project "${WANDB_PROJECT}"
    --results_base_folder "${RESULTS_BASE_FOLDER}"
    --model "${MODEL}"
    --dataset "${DATASET_NAME}"
    --datasets_dir "${DATASETS_DIR}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --n_layer "${N_LAYER}"
    --n_embd "${N_EMBD}"
    --n_head "${N_HEAD}"
    --dropout "${DROPOUT}"
    --batch_size "${BATCH_SIZE}"
    --acc_steps "${ACC_STEPS}"
    --sequence_length "${SEQUENCE_LENGTH}"
    --beta1 "${BETA1}"
    --beta2 "${BETA2}"
    --momentum "${MOMENTUM}"
    --weight_decay "${WEIGHT_DECAY}"
    --iterations "${ITERATIONS}"
    --warmup_steps "${WARMUP_STEPS}"
    --eval_interval "${EVAL_INTERVAL}"
    --log_interval "${LOG_INTERVAL}"
    --grad_clip "${GRAD_CLIP}"
    --scheduler "${SCHEDULER}"
    --distributed_backend nccl
)

if [[ -n "${EXTRA_COMMON_ARGS}" ]]; then
    read -r -a EXTRA_COMMON_PARTS <<< "${EXTRA_COMMON_ARGS}"
    COMMON_ARGS+=("${EXTRA_COMMON_PARTS[@]}")
fi

should_run_experiment() {
    local exp_id="$1"

    if [[ -z "${ONLY_EXPERIMENTS}" ]]; then
        return 0
    fi

    local item
    local trimmed_item
    IFS=',' read -r -a ONLY_ARRAY <<< "${ONLY_EXPERIMENTS}"
    for item in "${ONLY_ARRAY[@]}"; do
        trimmed_item="${item#"${item%%[![:space:]]*}"}"
        trimmed_item="${trimmed_item%"${trimmed_item##*[![:space:]]}"}"
        if [[ -z "${trimmed_item}" ]]; then
            continue
        fi

        if [[ "${exp_id}" == "${trimmed_item}" ]]; then
            return 0
        fi
        if [[ "${exp_id}" == "${trimmed_item}_"* ]]; then
            return 0
        fi

        local suffix="${exp_id#"$trimmed_item"}"
        if [[ "${suffix}" != "${exp_id}" && "${suffix}" == [0-9]* ]]; then
            return 0
        fi
    done
    return 1
}

build_experiment_name() {
    local run_tag="$1"
    local opt_name="$2"
    local zeropower_mode="$3"
    local ns_steps="$4"
    local -a name_parts=(
        "${run_tag}"
    )

    case "${opt_name}" in
        adamw)
            name_parts+=("beta1" "${BETA1}" "beta2" "${BETA2}")
            ;;
        muon|muonplus|adamuon|mousse|fismo|muoneq-rc|muoneq-r|muoneq-c|muon-splitlr)
            name_parts+=("mom" "${MOMENTUM}")
            ;;
    esac

    name_parts+=(
        "zp"
        "${zeropower_mode}"
        "ns${ns_steps}"
        "b${BATCH_SIZE}"
        "a${ACC_STEPS}"
        "g${NPROC_PER_NODE}"
        "s${SEED}"
    )

    if [[ -n "${RUN_NAME_PREFIX}" ]]; then
        name_parts=("${RUN_NAME_PREFIX}" "${name_parts[@]}")
    fi

    local IFS=_
    printf "%s" "${name_parts[*]}"
}

run_experiment() {
    local record="$1"
    local exp_id
    local opt_name
    local lr
    local run_tag
    local extra_args
    local zeropower_mode
    local ns_steps
    local experiment_name
    local safe_name
    local log_file
    local ts
    local cmd
    local extra_parts

    IFS='|' read -r exp_id opt_name lr run_tag extra_args zeropower_mode ns_steps <<< "${record}"

    experiment_name="$(build_experiment_name "${run_tag}" "${opt_name}" "${zeropower_mode}" "${ns_steps}")"
    safe_name="${experiment_name//\//_}"
    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/${safe_name}_${ts}.log"

    cmd=(
        torchrun
        --nproc_per_node="${NPROC_PER_NODE}"
        "${MAIN_PY}"
        "${COMMON_ARGS[@]}"
        --opt "${opt_name}"
        --lr "${lr}"
        --muon_ns_steps "${ns_steps}"
        --zeropower_mode "${zeropower_mode}"
        --experiment_name "${experiment_name}"
    )

    if [[ -n "${extra_args}" ]]; then
        read -r -a extra_parts <<< "${extra_args}"
        cmd+=("${extra_parts[@]}")
    fi

    echo "Launching ${experiment_name}"
    echo "  opt=${opt_name} lr=${lr} gpus=${GPU_IDS} zp=${zeropower_mode} ns=${ns_steps}"
    if [[ -n "${extra_args}" ]]; then
        echo "  extra_args=${extra_args}"
    fi
    printf "  cmd:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    echo "  log=${log_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo ""
        return
    fi

    (
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
        export TOKENIZERS_PARALLELISM=false
        "${cmd[@]}"
    ) 2>&1 \
        | awk -v ename="${experiment_name}" '{ print strftime("[%Y-%m-%d %H:%M:%S]"), "[" ename "]", $0; fflush(); }' \
        | tee "${log_file}"

    echo ""
}

echo "Optimizer comparison on ${DATASET_LABEL}"
echo "  repo=${REPO_ROOT}"
echo "  dataset_key=${DATASET_NAME}"
echo "  datasets_dir=${DATASETS_DIR}"
echo "  gpus=${GPU_IDS} (nproc=${NPROC_PER_NODE})"
echo "  lr_range=${LR_VALUES[*]}"
echo "  muoneq_phase_switch=${MUONEQ_PHASE_SWITCH}"
echo "  default_zeropower_mode=${ZEROPOWER_MODE}"
echo "  default_muon_ns_steps=${MUON_NS_STEPS}"
if [[ -n "${EXPERIMENT_SPECS}" ]]; then
    echo "  experiment_specs=${EXPERIMENT_SPECS}"
fi
echo "  muon_splitlr_sweep=${INCLUDE_MUON_SPLITLR_SWEEP}"
if [[ "${INCLUDE_MUON_SPLITLR_SWEEP}" == "1" ]]; then
    echo "  muon_factor_range=${MUON_FACTOR_VALUES[*]}"
    echo "  muon_adamw_backup_lr=${MUON_ADAMW_BACKUP_LR}"
fi
echo "  wandb_project=${WANDB_PROJECT}"
echo "  results_dir=${RESULTS_BASE_FOLDER}"
echo "  log_dir=${LOG_DIR}"
echo ""

for record in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_id _ <<< "${record}"
    if ! should_run_experiment "${exp_id}"; then
        continue
    fi
    run_experiment "${record}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        sleep "${SLEEP_BETWEEN_RUNS}"
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run completed."
else
    echo "All experiments completed."
fi
