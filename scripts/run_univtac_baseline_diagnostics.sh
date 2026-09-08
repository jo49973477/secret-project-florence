#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# ============================================================
# User-configurable settings
# ============================================================

RUN_MODE="${RUN_MODE:-full}"
EVAL_GPU="${EVAL_GPU:-0}"

DATASET_PATH="${DATASET_PATH:-/ssdg/spl_yeongyoo/univtac_gr00t}"

# Intentionally no hard-coded old run.
# Example:
#   RUN_DIR=outputs/univtac_rgb_joint_h16_20260908_183010 \
#   EVAL_GPU=2 \
#   RUN_MODE=full \
#   bash scripts/run_univtac_baseline_diagnostics.sh
RUN_DIR="${RUN_DIR:-}"

EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"

EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
VERIFY_REPRODUCIBILITY="${VERIFY_REPRODUCIBILITY:-0}"

# RGB ablation conditions
CONDITIONS="${CONDITIONS:-normal black shuffled}"


# ============================================================
# Checkpoint / evaluation presets
# ============================================================

case "${RUN_MODE}" in
    smoke)
        CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
        TRAJ_IDS="${TRAJ_IDS:-0}"
        SEEDS="${SEEDS:-0}"
        EVAL_STEPS="${EVAL_STEPS:-32}"
        ;;
    full)
        CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000 2000 3000 4000 5000}"
        TRAJ_IDS="${TRAJ_IDS:-0 1 2 3 4}"
        SEEDS="${SEEDS:-0 1 2 3 4}"
        EVAL_STEPS="${EVAL_STEPS:-400}"
        ;;
    *)
        echo "ERROR: RUN_MODE must be 'smoke' or 'full'; got '${RUN_MODE}'." >&2
        exit 2
        ;;
esac


# ============================================================
# Basic validation
# ============================================================

if [[ -z "${RUN_DIR}" ]]; then
    echo "ERROR: RUN_DIR must be specified." >&2
    echo >&2
    echo "Example:" >&2
    echo "  RUN_DIR=outputs/univtac_rgb_joint_h16_20260908_183010 \\" >&2
    echo "  EVAL_GPU=2 RUN_MODE=full \\" >&2
    echo "  bash scripts/run_univtac_baseline_diagnostics.sh" >&2
    exit 2
fi

RESULT_DIR="${RESULT_DIR:-${RUN_DIR}/diagnostics}"

if [[ ! "${EVAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: EVAL_GPU must be a physical integer GPU index from nvidia-smi." >&2
    exit 2
fi

for integer_value in "${EXECUTION_HORIZON}" "${DENOISING_STEPS}" "${EVAL_STEPS}"; do
    if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: horizon, denoising steps, and evaluation steps must be positive integers." >&2
        exit 2
    fi
done

if [[ "${VERIFY_REPRODUCIBILITY}" != "0" && "${VERIFY_REPRODUCIBILITY}" != "1" ]]; then
    echo "ERROR: VERIFY_REPRODUCIBILITY must be 0 or 1." >&2
    exit 2
fi


# ============================================================
# Parse lists
# ============================================================

read -r -a checkpoint_steps_array <<< "${CHECKPOINT_STEPS}"
read -r -a trajectory_ids_array <<< "${TRAJ_IDS}"
read -r -a seeds_array <<< "${SEEDS}"
read -r -a conditions_array <<< "${CONDITIONS}"

if (( ${#checkpoint_steps_array[@]} == 0 || \
      ${#trajectory_ids_array[@]} == 0 || \
      ${#seeds_array[@]} == 0 || \
      ${#conditions_array[@]} == 0 )); then
    echo "ERROR: CHECKPOINT_STEPS, TRAJ_IDS, SEEDS, and CONDITIONS must each contain at least one value." >&2
    exit 2
fi

checkpoint_paths=()

for step in "${checkpoint_steps_array[@]}"; do
    if [[ ! "${step}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid checkpoint step '${step}'." >&2
        exit 2
    fi
    checkpoint_paths+=("${RUN_DIR}/checkpoint-${step}")
done

for trajectory_id in "${trajectory_ids_array[@]}"; do
    if [[ ! "${trajectory_id}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid trajectory ID '${trajectory_id}'." >&2
        exit 2
    fi
done

for seed in "${seeds_array[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed '${seed}'." >&2
        exit 2
    fi
done

normal_found=0
for condition in "${conditions_array[@]}"; do
    case "${condition}" in
        normal)
            normal_found=1
            ;;
        black|shuffled)
            ;;
        *)
            echo "ERROR: invalid RGB condition '${condition}'." >&2
            exit 2
            ;;
    esac
done

if [[ "${normal_found}" != "1" ]]; then
    echo "ERROR: CONDITIONS must include normal for degradation calculations." >&2
    exit 2
fi


# ============================================================
# Logging
# ============================================================

mkdir -p "${RESULT_DIR}/logs"

LOG_FILE="${RESULT_DIR}/logs/diagnostics_${RUN_MODE}_$(date +%Y%m%d_%H%M%S).log"

{
echo "============================================================"
echo "UniVTAC baseline diagnostics"
echo "============================================================"
printf "%-20s: %s\n" "Run mode" "${RUN_MODE}"
printf "%-20s: %s\n" "Physical GPU" "${EVAL_GPU}"
printf "%-20s: %s\n" "Logical device" "cuda:0"
printf "%-20s: %s\n" "Dataset" "${DATASET_PATH}"
printf "%-20s: %s\n" "Run" "${RUN_DIR}"
printf "%-20s: %s\n" "Checkpoints" "${CHECKPOINT_STEPS}"
printf "%-20s: %s\n" "Seeds" "${SEEDS}"
printf "%-20s: %s\n" "Conditions" "${CONDITIONS}"
printf "%-20s: %s\n" "Trajectories" "${TRAJ_IDS}"
printf "%-20s: %s\n" "Evaluation steps" "${EVAL_STEPS}"
printf "%-20s: %s\n" "Execution horizon" "${EXECUTION_HORIZON}"
printf "%-20s: %s\n" "Denoising steps" "${DENOISING_STEPS}"
printf "%-20s: %s\n" "Repro check" "${VERIFY_REPRODUCIBILITY}"
printf "%-20s: %s\n" "Results" "${RESULT_DIR}"
echo "============================================================"


# ============================================================
# Preflight
# ============================================================

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not available; this script is intended for the GPU server." >&2
    exit 1
fi

if ! nvidia-smi --query-gpu=index --format=csv,noheader \
    | tr -d ' ' \
    | grep -Fxq "${EVAL_GPU}"; then
    echo "ERROR: physical GPU ${EVAL_GPU} is not listed by nvidia-smi." >&2
    exit 1
fi

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: UniVTAC dataset does not exist: ${DATASET_PATH}" >&2
    exit 1
fi

for required_metadata in \
    meta/info.json \
    meta/episodes.jsonl \
    meta/tasks.jsonl \
    meta/modality.json \
    meta/stats.json \
    meta/relative_stats.json
do
    if [[ ! -f "${DATASET_PATH}/${required_metadata}" ]]; then
        echo "ERROR: dataset metadata is missing: ${DATASET_PATH}/${required_metadata}" >&2
        exit 1
    fi
done

for checkpoint_path in "${checkpoint_paths[@]}"; do
    if [[ ! -d "${checkpoint_path}" ]]; then
        echo "ERROR: requested checkpoint directory does not exist: ${checkpoint_path}" >&2
        echo "Set RUN_DIR or CHECKPOINT_STEPS to match the server checkout." >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
uv run python -c \
'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA preflight:", torch.cuda.get_device_name(0), "as cuda:0")'


# ============================================================
# Run diagnostics
# ============================================================

diagnostic_args=(
    --dataset-path "${DATASET_PATH}"
    --checkpoints "${checkpoint_paths[@]}"
    --trajectory-ids "${trajectory_ids_array[@]}"
    --seeds "${seeds_array[@]}"
    --conditions "${conditions_array[@]}"
    --execution-horizon "${EXECUTION_HORIZON}"
    --steps "${EVAL_STEPS}"
    --denoising-steps "${DENOISING_STEPS}"
    --result-dir "${RESULT_DIR}"
    --embodiment-tag "${EMBODIMENT_TAG}"
    --device cuda:0
)

if [[ "${VERIFY_REPRODUCIBILITY}" == "1" ]]; then
    diagnostic_args+=(--verify-reproducibility)
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
uv run python \
    examples/UniVTAC/evaluate_baseline_diagnostics.py \
    "${diagnostic_args[@]}"


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "RESULT"
echo "============================================================"

uv run python -c \
'import pandas as pd, sys; p=sys.argv[1]; print(pd.read_csv(p)[["checkpoint_step", "condition", "mse_mean", "mse_std", "mae_mean", "mae_std"]].to_string(index=False))' \
    "${RESULT_DIR}/summary_metrics.csv"

echo
echo "Persistence:"

uv run python -c \
'import pandas as pd, sys; p=sys.argv[1]; print(pd.read_csv(p)[["baseline", "mse", "mae"]].to_string(index=False))' \
    "${RESULT_DIR}/persistence_metrics.csv"

echo
echo "Results:"
echo "  ${RESULT_DIR}/raw_metrics.csv"
echo "  ${RESULT_DIR}/summary_metrics.csv"
echo "  ${RESULT_DIR}/persistence_metrics.csv"

if [[ "${VERIFY_REPRODUCIBILITY}" == "1" ]]; then
    echo "  ${RESULT_DIR}/reproducibility_check.csv"
fi

echo "  ${RESULT_DIR}/logs/"
echo "  ${LOG_FILE}"
echo "============================================================"

} 2>&1 | tee "${LOG_FILE}"
