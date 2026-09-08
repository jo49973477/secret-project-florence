#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# User config
# ============================================================

# Physical GPUs shown in nvidia-smi
TRAIN_GPUS="${TRAIN_GPUS:-4,5}"
EVAL_GPU="${EVAL_GPU:-4}"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-/ssdg/spl_yeongyoo/univtac_gr00t}"

# 지금은 training dataset과 동일해도 baseline sanity check 용도로는 OK.
# 나중에 held-out episode dataset을 만들면 이 경로만 바꾸면 됨.
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/ssdg/spl_yeongyoo/univtac_gr00t}"

MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_config.py}"

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# examples/UniVTAC/univtac_config.py 에서 반드시:
#
# delta_indices=list(range(16))
#
# 으로 설정해둘 것.
#
# Model의 max action horizon=40은 그대로 둬도 됨.
# 실제 UniVTAC training horizon만 16으로 줄이는 것.
# ------------------------------------------------------------

# 평가할 trajectory IDs
EVAL_TRAJ_IDS=(0 1 2 3 4)

# 한 trajectory에서 평가할 최대 step
EVAL_STEPS="${EVAL_STEPS:-400}"

# 실제 실행/평가 horizon
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"

# effective batch:
#   GLOBAL_BATCH_SIZE 2 × GRAD_ACCUM_STEPS 16 = 32
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

RUN_NAME="${RUN_NAME:-univtac_rgb_joint_h16_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="outputs/${RUN_NAME}"
LOG_DIR="${OUTPUT_DIR}/logs"


# ============================================================
# Training environment
# ============================================================

# PyTorch CUDA allocator fragmentation mitigation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

export USE_WANDB=0
export NUM_GPUS=2

export GLOBAL_BATCH_SIZE
export DATALOADER_NUM_WORKERS
export MAX_STEPS
export SAVE_STEPS


# ============================================================
# Sanity checks
# ============================================================

if [[ ! -d "${TRAIN_DATASET_PATH}" ]]; then
    echo "[ERROR] Training dataset not found:"
    echo "        ${TRAIN_DATASET_PATH}"
    exit 1
fi

if [[ ! -d "${EVAL_DATASET_PATH}" ]]; then
    echo "[ERROR] Evaluation dataset not found:"
    echo "        ${EVAL_DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Modality config not found:"
    echo "        ${MODALITY_CONFIG}"
    exit 1
fi

mkdir -p "${LOG_DIR}"


# ============================================================
# Configuration summary
# ============================================================

EFFECTIVE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))

echo "============================================================"
echo "UniVTAC GR00T baseline training"
echo "============================================================"
echo "Train GPUs             : ${TRAIN_GPUS}"
echo "Eval GPU               : ${EVAL_GPU}"
echo "Train dataset          : ${TRAIN_DATASET_PATH}"
echo "Eval dataset           : ${EVAL_DATASET_PATH}"
echo "Modality config        : ${MODALITY_CONFIG}"
echo "Output                 : ${OUTPUT_DIR}"
echo
echo "Max steps              : ${MAX_STEPS}"
echo "Save interval          : ${SAVE_STEPS}"
echo "Global batch           : ${GLOBAL_BATCH_SIZE}"
echo "Gradient accumulation  : ${GRAD_ACCUM_STEPS}"
echo "Effective batch        : ${EFFECTIVE_BATCH_SIZE}"
echo "State dropout          : ${STATE_DROPOUT_PROB}"
echo "Execution horizon      : ${EXECUTION_HORIZON}"
echo "============================================================"


# ============================================================
# 1. Train
# ============================================================

TRAIN_LOG="${LOG_DIR}/train.log"

uv run bash examples/finetune.sh \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${TRAIN_DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-only-model \
    --state-dropout-prob "${STATE_DROPOUT_PROB}" \
    -- \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    2>&1 | tee "${TRAIN_LOG}"


# ============================================================
# 2. Evaluate checkpoints
# ============================================================

RESULT_CSV="${LOG_DIR}/metrics.csv"

echo "step,mse,mae" > "${RESULT_CSV}"

# SAVE_STEPS마다 checkpoint 평가:
# default = 1000 2000 3000 4000 5000
CHECKPOINTS=()

STEP="${SAVE_STEPS}"
while (( STEP <= MAX_STEPS )); do
    CHECKPOINTS+=("${STEP}")
    STEP=$((STEP + SAVE_STEPS))
done

echo
echo "============================================================"
echo "Open-loop evaluation"
echo "============================================================"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Eval GPU   : ${EVAL_GPU}"
echo "============================================================"


for STEP in "${CHECKPOINTS[@]}"; do

    CKPT="${OUTPUT_DIR}/checkpoint-${STEP}"
    EVAL_LOG="${LOG_DIR}/eval_${STEP}.log"

    if [[ ! -d "${CKPT}" ]]; then
        echo "[ERROR] Checkpoint not found: ${CKPT}"
        exit 1
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Evaluating checkpoint-${STEP}"
    echo "Physical GPU : ${EVAL_GPU}"
    echo "Logical GPU  : cuda:0"
    echo "------------------------------------------------------------"

    # CUDA_VISIBLE_DEVICES remaps physical EVAL_GPU -> logical cuda:0
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
    uv run python gr00t/eval/open_loop_eval.py \
        --dataset-path "${EVAL_DATASET_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path "${CKPT}" \
        --traj-ids "${EVAL_TRAJ_IDS[@]}" \
        --execution-horizon "${EXECUTION_HORIZON}" \
        --steps "${EVAL_STEPS}" \
        --modality-keys joint \
        2>&1 | tee "${EVAL_LOG}"

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    MSE=$(
        grep "Average MSE across all trajs:" "${EVAL_LOG}" \
        | tail -1 \
        | awk -F': ' '{print $NF}'
    )

    MAE=$(
        grep "Average MAE across all trajs:" "${EVAL_LOG}" \
        | tail -1 \
        | awk -F': ' '{print $NF}'
    )

    if [[ -z "${MSE}" || -z "${MAE}" ]]; then
        echo "[ERROR] Failed to parse evaluation result:"
        echo "        ${EVAL_LOG}"
        exit 1
    fi

    echo "${STEP},${MSE},${MAE}" >> "${RESULT_CSV}"

done


# ============================================================
# 3. Summary
# ============================================================

echo
echo "============================================================"
echo "RESULT"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
    column -s, -t "${RESULT_CSV}"
else
    cat "${RESULT_CSV}"
fi

echo
echo "Training log : ${TRAIN_LOG}"
echo "Metrics CSV  : ${RESULT_CSV}"
echo "Eval logs    : ${LOG_DIR}/eval_*.log"
echo "============================================================"