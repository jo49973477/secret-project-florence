#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# User config
# ============================================================

TRAIN_GPUS="${TRAIN_GPUS:-4,5}"
EVAL_GPU="${EVAL_GPU:-4}"

BASE_MODEL="nvidia/GR00T-N1.7-3B"

TRAIN_DATASET_PATH="/ssdg/spl_yeongyoo/univtac_gr00t"

# 지금 당장은 같은 dataset을 써도 diagnostic 용도로는 OK.
# 나중에 episode-level held-out dataset을 만들면 이 경로만 바꾸면 됨.
EVAL_DATASET_PATH="/ssdg/spl_yeongyoo/univtac_gr00t"

MODALITY_CONFIG="examples/UniVTAC/univtac_config.py"

# 평가할 trajectory IDs
# 필요하면 원하는 episode들로 바꾸기
EVAL_TRAJ_IDS=(0 1 2 3 4)

# 한 trajectory에서 평가할 최대 step
EVAL_STEPS=400

EXECUTION_HORIZON=16

RUN_NAME="${RUN_NAME:-univtac_rgb_joint_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="outputs/${RUN_NAME}"
LOG_DIR="${OUTPUT_DIR}/logs"


# ============================================================
# Training environment
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

export USE_WANDB=0
export NUM_GPUS=2

export GLOBAL_BATCH_SIZE=2
export DATALOADER_NUM_WORKERS=0

# 핵심:
# 2000 step까지 한 번 학습
# 매 500 step마다 checkpoint 저장
export MAX_STEPS=2000
export SAVE_STEPS=500


mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "UniVTAC GR00T baseline training"
echo "============================================================"
echo "Train GPUs      : ${TRAIN_GPUS}"
echo "Dataset         : ${TRAIN_DATASET_PATH}"
echo "Output          : ${OUTPUT_DIR}"
echo "Max steps       : ${MAX_STEPS}"
echo "Save interval   : ${SAVE_STEPS}"
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
  -- \
  --gradient-accumulation-steps 32 \
  2>&1 | tee "${TRAIN_LOG}"

# ============================================================
# 2. Evaluate checkpoints
# ============================================================

RESULT_CSV="${LOG_DIR}/metrics.csv"

echo "step,mse,mae" > "${RESULT_CSV}"

CHECKPOINTS=(500 1000 1500 2000)

echo
echo "============================================================"
echo "Open-loop evaluation"
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
    echo "------------------------------------------------------------"

    # inference는 GPU 한 장만 사용
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

    # GR00T open_loop_eval.py의 출력에서 평균값 추출
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
