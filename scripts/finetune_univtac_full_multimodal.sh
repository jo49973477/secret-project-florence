#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# GPU
# ============================================================

TRAIN_GPUS="${TRAIN_GPUS:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29627}"

# ============================================================
# Model / Dataset
# ============================================================

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

DATASET_PATH="${DATASET_PATH:-/ssdg/spl_yeongyoo/univtac_full}"

MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_multimodal_config.py}"

# ============================================================
# Training
# ============================================================

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"

RUN_NAME="${RUN_NAME:-univtac_full_multimodal_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"

# ============================================================
# Environment
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

# ============================================================
# Sanity checks
# ============================================================

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset does not exist:"
    echo "        ${DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
    echo "[ERROR] Missing meta/info.json"
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/meta/modality.json" ]]; then
    echo "[ERROR] Missing meta/modality.json"
    exit 1
fi

if [[ ! -d "${DATASET_PATH}/pointclouds" ]]; then
    echo "[ERROR] Missing pointclouds/"
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Missing modality config:"
    echo "        ${MODALITY_CONFIG}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

EFFECTIVE_BATCH=$((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))

echo "============================================================"
echo "UniVTAC FULL Multimodal Fine-tuning"
echo "============================================================"
echo "GPUs                : ${TRAIN_GPUS}"
echo "Dataset             : ${DATASET_PATH}"
echo "Config              : ${MODALITY_CONFIG}"
echo "Output              : ${OUTPUT_DIR}"
echo "Max steps           : ${MAX_STEPS}"
echo "Global batch        : ${GLOBAL_BATCH_SIZE}"
echo "Grad accumulation   : ${GRAD_ACCUM_STEPS}"
echo "Effective batch     : ${EFFECTIVE_BATCH}"
echo "Episode sampling    : ${EPISODE_SAMPLING_RATE}"
echo
echo "Modalities:"
echo "  RGB          : ON"
echo "  Joint        : ON"
echo "  Tactile      : ON"
echo "  Point cloud  : ON"
echo
echo "Training:"
echo "  Qwen/VLM backbone : FROZEN"
echo "  Action head       : TRAIN"
echo "  Point encoder     : TRAIN"
echo "  Tactile encoder   : TRAIN"
echo "  MM adapter        : TRAIN"
echo "============================================================"

# ============================================================
# Train
# ============================================================

TRAIN_ARGS=(
    --base-model-path "${BASE_MODEL}"
    --dataset-path "${DATASET_PATH}"
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path "${MODALITY_CONFIG}"

    --output-dir "${OUTPUT_DIR}"

    --num-gpus "${NUM_GPUS}"

    --max-steps "${MAX_STEPS}"
    --save-steps "${SAVE_STEPS}"

    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"

    --learning-rate "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"

    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --episode-sampling-rate "${EPISODE_SAMPLING_RATE}"

    --dit-type multimodal_conditioned_dit

    --use-point-conditioning
    --use-tactile-conditioning

    --point-input-dim 3

    --no-tune-llm
    --no-tune-visual

    --tune-projector
    --tune-vlln
    --tune-diffusion-model

    --tune-point-encoder
    --tune-tactile-encoder
    --tune-multimodal-adapter

    --save-only-model
)

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
uv run torchrun \
    --nproc-per-node="${NUM_GPUS}" \
    --master-port="${MASTER_PORT}" \
    gr00t/experiment/launch_finetune.py \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/train.log"
