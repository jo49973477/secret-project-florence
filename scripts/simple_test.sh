#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/checkpoint_resume.sh
source "${SCRIPT_DIR}/lib/checkpoint_resume.sh"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/home/yeongyoo/03_Dataset/07_UnivTac/univtac_gr00t}"
MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_WANDB="${USE_WANDB:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
export MAX_STEPS="${MAX_STEPS:-100}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_STEPS

checkpoint_resume_configure "${OUTPUT_DIR}" 5

echo "============================================================"
echo "UniVTAC GR00T simple training test"
echo "============================================================"
echo "DeepSpeed stage : ${DEEPSPEED_STAGE}"
checkpoint_resume_print_status "${OUTPUT_DIR}"
echo "============================================================"

uv run bash examples/finetune.sh \
  --deepspeed-stage "${DEEPSPEED_STAGE}" \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}"
