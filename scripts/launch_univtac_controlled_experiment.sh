#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT to A, B, or C}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the UniVTAC LeRobot dataset}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/univtac_controlled}"

if (( NUM_GPUS <= 0 || GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS" >&2
    exit 1
fi

case "${EXPERIMENT}" in
    A)
        RUN_NAME="A_rgb_baseline"
        DIT_TYPE="alternate_vl_dit"
        MODALITY_CONFIG="examples/UniVTAC/univtac_config.py"
        SENSOR_ARGS=()
        ;;
    B)
        RUN_NAME="B_architecture_parity"
        DIT_TYPE="multimodal_conditioned_dit"
        MODALITY_CONFIG="examples/UniVTAC/univtac_config.py"
        SENSOR_ARGS=(--no-use-point-conditioning --no-use-tactile-conditioning)
        ;;
    C)
        RUN_NAME="C_full_multimodal"
        DIT_TYPE="multimodal_conditioned_dit"
        MODALITY_CONFIG="examples/UniVTAC/univtac_multimodal_config.py"
        SENSOR_ARGS=(--use-point-conditioning --use-tactile-conditioning)
        ;;
    *)
        echo "EXPERIMENT must be A, B, or C" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WARMUP_STEPS=$(awk -v steps="${MAX_STEPS}" -v ratio="${WARMUP_RATIO}" \
    'BEGIN { printf "%d", int(steps * ratio + 0.999999) }')
echo "Experiment ${EXPERIMENT}: ${RUN_NAME}"
echo "GPU IDs=${TRAIN_GPUS}; GPUs=${NUM_GPUS}"
echo "Per-device microbatch=$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
echo "Global batch=${GLOBAL_BATCH_SIZE}; accumulation=${GRAD_ACCUM_STEPS}"
echo "Effective optimizer batch=$((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "warmup ratio=${WARMUP_RATIO}; warmup steps=${WARMUP_STEPS}; LR=${LR}; max steps=${MAX_STEPS}"
echo "DiT=${DIT_TYPE}; modalities=${MODALITY_CONFIG}; output=${OUTPUT_DIR}; seed=${SEED}"

uv run torchrun --nproc-per-node="${NUM_GPUS}" \
    --master-port="${MASTER_PORT:-29637}" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus "${NUM_GPUS}" \
    --deepspeed-stage "${DEEPSPEED_STAGE:-3}" \
    --dit-type "${DIT_TYPE}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --learning-rate "${LR}" \
    --vlm-learning-rate "${LR}" \
    --action-head-learning-rate "${LR}" \
    --state-dropout-prob 0.0 \
    --no-tune-llm --no-tune-visual \
    --tune-projector --tune-vlln --tune-diffusion-model \
    --point-input-dim 6 \
    --seed "${SEED}" \
    "${SENSOR_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/train.log"
