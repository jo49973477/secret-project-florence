#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
EVAL_GPU="${EVAL_GPU:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/ssdg/spl_yeongyoo/univtac_gr00t}"
MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_config.py}"

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_BIAS="${LORA_BIAS:-none}"

EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
USE_WANDB="${USE_WANDB:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
EVAL_STEPS="${EVAL_STEPS:-400}"
EVAL_TRAJ_IDS=(0 1 2 3 4)

OUTPUT_DIR="${OUTPUT_DIR:-outputs/univtac_lora_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${OUTPUT_DIR}/logs"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
    exit 1
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_STEPS and SAVE_STEPS must be positive integers." >&2
    exit 1
fi
if [[ -z "${SAVE_TOTAL_LIMIT}" ]]; then
    SAVE_TOTAL_LIMIT=$(((MAX_STEPS + SAVE_STEPS - 1) / SAVE_STEPS))
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_PATH}" >&2
    exit 1
fi
if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Modality config not found: ${MODALITY_CONFIG}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

WANDB_ARG="--no-use-wandb"
if [[ "${USE_WANDB}" == "1" ]]; then
    WANDB_ARG="--use-wandb"
fi

EFFECTIVE_BATCH=$((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))

echo
echo "============================================================"
echo "UniVTAC GR00T N1.7 LoRA fine-tuning"
echo "============================================================"
echo "LoRA enabled             : true"
echo "LoRA r                   : ${LORA_R}"
echo "LoRA alpha               : ${LORA_ALPHA}"
echo "LoRA dropout             : ${LORA_DROPOUT}"
echo "GPUs                     : ${TRAIN_GPUS} (${NUM_GPUS} process(es))"
echo "Dataset                  : ${DATASET_PATH}"
echo "Base model               : ${BASE_MODEL}"
echo "Global batch             : ${GLOBAL_BATCH_SIZE}"
echo "Gradient accumulation    : ${GRAD_ACCUM_STEPS}"
echo "Effective batch          : ${EFFECTIVE_BATCH}"
echo "LR                       : ${LR}"
echo "Max steps                : ${MAX_STEPS}"
echo "Output directory         : ${OUTPUT_DIR}"
echo "Percentile normalization : enabled"
echo "Run evaluation           : ${RUN_EVAL}"
echo "============================================================"

TRAIN_ARGS=(
    --base-model-path "${BASE_MODEL}"
    --dataset-path "${DATASET_PATH}"
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path "${MODALITY_CONFIG}"
    --num-gpus "${NUM_GPUS}"
    --output-dir "${OUTPUT_DIR}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --max-steps "${MAX_STEPS}"
    --warmup-ratio "${WARMUP_RATIO}"
    --weight-decay "${WEIGHT_DECAY}"
    --learning-rate "${LR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --shard-size "${SHARD_SIZE}"
    --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}"
    --episode-sampling-rate "${EPISODE_SAMPLING_RATE}"
    --state-dropout-prob "${STATE_DROPOUT_PROB}"
    --use-percentiles
    --use-lora
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-bias "${LORA_BIAS}"
    --no-tune-llm
    --no-tune-visual
    --dit-type alternate_vl_dit
    --no-use-point-conditioning
    --no-use-tactile-conditioning
    --tune-projector
    --tune-diffusion-model
    --tune-vlln
    --no-tune-point-encoder
    --no-tune-tactile-encoder
    --no-tune-multimodal-adapter
    "${WANDB_ARG}"
)

TRAIN_LOG="${LOG_DIR}/train.log"
if (( NUM_GPUS == 1 )); then
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        uv run python gr00t/experiment/launch_finetune.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${TRAIN_LOG}"
else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        uv run torchrun \
        --nproc-per-node="${NUM_GPUS}" \
        --master-port="${MASTER_PORT}" \
        gr00t/experiment/launch_finetune.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${TRAIN_LOG}"
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "Training complete. Evaluation disabled (RUN_EVAL=${RUN_EVAL})."
    exit 0
fi

RESULT_CSV="${LOG_DIR}/memorization_metrics.csv"
echo "step,mse,mae" > "${RESULT_CSV}"

STEP="${SAVE_STEPS}"
while (( STEP <= MAX_STEPS )); do
    CKPT="${OUTPUT_DIR}/checkpoint-${STEP}"
    if [[ ! -d "${CKPT}" ]]; then
        echo "[WARN] Missing checkpoint, skipping: ${CKPT}"
        STEP=$((STEP + SAVE_STEPS))
        continue
    fi

    EVAL_LOG="${LOG_DIR}/eval_${STEP}.log"
    echo "Evaluating checkpoint-${STEP} on trajectories 0 1 2 3 4"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        uv run python gr00t/eval/open_loop_eval.py \
        --dataset-path "${DATASET_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path "${CKPT}" \
        --traj-ids "${EVAL_TRAJ_IDS[@]}" \
        --execution-horizon "${EXECUTION_HORIZON}" \
        --steps "${EVAL_STEPS}" \
        --modality-keys joint \
        2>&1 | tee "${EVAL_LOG}"

    MSE="$(grep "Average MSE across all trajs:" "${EVAL_LOG}" | tail -1 | awk -F': ' '{print $NF}' || true)"
    MAE="$(grep "Average MAE across all trajs:" "${EVAL_LOG}" | tail -1 | awk -F': ' '{print $NF}' || true)"
    if [[ -n "${MSE}" && -n "${MAE}" ]]; then
        echo "${STEP},${MSE},${MAE}" >> "${RESULT_CSV}"
    else
        echo "[WARN] Could not parse MSE/MAE from ${EVAL_LOG}"
    fi
    STEP=$((STEP + SAVE_STEPS))
done

echo "Training log: ${TRAIN_LOG}"
echo "Evaluation metrics: ${RESULT_CSV}"
