#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Official GR00T N1.7 SO100 5-episode sanity / memorization test
#
# Purpose:
#   1) Verify that the current GR00T training stack can learn NVIDIA's
#      known-good NEW_EMBODIMENT 5-episode example.
#   2) Evaluate the SAME five training episodes at every checkpoint.
#
# Default MODE=official:
#   - Keeps NVIDIA's official fine-tuning recipe as closely as practical.
#   - On small GPUs, GLOBAL_BATCH_SIZE=2 + GRAD_ACCUM_STEPS=16
#     approximates effective batch 32 without needing batch 16/GPU.
#
# Optional MODE=overfit:
#   - Removes augmentation/regularization and grad accumulation to make
#     memorization easier.
#
# Examples:
#   MODE=official \
#   TRAIN_GPUS=2,3 EVAL_GPU=2 NUM_GPUS=2 MASTER_PORT=29611 \
#   bash scripts/overfit_so100_5eps.sh
#
#   MODE=overfit \
#   TRAIN_GPUS=2,3 EVAL_GPU=2 NUM_GPUS=2 MASTER_PORT=29612 \
#   bash scripts/overfit_so100_5eps.sh
# ============================================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${MODE:-official}"

TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
EVAL_GPU="${EVAL_GPU:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29611}"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-demo_data/cube_to_bowl_5}"
MODALITY_CONFIG="${MODALITY_CONFIG:-examples/SO100/so100_config.py}"

MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# Dataset loader knobs. Keep all five episodes available.
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"

EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
EVAL_STEPS="${EVAL_STEPS:-400}"
EVAL_TRAJ_IDS=(0 1 2 3 4)

case "${MODE}" in
  official)
    # NVIDIA's documented recipe uses global batch 32.
    # Two 24GB GPUs are unlikely to hold 16 samples/GPU, so emulate an
    # effective batch of 32 with micro/global batch 2 x accumulation 16.
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
    LR="${LR:-1e-4}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
    STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"

    # Official guide augmentation.
    CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.3}"
    CJ_CONTRAST="${CJ_CONTRAST:-0.4}"
    CJ_SATURATION="${CJ_SATURATION:-0.5}"
    CJ_HUE="${CJ_HUE:-0.08}"
    RANDOM_ROTATION="${RANDOM_ROTATION:-0}"
    USE_PERCENTILES="${USE_PERCENTILES:-1}"
    ;;

  overfit)
    # Deliberately make memorization easy while preserving the official
    # SO100 dataset/config and percentile normalization.
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
    LR="${LR:-1e-4}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
    STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

    CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.0}"
    CJ_CONTRAST="${CJ_CONTRAST:-0.0}"
    CJ_SATURATION="${CJ_SATURATION:-0.0}"
    CJ_HUE="${CJ_HUE:-0.0}"
    RANDOM_ROTATION="${RANDOM_ROTATION:-0}"
    USE_PERCENTILES="${USE_PERCENTILES:-1}"
    ;;

  *)
    echo "[ERROR] MODE must be 'official' or 'overfit', got: ${MODE}" >&2
    exit 1
    ;;
esac

RUN_NAME="${RUN_NAME:-so100_5eps_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
LOG_DIR="${OUTPUT_DIR}/logs"
PLOT_DIR="${LOG_DIR}/plots"

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] SO100 demo dataset not found: ${DATASET_PATH}" >&2
    echo "        Make sure git-lfs data/submodules are available." >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/meta/info.json" ]] || \
   [[ ! -f "${DATASET_PATH}/meta/episodes.jsonl" ]] || \
   [[ ! -f "${DATASET_PATH}/meta/stats.json" ]]; then
    echo "[ERROR] ${DATASET_PATH} does not look like the complete official SO100 demo dataset." >&2
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Modality config not found: ${MODALITY_CONFIG}" >&2
    exit 1
fi

PARQUET_COUNT="$(
  find -L "${DATASET_PATH}/data" -type f -name 'episode_*.parquet' 2>/dev/null | wc -l
)"
if [[ "${PARQUET_COUNT}" -ne 5 ]]; then
    echo "[ERROR] Expected exactly 5 SO100 parquet episodes, found ${PARQUET_COUNT}." >&2
    echo "        If the files are Git-LFS pointer stubs, run: git lfs pull" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

echo
echo "============================================================"
echo "Official SO100 5-episode sanity / memorization test"
echo "============================================================"
echo "Mode                     : ${MODE}"
echo "GPUs                     : ${TRAIN_GPUS}"
echo "Eval GPU                 : ${EVAL_GPU}"
echo "Master port              : ${MASTER_PORT}"
echo "Dataset                  : ${DATASET_PATH}"
echo "Modality config          : ${MODALITY_CONFIG}"
echo "Base model               : ${BASE_MODEL}"
echo "Output                    : ${OUTPUT_DIR}"
echo "Max steps                 : ${MAX_STEPS}"
echo "Save steps                : ${SAVE_STEPS}"
echo "Learning rate             : ${LR}"
echo "Global/micro batch        : ${GLOBAL_BATCH_SIZE}"
echo "Gradient accumulation     : ${GRAD_ACCUM_STEPS}"
echo "Effective batch           : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Episode sampling rate     : ${EPISODE_SAMPLING_RATE}"
echo "State dropout             : ${STATE_DROPOUT_PROB}"
echo "Weight decay              : ${WEIGHT_DECAY}"
echo "Warmup ratio              : ${WARMUP_RATIO}"
echo "Color jitter              : ${CJ_BRIGHTNESS}/${CJ_CONTRAST}/${CJ_SATURATION}/${CJ_HUE}"
echo "Use percentiles           : ${USE_PERCENTILES}"
echo "============================================================"

TRAIN_LOG="${LOG_DIR}/train.log"

PERCENTILE_ARG=(--use-percentiles)
if [[ "${USE_PERCENTILES}" == "0" ]]; then
    PERCENTILE_ARG=(--no-use-percentiles)
fi

if [[ "${NUM_GPUS}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    uv run python \
        gr00t/experiment/launch_finetune.py \
        --base-model-path "${BASE_MODEL}" \
        --dataset-path "${DATASET_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${MODALITY_CONFIG}" \
        --num-gpus 1 \
        --output-dir "${OUTPUT_DIR}" \
        --save-steps "${SAVE_STEPS}" \
        --save-total-limit 10 \
        --max-steps "${MAX_STEPS}" \
        --learning-rate "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-ratio "${WARMUP_RATIO}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
        --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
        --shard-size "${SHARD_SIZE}" \
        --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}" \
        --episode-sampling-rate "${EPISODE_SAMPLING_RATE}" \
        --state-dropout-prob "${STATE_DROPOUT_PROB}" \
        --random-rotation-angle "${RANDOM_ROTATION}" \
        --color-jitter-params \
            brightness "${CJ_BRIGHTNESS}" \
            contrast "${CJ_CONTRAST}" \
            saturation "${CJ_SATURATION}" \
            hue "${CJ_HUE}" \
        "${PERCENTILE_ARG[@]}" \
        --save-only-model \
        2>&1 | tee "${TRAIN_LOG}"
else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    uv run torchrun \
        --nproc-per-node="${NUM_GPUS}" \
        --master-port="${MASTER_PORT}" \
        gr00t/experiment/launch_finetune.py \
        --base-model-path "${BASE_MODEL}" \
        --dataset-path "${DATASET_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${MODALITY_CONFIG}" \
        --num-gpus "${NUM_GPUS}" \
        --output-dir "${OUTPUT_DIR}" \
        --save-steps "${SAVE_STEPS}" \
        --save-total-limit 10 \
        --max-steps "${MAX_STEPS}" \
        --learning-rate "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-ratio "${WARMUP_RATIO}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
        --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
        --shard-size "${SHARD_SIZE}" \
        --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}" \
        --episode-sampling-rate "${EPISODE_SAMPLING_RATE}" \
        --state-dropout-prob "${STATE_DROPOUT_PROB}" \
        --random-rotation-angle "${RANDOM_ROTATION}" \
        --color-jitter-params \
            brightness "${CJ_BRIGHTNESS}" \
            contrast "${CJ_CONTRAST}" \
            saturation "${CJ_SATURATION}" \
            hue "${CJ_HUE}" \
        "${PERCENTILE_ARG[@]}" \
        --save-only-model \
        2>&1 | tee "${TRAIN_LOG}"
fi

# ------------------------------------------------------------
# Evaluate every checkpoint on the SAME five training episodes.
# ------------------------------------------------------------

RESULT_CSV="${LOG_DIR}/memorization_metrics.csv"
echo "step,mse,mae" > "${RESULT_CSV}"

echo
echo "============================================================"
echo "SO100 same-training-episode open-loop evaluation"
echo "============================================================"

STEP="${SAVE_STEPS}"
while (( STEP <= MAX_STEPS )); do
    CKPT="${OUTPUT_DIR}/checkpoint-${STEP}"

    if [[ ! -d "${CKPT}" ]]; then
        echo "[WARN] Missing checkpoint: ${CKPT}"
        STEP=$((STEP + SAVE_STEPS))
        continue
    fi

    EVAL_LOG="${LOG_DIR}/eval_${STEP}.log"
    STEP_PLOT_DIR="${PLOT_DIR}/checkpoint_${STEP}"
    mkdir -p "${STEP_PLOT_DIR}"

    echo
    echo "Evaluating checkpoint-${STEP} on SO100 trajectories 0..4"

    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
    uv run python gr00t/eval/open_loop_eval.py \
        --dataset-path "${DATASET_PATH}" \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path "${CKPT}" \
        --traj-ids "${EVAL_TRAJ_IDS[@]}" \
        --execution-horizon "${EXECUTION_HORIZON}" \
        --steps "${EVAL_STEPS}" \
        --modality-keys single_arm gripper \
        --save-plot-path "${STEP_PLOT_DIR}" \
        2>&1 | tee "${EVAL_LOG}"

    MSE="$(
        grep "Average MSE across all trajs:" "${EVAL_LOG}" \
        | tail -1 | awk -F': ' '{print $NF}' || true
    )"
    MAE="$(
        grep "Average MAE across all trajs:" "${EVAL_LOG}" \
        | tail -1 | awk -F': ' '{print $NF}' || true
    )"

    if [[ -n "${MSE}" && -n "${MAE}" ]]; then
        echo "${STEP},${MSE},${MAE}" >> "${RESULT_CSV}"
    else
        echo "[WARN] Could not parse MSE/MAE from ${EVAL_LOG}"
    fi

    STEP=$((STEP + SAVE_STEPS))
done

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
echo "Train log : ${TRAIN_LOG}"
echo "Metrics   : ${RESULT_CSV}"
echo "Plots     : ${PLOT_DIR}"
echo
echo "Reference shape from NVIDIA's SO100 NEW_EMBODIMENT guide:"
echo "  training-trajectory MSE/MAE should fall clearly with checkpoint step."
echo "  If SO100 learns but UniVTAC stays flat, focus on UniVTAC conversion/"
echo "  action representation/normalization/temporal alignment."
echo "============================================================"
