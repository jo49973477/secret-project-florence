#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# UniVTAC 5-episode memorization test
# VLM LoRA + FULL GR00T action-head fine-tuning
#
# Purpose:
#   Deliberately overfit episodes 0..4.
#   If this cannot memorize 5 episodes, investigate the pipeline
#   (targets / normalization / masks / checkpoint loading / etc.)
#   rather than blaming full-dataset capacity first.
# ============================================================

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=scripts/lib/checkpoint_resume.sh
source "${REPO_ROOT}/scripts/lib/checkpoint_resume.sh"

# --------------------------
# User-configurable settings
# --------------------------
TRAIN_GPUS="${TRAIN_GPUS:-3,4}"
EVAL_GPU="${EVAL_GPU:-3}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29631}"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

# RGB UniVTAC dataset currently used by finetune_univtac_lora.sh
SOURCE_DATASET="${SOURCE_DATASET:-/ssdg/spl_yeongyoo/univtac_gr00t}"

# Tiny 5-episode subset created with symlinks.
OVERFIT_DATASET="${OVERFIT_DATASET:-/ssdg/spl_yeongyoo/univtac_gr00t_overfit5_lora}"

MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_config.py}"

# Intentionally short/aggressive memorization run.
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"

# 2 GPUs => one sample/GPU at global batch 2.
# Accumulation is deliberately 1 for a memorization test.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
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
REBUILD_SUBSET="${REBUILD_SUBSET:-1}"

EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
EVAL_STEPS="${EVAL_STEPS:-400}"
EVAL_TRAJ_IDS=(0 1 2 3 4)

OUTPUT_DIR="${OUTPUT_DIR:-outputs/univtac_overfit5_lora_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${OUTPUT_DIR}/logs"

if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_STEPS and SAVE_STEPS must be positive integers." >&2
    exit 1
fi
ALL_CHECKPOINTS_LIMIT=$(((MAX_STEPS + SAVE_STEPS - 1) / SAVE_STEPS))
checkpoint_resume_configure "${OUTPUT_DIR}" "${ALL_CHECKPOINTS_LIMIT}"

# ============================================================
# Basic checks
# ============================================================

if [[ ! -d "${SOURCE_DATASET}" ]]; then
    echo "[ERROR] SOURCE_DATASET not found: ${SOURCE_DATASET}" >&2
    exit 1
fi

if [[ ! -f "${SOURCE_DATASET}/meta/info.json" ]] || \
   [[ ! -f "${SOURCE_DATASET}/meta/episodes.jsonl" ]]; then
    echo "[ERROR] SOURCE_DATASET does not look like a LeRobot dataset." >&2
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Modality config not found: ${MODALITY_CONFIG}" >&2
    exit 1
fi

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_GPUS must be a positive integer: ${NUM_GPUS}" >&2
    exit 1
fi

SOURCE_REAL="$(realpath "${SOURCE_DATASET}")"
OVERFIT_PARENT="$(dirname "${OVERFIT_DATASET}")"
mkdir -p "${OVERFIT_PARENT}"
OVERFIT_REAL="$(realpath -m "${OVERFIT_DATASET}")"

if [[ "${SOURCE_REAL}" == "${OVERFIT_REAL}" ]]; then
    echo "[ERROR] SOURCE_DATASET and OVERFIT_DATASET must differ." >&2
    exit 1
fi

if [[ -z "${OVERFIT_REAL}" || "${OVERFIT_REAL}" == "/" || "${OVERFIT_REAL}" == "${HOME}" ]]; then
    echo "[ERROR] Refusing unsafe OVERFIT_DATASET path: ${OVERFIT_REAL}" >&2
    exit 1
fi

# ============================================================
# 1. Build 5-episode subset: episodes 0..4
# ============================================================

if [[ "${REBUILD_SUBSET}" == "1" ]]; then
    echo "============================================================"
    echo "Building UniVTAC 5-episode subset"
    echo "============================================================"
    echo "Source : ${SOURCE_REAL}"
    echo "Target : ${OVERFIT_REAL}"

    rm -rf "${OVERFIT_REAL}"
    mkdir -p "${OVERFIT_REAL}/meta"

    python - "${SOURCE_REAL}" "${OVERFIT_REAL}" <<'PY'
import json
import math
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
src_meta = src / "meta"
dst_meta = dst / "meta"

selected_ids = list(range(5))

with (src_meta / "episodes.jsonl").open("r", encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f if line.strip()]

by_id = {int(ep["episode_index"]): ep for ep in episodes}
missing = [idx for idx in selected_ids if idx not in by_id]
if missing:
    raise SystemExit(f"[ERROR] Missing episode metadata for IDs: {missing}")

selected = [by_id[idx] for idx in selected_ids]

# Full-dataset statistics must not leak into the tiny memorization test.
skip = {"info.json", "episodes.jsonl", "stats.json", "relative_stats.json"}

for item in src_meta.iterdir():
    if item.name in skip:
        continue
    target = dst_meta / item.name
    if item.is_dir():
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)

with (dst_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
    for ep in selected:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")

with (src_meta / "info.json").open("r", encoding="utf-8") as f:
    info = json.load(f)

total_frames = sum(int(ep["length"]) for ep in selected)
chunks_size = int(info.get("chunks_size", 1000))

info["total_episodes"] = 5
info["total_frames"] = total_frames
info["splits"] = {"train": "0:5"}
info["total_chunks"] = math.ceil(5 / chunks_size)

video_features = [
    value
    for value in info.get("features", {}).values()
    if isinstance(value, dict) and value.get("dtype") == "video"
]
if video_features:
    info["total_videos"] = 5 * len(video_features)

with (dst_meta / "info.json").open("w", encoding="utf-8") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)
    f.write("\n")

# Preserve any dataset-root sidecars.
for item in src.iterdir():
    if item.name == "meta" or item.is_dir():
        continue
    target = dst / item.name
    if not target.exists():
        target.symlink_to(item.resolve())

print(f"Selected episodes: {selected_ids}")
print(f"Selected frames:   {total_frames}")
PY

    # Symlink episode-specific artifacts: parquet, mp4, npz, etc.
    while IFS= read -r -d '' SRC_FILE; do
        BASE="$(basename "${SRC_FILE}")"

        if [[ "${BASE}" =~ ^episode_([0-9]{6})\. ]]; then
            IDX=$((10#${BASH_REMATCH[1]}))
            if (( IDX >= 0 && IDX <= 4 )); then
                REL="${SRC_FILE#${SOURCE_REAL}/}"
                DST_FILE="${OVERFIT_REAL}/${REL}"
                mkdir -p "$(dirname "${DST_FILE}")"
                ln -s "${SRC_FILE}" "${DST_FILE}"
            fi
        fi
    done < <(
        find -L "${SOURCE_REAL}" \
            -path "${SOURCE_REAL}/meta" -prune -o \
            -type f -name 'episode_*' -print0
    )

    PARQUET_COUNT="$(find -L "${OVERFIT_REAL}/data" -type f -name 'episode_*.parquet' 2>/dev/null | wc -l)"
    if [[ "${PARQUET_COUNT}" -ne 5 ]]; then
        echo "[ERROR] Expected 5 parquet episodes, found ${PARQUET_COUNT}" >&2
        exit 1
    fi

    echo
    echo "Recomputing state/action statistics on only these 5 episodes..."
    rm -f "${OVERFIT_REAL}/meta/stats.json" \
          "${OVERFIT_REAL}/meta/relative_stats.json"

    uv run python gr00t/data/stats.py \
        --dataset-path "${OVERFIT_REAL}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${MODALITY_CONFIG}"
else
    echo "[INFO] Reusing subset: ${OVERFIT_REAL}"
fi

# ============================================================
# 2. Fine-tune: VLM LoRA + FULL action head
# ============================================================

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
echo "UniVTAC 5-episode OVERFIT: LoRA + full action head"
echo "============================================================"
echo "GPUs                  : ${TRAIN_GPUS}"
echo "Subset                 : ${OVERFIT_REAL}"
echo "Max steps              : ${MAX_STEPS}"
echo "Save every             : ${SAVE_STEPS}"
echo "Global batch           : ${GLOBAL_BATCH_SIZE}"
echo "Grad accumulation      : ${GRAD_ACCUM_STEPS}"
echo "Effective batch        : ${EFFECTIVE_BATCH}"
echo "LR                     : ${LR}"
echo "Warmup                 : ${WARMUP_RATIO}"
echo "Weight decay           : ${WEIGHT_DECAY}"
echo "State dropout          : ${STATE_DROPOUT_PROB}"
echo "LoRA r / alpha         : ${LORA_R} / ${LORA_ALPHA}"
checkpoint_resume_print_status "${OUTPUT_DIR}"
echo
echo "TRAIN:"
echo "  Qwen base weights    : FROZEN"
echo "  Qwen LoRA            : TRAIN"
echo "  Projector            : TRAIN"
echo "  VLLN                 : TRAIN"
echo "  Diffusion / DiT      : TRAIN"
echo "  Point/tactile        : OFF"
echo "============================================================"

TRAIN_ARGS=(
    --base-model-path "${BASE_MODEL}"
    --dataset-path "${OVERFIT_REAL}"
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

    "${CHECKPOINT_SAVE_ARGS[@]}"
    "${CHECKPOINT_RESUME_ARGS[@]}"
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

# ============================================================
# 3. Same-episode open-loop eval
# ============================================================

if [[ "${RUN_EVAL}" != "1" ]]; then
    echo
    echo "Training complete. RUN_EVAL=${RUN_EVAL}, skipping evaluation."
    exit 0
fi

RESULT_CSV="${LOG_DIR}/memorization_metrics.csv"
echo "step,mse,mae" > "${RESULT_CSV}"

STEP="${SAVE_STEPS}"
while (( STEP <= MAX_STEPS )); do
    CKPT="${OUTPUT_DIR}/checkpoint-${STEP}"

    if [[ ! -d "${CKPT}" ]]; then
        echo "[WARN] Missing checkpoint: ${CKPT}"
        STEP=$((STEP + SAVE_STEPS))
        continue
    fi

    EVAL_LOG="${LOG_DIR}/eval_${STEP}.log"

    echo
    echo "============================================================"
    echo "Evaluating checkpoint-${STEP} on SAME episodes 0..4"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        uv run python gr00t/eval/open_loop_eval.py \
        --dataset-path "${OVERFIT_REAL}" \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path "${CKPT}" \
        --traj-ids "${EVAL_TRAJ_IDS[@]}" \
        --execution-horizon "${EXECUTION_HORIZON}" \
        --steps "${EVAL_STEPS}" \
        --modality-keys joint \
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
echo "MEMORIZATION RESULT"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
    column -s, -t "${RESULT_CSV}"
else
    cat "${RESULT_CSV}"
fi

echo
echo "Train log : ${TRAIN_LOG}"
echo "Metrics   : ${RESULT_CSV}"
echo
echo "Interpretation:"
echo "  - Large loss / MSE / MAE reduction => training pipeline can memorize."
echo "  - Failure to memorize 5 episodes => inspect target alignment,"
echo "    normalization, masks/padding, checkpoint loading, and gradients."
