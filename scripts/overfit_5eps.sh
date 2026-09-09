#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# UniVTAC 5-episode memorization / overfit test for rgbd_test
#
# Goal:
#   If the current GR00T pipeline is healthy, it should be able to
#   memorize episodes 0..4 much more aggressively than the full dataset.
#
# Run:
#   bash scripts/overfit_univtac_5eps.sh
#
# Common overrides:
#   TRAIN_GPUS=4,5 \
#   SOURCE_DATASET=/ssdg/spl_yeongyoo/univtac_gr00t \
#   MAX_STEPS=5000 \
#   LR=3e-4 \
#   bash scripts/overfit_univtac_5eps.sh
# ============================================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# --------------------------
# User-configurable settings
# --------------------------
TRAIN_GPUS="${TRAIN_GPUS:-5,6}"
EVAL_GPU="${EVAL_GPU:-5}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29601}"

BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

# Existing full LeRobot-format UniVTAC dataset.
SOURCE_DATASET="${SOURCE_DATASET:-/ssdg/spl_yeongyoo/univtac_gr00t}"

# A tiny dataset containing only episode indices 0,1,2,3,4.
# Episode files are symlinked, so this consumes little extra disk space.
OVERFIT_DATASET="${OVERFIT_DATASET:-/ssdg/spl_yeongyoo/univtac_gr00t_overfit5}"

MODALITY_CONFIG="${MODALITY_CONFIG:-examples/UniVTAC/univtac_config.py}"

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

# Two 3090s -> global batch 2 means roughly 1 sample/GPU.
# For memorization, do NOT use the previous grad accumulation = 16.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# Deliberately aggressive memorization settings.
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"

SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"

EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
EVAL_STEPS="${EVAL_STEPS:-400}"
EVAL_TRAJ_IDS=(0 1 2 3 4)

RUN_NAME="${RUN_NAME:-univtac_overfit5_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
LOG_DIR="${OUTPUT_DIR}/logs"

# Rebuild the tiny metadata/symlink dataset each run by default.
REBUILD_SUBSET="${REBUILD_SUBSET:-1}"

# ============================================================
# Safety checks
# ============================================================

if [[ ! -d "${SOURCE_DATASET}" ]]; then
    echo "[ERROR] SOURCE_DATASET not found: ${SOURCE_DATASET}" >&2
    exit 1
fi

if [[ ! -f "${SOURCE_DATASET}/meta/info.json" ]] || \
   [[ ! -f "${SOURCE_DATASET}/meta/episodes.jsonl" ]]; then
    echo "[ERROR] SOURCE_DATASET does not look like a LeRobot dataset." >&2
    echo "        Missing meta/info.json or meta/episodes.jsonl" >&2
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "[ERROR] Modality config not found: ${MODALITY_CONFIG}" >&2
    exit 1
fi

SOURCE_REAL="$(realpath "${SOURCE_DATASET}")"
OVERFIT_PARENT="$(dirname "${OVERFIT_DATASET}")"
mkdir -p "${OVERFIT_PARENT}"

# realpath -m works even if target does not exist.
OVERFIT_REAL="$(realpath -m "${OVERFIT_DATASET}")"

if [[ "${SOURCE_REAL}" == "${OVERFIT_REAL}" ]]; then
    echo "[ERROR] SOURCE_DATASET and OVERFIT_DATASET must be different." >&2
    exit 1
fi

if [[ -z "${OVERFIT_REAL}" || "${OVERFIT_REAL}" == "/" || "${OVERFIT_REAL}" == "${HOME}" ]]; then
    echo "[ERROR] Refusing unsafe OVERFIT_DATASET path: ${OVERFIT_REAL}" >&2
    exit 1
fi

# ============================================================
# 1. Build a 5-episode LeRobot subset (episodes 0..4)
# ============================================================

if [[ "${REBUILD_SUBSET}" == "1" ]]; then
    echo "============================================================"
    echo "Building 5-episode subset"
    echo "============================================================"
    echo "Source : ${SOURCE_REAL}"
    echo "Target : ${OVERFIT_REAL}"

    rm -rf "${OVERFIT_REAL}"
    mkdir -p "${OVERFIT_REAL}/meta"

    # Copy metadata except stats; stats MUST be recomputed from the 5 episodes.
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

# Copy every metadata item except files that depend on the full dataset.
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

# Keep total_tasks/tasks.jsonl unchanged because parquet task_index values
# should not be remapped. Only update total_videos when it can be inferred.
video_features = [
    value for value in info.get("features", {}).values()
    if isinstance(value, dict) and value.get("dtype") == "video"
]
if video_features:
    info["total_videos"] = 5 * len(video_features)

with (dst_meta / "info.json").open("w", encoding="utf-8") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)
    f.write("\n")

# Preserve non-directory dataset-root sidecars, if any.
for item in src.iterdir():
    if item.name == "meta" or item.is_dir():
        continue
    target = dst / item.name
    if not target.exists():
        target.symlink_to(item.resolve())

print(f"Selected episodes: {selected_ids}")
print(f"Selected frames:   {total_frames}")
PY

    # Symlink every episode-specific artifact for episode 000000..000004.
    # This is intentionally generic: parquet, mp4, depth/mask sidecars, etc.
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
        echo "[ERROR] Expected exactly 5 parquet episodes, found ${PARQUET_COUNT}." >&2
        exit 1
    fi

    echo
    echo "Recomputing normalization statistics ONLY from the 5 episodes..."
    rm -f "${OVERFIT_REAL}/meta/stats.json" \
          "${OVERFIT_REAL}/meta/relative_stats.json"

    uv run python gr00t/data/stats.py \
        --dataset-path "${OVERFIT_REAL}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${MODALITY_CONFIG}"
else
    echo "[INFO] Reusing existing subset: ${OVERFIT_REAL}"
fi

# ============================================================
# 2. Train: intentionally try to memorize the five episodes
# ============================================================

mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

echo
echo "============================================================"
echo "UniVTAC 5-episode OVERFIT TEST"
echo "============================================================"
echo "GPUs                    : ${TRAIN_GPUS}"
echo "Dataset                 : ${OVERFIT_REAL}"
echo "Base model              : ${BASE_MODEL}"
echo "Output                   : ${OUTPUT_DIR}"
echo "Max steps                : ${MAX_STEPS}"
echo "Learning rate            : ${LR}"
echo "Global batch             : ${GLOBAL_BATCH_SIZE}"
echo "Gradient accumulation    : ${GRAD_ACCUM_STEPS}"
echo "Effective batch          : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Episode sampling rate    : ${EPISODE_SAMPLING_RATE}"
echo "State dropout            : ${STATE_DROPOUT_PROB}"
echo "Weight decay             : ${WEIGHT_DECAY}"
echo "Warmup ratio             : ${WARMUP_RATIO}"
echo "Color jitter             : all zeros"
echo "Use percentiles          : false (full min/max)"
echo "Trainable by default     : projector + diffusion/action model"
echo "Frozen by default        : LLM + visual backbone"
echo "============================================================"

TRAIN_LOG="${LOG_DIR}/train.log"

uv run torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "${BASE_MODEL}" \
    --dataset_path "${OVERFIT_REAL}" \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path "${MODALITY_CONFIG}" \
    --num_gpus "${NUM_GPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 5 \
    --max_steps "${MAX_STEPS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --learning_rate "${LR}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --shard_size "${SHARD_SIZE}" \
    --num_shards_per_epoch "${NUM_SHARDS_PER_EPOCH}" \
    --episode_sampling_rate "${EPISODE_SAMPLING_RATE}" \
    --state_dropout_prob "${STATE_DROPOUT_PROB}" \
    --random_rotation_angle 0 \
    --color_jitter_params \
        brightness 0.0 \
        contrast 0.0 \
        saturation 0.0 \
        hue 0.0 \
    --no-use-percentiles \
    --save_only_model \
    2>&1 | tee "${TRAIN_LOG}"

# ============================================================
# 3. Evaluate on the SAME five episodes
#    This is memorization, NOT generalization evaluation.
# ============================================================

RESULT_CSV="${LOG_DIR}/memorization_metrics.csv"
echo "step,mse,mae" > "${RESULT_CSV}"

echo
echo "============================================================"
echo "Same-episode open-loop memorization evaluation"
echo "============================================================"

STEP="${SAVE_STEPS}"
while (( STEP <= MAX_STEPS )); do
    CKPT="${OUTPUT_DIR}/checkpoint-${STEP}"

    if [[ ! -d "${CKPT}" ]]; then
        echo "[WARN] Missing checkpoint, skipping: ${CKPT}"
        STEP=$((STEP + SAVE_STEPS))
        continue
    fi

    EVAL_LOG="${LOG_DIR}/eval_${STEP}.log"

    echo
    echo "Evaluating checkpoint-${STEP} on trajectories 0..4"

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
        | tail -1 \
        | awk -F': ' '{print $NF}' || true
    )"
    MAE="$(
        grep "Average MAE across all trajs:" "${EVAL_LOG}" \
        | tail -1 \
        | awk -F': ' '{print $NF}' || true
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
echo
echo "Interpretation:"
echo "  - Same-episode MSE/MAE should drop hard if the pipeline can memorize."
echo "  - Do NOT judge only by diffusion/flow-matching train loss;"
echo "    same-episode action error is the more useful memorization signal."
echo "============================================================"
