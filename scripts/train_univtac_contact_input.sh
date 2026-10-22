#!/usr/bin/env bash
# Common, explicit comparison budget. No automatic checkpoint discovery or resume.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:?Usage: bash scripts/train_univtac_contact_input.sh B0|B1|B2|B3|B4 [finetune flags]}"
shift
mode=none
extra=()
case "$variant" in
  B0) mode=none ;;
  B1) mode=direct ;;
  B2) mode=scalar_gate ;;
  B3) mode=moe_soft ;;
  B4)
    mode=moe_soft
    : "${CONTACT_PREDICTOR:?B4 unavailable until a validated predictor bundle is supplied}"
    if [[ ! -f "$CONTACT_PREDICTOR" ]]; then
      echo "Predictor bundle does not exist: $CONTACT_PREDICTOR" >&2
      exit 2
    fi
    extra+=(--contact-moe-use-surprise --contact-predictor-checkpoint "$CONTACT_PREDICTOR")
    ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
: "${BASE_MODEL:?Set the original GR00T checkpoint path}"
: "${DATASET_PATH:?Set a converted UniVTAC dataset or independent five-episode subset}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-contact_input_moe_20261021}"
export WANDB_NAME="${WANDB_NAME:-contact_input_${variant}}"
export WANDB_DIR="${WANDB_DIR:-$PWD/outputs/contact_input/$variant/wandb}"
mkdir -p "$WANDB_DIR"
# --no-sync preserves the explicitly verified local environment.
exec uv run --no-sync python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL" --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_contact_config.py \
  --output-dir "outputs/contact_input/$variant" \
  --dit-type alternate_vl_dit --contact-input-mode "$mode" \
  --no-use-point-conditioning --use-tactile-conditioning \
  --no-tune-llm --no-tune-visual --tune-projector --tune-vlln \
  --freeze-pretrained-dit --no-tune-tactile-encoder \
  --train-contact-input-adapter --train-contact-router \
  --seed 42 --global-batch-size 1 --gradient-accumulation-steps 16 \
  --num-gpus 1 --deepspeed-stage 2 --max-steps 5000 --save-steps 100 \
  --action-head-learning-rate 1e-4 --weight-decay 1e-5 \
  --episode-sampling-rate 1.0 --dataloader-num-workers 0 \
  "${extra[@]}" "$@"
