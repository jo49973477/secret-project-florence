#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: train_florence_ablations.sh MODE [Hydra overrides...]}"
shift
case "$mode" in
  temporal_frozen) config=temporal_sidecar_ditfrozen-overfit5 ;;
  temporal_lora) config=temporal_sidecar_ditlora-overfit5 ;;
  temporal_full) config=temporal_sidecar_ditfull-overfit5 ;;
  sidecar_moe_frozen) config=moe_sidecar_ditfrozen-overfit5 ;;
  sidecar_moe_lora) config=moe_sidecar_ditlora-overfit5 ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
exec uv run --no-sync python scripts/train_hydra.py --config-name "experiment/$config" "$@"
