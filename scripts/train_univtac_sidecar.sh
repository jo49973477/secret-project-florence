#!/usr/bin/env bash
# Explicit staged experiments. Additional Hydra overrides follow the stage argument.
set -euo pipefail
stage="${1:?Usage: bash scripts/train_univtac_sidecar.sh A|B|C|overfit5 [Hydra overrides]}"
shift
experiment=univtac_full
args=(model=gr00t_sidecar data=univtac_multimodal)
case "$stage" in
  A) args+=(model.tactile_fusion_mode=none model.use_tactile_conditioning=false) ;;
  B) args+=(model.contact_predictor_warmup=true model.contact_injection_enabled=false) ;;
  C) args+=(model.contact_predictor_warmup=false model.contact_injection_enabled=true) ;;
  overfit5)
    experiment=univtac_overfit5
    args+=(model.tactile_fusion_mode=none model.use_tactile_conditioning=false)
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
uv run python scripts/train_hydra.py --config-name "experiment/$experiment" "${args[@]}" "$@"
