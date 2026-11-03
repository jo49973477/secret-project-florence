#!/usr/bin/env bash
set -euo pipefail

# Run one Florence experiment at a time. Hydra overrides pass through unchanged.
# The four primary modes share the same overfit5 base and hardware defaults.
usage() {
  cat <<'EOF'
Usage: bash scripts/train_florence_ablations.sh MODE [Hydra overrides...]

  A / temporal_lora      Temporal + DiT LoRA
  B / sidecar_moe_lora   Contact MoE + DiT LoRA
  C / temporal_full      Temporal + DiT Full
  D / sidecar_moe_full   Contact MoE + DiT Full

Additional frozen-DiT controls: temporal_frozen, sidecar_moe_frozen

Examples:
  bash scripts/train_florence_ablations.sh A dry_run=true
  bash scripts/train_florence_ablations.sh B training.max_steps=10000
  bash scripts/train_florence_ablations.sh C hardware=dual_3090_zero3
  bash scripts/train_florence_ablations.sh D training.max_steps=10000

Set FLORENCE_OVERFIT_DATASET to the prepared five-episode LeRobot subset.
Set FLORENCE_CONCERTO_CHECKPOINT if the point encoder needs a local checkpoint.
EOF
}

mode="${1:-}"
case "$mode" in
  -h|--help|help) usage; exit 0 ;;
  A|a|temporal_lora) config=temporal_sidecar_ditlora-overfit5 ;;
  B|b|sidecar_moe_lora) config=moe_sidecar_ditlora-overfit5 ;;
  C|c|temporal_full) config=temporal_sidecar_ditfull-overfit5 ;;
  D|d|sidecar_moe_full) config=moe_sidecar_ditfull-overfit5 ;;
  temporal_frozen) config=temporal_sidecar_ditfrozen-overfit5 ;;
  sidecar_moe_frozen) config=moe_sidecar_ditfrozen-overfit5 ;;
  *)
    echo "Unknown or missing experiment mode: ${mode:-<none>}" >&2
    usage >&2
    exit 2
    ;;
esac
shift

# Allow invoking this launcher from anywhere, not only the repository root.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."

printf 'Florence experiment: %s -> %s\n' "$mode" "$config"
exec uv run --no-sync python scripts/train_hydra.py --config-name "experiment/$config" "$@"
