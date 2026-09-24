#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=""
embodiment_tag="NEW_EMBODIMENT"
device="cuda:0"
host="127.0.0.1"
port="5555"
extra_args=()

while (($#)); do
  case "$1" in
    --checkpoint)
      checkpoint="${2:?--checkpoint requires a path}"
      shift 2
      ;;
    --embodiment-tag)
      embodiment_tag="${2:?--embodiment-tag requires a value}"
      shift 2
      ;;
    --device)
      device="${2:?--device requires a value}"
      shift 2
      ;;
    --host)
      host="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      port="${2:?--port requires a value}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/run_gr00t_univtac_server.sh --checkpoint PATH [options]

Options:
  --embodiment-tag TAG  Default: NEW_EMBODIMENT
  --device DEVICE       Default: cuda:0
  --host HOST           Default: 127.0.0.1
  --port PORT           Default: 5555

Unrecognized options are forwarded to run_gr00t_server.py.
EOF
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${checkpoint}" ]]; then
  echo "--checkpoint is required." >&2
  exit 2
fi

cd "${repo_root}"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${checkpoint}" \
  --embodiment-tag "${embodiment_tag}" \
  --device "${device}" \
  --host "${host}" \
  --port "${port}" \
  "${extra_args[@]}"
