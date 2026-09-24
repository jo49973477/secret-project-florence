#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${UNIVTAC_ROOT:-}" ]]; then
  echo "UNIVTAC_ROOT must point to the UniVTAC checkout." >&2
  exit 2
fi

python_command="${UNIVTAC_PYTHON_COMMAND:-python}"
read -r -a driver_command <<< "${UNIVTAC_DRIVER_PYTHON_COMMAND:-python}"

exec "${driver_command[@]}" "${repo_root}/examples/UniVTAC/run_closed_loop_eval.py" \
  --univtac-root "${UNIVTAC_ROOT}" \
  --python-command "${python_command}" \
  "$@"
