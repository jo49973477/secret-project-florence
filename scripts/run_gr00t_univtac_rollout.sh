#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${UNIVTAC_ROOT:-}" ]]; then
  echo "UNIVTAC_ROOT must point to the UniVTAC checkout." >&2
  exit 2
fi

if [[ -z "${UNIVTAC_PYTHON_COMMAND:-}" ]]; then
  echo "UNIVTAC_PYTHON_COMMAND must name the Isaac/UniVTAC Python launcher." >&2
  exit 2
fi
if [[ -z "${UNIVTAC_EXPECTED_PYTHON:-}" ]]; then
  echo "UNIVTAC_EXPECTED_PYTHON must be the exact Python executable used by that launcher." >&2
  exit 2
fi
if [[ ! -x "${UNIVTAC_EXPECTED_PYTHON}" ]]; then
  echo "UNIVTAC_EXPECTED_PYTHON is not executable: ${UNIVTAC_EXPECTED_PYTHON}" >&2
  exit 2
fi

python_command="${UNIVTAC_PYTHON_COMMAND}"
read -r -a driver_command <<< "${UNIVTAC_DRIVER_PYTHON_COMMAND:-${UNIVTAC_PYTHON_COMMAND}}"

exec "${driver_command[@]}" "${repo_root}/examples/UniVTAC/run_closed_loop_eval.py" \
  --univtac-root "${UNIVTAC_ROOT}" \
  --python-command "${python_command}" \
  --expected-python "${UNIVTAC_EXPECTED_PYTHON}" \
  "$@"
