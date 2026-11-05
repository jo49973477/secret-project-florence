#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UNIVTAC_EXPECTED_PYTHON:?Set UNIVTAC_EXPECTED_PYTHON to the Isaac-compatible Python executable}"
read -r -a driver_command <<< "${UNIVTAC_DRIVER_PYTHON_COMMAND:-${UNIVTAC_PYTHON_COMMAND:-${UNIVTAC_EXPECTED_PYTHON}}}"
exec "${driver_command[@]}" "${repo_root}/scripts/univtac_deployment.py" --deployment split "$@"
