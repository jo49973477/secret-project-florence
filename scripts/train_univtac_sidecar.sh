#!/usr/bin/env bash
# Experiment presets are in scripts/train_univtac_sidecar.py.
set -euo pipefail
uv run python scripts/train_univtac_sidecar.py "$@"
