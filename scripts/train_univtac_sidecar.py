#!/usr/bin/env python3
"""Readable experiment presets for UniVTAC Sidecar training.

Usage:
    bash scripts/train_univtac_sidecar.sh overfit5 training.max_steps=500 name=my_run

Edit PRESETS for stage behavior; edit YAML for model/training defaults.
Extra Hydra overrides always win over the preset.
"""

import shlex
import subprocess
import sys
from pathlib import Path


PRESETS = {
    # Point-only ablation.
    "A": {
        "experiment": "univtac_full",
        "overrides": [
            "model.tactile_fusion_mode=none",
            "model.use_tactile_conditioning=false",
        ],
    },
    # Train the contact predictor without injecting contact into actions.
    "B": {
        "experiment": "univtac_full",
        "overrides": [
            "model.contact_predictor_warmup=true",
            "model.contact_injection_enabled=false",
        ],
    },
    # Full Point + Tactile Sidecar.
    "C": {
        "experiment": "univtac_full",
        "overrides": [
            "model.contact_predictor_warmup=false",
            "model.contact_injection_enabled=true",
        ],
    },
    # Five-episode overfit with BOTH Point + Tactile enabled.
    "overfit5": {
        "experiment": "univtac_overfit5",
        "overrides": [
            "model.use_point_conditioning=true",
            "model.use_tactile_conditioning=true",
            "model.tactile_fusion_mode=surprise",
        ],
    },
}


def build_command(stage: str, overrides: list[str]) -> list[str]:
    if stage not in PRESETS:
        choices = ", ".join(PRESETS)
        raise ValueError(f"Choose a stage: {choices}")

    preset = PRESETS[stage]
    return [
        sys.executable,
        str(Path(__file__).with_name("train_hydra.py")),
        "--config-name",
        f"experiment/{preset['experiment']}",
        "model=gr00t_sidecar",
        "data=univtac_multimodal",
        *preset["overrides"],
        *overrides,
    ]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: bash scripts/train_univtac_sidecar.sh A|B|C|overfit5 [Hydra overrides]")
        return 2

    try:
        command = build_command(sys.argv[1], sys.argv[2:])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    print("Running:", shlex.join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
