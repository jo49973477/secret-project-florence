#!/usr/bin/env python3
"""Intentionally consolidate a native DeepSpeed ZeRO checkpoint for inference.

This operation materializes the full FP32 state dict in host RAM. Run it outside
the training job on a host with enough memory for the complete model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Native checkpoint-N directory")
    parser.add_argument("output", type=Path, help="New inference checkpoint directory")
    parser.add_argument(
        "--model-config-dir",
        required=True,
        type=Path,
        help="Original HF model directory containing config.json and model code",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    model_config = args.model_config_dir.resolve()
    if not checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {checkpoint}")
    if not (model_config / "config.json").is_file():
        parser.error(f"--model-config-dir must contain config.json: {model_config}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        from deepspeed.utils.zero_to_fp32 import (
            convert_zero_checkpoint_to_fp32_state_dict,
        )
    except ImportError as exc:
        raise SystemExit("Install the repository's DeepSpeed dependency to export ZeRO-3.") from exc

    # Writes the standard HF pytorch_model.bin from the rank-partitioned shards.
    convert_zero_checkpoint_to_fp32_state_dict(
        str(checkpoint), str(output / "pytorch_model.bin")
    )
    for item in model_config.iterdir():
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        elif not (
            item.name.endswith((".bin", ".safetensors"))
            or item.name.startswith(("pytorch_model-", "model-"))
        ):
            shutil.copy2(item, destination)

    # Keep the processor/config artifacts saved with the training run.
    run_directory = checkpoint.parent
    processor = checkpoint / "processor"
    if not processor.is_dir():
        processor = run_directory / "processor"
    if processor.is_dir():
        shutil.copytree(processor, output / "processor", dirs_exist_ok=True)
    experiment_cfg = checkpoint / "experiment_cfg"
    if not experiment_cfg.is_dir():
        experiment_cfg = run_directory / "experiment_cfg"
    if experiment_cfg.is_dir():
        shutil.copytree(experiment_cfg, output / "experiment_cfg", dirs_exist_ok=True)
    print(f"Inference checkpoint written to {output}")


if __name__ == "__main__":
    main()
