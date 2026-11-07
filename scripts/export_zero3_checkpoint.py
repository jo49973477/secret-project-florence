#!/usr/bin/env python3
"""Export trusted native ZeRO-3 checkpoints; --inspect never consolidates weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil


def artifact(checkpoint: Path, name: str) -> Path:
    for root in (checkpoint, checkpoint.parent):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing training artifact: {name}")


def export_config(checkpoint: Path, base: Path | None = None) -> dict:
    # This is written AFTER pretrained loading, architecture conversion, and LoRA insertion.
    # Raw Hydra model defaults are insufficient (e.g. select_layer and DiT dimensions).
    resolved = artifact(checkpoint, "experiment_cfg/final_model_config.json")
    config = json.loads(resolved.read_text())
    if base:
        original = json.loads((base / "config.json").read_text())
        original.update(config)
        config = original
    if config.get("dit_type") != "sidecar_conditioned_dit":
        raise ValueError("Expected resolved Sidecar configuration")
    for key in ("diffusion_model_cfg", "dit_training_mode", "contact_input_mode"):
        if key not in config:
            raise ValueError(f"Resolved training configuration lacks {key}")
    return config


def inspect_checkpoint(checkpoint: Path) -> dict:
    import torch

    tag = (checkpoint / "latest").read_text().strip()
    if not tag or Path(tag).name != tag:
        raise ValueError("Invalid DeepSpeed latest tag")
    directory = checkpoint / tag
    models = sorted(directory.glob("*_model_states.pt"))
    optimizers = sorted(directory.glob("*_optim_states.pt"))
    if not models or not optimizers:
        raise ValueError("Missing DeepSpeed model/optimizer shards")
    shapes = None
    names = set()
    partition_sizes = []
    for rank, path in enumerate(optimizers):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        optim = state["optimizer_state_dict"]
        if optim["zero_stage"] != 3 or optim["partition_count"] != len(optimizers):
            raise ValueError("Unsupported stage or incomplete partition count")
        if not optim.get("fp32_flat_groups"):
            raise ValueError("Missing FP32 trainable partitions")
        partition_sizes.append([group.numel() for group in optim["fp32_flat_groups"]])
        if not path.name.endswith(f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"):
            raise ValueError("Only contiguous data-parallel ranks with mp_rank_00 are supported")
    if len(models) != len(optimizers):
        raise ValueError("Incomplete model shards")
    for rank, path in enumerate(models):
        if path.name != f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt":
            raise ValueError("Unexpected model rank layout")
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        expected_sizes = [
            sum(math.ceil(shape.numel() / len(models)) for shape in group.values())
            for group in state["param_shapes"]
        ]
        if partition_sizes[rank] != expected_sizes:
            raise ValueError("Trainable partition sizes do not match parameter shapes")
        current = {k: tuple(v) for group in state["param_shapes"] for k, v in group.items()}
        frozen = state.get("frozen_param_shapes", {})
        current.update({k: tuple(v) for k, v in frozen.items()})
        if frozen and set(frozen) != set(state.get("frozen_param_fragments", {})):
            raise ValueError("Frozen parameter fragments missing; cannot export complete model")
        for name, shape in frozen.items():
            if state["frozen_param_fragments"][name].numel() != math.ceil(
                shape.numel() / len(models)
            ):
                raise ValueError(f"Incomplete frozen fragment: {name}")
        for name in state.get("buffer_names", []):
            if name not in state.get("module", {}):
                raise ValueError(f"Missing persistent buffer: {name}")
        if shapes is not None and current != shapes:
            raise ValueError("Rank parameter layouts differ")
        shapes = current
        names.update(current)
    return {
        "tag": tag,
        "partitions": len(models),
        "parameters": len(names),
        "sidecar_parameters": sum("sidecar_" in n for n in names),
        "lora_parameters": sum(".lora_" in n for n in names),
        "moe_parameters": sum("input_contact_adapter" in n for n in names),
    }


def validate_loading(info: dict) -> None:
    errors = {
        k: info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(k)
    }
    if errors:
        raise RuntimeError(f"Incomplete inference weight restoration: {errors}")


def compare_tensors(restored: dict, native: dict, names: list[str]) -> None:
    import torch

    for name in names:
        if name not in native:
            raise RuntimeError(f"Native checkpoint lacks restored tensor: {name}")
        expected = native[name].contiguous().to(restored[name].dtype)
        if not torch.equal(expected, restored[name].cpu()):
            raise RuntimeError(f"Restored tensor differs from native checkpoint: {name}")


def verify_export(output: Path, source: Path | None = None) -> dict:
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model, info = Gr00tN1d7.from_pretrained(str(output), output_loading_info=True)
    validate_loading(info)
    restored = model.state_dict()
    groups = {
        "sidecar": [n for n in restored if "sidecar_" in n],
        "lora": [n for n in restored if ".lora_" in n],
        "moe": [n for n in restored if "input_contact_adapter" in n],
    }
    config = json.loads((output / "config.json").read_text())
    for group, required in (
        ("sidecar", True),
        ("lora", config["dit_training_mode"] == "lora"),
        ("moe", config["contact_input_mode"].startswith("moe")),
    ):
        if required and not groups[group]:
            raise RuntimeError(f"No restored {group} parameters")
    if source:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        native = get_fp32_state_dict_from_zero_checkpoint(str(source), lazy_mode=True)
        for names in groups.values():
            compare_tensors(restored, native, names)
    report = {
        "status": "PASS",
        "restored": {k: len(v) for k, v in groups.items()},
        "native_tensor_comparison": source is not None,
    }
    (output / "export_verification.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        help="Optional original pretrained config; resolved training config takes precedence",
    )
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if args.verify_only and args.output is None:
        parser.error("output is required for verification")
    if args.verify_only:
        print(verify_export(args.output.resolve(), checkpoint))
        return
    config = export_config(checkpoint, args.model_config_dir)
    report = inspect_checkpoint(checkpoint)
    print(json.dumps(report, indent=2))
    if args.inspect:
        return
    if args.output is None:
        parser.error("output is required for export")
    output = args.output.resolve()
    if output == checkpoint or checkpoint in output.parents or output.exists():
        parser.error("Output must be a new directory outside the original checkpoint")
    # Check all inference artifacts before the expensive operation.
    artifacts = {
        name: artifact(checkpoint, name)
        for name in (
            "experiment_cfg",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        )
    }
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    # DeepSpeed 0.17 takes an OUTPUT DIRECTORY, not a pytorch_model.bin filename.
    # Frozen fragments must be included: most of this model is frozen during training.
    convert_zero_checkpoint_to_fp32_state_dict(
        str(checkpoint), str(output), safe_serialization=False, exclude_frozen_parameters=False
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for name, path in artifacts.items():
        if path.is_dir():
            shutil.copytree(path, output / name)
        else:
            shutil.copy2(path, output / name)
    for name in ("processor", "metadata.json"):
        try:
            path = artifact(checkpoint, name)
        except FileNotFoundError:
            continue
        if path.is_dir():
            shutil.copytree(path, output / name)
        else:
            shutil.copy2(path, output / name)
    (output / "export_manifest.json").write_text(
        json.dumps({"source": str(checkpoint), **report}, indent=2)
    )
    print(verify_export(output, checkpoint))


if __name__ == "__main__":
    main()
