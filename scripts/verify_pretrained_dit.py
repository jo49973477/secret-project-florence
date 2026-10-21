#!/usr/bin/env python3
"""Audit real local HF checkpoint tensors and optionally run real Action DiT parity."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys


# Support direct invocation from the repository without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.model.gr00t_n1d7.dit_integrity import (
    compare_shared_tensors,
    forward_parity,
    overall_status,
    validate_loading_info,
)
from gr00t.model.gr00t_n1d7.sensor_loading import (
    MULTIMODAL_ADAPTER_PREFIXES,
    SIDECAR_ADAPTER_PREFIXES,
)
import torch


class CheckpointTensors(Mapping):
    """Read individual safetensors; retain at most one PyTorch shard on CPU."""

    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path).resolve(strict=True)
        self.cache_name, self.cache = None, None
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = self.path / name
            if index.exists():
                self.weight_map = json.loads(index.read_text())["weight_map"]
                return
        for name in ("model.safetensors", "pytorch_model.bin"):
            file = self.path / name
            if file.exists():
                if name.endswith("safetensors"):
                    with safe_open(str(file), framework="pt", device="cpu") as stream:
                        keys = list(stream.keys())
                else:
                    keys = list(torch.load(file, map_location="cpu", weights_only=True, mmap=True))
                self.weight_map = dict.fromkeys(keys, name)
                return
        raise FileNotFoundError(f"No HF weight files in {path}")

    def __iter__(self):
        return iter(self.weight_map)

    def __len__(self):
        return len(self.weight_map)

    def __getitem__(self, key):
        from safetensors import safe_open

        name = self.weight_map[key]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Checkpoint shard outside checkpoint directory: {name}")
        # HF snapshot files legitimately symlink into the cache's blob directory.
        file = self.path / name
        if name.endswith("safetensors"):
            with safe_open(str(file), framework="pt", device="cpu") as stream:
                return stream.get_tensor(key)
        if self.cache_name != name:
            self.cache = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
            self.cache_name = name
        return self.cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--candidate-checkpoint",
        type=Path,
        help="Optional initialized candidate checkpoint; otherwise load Action DiT from reference",
    )
    source.add_argument(
        "--training-config",
        type=Path,
        help="Optional repository Config YAML: invoke the actual pipeline _create_model loading path (includes VLM/encoders)",
    )
    parser.add_argument(
        "--forward-parity",
        action="store_true",
        help="Run three real DiT forward cases, sequentially on --device",
    )
    parser.add_argument(
        "--dit-type",
        choices=("multimodal_conditioned_dit", "sidecar_conditioned_dit"),
        default="multimodal_conditioned_dit",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--report", type=Path, default=Path("pretrained_dit_report.json"))
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "forward_check": {
            "status": "SKIP",
            "reason": "Enable --forward-parity to run real DiT inputs.",
        },
    }
    try:
        from gr00t.model.extension.multimodal_dit import MultiModalConditionedDiT
        from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
        from gr00t.model.modules.dit import AlternateVLDiT

        adapter_class = (
            SidecarConditionedDiT
            if args.dit_type == "sidecar_conditioned_dit"
            else MultiModalConditionedDiT
        )
        adapter_prefixes = (
            SIDECAR_ADAPTER_PREFIXES
            if args.dit_type == "sidecar_conditioned_dit"
            else MULTIMODAL_ADAPTER_PREFIXES
        )
        reference = CheckpointTensors(args.reference_checkpoint)
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

        checkpoint_config = Gr00tN1d7Config.from_pretrained(
            args.reference_checkpoint, local_files_only=True
        )
        if checkpoint_config.dit_type not in {"alternate_vl_dit", "multimodal_conditioned_dit"}:
            raise ValueError("Reference must use AlternateVLDiT or its multimodal extension")
        config = dict(checkpoint_config.diffusion_model_cfg)
        if isinstance(config.get("compute_dtype"), str):
            config["compute_dtype"] = getattr(torch, config["compute_dtype"].removeprefix("torch."))
        config["cross_attention_dim"] = checkpoint_config.backbone_embedding_dim
        config["attend_text_every_n_blocks"] = checkpoint_config.attend_text_every_n_blocks
        prefix = "action_head.model."
        dtype = getattr(torch, args.dtype)
        with torch.device("meta"):
            schema = AlternateVLDiT(**config)
        expected = [prefix + k for k in schema.state_dict()]
        report["expected_original_parameter_tensors"] = len(list(schema.named_parameters()))
        report["architecture"] = {"class": "MultiModalConditionedDiT", "configuration": config}
        report["skip_weight_loading"] = False
        runtime_model = None
        if args.training_config:
            from gr00t.configs.base_config import Config
            from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

            cfg = Config()
            cfg.load(args.training_config)
            report["skip_weight_loading"] = cfg.training.skip_weight_loading
            report["effective_checkpoint"] = str(cfg.training.start_from_checkpoint)
            runtime_model = Gr00tN1d7Pipeline(cfg, args.report.parent)._create_model()
            candidate_dit = runtime_model.action_head.model
            candidate = {prefix + k: v for k, v in candidate_dit.state_dict().items()}
            report["loading_check"] = getattr(
                runtime_model,
                "pretrained_loading_report",
                {
                    "status": "FAIL",
                    "reason": "No pretrained loading report; weights may have been skipped.",
                },
            )
            report["sensor_loading"] = report["loading_check"].get(
                "sensors", {"status": "SKIP", "reason": "No sensor loading report"}
            )
            report["architecture"] = {
                "class": type(candidate_dit).__name__,
                "configuration": dict(candidate_dit.config),
            }
        elif args.candidate_checkpoint:
            report["skip_weight_loading"] = None
            report["candidate_checkpoint"] = str(args.candidate_checkpoint.resolve())
            candidate = CheckpointTensors(args.candidate_checkpoint)
            candidate_dit = None
            report["loading_check"] = {
                "status": "SKIP",
                "reason": "Saved-file comparison does not exercise the training loader.",
            }
            report["sensor_loading"] = {
                "status": "SKIP",
                "reason": "Encoders excluded from Action DiT audit.",
            }
        else:
            candidate_dit = adapter_class(**config).to(dtype=dtype)
            candidate_state = candidate_dit.state_dict()
            shared, mismatched = {}, []
            source_keys = [k for k in reference if k.startswith(prefix)]
            for key in source_keys:
                local_key = key.removeprefix(prefix)
                value = reference[key]
                if local_key in candidate_state and value.shape != candidate_state[local_key].shape:
                    mismatched.append(
                        {
                            "name": key,
                            "checkpoint_shape": list(value.shape),
                            "model_shape": list(candidate_state[local_key].shape),
                        }
                    )
                else:
                    shared[local_key] = value
            info = candidate_dit.load_state_dict(shared, strict=False)
            loading = {
                "missing_keys": [prefix + k for k in info.missing_keys],
                "unexpected_keys": [prefix + k for k in info.unexpected_keys],
                "mismatched_keys": mismatched,
            }
            report["loading_check"] = validate_loading_info(loading, adapter_prefixes)
            report["loading_scope"] = (
                "Action DiT only: same architecture and key validation as training; no VLM or sensor encoder allocation. Use --training-config to audit full pipeline."
            )
            report["sensor_loading"] = {
                "status": "SKIP",
                "reason": "Action-DiT-only mode intentionally does not initialize encoders.",
            }
            candidate = {prefix + k: v for k, v in candidate_dit.state_dict().items()}
            del shared
        if candidate_dit is not None:
            report["architecture"] = {
                "class": type(candidate_dit).__name__,
                "configuration": dict(candidate_dit.config),
                "attend_text_every_n_blocks": candidate_dit.attend_text_every_n_blocks,
            }
        report["weight_check"] = compare_shared_tensors(
            reference, candidate, expected_keys=expected, atol=args.atol, rtol=args.rtol
        )
        if args.forward_parity and report["weight_check"]["status"] != "PASS":
            report["forward_check"] = {
                "status": "SKIP",
                "reason": "Weight integrity failed; parity on incomplete weights is not meaningful.",
            }
        elif (
            args.forward_parity
            and str(args.device).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            report["forward_check"] = {
                "status": "SKIP",
                "reason": "CUDA requested but no CUDA device is available.",
            }
        elif args.forward_parity:
            if candidate_dit is None:
                candidate_dit = adapter_class(**config).to(dtype=dtype)
                candidate_dit.load_state_dict(
                    {
                        k.removeprefix(prefix): candidate[k]
                        for k in candidate
                        if k.startswith(prefix)
                    },
                    strict=True,
                )
            try:
                original = AlternateVLDiT(**config).to(dtype=dtype)
                original.load_state_dict(
                    {
                        k.removeprefix(prefix): reference[k].to(dtype=candidate[k].dtype)
                        for k in expected
                    },
                    strict=True,
                )
                report["forward_check"] = forward_parity(
                    original, candidate_dit, config, args.device, dtype
                )
            except torch.OutOfMemoryError as exc:
                report["forward_check"] = {
                    "status": "SKIP",
                    "reason": f"Insufficient memory: {exc}",
                }
        statuses = [report[k]["status"] for k in ("weight_check", "loading_check", "forward_check")]
        if report["skip_weight_loading"]:
            statuses.append("FAIL")
        report["overall"] = overall_status(statuses)
    except Exception as exc:
        report["overall"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    print("════════ Pretrained DiT Integrity Check ════════")
    print(f"Checkpoint: {report['reference_checkpoint']}")
    weights = report.get("weight_check", {})
    for key in (
        "expected_tensors",
        "matched_tensors",
        "coverage",
        "max_abs_error",
        "max_relative_error",
    ):
        print(f"{key}: {weights.get(key, 'N/A')}")
    print(
        f"Missing: {len(weights.get('missing', []))}  Mismatched: {len(weights.get('mismatched', []))}"
    )
    print(
        f"Weights: {weights.get('status', 'FAIL')}  Zero gate: {report['forward_check']['status']}"
    )
    print(f"Overall: {report['overall']}\nJSON report: {args.report}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["overall"]]


if __name__ == "__main__":
    raise SystemExit(main())
