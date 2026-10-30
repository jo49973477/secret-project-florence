#!/usr/bin/env python3
"""Fail-closed, CPU-safe Florence checks before any distributed training launch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.experiment.hydra_launch import validate_dataset_inputs
from gr00t.model.extension.dit_lora import DiTLoRALinear
from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
from gr00t.model.extension.temporal_fusion import LatentRingBuffer, TemporalFusion


def synthetic_contracts(mode: str, window: int):
    """Behavioral checks; never presented as actual checkpoint proof."""
    torch.manual_seed(71)
    baseline = torch.nn.Linear(7, 7)
    adapter = DiTLoRALinear(baseline, 2, 4, 0.0)
    x = torch.randn(3, 7, requires_grad=True)
    before = baseline(x).detach()
    parity = torch.equal(adapter(x).detach(), before)
    base_weight = adapter.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=0.01)
    loss = adapter(x).square().mean()
    loss.backward()
    gradient_finite = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()
    )
    optimizer.step()
    lora_updated = not torch.equal(
        adapter.lora_B.weight.detach(), torch.zeros_like(adapter.lora_B.weight)
    )
    frozen_gradient = x.grad is not None and bool(x.grad.abs().sum() > 0)
    base_unchanged = torch.equal(adapter.weight.detach(), base_weight)
    full = torch.nn.Linear(7, 7)
    full.load_state_dict(baseline.state_dict(), strict=True)
    full_initialization = all(
        torch.equal(a, b) for a, b in zip(full.parameters(), baseline.parameters())
    )
    full_before = full.weight.detach().clone()
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=0.01)
    full(x.detach()).square().mean().backward()
    full_optimizer.step()
    full_updated = not torch.equal(full.weight.detach(), full_before)
    results = {
        "scope": "SYNTHETIC ONLY",
        "lora_initial_output_parity": parity,
        "lora_base_weights_unchanged": base_unchanged,
        "lora_optimizer_updates": lora_updated,
        "frozen_gradient_flow": frozen_gradient,
        "full_initialization": full_initialization,
        "full_optimizer_updates": full_updated,
        "gradients_finite": gradient_finite,
    }
    sidecar = SidecarConditionedDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=6,
        sidecar_num_stages=2,
        sidecar_hidden_dim=8,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        tactile_fusion_mode="none",
        freeze_pretrained_dit=True,
        dit_training_mode="frozen",
    )
    sidecar_output = sidecar(
        hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 4, 6),
        timestep=torch.tensor([7]),
        image_mask=torch.tensor([[False, False, True, True]]),
        backbone_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        point_tokens=torch.randn(1, 5, 8),
        point_attention_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    sidecar_output.square().mean().backward()
    results["frozen_dit_to_sidecar_gradient"] = all(
        p.grad is None for p in sidecar.transformer_blocks.parameters()
    ) and any(
        stage.action_gate.grad is not None and bool(stage.action_gate.grad.abs().sum() > 0)
        for stage in sidecar.sidecar_stages
    )
    if window:
        fusion = TemporalFusion(8, 3, window)
        points = torch.randn(2, window, 4, 8, requires_grad=True)
        tactile = torch.randn(2, window, 2, 8, requires_grad=True)
        state = torch.randn(2, window, 3, requires_grad=True)
        valid = torch.ones(2, window, dtype=torch.bool)
        valid[0, :-2] = False
        output, mask = fusion(points, tactile, state, valid)
        output.sum().backward()
        results["temporal_shapes_masks"] = output.shape == (2, window, 8) and torch.equal(
            mask, valid
        )
        results["temporal_gradients"] = points.grad is not None and bool(
            torch.isfinite(points.grad).all()
        )
        ring = LatentRingBuffer(window)
        for index in range(window + 2):
            ring.push("env0", torch.tensor([float(index)]))
        values, mask = ring.window("env0", torch.zeros(1))
        results["ring_wraparound"] = torch.equal(
            values[:, 0], torch.arange(2, window + 2).float()
        ) and bool(mask.all())
        ring.reset("env0")
        _, mask = ring.window("env0", torch.zeros(1))
        results["ring_reset"] = not bool(mask.any())
    return results


def run(cfg, config, report_path: Path = Path("outputs/preflight/florence_preflight.json")):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAILED", "checks": {}, "synthetic": {}}
    try:
        if not config.training.start_from_checkpoint:
            raise ValueError("Pretrained checkpoint is required")
        if config.training.save_steps < 1 or config.training.max_steps < 1:
            raise ValueError("save_steps and max_steps must be positive")
        validate_dataset_inputs(cfg)
        report["checks"]["resolved_hydra_and_dataset"] = "PASSED"
        output = Path(cfg.output_dir)
        if not cfg.resume.enabled and output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}")
        report["checks"]["output_path"] = "PASSED"
        report["synthetic"] = synthetic_contracts(
            config.model.dit_training_mode, config.model.temporal_window_size
        )
        if not all(value for key, value in report["synthetic"].items() if key != "scope"):
            raise RuntimeError("Synthetic training contracts failed")
        report["checks"]["synthetic_contracts"] = "PASSED"
        verifier = Path(__file__).with_name("verify_pretrained_dit.py")
        dit_report = report_path.parent / "dit_verification.json"
        command = [
            sys.executable,
            str(verifier),
            "--reference-checkpoint",
            str(config.training.start_from_checkpoint),
            "--dit-type",
            "sidecar_conditioned_dit",
            "--dit-training-mode",
            config.model.dit_training_mode,
            "--lora-rank",
            str(config.model.dit_lora_rank),
            "--lora-alpha",
            str(config.model.dit_lora_alpha),
            "--dtype",
            "bfloat16" if config.training.bf16 else "float32",
            "--forward-parity",
            "--report",
            str(dit_report),
        ]
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env={**os.environ, "NO_ALBUMENTATIONS_UPDATE": "1"},
        )
        report["dit_verifier_stdout"] = proc.stdout[-5000:]
        report["dit_verifier_stderr"] = proc.stderr[-5000:]
        if not dit_report.is_file():
            raise RuntimeError("DiT verifier did not write a report")
        actual = json.loads(dit_report.read_text())
        report["actual_pretrained_checkpoint"] = {
            "checkpoint": actual.get("reference_checkpoint"),
            "revision": actual.get("checkpoint_revision"),
            "weights": actual.get("weight_check", {}),
            "forward": actual.get("forward_check", {}),
            "lora_parameter_count": actual.get("lora_parameter_count"),
        }
        if proc.returncode or actual.get("overall") != "PASS":
            raise RuntimeError(
                f"Mandatory actual pretrained DiT verification failed: {actual.get('overall')}"
            )
        report["checks"]["actual_pretrained_dit"] = "PASSED"
        report["checks"]["runtime_training_model"] = "NOT RUN"
        report["checks"]["optimizer_groups"] = "NOT RUN"
        report["checks"]["gpu_distributed_smoke"] = (
            "NOT RUN" if torch.cuda.is_available() else "BLOCKED"
        )
        report["experiment_ready"] = False
        report["status"] = "PASSED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/preflight/florence_preflight.json")
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.resolved_config)
    config = build_gr00t_config(cfg)
    report = run(cfg, config, args.report)
    print(f"Florence preflight: {report['status']} ({args.report})", flush=True)
    if report["status"] != "PASSED":
        print(report.get("error"), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
