# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect spconv dtypes and optionally run a Concerto two-step BF16/ZeRO-3 smoke test."""

import argparse
from importlib import metadata
import inspect
import logging
import os

from gr00t.model.extension.point_encoder import ConcertoPointEncoder
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-npy", help="One real UniVTAC XYZRGB scene as an [N, 6] .npy file")
    parser.add_argument("--checkpoint-path", help="Local official Concerto checkpoint (optional)")
    parser.add_argument(
        "--steps", type=int, default=2, help="Training steps with a diagnostic SGD optimizer"
    )
    parser.add_argument(
        "--zero3", action="store_true", help="Run under two-rank torchrun with BF16 ZeRO-3"
    )
    parser.add_argument(
        "--frozen-backbone", action="store_true", help="Also exercise the fused eval-bias path"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.zero3 and (not args.points_npy or not args.checkpoint_path):
        parser.error("--zero3 requires --points-npy and a local --checkpoint-path")

    import spconv
    import spconv.pytorch.cppcore as cppcore

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA: {torch.version.cuda}")
    print(f"spconv: {spconv.__version__}")
    print(f"spconv package: {metadata.packages_distributions().get('spconv', ['unknown'])}")
    print(f"spconv mapped torch dtypes: {[str(dtype) for dtype in cppcore._TORCH_DTYPE_TO_TV]}")
    print(f"FP16 mapped: {torch.float16 in cppcore._TORCH_DTYPE_TO_TV}")
    print(f"BF16 mapped: {torch.bfloat16 in cppcore._TORCH_DTYPE_TO_TV}")
    print(
        "outer CUDA autocast: "
        f"enabled={torch.is_autocast_enabled('cuda')}, "
        f"dtype={torch.get_autocast_dtype('cuda')}"
    )

    if not args.points_npy:
        return
    if not torch.cuda.is_available():
        parser.error("--points-npy requires an available CUDA device")
    if torch.float16 not in cppcore._TORCH_DTYPE_TO_TV:
        parser.error("the installed spconv build does not map torch.float16")

    points_array = np.load(args.points_npy, allow_pickle=False)
    if points_array.ndim != 2 or points_array.shape[1] != 6:
        parser.error("--points-npy must contain one [N, 6] XYZRGB scene")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    points = torch.as_tensor(points_array, device=f"cuda:{rank}", dtype=torch.float32).unsqueeze(0)
    if args.zero3:
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            parser.error("--zero3 requires torchrun --nproc-per-node=2")
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")

    # Match DeepSpeed's BF16 parameter storage without changing parameters in forward.
    encoder = ConcertoPointEncoder(
        input_dim=6,
        point_dim=1536,
        model_name="concerto_small",
        checkpoint_path=args.checkpoint_path,
    ).to("cuda", dtype=torch.bfloat16)
    encoder.train()
    encoder.backbone.requires_grad_(not args.frozen_backbone)
    if args.frozen_backbone:
        encoder.backbone.eval()
    parameter_ids = {name: id(parameter) for name, parameter in encoder.named_parameters()}
    # Diagnostic optimizer only; production training's optimizer is unchanged.
    optimizer = torch.optim.SGD([p for p in encoder.parameters() if p.requires_grad], lr=1e-4)
    runner = encoder
    if args.zero3:
        runner, optimizer, _, _ = deepspeed.initialize(
            model=encoder,
            optimizer=optimizer,
            config={
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "train_batch_size": 2,
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
                "zero_optimization": {"stage": 3},
                "zero_allow_untested_optimizer": True,
            },
        )
    os.environ["GROOT_DEBUG_CONCERTO_PRECISION"] = "1"
    logging.basicConfig(level=logging.INFO)

    import spconv.pytorch.ops as spconv_ops

    observed_dtypes = []
    original_implicit_gemm = spconv_ops.implicit_gemm
    sparse_parameter_ids = {
        id(parameter)
        for module in encoder.backbone.modules()
        if getattr(module, "_gr00t_sparse_precision", False)
        for parameter in module.parameters(recurse=False)
    }

    def record_implicit_gemm(features, filters, *args, **kwargs):
        arguments = (
            inspect.signature(original_implicit_gemm)
            .bind(features, filters, *args, **kwargs)
            .arguments
        )
        bias = arguments.get("bias")
        output = original_implicit_gemm(features, filters, *args, **kwargs)
        observed_dtypes.append(
            (
                features.dtype,
                filters.dtype,
                bias.dtype if bias is not None else None,
                output[0].dtype,
            )
        )
        return output

    try:
        # Scoped diagnostic observation only: never change dtype mappings or files.
        spconv_ops.implicit_gemm = record_implicit_gemm
        for step in range(args.steps):
            if not args.zero3:
                optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = runner(points)
                loss = (tokens.float()[..., 0] - 0.5).square().mean()
            if not torch.isfinite(tokens).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite Concerto output/loss")
            if args.zero3:
                runner.backward(loss)
                from deepspeed.utils import safe_get_full_grad
            else:
                loss.backward()
            sparse_gradients = 0
            # All ranks must participate in safe_get_full_grad collectives.
            for name, parameter in encoder.named_parameters():
                if id(parameter) != parameter_ids[name] or parameter.dtype != torch.bfloat16:
                    raise RuntimeError(f"Parameter identity or BF16 storage changed: {name}")
                if not parameter.requires_grad:
                    continue
                gradient = safe_get_full_grad(parameter) if args.zero3 else parameter.grad
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError(f"Non-finite gradient: {name}")
                    if id(parameter) in sparse_parameter_ids and gradient.abs().sum().item() > 0:
                        sparse_gradients += 1
            if not args.frozen_backbone and not sparse_gradients:
                raise RuntimeError("No nonzero sparse parameter gradients")
            if args.zero3:
                runner.step()
            else:
                optimizer.step()
            print(
                f"rank={rank} step={step + 1} loss={loss.item():.6g} finite_sparse_gradients={sparse_gradients}"
            )
    finally:
        spconv_ops.implicit_gemm = original_implicit_gemm

    print(f"Concerto input geometry: {points.dtype}; output tokens: {tokens.dtype}")
    print(
        f"spconv GEMM (features, weight, fused bias, output): {sorted(set(observed_dtypes), key=str)}"
    )
    if not observed_dtypes or any(
        feature != torch.float16
        or weight != torch.float16
        or bias not in (None, torch.float16)
        or output != torch.float16
        for feature, weight, bias, output in observed_dtypes
    ):
        raise RuntimeError(
            "spconv did not receive supported FP16 features/weights/bias or return FP16"
        )
    print(f"Concerto {args.steps}-step {'ZeRO-3' if args.zero3 else 'single-GPU'} smoke test: PASS")
    if args.zero3:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
