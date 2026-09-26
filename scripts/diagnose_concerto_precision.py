# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the installed spconv dtype boundary and optionally run one Concerto batch."""

import argparse
from importlib import metadata
import logging
import os

from gr00t.model.extension.point_encoder import ConcertoPointEncoder
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-npy", help="One real UniVTAC XYZRGB scene as an [N, 6] .npy file")
    parser.add_argument("--checkpoint-path", help="Local official Concerto checkpoint (optional)")
    args = parser.parse_args()

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
    points = torch.as_tensor(points_array, device="cuda", dtype=torch.bfloat16).unsqueeze(0)

    # Match DeepSpeed's BF16 parameter storage without changing parameters in forward.
    encoder = ConcertoPointEncoder(
        input_dim=6,
        point_dim=1536,
        model_name="concerto_small",
        checkpoint_path=args.checkpoint_path,
    ).to("cuda", dtype=torch.bfloat16)
    encoder.train()
    os.environ["GROOT_DEBUG_CONCERTO_PRECISION"] = "1"
    logging.basicConfig(level=logging.INFO)

    import spconv.pytorch.ops as spconv_ops

    observed_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
    original_implicit_gemm = spconv_ops.implicit_gemm

    def record_implicit_gemm(features, filters, *args, **kwargs):
        observed_dtypes.append((features.dtype, filters.dtype))
        return original_implicit_gemm(features, filters, *args, **kwargs)

    try:
        # This local diagnostic spy observes the actual tensors entering spconv's
        # operation; it is removed before the process exits.
        spconv_ops.implicit_gemm = record_implicit_gemm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(points)
            loss = tokens.float().square().mean()
        loss.backward()
    finally:
        spconv_ops.implicit_gemm = original_implicit_gemm

    sparse_gradients = [
        name
        for name, parameter in encoder.backbone.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum().item() > 0
    ]
    print(f"Concerto input: {points.dtype}; output tokens: {tokens.dtype}")
    print(f"spconv implicit GEMM feature/weight dtypes: {sorted(set(observed_dtypes), key=str)}")
    print(f"finite output: {bool(torch.isfinite(tokens).all())}")
    print(f"nonzero Concerto parameter gradients: {len(sparse_gradients)}")
    if not observed_dtypes or any(
        pair != (torch.float16, torch.float16) for pair in observed_dtypes
    ):
        raise RuntimeError("spconv implicit GEMM did not receive FP16 features and weights")
    if not torch.isfinite(tokens).all() or not sparse_gradients:
        raise RuntimeError("Concerto smoke test did not produce finite output and gradients")


if __name__ == "__main__":
    main()
