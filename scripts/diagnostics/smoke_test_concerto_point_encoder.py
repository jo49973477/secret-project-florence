#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Load an official Concerto checkpoint and run one GR00T-adapter batch."""

from __future__ import annotations

import argparse

from gr00t.model.extension.point_encoder import ConcertoPointEncoder
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("concerto_small", "concerto_base"),
        default="concerto_small",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Optional local official .pth; otherwise use the official Hugging Face repository.",
    )
    parser.add_argument("--repo-id", default="Pointcept/Concerto")
    parser.add_argument("--download-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--dit-dim", type=int, default=1536)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument(
        "--enable-flash",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the official checkpoint setting only when explicitly supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_points <= 0 or args.dit_dim <= 0:
        raise SystemExit("batch-size, num-points, and dit-dim must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested but CUDA is unavailable: {device}")

    torch.manual_seed(0)
    encoder = ConcertoPointEncoder(
        input_dim=6,
        point_dim=args.dit_dim,
        model_name=args.model,
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint_path,
        download_root=args.download_root,
        grid_size=args.grid_size,
        enable_flash=args.enable_flash,
    ).to(device)
    encoder.eval()

    xyz = torch.rand(args.batch_size, args.num_points, 3, device=device)
    rgb = torch.randint(
        0,
        256,
        (args.batch_size, args.num_points, 3),
        device=device,
        dtype=torch.int64,
    ).float()
    points = torch.cat((xyz, rgb), dim=-1)

    with torch.inference_mode():
        tokens, mask = encoder(points, return_mask=True)

    if tokens.ndim != 3 or tokens.shape[0] != args.batch_size or tokens.shape[-1] != args.dit_dim:
        raise RuntimeError(f"Unexpected Concerto token shape: {tuple(tokens.shape)}")
    if mask.shape != tokens.shape[:2] or not mask.any(dim=1).all():
        raise RuntimeError(f"Unexpected Concerto attention mask: {tuple(mask.shape)}")
    if not torch.isfinite(tokens).all():
        raise RuntimeError("Concerto produced NaN or Inf tokens")

    first_parameter = next(encoder.backbone.parameters()).detach().float()
    print("Concerto checkpoint smoke test: PASS")
    print(f"model: {args.model}")
    print(f"checkpoint: {args.checkpoint_path or args.repo_id}")
    print(f"input: {tuple(points.shape)} {points.dtype} {points.device}")
    print(f"tokens: {tuple(tokens.shape)} {tokens.dtype}")
    print(f"attention mask: {tuple(mask.shape)} valid={int(mask.sum())}")
    print(
        "pretrained parameter fingerprint: "
        f"mean={first_parameter.mean().item():.8g} "
        f"std={first_parameter.std().item():.8g} "
        f"nonzero={int(torch.count_nonzero(first_parameter))}"
    )


if __name__ == "__main__":
    main()
