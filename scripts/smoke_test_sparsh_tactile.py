#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one CUDA forward/backward batch through the real Sparsh tactile branch."""

import argparse

from gr00t.model.extension.tactile_encoder import SparshDinoTactileEncoder
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-model",
        default="facebook/sparsh-dino-base",
        help="Hugging Face ID, local directory, or checkpoint file.",
    )
    parser.add_argument("--dit-dim", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA.")
    device = torch.device("cuda")
    encoder = SparshDinoTactileEncoder(
        output_dim=args.dit_dim,
        pretrained_model=args.pretrained_model,
    ).to(device=device, dtype=torch.bfloat16)
    encoder.train()
    tactile = torch.randint(
        0,
        256,
        (args.batch_size, 2, 3, 240, 320),
        dtype=torch.uint8,
        device=device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = encoder(tactile)
        loss = tokens.float().square().mean()
    loss.backward()
    if not torch.isfinite(tokens).all() or not torch.isfinite(loss):
        raise RuntimeError("Sparsh BF16 smoke test produced non-finite tokens or loss.")
    for name, parameter in (
        ("patch embedding", encoder.backbone.patch_embed.proj.weight),
        ("first attention QKV", encoder.backbone.blocks[0].attn.qkv.weight),
        ("projection", encoder.projection[1].weight),
    ):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Sparsh BF16 smoke test did not produce a finite {name} gradient.")
    print(
        f"PASS input={tuple(tactile.shape)} tokens={tuple(tokens.shape)} "
        f"dtype={tokens.dtype} loss={loss.item():.6g}"
    )


if __name__ == "__main__":
    main()
