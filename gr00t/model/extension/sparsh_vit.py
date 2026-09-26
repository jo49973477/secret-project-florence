# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Minimal official Sparsh ViT-B/16 backbone used by Sparsh-DINO.

This is a dependency-light extraction of Meta's official implementation at
https://github.com/facebookresearch/sparsh (commit fee6a05).  Module and parameter
names intentionally match ``dino_vitbase.safetensors`` exactly.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_


def _create_ndgrid(resolution: list[int], device: torch.device) -> torch.Tensor:
    axes = [torch.arange(0, size, dtype=torch.float32, device=device) for size in resolution]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, len(resolution))


class SinusoidalEmbed(nn.Module):
    """Official Sparsh/DINO two-dimensional sinusoidal position embedding."""

    def __init__(self, size: tuple[int, int], patch_size: int, embed_dim: int) -> None:
        super().__init__()
        patch_grid_size = [side // patch_size for side in size]
        num_bands = math.ceil(embed_dim / (2 * len(size)))
        frequency_bands = torch.stack(
            [torch.linspace(0, 1.0, steps=num_bands + 1)[:-1] for _ in range(len(size))],
            dim=0,
        )
        frequency_bands = 10000**-frequency_bands
        self.embed_dim = embed_dim
        self.patches_resolution = patch_grid_size
        self.register_buffer("frequency_bands", frequency_bands)
        self.register_buffer("cached_encoding", None, persistent=False)

    def forward(self, device: torch.device) -> torch.Tensor:
        if self.cached_encoding is not None:
            return self.cached_encoding.to(device=device, non_blocking=True)
        grid = _create_ndgrid(self.patches_resolution, device=device)
        features = grid[..., None] * self.frequency_bands
        encoded = torch.cat((torch.sin(features), torch.cos(features)), dim=-1)
        self.cached_encoding = encoded.flatten(-2, -1)[..., : self.embed_dim]
        return self.cached_encoding


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (320, 240),
        patch_size: int = 16,
        in_chans: int = 6,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError(
                f"Sparsh input {(height, width)} must be divisible by patch size {self.patch_size}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class SparshLayerNorm(nn.LayerNorm):
    """Accumulate normalization in FP32 without changing checkpoint parameters.

    ZeRO/BF16 may store affine weights in BF16 while convolution/attention
    autocast and positional embeddings produce a different activation dtype.
    Cast the operation's inputs, not the registered Parameters, then return to
    the incoming activation dtype.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)).float().softmax(dim=-1)
        attention = attention.to(dtype=value.dtype)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(dtype=x.dtype)


class Mlp(nn.Module):
    def __init__(self, dim: int = 768, hidden_dim: int = 3072) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = SparshLayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=1.0)
        self.drop_path1 = nn.Identity()
        self.norm2 = SparshLayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim=dim, hidden_dim=dim * 4)
        self.ls2 = LayerScale(dim, init_values=1.0)
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class SparshVisionTransformer(nn.Module):
    """Exact backbone topology for ``facebook/sparsh-dino-base``."""

    img_size = (320, 240)
    patch_size = 16
    in_chans = 6
    embed_dim = 768
    num_register_tokens = 1

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.num_register_tokens, self.embed_dim)
        )
        self.pos_embed = SinusoidalEmbed(
            self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=12) for _ in range(12)])
        self.norm = SparshLayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        nn.init.normal_(self.register_tokens, std=1e-6)
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(f"Sparsh backbone expects [B,6,H,W], got {tuple(x.shape)}.")
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"Sparsh backbone expects spatial size {self.img_size}, got {tuple(x.shape[-2:])}."
            )
        x = self.patch_embed(x)
        # Generate the sinusoidal encoding in FP32, then join the BF16 compute
        # stream explicitly. A bare BF16 + FP32 add promotes the entire ViT to
        # FP32 and fails at LayerNorm with ZeRO's BF16 affine parameters.
        positions = self.pos_embed(x.device).float().to(dtype=x.dtype)
        x = x + positions.unsqueeze(0)
        register_tokens = self.register_tokens.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat((register_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {
            "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
            "x_prenorm": x,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["x_norm_patchtokens"]


__all__ = ["SparshLayerNorm", "SparshVisionTransformer"]
