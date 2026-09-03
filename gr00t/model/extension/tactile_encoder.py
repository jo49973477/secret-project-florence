# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ResNet-based tactile image encoder."""

from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class TactileEncoder(nn.Module):
    """Convert a tactile image into ResNet-18 spatial feature tokens.

    The classifier and global average pooling are omitted, so a 224 x 224
    image produces a 7 x 7 feature map and therefore 49 tactile tokens.

    Input:
        tactile_images: ``[B, input_channels, H, W]``.

    Output:
        tactile_tokens: ``[B, H' * W', tactile_dim]``.
    """

    def __init__(
        self,
        input_channels: int = 3,
        token_dim: Optional[int] = None,
        patch_size: Optional[int] = None,
        dropout: float = 0.0,
        pretrained: bool = False,
        tactile_dim: int = 256,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")

        # token_dim and patch_size preserve the old constructor surface.
        # ResNet controls the spatial stride, so patch_size is no longer used.
        del patch_size
        if token_dim is not None:
            tactile_dim = token_dim

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        if input_channels != 3:
            original_convolution = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                input_channels,
                original_convolution.out_channels,
                kernel_size=original_convolution.kernel_size,
                stride=original_convolution.stride,
                padding=original_convolution.padding,
                bias=False,
            )
            if pretrained:
                self._initialize_input_convolution(resnet.conv1, original_convolution.weight)

        # Keep conv1 through layer4; omit avgpool and fc.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.input_channels = input_channels
        self.tactile_dim = tactile_dim
        self.output_projection = nn.Linear(512, tactile_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(tactile_dim)

    @staticmethod
    def _initialize_input_convolution(
        convolution: nn.Conv2d, pretrained_weight: torch.Tensor
    ) -> None:
        """Adapt ImageNet RGB weights to a different tactile channel count."""
        with torch.no_grad():
            if convolution.in_channels == 1:
                convolution.weight.copy_(pretrained_weight.mean(dim=1, keepdim=True))
            else:
                repeated_weight = pretrained_weight.repeat(
                    1, (convolution.in_channels + 2) // 3, 1, 1
                )[:, : convolution.in_channels]
                repeated_weight.mul_(3.0 / convolution.in_channels)
                convolution.weight.copy_(repeated_weight)

    def forward(self, tactile_images: torch.Tensor) -> torch.Tensor:
        if tactile_images.ndim != 4:
            raise ValueError(
                f"tactile_images must have shape [B, C, H, W], got {tuple(tactile_images.shape)}."
            )
        if tactile_images.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} tactile channels, got {tactile_images.shape[1]}."
            )
        if tactile_images.shape[-2] == 0 or tactile_images.shape[-1] == 0:
            raise ValueError("tactile_images must have non-empty height and width dimensions.")

        # spatial_features: [B, 512, H', W']
        spatial_features = self.backbone(tactile_images)

        # tactile_tokens: [B, H' * W', tactile_dim]
        tactile_tokens = spatial_features.flatten(start_dim=2).transpose(1, 2)
        tactile_tokens = self.output_projection(tactile_tokens)
        tactile_tokens = self.output_dropout(tactile_tokens)
        tactile_tokens = self.output_norm(tactile_tokens)
        return tactile_tokens


__all__ = ["TactileEncoder"]
