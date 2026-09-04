# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight point-cloud encoders for multimodal action conditioning."""

from typing import Optional

import torch
from torch import nn


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, ...]`` point indices from a ``[B, N, C]`` tensor."""
    batch_indices = torch.arange(points.shape[0], device=points.device)
    batch_indices = batch_indices.view(-1, *([1] * (indices.ndim - 1)))
    return points[batch_indices, indices]


def _farthest_point_sample(
    xyz: torch.Tensor,
    num_samples: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select representative points with deterministic farthest-point sampling."""
    batch_size, num_points, _ = xyz.shape
    num_samples = min(num_samples, num_points)

    sampled_indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=xyz.device)
    sampled_mask = torch.zeros(batch_size, num_samples, dtype=torch.bool, device=xyz.device)
    min_distances = torch.full(
        (batch_size, num_points), torch.inf, dtype=xyz.dtype, device=xyz.device
    )
    selected_mask = torch.zeros_like(valid_mask)
    valid_counts = valid_mask.sum(dim=1)

    # Start with the first valid point. Rows without a valid point use index 0
    # as a harmless placeholder and remain False in sampled_mask.
    farthest_indices = valid_mask.to(dtype=torch.int64).argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=xyz.device)

    for sample_index in range(num_samples):
        sample_is_valid = sample_index < valid_counts
        sampled_indices[:, sample_index] = farthest_indices
        sampled_mask[:, sample_index] = sample_is_valid
        selected_mask[batch_indices, farthest_indices] |= sample_is_valid

        centroids = xyz[batch_indices, farthest_indices]
        squared_distances = ((xyz - centroids[:, None, :]) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, squared_distances)

        candidate_scores = min_distances.masked_fill(~valid_mask | selected_mask, -1.0)
        farthest_indices = candidate_scores.argmax(dim=1)

    return sampled_indices, sampled_mask


class SetAbstraction(nn.Module):
    """One PointNet++ set-abstraction stage using k-nearest neighborhoods."""

    def __init__(
        self,
        num_samples: int,
        num_neighbors: int,
        input_feature_dim: int,
        mlp_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if num_samples <= 0 or num_neighbors <= 0:
            raise ValueError("num_samples and num_neighbors must be positive.")

        layers: list[nn.Module] = []
        input_channels = input_feature_dim + 3  # point features plus relative XYZ
        for output_channels in mlp_channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            input_channels = output_channels

        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.shared_mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample centroids and aggregate their local neighborhoods."""
        centroid_indices, centroid_mask = _farthest_point_sample(xyz, self.num_samples, valid_mask)
        centroid_xyz = _index_points(xyz, centroid_indices)

        # squared_distances: [B, N_centroid, N_input]
        squared_distances = ((centroid_xyz[:, :, None] - xyz[:, None, :]) ** 2).sum(dim=-1)
        squared_distances = squared_distances.masked_fill(~valid_mask[:, None, :], torch.inf)

        num_neighbors = min(self.num_neighbors, xyz.shape[1])
        neighbor_distances, neighbor_indices = squared_distances.topk(
            k=num_neighbors, dim=-1, largest=False, sorted=False
        )
        neighbor_mask = torch.isfinite(neighbor_distances) & centroid_mask[:, :, None]

        # topk fills short masked neighborhoods with arbitrary invalid indices.
        # Repeat a valid neighbor instead so padded point values do not enter
        # the shared MLP or its batch-normalization statistics.
        fallback_indices = squared_distances.argmin(dim=-1, keepdim=True)
        neighbor_indices = torch.where(neighbor_mask, neighbor_indices, fallback_indices)

        # grouped_xyz: [B, N_centroid, K, 3]
        grouped_xyz = _index_points(xyz, neighbor_indices) - centroid_xyz[:, :, None]
        if point_features is None:
            grouped_features = grouped_xyz
        else:
            grouped_features = torch.cat(
                (grouped_xyz, _index_points(point_features, neighbor_indices)), dim=-1
            )
        grouped_features = grouped_features.masked_fill(~centroid_mask[:, :, None, None], 0.0)

        # shared_features: [B, C_out, N_centroid, K]
        shared_features = self.shared_mlp(grouped_features.permute(0, 3, 1, 2))
        minimum_value = torch.finfo(shared_features.dtype).min
        shared_features = shared_features.masked_fill(~neighbor_mask[:, None, :, :], minimum_value)
        pooled_features = shared_features.max(dim=-1).values.transpose(1, 2)
        pooled_features = pooled_features.masked_fill(~centroid_mask[:, :, None], 0.0)

        return centroid_xyz, pooled_features, centroid_mask


class PointNet2Encoder(nn.Module):
    """Encode XYZ and optional per-point features with two PointNet++ stages.

    The first three input values are always interpreted as XYZ. Remaining
    values, normally RGB, are local point features.

    Input:
        points: ``[B, N, input_dim]``.

    Output:
        point_tokens: ``[B, min(N, num_samples[-1]), point_dim]``.
    """

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        num_samples: tuple[int, int] = (256, 64),
        num_neighbors: tuple[int, int] = (32, 32),
        dropout: float = 0.0,
        *,
        token_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError(f"input_dim must include XYZ and be at least 3, got {input_dim}.")
        if len(num_samples) != 2 or len(num_neighbors) != 2:
            raise ValueError("num_samples and num_neighbors must each contain two values.")

        # token_dim and hidden_dim are retained for compatibility with the old
        # PointEncoder constructor used by early extension checkpoints.
        if token_dim is not None:
            point_dim = token_dim
        first_stage_dim = hidden_dim or min(128, point_dim)

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.stage1 = SetAbstraction(
            num_samples=num_samples[0],
            num_neighbors=num_neighbors[0],
            input_feature_dim=input_dim - 3,
            mlp_channels=(64, 64, first_stage_dim),
        )
        self.stage2 = SetAbstraction(
            num_samples=num_samples[1],
            num_neighbors=num_neighbors[1],
            input_feature_dim=first_stage_dim,
            mlp_channels=(first_stage_dim, point_dim, point_dim),
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(point_dim)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError(
                f"points must have shape [B, N, input_dim], got {tuple(points.shape)}."
            )
        if points.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} values per point, got {points.shape[-1]}.")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point.")

        batch_size, num_points, _ = points.shape
        if point_mask is None:
            point_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=points.device)
        elif point_mask.shape != (batch_size, num_points):
            raise ValueError(
                "point_mask must have shape "
                f"{(batch_size, num_points)}, got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        xyz = points[..., :3]
        input_features = points[..., 3:] if self.input_dim > 3 else None
        stage1_xyz, stage1_features, stage1_mask = self.stage1(xyz, input_features, point_mask)
        _, stage2_features, stage2_mask = self.stage2(stage1_xyz, stage1_features, stage1_mask)

        # point_tokens: [B, N_out, point_dim]
        point_tokens = self.output_norm(self.output_dropout(stage2_features))
        point_tokens = point_tokens.masked_fill(~stage2_mask[:, :, None], 0.0)

        if return_mask:
            return point_tokens, stage2_mask
        return point_tokens


class PointTransformerEncoder(nn.Module):
    """Small point transformer that keeps one output token per input point.

    This intentionally simple backend is a replaceable baseline, not a port of
    a larger sparse-convolution or Point Transformer framework.
    """

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.0,
        *,
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError(f"input_dim must include XYZ and be at least 3, got {input_dim}.")
        if token_dim is not None:
            point_dim = token_dim
        if point_dim % num_heads != 0:
            raise ValueError(
                f"point_dim ({point_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.input_projection = nn.Linear(input_dim, point_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(3, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=point_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or point_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(point_dim)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError(
                f"points must have shape [B, N, input_dim], got {tuple(points.shape)}."
            )
        if points.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} values per point, got {points.shape[-1]}.")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point.")

        batch_size, num_points, _ = points.shape
        if point_mask is None:
            point_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=points.device)
        elif point_mask.shape != (batch_size, num_points):
            raise ValueError(
                "point_mask must have shape "
                f"{(batch_size, num_points)}, got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        safe_mask = point_mask.clone()
        rows_without_points = ~safe_mask.any(dim=1)
        if rows_without_points.any():
            safe_mask[rows_without_points, 0] = True

        # point_tokens: [B, N, point_dim]
        point_tokens = self.input_projection(points)
        point_tokens = point_tokens + self.position_projection(points[..., :3])
        point_tokens = self.transformer(point_tokens, src_key_padding_mask=~safe_mask)
        point_tokens = self.output_norm(point_tokens)
        point_tokens = point_tokens.masked_fill(~point_mask[:, :, None], 0.0)

        if return_mask:
            return point_tokens, point_mask
        return point_tokens


def build_point_encoder(
    encoder_type: str = "pointnet2",
    **encoder_kwargs,
) -> PointNet2Encoder | PointTransformerEncoder:
    """Build one of the lightweight point encoder backends."""
    normalized_type = encoder_type.lower().replace("-", "_")
    if normalized_type in {"pointnet", "pointnet2", "pointnet++"}:
        return PointNet2Encoder(**encoder_kwargs)
    if normalized_type in {"point_transformer", "transformer"}:
        return PointTransformerEncoder(**encoder_kwargs)
    raise ValueError(
        f"Unknown point encoder type {encoder_type!r}. Expected 'pointnet2' or 'point_transformer'."
    )


class PointEncoder(PointNet2Encoder):
    """Backward-compatible PointNet++ replacement for the old ``PointEncoder``.

    The argument order matches the former point-wise MLP encoder while the
    implementation now uses :class:`PointNet2Encoder`.
    """

    def __init__(
        self,
        input_dim: int = 6,
        token_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            point_dim=token_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


__all__ = [
    "PointEncoder",
    "PointNet2Encoder",
    "PointTransformerEncoder",
    "build_point_encoder",
]
