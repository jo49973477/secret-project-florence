# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud encoders for multimodal action conditioning.

Concerto is an optional dependency.  It is imported only when a Concerto
backend is selected so RGB-only and lightweight point-cloud workflows do not
need the sparse-convolution stack.
"""

import importlib
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


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


class ConcertoPointEncoder(nn.Module):
    """Adapt an official pretrained Concerto PTv3 model to GR00T point tokens.

    Input scenes are dense ``[B, N, 6]`` XYZRGB arrays. XYZ is in one common
    metric frame and RGB is stored in the dataset as ``[0, 255]``. The wrapper
    applies Concerto's published preprocessing: per-scene ``CenterShift``,
    grid coordinates at ``grid_size``, RGB division by 255, and zero normals
    (the official demo's supported ``--wo_normal`` path). Concerto receives
    ``[centered XYZ, RGB/255, zero normals]`` features.
    """

    SUPPORTED_MODELS = {"concerto_small", "concerto_base"}

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        *,
        model_name: str = "concerto_small",
        repo_id: str = "Pointcept/Concerto",
        checkpoint_path: str | None = None,
        download_root: str | None = None,
        grid_size: float = 0.02,
        enable_flash: bool | None = None,
        backbone: nn.Module | None = None,
        backbone_output_dim: int | None = None,
        pretrained_loaded: bool = False,
    ) -> None:
        super().__init__()
        if input_dim != 6:
            raise ValueError(
                "Concerto requires one unified XYZRGB scene field with 6 values per point; "
                f"received point_input_dim={input_dim}. Use pointnet2/point_transformer for "
                "legacy XYZ-only data."
            )
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Concerto model {model_name!r}; expected one of "
                f"{sorted(self.SUPPORTED_MODELS)}."
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}.")

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.model_name = model_name
        self.repo_id = repo_id
        self.grid_size = float(grid_size)

        if backbone is None:
            backbone, backbone_output_dim, checkpoint_label, parameter_count = (
                self._load_official_backbone(
                    model_name=model_name,
                    repo_id=repo_id,
                    checkpoint_path=checkpoint_path,
                    download_root=download_root,
                    enable_flash=enable_flash,
                )
            )
            pretrained_loaded = True
            LOGGER.info("Point encoder: %s", model_name.replace("_", "-").title())
            LOGGER.info("Checkpoint: %s", checkpoint_label)
            LOGGER.info("Loaded pretrained parameters: %s", f"{parameter_count:,}")
            LOGGER.info("Missing keys: []")
            LOGGER.info("Unexpected keys: []")
        elif backbone_output_dim is None:
            raise ValueError("backbone_output_dim is required when injecting a Concerto backbone")

        if not pretrained_loaded:
            raise RuntimeError(
                "Concerto was requested without verified pretrained weights. Randomly initialized "
                "Concerto backbones are intentionally unsupported."
            )
        assert backbone_output_dim is not None
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(backbone_output_dim, point_dim),
            nn.LayerNorm(point_dim),
        )
        self.pretrained_loaded = True
        LOGGER.info(
            "Trainable: %s", any(parameter.requires_grad for parameter in self.parameters())
        )

    @staticmethod
    def _load_official_backbone(
        *,
        model_name: str,
        repo_id: str,
        checkpoint_path: str | None,
        download_root: str | None,
        enable_flash: bool | None,
    ) -> tuple[nn.Module, int, str, int]:
        try:
            concerto = importlib.import_module("concerto")
        except ImportError as exc:
            raise ImportError(
                "Concerto point encoding requires the official Pointcept/Concerto package and "
                "its spconv + torch-scatter dependencies. Install it as documented in "
                "examples/UniVTAC/README.md."
            ) from exc

        checkpoint_name = str(Path(checkpoint_path).expanduser()) if checkpoint_path else model_name
        custom_config = {}
        if enable_flash is not None:
            custom_config["enable_flash"] = enable_flash
        try:
            checkpoint = concerto.load(
                checkpoint_name,
                repo_id=repo_id,
                download_root=download_root,
                custom_config=custom_config or None,
                ckpt_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load requested pretrained Concerto checkpoint {checkpoint_name!r} "
                f"from {repo_id!r}. No random-init fallback is allowed."
            ) from exc

        if not isinstance(checkpoint, dict) or not {"config", "state_dict"}.issubset(checkpoint):
            raise RuntimeError(
                "Official Concerto checkpoint must contain 'config' and 'state_dict' entries."
            )
        config = dict(checkpoint["config"])
        if config.get("in_channels") != 9:
            raise RuntimeError(
                "This wrapper follows official XYZ+RGB+normal feature construction and expects "
                f"a 9-channel Concerto checkpoint, got in_channels={config.get('in_channels')}."
            )
        if config.get("enc_mode", False):
            output_dim = int(config["enc_channels"][-1])
        else:
            output_dim = int(config["dec_channels"][0])

        try:
            model = concerto.model.PointTransformerV3(**config)
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not construct Concerto from pretrained checkpoint {checkpoint_name!r}."
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Concerto checkpoint is incompatible with its declared architecture: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count == 0 or not any(
            torch.count_nonzero(parameter.detach()).item() > 0 for parameter in model.parameters()
        ):
            raise RuntimeError("Concerto checkpoint loaded no non-zero pretrained parameters.")
        return model, output_dim, checkpoint_name, parameter_count

    def backbone_parameters(self):
        """Return only pretrained backbone parameters for optimizer LR grouping."""
        return self.backbone.parameters()

    def adapter_parameters(self):
        """Return newly initialized GR00T projection parameters."""
        return self.projection.parameters()

    def _prepare_batch(
        self, points: torch.Tensor, point_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor | float], list[int]]:
        coords = []
        features = []
        grid_coords = []
        counts = []
        for batch_index in range(points.shape[0]):
            scene = points[batch_index, point_mask[batch_index]]
            if scene.shape[0] == 0:
                raise ValueError(f"Concerto scene {batch_index} contains no valid points.")
            coord = scene[:, :3]
            coord_min = coord.amin(dim=0)
            coord_max = coord.amax(dim=0)
            shift = torch.stack(
                (
                    (coord_min[0] + coord_max[0]) / 2,
                    (coord_min[1] + coord_max[1]) / 2,
                    coord_min[2],
                )
            )
            coord = coord - shift
            grid_coord = torch.floor(coord / self.grid_size).to(torch.int32)
            grid_coord = grid_coord - grid_coord.amin(dim=0)
            color = scene[:, 3:6] / 255.0
            normal = torch.zeros_like(coord)
            coords.append(coord)
            grid_coords.append(grid_coord)
            features.append(torch.cat((coord, color, normal), dim=-1))
            counts.append(int(scene.shape[0]))

        count_tensor = torch.tensor(counts, dtype=torch.long, device=points.device)
        offset = count_tensor.cumsum(dim=0)
        batch = torch.arange(points.shape[0], device=points.device).repeat_interleave(count_tensor)
        return {
            "coord": torch.cat(coords, dim=0),
            "grid_coord": torch.cat(grid_coords, dim=0),
            "feat": torch.cat(features, dim=0),
            "offset": offset,
            "batch": batch,
            "grid_size": self.grid_size,
        }, counts

    @staticmethod
    def _pad_tokens(
        flat_tokens: torch.Tensor, counts: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_tokens = max(counts)
        batch_size = len(counts)
        tokens = flat_tokens.new_zeros((batch_size, max_tokens, flat_tokens.shape[-1]))
        mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=flat_tokens.device)
        start = 0
        for batch_index, count in enumerate(counts):
            tokens[batch_index, :count] = flat_tokens[start : start + count]
            mask[batch_index, :count] = True
            start += count
        if start != flat_tokens.shape[0]:
            raise RuntimeError(
                f"Concerto output offsets account for {start} tokens, got {flat_tokens.shape[0]}."
            )
        return tokens, mask

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError(
                f"Concerto points must have shape [B, N, 6], got {tuple(points.shape)}."
            )
        if not torch.isfinite(points).all():
            raise ValueError("Concerto points contain NaN or Inf.")
        if point_mask is None:
            point_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        elif point_mask.shape != points.shape[:2]:
            raise ValueError(
                f"point_mask must have shape {tuple(points.shape[:2])}, "
                f"got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        concerto_input, _input_counts = self._prepare_batch(points, point_mask)
        output = self.backbone(concerto_input)
        if not hasattr(output, "feat") or not hasattr(output, "offset"):
            raise RuntimeError("Concerto backbone output must expose .feat and .offset tensors.")
        output_counts = torch.diff(
            torch.cat((output.offset.new_zeros(1), output.offset.to(torch.long)))
        ).tolist()
        point_tokens = self.projection(output.feat)
        point_tokens, output_mask = self._pad_tokens(point_tokens, output_counts)
        if return_mask:
            return point_tokens, output_mask
        return point_tokens


def build_point_encoder(
    encoder_type: str = "pointnet2",
    **encoder_kwargs,
) -> PointNet2Encoder | PointTransformerEncoder | ConcertoPointEncoder:
    """Build one of the lightweight point encoder backends."""
    normalized_type = encoder_type.lower().replace("-", "_")
    if normalized_type in {"pointnet", "pointnet2", "pointnet++"}:
        return PointNet2Encoder(**encoder_kwargs)
    if normalized_type in {"point_transformer", "transformer"}:
        return PointTransformerEncoder(**encoder_kwargs)
    if normalized_type in ConcertoPointEncoder.SUPPORTED_MODELS:
        return ConcertoPointEncoder(model_name=normalized_type, **encoder_kwargs)
    raise ValueError(
        f"Unknown point encoder type {encoder_type!r}. Expected 'pointnet2', "
        "'point_transformer', 'concerto_small', or 'concerto_base'."
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
    "ConcertoPointEncoder",
    "PointNet2Encoder",
    "PointTransformerEncoder",
    "build_point_encoder",
]
