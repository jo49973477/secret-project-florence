# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile encoders for action-token conditioning."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .sparsh_vit import SparshVisionTransformer


LOGGER = logging.getLogger(__name__)
SPARSH_DINO_BASE_REPO_ID = "facebook/sparsh-dino-base"
SPARSH_DINO_BASE_FILENAME = "dino_vitbase.safetensors"


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

    def backbone_parameters(self):
        return self.backbone.parameters()

    def adapter_parameters(self):
        yield from self.output_projection.parameters()
        yield from self.output_norm.parameters()

    def forward(self, tactile_images: torch.Tensor) -> torch.Tensor:
        # The legacy ablation consumes the current frame from a temporal input.
        if tactile_images.ndim == 5:
            tactile_images = tactile_images[:, -1]
        if tactile_images.ndim != 4:
            raise ValueError(
                "tactile_images must have shape [B,C,H,W] or [B,T,C,H,W], got "
                f"{tuple(tactile_images.shape)}."
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


class SparshDinoTactileEncoder(nn.Module):
    """Official temporal Sparsh-DINO-Base backbone plus a GR00T projection.

    GR00T supplies chronological frames ``[previous, current]`` as
    ``[B,2,3,H,W]``. Meta's official Sparsh DINO input is a six-channel image
    ordered ``I_t ⊕ I_{t-5}``, so this wrapper resizes and concatenates the
    current frame before the previous frame. The returned representation is the
    official set of 300 normalized patch tokens; the register token is not
    returned by the official backbone forward method.
    """

    model_name = "Sparsh-DINO-Base"
    temporal_frames = 2
    official_image_size = (320, 240)
    backbone_dim = 768

    def __init__(
        self,
        output_dim: int,
        pretrained_model: str = SPARSH_DINO_BASE_REPO_ID,
        *,
        checkpoint_filename: str = SPARSH_DINO_BASE_FILENAME,
        background_path: str | None = None,
        load_pretrained: bool = True,
        backbone_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.checkpoint_filename = checkpoint_filename
        self.backbone = SparshVisionTransformer()
        self.projection = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, output_dim),
        )
        # A fine-tuned GR00T checkpoint embeds this calibration tensor. When
        # bootstrapping Sparsh, load it from the user-supplied no-contact image.
        background = self._load_background(background_path) if load_pretrained else None
        self.register_buffer(
            "background",
            background if background is not None else torch.empty(0),
            persistent=True,
        )
        self.background_path = background_path
        self.pretrained_loaded = False
        self.resolved_checkpoint: str | None = None

        if load_pretrained:
            checkpoint = self._resolve_checkpoint(pretrained_model, checkpoint_filename)
            self._load_pretrained_weights(checkpoint)
            # Transformers skips its generic missing-key initialization for modules
            # marked this way, preserving weights loaded from the external checkpoint.
            for module in self.backbone.modules():
                module._is_hf_initialized = True
            self.pretrained_loaded = True
            self.resolved_checkpoint = str(checkpoint)

        # Apply the requested fine-tuning mode before logging it. Previously the
        # constructor always reported the module default (trainable) and the
        # action head froze it only afterwards, so --no-tune-tactile-encoder
        # produced a misleading startup message.
        self.backbone.requires_grad_(backbone_trainable)

        LOGGER.info("Tactile encoder: %s", self.model_name)
        LOGGER.info(
            "Checkpoint/model: %s",
            self.resolved_checkpoint or f"embedded GR00T checkpoint ({pretrained_model})",
        )
        LOGGER.info("Pretrained weights loaded: %s", "yes" if load_pretrained else "from GR00T")
        LOGGER.info("Trainable: %s", "yes" if backbone_trainable else "no")
        if load_pretrained and background_path is None:
            LOGGER.warning(
                "Sparsh preprocessing has no background reference. Official GelSight Mini/DIGIT "
                "pretraining subtracts a sensor-specific no-contact background; pass "
                "--tactile-background-path when one is available."
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Background resolution is sensor-specific. Resize the placeholder
        # buffer before PyTorch performs its strict shape check.
        background_key = f"{prefix}background"
        if (
            background_key in state_dict
            and self.background.shape != state_dict[background_key].shape
        ):
            self.background = torch.empty_like(state_dict[background_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _load_background(background_path: str | None) -> torch.Tensor | None:
        if background_path is None:
            return None
        path = Path(background_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Tactile background image does not exist: {path}")
        with Image.open(path) as image:
            array = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
        return array.permute(2, 0, 1).to(torch.float32).div_(255.0)[None, None]

    @staticmethod
    def _resolve_checkpoint(pretrained_model: str, filename: str) -> Path:
        model_path = Path(pretrained_model).expanduser()
        if model_path.is_file():
            return model_path
        if model_path.is_dir():
            candidate = model_path / filename
            if candidate.is_file():
                return candidate
            alternatives = [
                model_path / name for name in (SPARSH_DINO_BASE_FILENAME, "dino_vitbase.ckpt")
            ]
            existing = [path for path in alternatives if path.is_file()]
            if len(existing) == 1:
                return existing[0]
            raise FileNotFoundError(
                f"Could not find official Sparsh checkpoint {filename!r} in {model_path}."
            )
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=pretrained_model, filename=filename))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download requested pretrained Sparsh checkpoint "
                f"{pretrained_model}/{filename}. Random initialization is not allowed."
            ) from exc

    @staticmethod
    def _read_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        suffix = checkpoint_path.suffix.lower()
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                return load_file(str(checkpoint_path), device="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read Sparsh safetensors file {checkpoint_path}."
                ) from exc
        if suffix not in {".ckpt", ".pt", ".pth"}:
            raise ValueError(
                f"Unsupported Sparsh checkpoint extension {suffix!r}; expected .safetensors, "
                ".ckpt, .pt, or .pth."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Sparsh checkpoint {checkpoint_path} does not contain a state dict.")
        encoder_prefix = "teacher_encoder.backbone."
        matching = {}
        for key, value in checkpoint.items():
            if encoder_prefix in key:
                stripped_key = key.split(encoder_prefix, maxsplit=1)[1]
                if stripped_key in matching:
                    raise RuntimeError(
                        f"Duplicate Sparsh backbone key {stripped_key!r} in {checkpoint_path}."
                    )
                matching[stripped_key] = value
        return matching or checkpoint

    def _load_pretrained_weights(self, checkpoint_path: Path) -> None:
        state_dict = self._read_checkpoint(checkpoint_path)
        try:
            incompatible = self.backbone.load_state_dict(state_dict, strict=True, assign=True)
        except Exception as exc:
            raise RuntimeError(
                f"Official Sparsh checkpoint {checkpoint_path} is incompatible with "
                "Sparsh-DINO-Base. No partial/random fallback is allowed."
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Sparsh checkpoint load was not strict: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
        if not state_dict or "patch_embed.proj.weight" not in state_dict:
            raise RuntimeError("Sparsh checkpoint did not contain the official patch embedding.")

    def backbone_parameters(self):
        return self.backbone.parameters()

    def adapter_parameters(self):
        return self.projection.parameters()

    def set_trainable(self, *, backbone: bool, projection: bool) -> None:
        self.backbone.requires_grad_(backbone)
        self.projection.requires_grad_(projection)

    @staticmethod
    def _official_orientation_and_crop(images: torch.Tensor) -> torch.Tensor:
        # Meta's loader rotates landscape tactile frames clockwise, then center
        # crops to a 4:3 portrait aspect ratio before Resize((320, 240)).
        if images.shape[-2] < images.shape[-1]:
            images = torch.rot90(images, k=-1, dims=(-2, -1))
        height, width = images.shape[-2:]
        if height / width == 4 / 3:
            return images
        # Preserve the official loader's exact crop calculation.
        target_height = int(height / (4 / 3))
        target_width = width
        top = (height - target_height) // 2
        return images[..., top : top + target_height, :target_width]

    def preprocess(self, tactile_sequence: torch.Tensor) -> torch.Tensor:
        """Return the exact six-channel Sparsh input before patch embedding."""
        if tactile_sequence.ndim != 5:
            raise ValueError(
                "Sparsh tactile input must have shape [B,2,3,H,W], got "
                f"{tuple(tactile_sequence.shape)}."
            )
        if tactile_sequence.shape[1:3] != (self.temporal_frames, 3):
            raise ValueError(
                "Sparsh-DINO-Base requires two RGB frames [previous,current], got "
                f"{tuple(tactile_sequence.shape)}."
            )
        images = tactile_sequence.to(dtype=torch.float32)
        if tactile_sequence.dtype == torch.uint8:
            images = images.div(255.0)
        if not torch.isfinite(images).all():
            raise ValueError("Tactile images contain NaN or Inf.")
        if images.amin() < 0 or images.amax() > 1:
            raise ValueError("Sparsh tactile pixels must be in [0,1] (or uint8 before conversion).")
        if self.background.numel() > 0:
            background = self.background.to(device=images.device, dtype=images.dtype)
            if tuple(background.shape[-2:]) != tuple(images.shape[-2:]):
                background = F.interpolate(
                    background.flatten(0, 1),
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).view(1, 1, 3, *images.shape[-2:])
            # Match Meta's compute_diff: subtract in pixel space, offset by
            # 0.5, clip, and quantize back to uint8 before ToTensor scaling.
            images = (images - background + 0.5).clamp_(0.0, 1.0)
            images = images.mul(255.0).to(torch.uint8).to(torch.float32).div_(255.0)
        images = self._official_orientation_and_crop(images)
        batch, time, channels, height, width = images.shape
        images = F.interpolate(
            images.reshape(batch * time, channels, height, width),
            size=self.official_image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(batch, time, channels, *self.official_image_size)
        # Input from GR00T is chronological [I_(t-1), I_t]; official Sparsh is I_t ⊕ I_(t-5).
        return torch.cat((images[:, 1], images[:, 0]), dim=1)

    def forward(self, tactile_sequence: torch.Tensor) -> torch.Tensor:
        sparsh_input = self.preprocess(tactile_sequence)
        input_dtype = next(self.backbone.parameters()).dtype
        patch_tokens = self.backbone(sparsh_input.to(dtype=input_dtype))
        return self.projection(patch_tokens)


def build_tactile_encoder(
    encoder_cfg: str,
    *,
    output_dim: int,
    input_channels: int = 3,
    pretrained_model: str = SPARSH_DINO_BASE_REPO_ID,
    checkpoint_filename: str = SPARSH_DINO_BASE_FILENAME,
    background_path: str | None = None,
    load_pretrained: bool = True,
    backbone_trainable: bool = True,
) -> nn.Module:
    normalized = encoder_cfg.lower().replace("-", "_")
    if normalized == "resnet18":
        return TactileEncoder(input_channels=input_channels, token_dim=output_dim)
    if normalized == "sparsh_dino_base":
        if input_channels != 3:
            raise ValueError("Sparsh-DINO-Base requires RGB tactile frames (3 channels).")
        return SparshDinoTactileEncoder(
            output_dim=output_dim,
            pretrained_model=pretrained_model,
            checkpoint_filename=checkpoint_filename,
            background_path=background_path,
            load_pretrained=load_pretrained,
            backbone_trainable=backbone_trainable,
        )
    raise ValueError(
        f"Unsupported tactile encoder {encoder_cfg!r}; expected 'resnet18' or 'sparsh_dino_base'."
    )


__all__ = [
    "SPARSH_DINO_BASE_FILENAME",
    "SPARSH_DINO_BASE_REPO_ID",
    "SparshDinoTactileEncoder",
    "TactileEncoder",
    "build_tactile_encoder",
]
