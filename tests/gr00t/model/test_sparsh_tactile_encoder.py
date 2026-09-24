# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from gr00t.data.types import ModalityConfig, VLAStepData
import gr00t.model.extension.tactile_encoder as tactile_module
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn


class _TinySparshBackbone(nn.Module):
    embed_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(6, 768, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.patch_embed.proj(images).mean(dim=(-2, -1))
        return torch.stack((pooled, pooled), dim=1)


@pytest.fixture
def tiny_sparsh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tactile_module, "SparshVisionTransformer", _TinySparshBackbone)
    return tactile_module


def _checkpoint(path: Path, value: float = 0.25) -> Path:
    backbone = _TinySparshBackbone()
    with torch.no_grad():
        backbone.patch_embed.proj.weight.fill_(value)
        backbone.patch_embed.proj.bias.zero_()
    torch.save(backbone.state_dict(), path)
    return path


def test_official_preprocessing_shape_range_and_temporal_order(tiny_sparsh, tmp_path: Path) -> None:
    encoder = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=32,
        pretrained_model=str(_checkpoint(tmp_path / "sparsh.pth")),
    )
    previous = torch.zeros(1, 3, 240, 320)
    current = torch.ones(1, 3, 240, 320)

    processed = encoder.preprocess(torch.stack((previous, current), dim=1))

    assert processed.shape == (1, 6, 320, 240)
    assert processed.dtype == torch.float32
    assert processed.min().item() == 0.0
    assert processed.max().item() == 1.0
    torch.testing.assert_close(processed[:, :3], torch.ones_like(processed[:, :3]))
    torch.testing.assert_close(processed[:, 3:], torch.zeros_like(processed[:, 3:]))


def test_pretrained_checkpoint_is_strictly_loaded(tiny_sparsh, tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "sparsh.pth", value=0.375)
    encoder = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=24,
        pretrained_model=str(checkpoint),
    )

    assert encoder.pretrained_loaded is True
    assert encoder.resolved_checkpoint == str(checkpoint)
    torch.testing.assert_close(
        encoder.backbone.patch_embed.proj.weight,
        torch.full_like(encoder.backbone.patch_embed.proj.weight, 0.375),
    )

    broken = tmp_path / "broken.pth"
    torch.save({"wrong.weight": torch.ones(1)}, broken)
    with pytest.raises(RuntimeError, match="No partial/random fallback"):
        tiny_sparsh.SparshDinoTactileEncoder(
            output_dim=24,
            pretrained_model=str(broken),
        )


def test_patch_tokens_project_to_dit_dimension(tiny_sparsh, tmp_path: Path) -> None:
    encoder = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=40,
        pretrained_model=str(_checkpoint(tmp_path / "sparsh.pth")),
    )
    tactile = torch.randint(0, 256, (2, 2, 3, 24, 32), dtype=torch.uint8)

    tokens = encoder(tactile)

    assert tokens.shape == (2, 2, 40)
    assert torch.isfinite(tokens).all()


def test_temporal_pair_survives_vla_step_processor_and_encoder(tiny_sparsh, tmp_path: Path) -> None:
    previous = np.zeros((24, 32, 3), dtype=np.uint8)
    current = np.full((24, 32, 3), 255, dtype=np.uint8)
    step = VLAStepData(
        images={},
        states={},
        actions={},
        tactile={"rgb": [previous, current]},
    )
    modality_config = {"tactile": ModalityConfig(delta_indices=[-1, 0], modality_keys=["rgb"])}
    processed = Gr00tN1d7Processor._training_tactile(step, modality_config)
    encoder = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=40,
        pretrained_model=str(_checkpoint(tmp_path / "sparsh.pth")),
    )

    tokens = encoder(processed.unsqueeze(0))

    assert processed.shape == (2, 3, 24, 32)
    assert tokens.shape == (1, 2, 40)
    prepared = encoder.preprocess(processed.unsqueeze(0))
    torch.testing.assert_close(prepared[:, :3], torch.ones_like(prepared[:, :3]))
    torch.testing.assert_close(prepared[:, 3:], torch.zeros_like(prepared[:, 3:]))


def test_freeze_backbone_keeps_projection_trainable(tiny_sparsh) -> None:
    encoder = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=16,
        load_pretrained=False,
    )
    encoder.set_trainable(backbone=False, projection=True)

    assert not any(parameter.requires_grad for parameter in encoder.backbone_parameters())
    assert all(parameter.requires_grad for parameter in encoder.adapter_parameters())


def test_frozen_backbone_is_configured_before_status_logging(
    tiny_sparsh, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=tactile_module.__name__):
        encoder = tiny_sparsh.SparshDinoTactileEncoder(
            output_dim=16,
            pretrained_model=str(_checkpoint(tmp_path / "sparsh.pth")),
            backbone_trainable=False,
        )

    assert not any(parameter.requires_grad for parameter in encoder.backbone_parameters())
    assert all(parameter.requires_grad for parameter in encoder.adapter_parameters())
    assert "Trainable: no" in caplog.text


def test_saved_state_reconstructs_without_original_checkpoint(tiny_sparsh, tmp_path: Path) -> None:
    background_path = tmp_path / "no_contact.png"
    Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8)).save(background_path)
    source = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=16,
        pretrained_model=str(_checkpoint(tmp_path / "sparsh.pth")),
        background_path=str(background_path),
    )
    saved_state = source.state_dict()
    background_path.unlink()
    restored = tiny_sparsh.SparshDinoTactileEncoder(
        output_dim=16,
        pretrained_model="facebook/sparsh-dino-base",
        background_path=str(background_path),
        load_pretrained=False,
    )

    restored.load_state_dict(saved_state, strict=True)

    for expected, actual in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(source.background, restored.background)
