# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for backbone-scoped LoRA attachment and checkpoint structure."""

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.qwen3_backbone import attach_lora_adapters
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from transformers import PreTrainedModel


class _ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.visual = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))


class _ToyCategorySpecific(nn.Module):
    """Mimic CategorySpecificLinear's direct W/b parameters (not nn.Linear)."""

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2, 8, 8))
        self.b = nn.Parameter(torch.randn(2, 8))


class _ToyGr00t(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ToyQwen()
        self.action_head = nn.ModuleDict(
            {
                "state_encoder": _ToyCategorySpecific(),
                "action_encoder": _ToyCategorySpecific(),
                "action_decoder": _ToyCategorySpecific(),
                "model": nn.Linear(8, 8),
            }
        )


class _ToyCheckpointModel(PreTrainedModel):
    """Small HF model exercising the same config-driven adapter reconstruction."""

    config_class = Gr00tN1d7Config

    def __init__(self, config):
        super().__init__(config)
        self.backbone = _ToyQwen()
        self.action_head = _ToyCategorySpecific()
        if config.use_lora:
            attach_lora_adapters(
                self.backbone,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.lora_bias,
            )


def _attach(model: _ToyGr00t) -> list[str]:
    return attach_lora_adapters(
        model.backbone,
        r=2,
        alpha=4,
        dropout=0.0,
        bias="none",
    )


def test_lora_targets_language_and_visual_but_not_action_head():
    model = _ToyGr00t()
    lora_names = _attach(model)

    assert any(name.startswith("language_model.") for name in lora_names)
    assert any(name.startswith("visual.") for name in lora_names)
    assert all("lora_" in name for name in lora_names)
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())
    assert model.action_head["state_encoder"].W.requires_grad

    backbone_base = [
        parameter for name, parameter in model.backbone.named_parameters() if "lora_" not in name
    ]
    assert backbone_base
    assert not any(parameter.requires_grad for parameter in backbone_base)


def test_lora_state_dict_round_trips_with_structure_recreated_first():
    trained = _ToyGr00t()
    _attach(trained)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_" in name:
                parameter.add_(0.25)
    checkpoint = {name: tensor.detach().clone() for name, tensor in trained.state_dict().items()}

    reloaded = _ToyGr00t()
    _attach(reloaded)
    incompatible = reloaded.load_state_dict(checkpoint, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for name, tensor in reloaded.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint[name])


def test_non_lora_mode_has_unchanged_module_structure():
    model = _ToyGr00t()

    assert not any("lora_" in name for name, _ in model.named_parameters())
    assert isinstance(model.backbone.language_model[0], nn.Linear)
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())


def test_lora_metadata_round_trips_through_hugging_face_config(tmp_path):
    config = Gr00tN1d7Config(
        use_lora=True,
        lora_r=8,
        lora_alpha=24,
        lora_dropout=0.1,
        lora_bias="none",
    )
    config.save_pretrained(tmp_path)

    loaded = Gr00tN1d7Config.from_pretrained(tmp_path)

    assert loaded.use_lora is True
    assert loaded.lora_r == 8
    assert loaded.lora_alpha == 24
    assert loaded.lora_dropout == pytest.approx(0.1)
    assert loaded.lora_bias == "none"


def test_normal_hugging_face_loading_recreates_and_loads_lora(
    tmp_path, load_hf_model_weights
):
    config = Gr00tN1d7Config(
        use_lora=True,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias="none",
    )
    trained = _ToyCheckpointModel(config)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_" in name:
                parameter.add_(0.5)
    expected = {name: tensor.detach().clone() for name, tensor in trained.state_dict().items()}
    trained.save_pretrained(tmp_path)

    with load_hf_model_weights():
        reloaded, loading_info = _ToyCheckpointModel.from_pretrained(
            tmp_path, output_loading_info=True
        )

    assert loading_info["missing_keys"] == []
    assert loading_info["unexpected_keys"] == []
    assert loading_info["mismatched_keys"] == []
    assert any("lora_" in name for name, _ in reloaded.named_parameters())
    for name, tensor in reloaded.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_finetune_config_uses_omegaconf_compatible_plain_types():
    annotations = FinetuneConfig.__annotations__
    structured = OmegaConf.structured(FinetuneConfig)

    assert annotations["use_lora"] is bool
    assert annotations["lora_r"] is int
    assert annotations["lora_alpha"] is int
    assert annotations["lora_dropout"] is float
    assert annotations["lora_bias"] is str
    assert structured.use_lora is False
    assert structured.lora_r == 16


def test_lora_rejects_training_original_backbone_biases():
    with pytest.raises(ValueError, match="parameters stay frozen"):
        attach_lora_adapters(
            _ToyQwen(),
            r=2,
            alpha=4,
            dropout=0.0,
            bias="all",
        )
