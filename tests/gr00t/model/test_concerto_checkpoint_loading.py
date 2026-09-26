# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest
import torch
from torch import nn
from transformers.integrations.accelerate import init_empty_weights


_ROOT = Path(__file__).resolve().parents[3]
_POINT_ENCODER_PATH = _ROOT / "gr00t/model/extension/point_encoder.py"
_SPEC = importlib.util.spec_from_file_location(
    "concerto_point_encoder_under_test", _POINT_ENCODER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
point_encoder_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(point_encoder_module)
ConcertoPointEncoder = point_encoder_module.ConcertoPointEncoder

_CONFIG = {
    "in_channels": 9,
    "enc_mode": True,
    "enc_channels": [4],
}


class _TinyConcertoBackbone(nn.Module):
    def __init__(self, *, in_channels: int, enc_channels: list[int], **kwargs) -> None:
        super().__init__()
        output_dim = enc_channels[-1]
        self.last_assign: bool | None = None
        self.projection = nn.Linear(in_channels, output_dim)
        self.register_buffer("running_scale", torch.zeros(output_dim))
        self.register_buffer(
            "nonpersistent_offset",
            torch.arange(output_dim, dtype=torch.float32),
            persistent=False,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.last_assign = assign
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class _NonMaterializingBackbone(_TinyConcertoBackbone):
    def to_empty(self, *, device, recurse=True):
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.last_assign = assign
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def test_sparse_conv_container_is_zero3_leaf_without_replacing_parameters() -> None:
    class SparseTensor:
        def __init__(self, features):
            self.features = features

        def replace_feature(self, features):
            return SparseTensor(features)

    class SparseConv(nn.Module):
        __module__ = "spconv.pytorch.conv"

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 3, 3, 3, 4))
            self.seen_features = None

        def forward(self, sparse):
            self.seen_features = sparse.features
            return SparseTensor(sparse.features * self.weight.mean())

    class PointSequential(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SparseConv()

        def forward(self, point):
            point.sparse_conv_feat = self.conv(point.sparse_conv_feat)
            point.feat = point.sparse_conv_feat.features
            return point

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Module()
            self.block.cpe = PointSequential()
            self.other = nn.Linear(4, 4)

    backbone = FakeBackbone()
    sparse_weight = backbone.block.cpe.conv.weight
    state_keys = set(backbone.state_dict())
    encoder = ConcertoPointEncoder(
        backbone=backbone,
        backbone_output_dim=4,
        load_pretrained=False,
    )

    assert encoder.backbone.block.cpe._z3_leaf is True
    assert not hasattr(encoder.backbone.block.cpe.conv, "_z3_leaf")
    assert not hasattr(encoder.backbone.other, "_z3_leaf")
    assert encoder.backbone.block.cpe.conv.weight is sparse_weight
    assert set(encoder.backbone.state_dict()) == state_keys

    # Simulate ZeRO's input hook replacing Point.feat but not the sparse
    # structure's feature reference. The conv must consume the hooked tensor.
    old_feature = torch.randn(3, 4, requires_grad=True)
    hooked_feature = torch.randn(3, 4, requires_grad=True)
    point = SimpleNamespace(feat=hooked_feature, sparse_conv_feat=SparseTensor(old_feature))
    output = encoder.backbone.block.cpe(point)
    assert encoder.backbone.block.cpe.conv.seen_features is hooked_feature
    output.feat.sum().backward()
    assert hooked_feature.grad is not None
    assert old_feature.grad is None
    assert sparse_weight.grad is not None


def _checkpoint(value: float) -> dict:
    model = _TinyConcertoBackbone(**_CONFIG)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
        model.running_scale.fill_(value + 1.0)
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    return {"config": _CONFIG, "state_dict": state_dict}


def _install_fake_concerto(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: dict,
    model_class: type[nn.Module] = _TinyConcertoBackbone,
) -> Mock:
    load = Mock(return_value=checkpoint)
    concerto = SimpleNamespace(
        load=load,
        model=SimpleNamespace(PointTransformerV3=model_class),
    )

    def import_module(name: str):
        assert name == "concerto"
        return concerto

    monkeypatch.setattr(
        point_encoder_module,
        "importlib",
        SimpleNamespace(import_module=import_module),
    )
    return load


def _load_backbone():
    return ConcertoPointEncoder._load_official_backbone(
        model_name="concerto_small",
        repo_id="Pointcept/Concerto",
        checkpoint_path=None,
        download_root=None,
        enable_flash=None,
    )


def _assert_materialized_checkpoint(model: _TinyConcertoBackbone, value: float) -> None:
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())
    torch.testing.assert_close(
        model.projection.weight,
        torch.full_like(model.projection.weight, value),
    )
    torch.testing.assert_close(
        model.projection.bias,
        torch.full_like(model.projection.bias, value),
    )
    torch.testing.assert_close(
        model.running_scale,
        torch.full_like(model.running_scale, value + 1.0),
    )
    torch.testing.assert_close(
        model.nonpersistent_offset,
        torch.arange(4, dtype=torch.float32),
    )


def test_official_checkpoint_loads_during_normal_cpu_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(0.25)
    assert all(
        tensor.device.type == "cpu" and not tensor.is_meta
        for tensor in checkpoint["state_dict"].values()
    )
    load = _install_fake_concerto(monkeypatch, checkpoint)

    model, output_dim, checkpoint_name, parameter_count, architecture_config = _load_backbone()

    load.assert_called_once()
    assert output_dim == 4
    assert checkpoint_name == "concerto_small"
    assert parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert architecture_config == _CONFIG
    assert model.last_assign is True
    _assert_materialized_checkpoint(model, 0.25)


def test_official_checkpoint_escapes_transformers_meta_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(0.375)
    _install_fake_concerto(monkeypatch, checkpoint)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        # This is the register_parameter-based meta context used by Transformers. Unlike a
        # plain torch.device("meta") block, it overrides a nested CPU construction context.
        with init_empty_weights(include_buffers=False):
            model, _, _, _, _ = _load_backbone()

    assert model.last_assign is False
    _assert_materialized_checkpoint(model, 0.375)
    assert not any(
        "copying from a non-meta parameter" in str(warning.message) for warning in caught_warnings
    )


def test_embedded_checkpoint_reconstructs_without_external_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(0.625)
    load = _install_fake_concerto(monkeypatch, checkpoint)
    encoder = ConcertoPointEncoder(
        input_dim=6,
        point_dim=8,
        model_name="concerto_small",
        load_pretrained=False,
        architecture_config=_CONFIG,
    )

    incompatible = encoder.backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder.validate_pretrained_backbone("embedded GR00T checkpoint")

    load.assert_not_called()
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert encoder.pretrained_loaded is True
    assert not any(parameter.is_meta for parameter in encoder.parameters())
    assert not any(buffer.is_meta for buffer in encoder.buffers())
    torch.testing.assert_close(
        encoder.backbone.projection.weight,
        torch.full_like(encoder.backbone.projection.weight, 0.625),
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "missing=['projection.bias']"),
        ("unexpected", "unexpected=['extra.weight']"),
    ],
)
def test_checkpoint_key_validation_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    checkpoint = _checkpoint(0.25)
    if mutation == "missing":
        del checkpoint["state_dict"]["projection.bias"]
    else:
        checkpoint["state_dict"]["extra.weight"] = torch.ones(1)
    _install_fake_concerto(monkeypatch, checkpoint)

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _load_backbone()


def test_meta_tensor_in_checkpoint_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _checkpoint(0.25)
    checkpoint["state_dict"]["projection.weight"] = torch.empty(
        checkpoint["state_dict"]["projection.weight"].shape,
        device="meta",
    )
    _install_fake_concerto(monkeypatch, checkpoint)

    with pytest.raises(
        RuntimeError,
        match="Concerto checkpoint itself contains meta tensors:.*projection.weight",
    ):
        _load_backbone()


def test_unresolved_meta_parameters_fail_before_storage_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_concerto(
        monkeypatch,
        _checkpoint(0.25),
        model_class=_NonMaterializingBackbone,
    )
    count_nonzero = Mock(side_effect=AssertionError("storage validation must not run"))
    monkeypatch.setattr(point_encoder_module.torch, "count_nonzero", count_nonzero)

    with (
        init_empty_weights(include_buffers=False),
        pytest.raises(
            RuntimeError,
            match="Concerto checkpoint loading left meta parameters unresolved",
        ),
    ):
        _load_backbone()

    count_nonzero.assert_not_called()


def test_all_zero_checkpoint_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_concerto(monkeypatch, _checkpoint(0.0))

    with pytest.raises(RuntimeError, match="no non-zero pretrained parameters"):
        _load_backbone()
