# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concerto's local FP16 spconv boundary under surrounding BF16 execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


_PATH = Path(__file__).resolve().parents[3] / "gr00t/model/extension/point_encoder.py"
_SPEC = importlib.util.spec_from_file_location("concerto_precision_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
point_encoder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(point_encoder)


class _FakeConcertoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(9, 4)
        self.input_dtypes = None

    def forward(self, data):
        self.input_dtypes = {
            key: data[key].dtype for key in ("feat", "coord", "grid_coord", "offset", "batch")
        }
        return SimpleNamespace(feat=self.linear(data["feat"]), offset=data["offset"])


def _points(device: str, dtype: torch.dtype) -> torch.Tensor:
    xyz = torch.arange(16, device=device, dtype=torch.float32)[:, None] * 0.04
    xyz = torch.cat((xyz, torch.zeros(16, 2, device=device)), dim=1)
    rgb = torch.full((16, 3), 128, device=device)
    return torch.cat((xyz, rgb), dim=1).to(dtype).unsqueeze(0)


def test_cpu_bf16_input_uses_fp32_geometry_and_preserves_gradients() -> None:
    backbone = _FakeConcertoBackbone()
    encoder = point_encoder.ConcertoPointEncoder(
        input_dim=6,
        point_dim=8,
        backbone=backbone,
        backbone_output_dim=4,
        pretrained_loaded=True,
    )
    points = _points("cpu", torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens, mask = encoder(points, return_mask=True)
        loss = tokens.float().square().mean()
    loss.backward()

    assert backbone.input_dtypes["feat"] == torch.float32
    assert backbone.input_dtypes["coord"] == torch.float32
    assert backbone.input_dtypes["grid_coord"] == torch.int32
    assert backbone.input_dtypes["offset"] == torch.int64
    assert backbone.input_dtypes["batch"] == torch.int64
    assert tokens.shape == (1, 16, 8)
    assert tokens.dtype in {torch.bfloat16, torch.float32}
    assert mask.shape == (1, 16) and mask.all()
    assert torch.isfinite(tokens).all()
    assert backbone.linear.weight.grad is not None
    assert torch.isfinite(backbone.linear.weight.grad).all()
    assert backbone.linear.weight.grad.abs().sum() > 0


def test_non_concerto_encoder_still_uses_its_existing_path() -> None:
    encoder = point_encoder.PointNet2Encoder(
        input_dim=6, point_dim=8, num_samples=(8, 4), num_neighbors=(4, 4)
    )
    tokens = encoder(_points("cpu", torch.float32))
    assert tokens.shape == (1, 4, 8)
    assert torch.isfinite(tokens).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("trainable", [True, False])
def test_spconv_implicit_gemm_receives_fp16_and_backpropagates(monkeypatch, trainable) -> None:
    spconv = pytest.importorskip("spconv.pytorch")
    cppcore = pytest.importorskip("spconv.pytorch.cppcore")
    spconv_ops = pytest.importorskip("spconv.pytorch.ops")
    assert torch.float16 in cppcore._TORCH_DTYPE_TO_TV

    class TinySparseBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(9, 32)
            self.conv = spconv.SubMConv3d(32, 32, kernel_size=3, padding=1, bias=True)
            self.output_dtype = None

        def forward(self, data):
            indices = (
                torch.cat((data["batch"][:, None], data["grid_coord"].long()), dim=1)
                .to(torch.int32)
                .contiguous()
            )
            spatial_shape = [int(value) + 3 for value in data["grid_coord"].amax(dim=0).tolist()]
            sparse = spconv.SparseConvTensor(
                features=self.stem(data["feat"]),
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=1,
            )
            output = self.conv(sparse)
            self.output_dtype = output.features.dtype
            return SimpleNamespace(feat=output.features, offset=data["offset"])

    backbone = TinySparseBackbone()
    encoder = point_encoder.ConcertoPointEncoder(
        input_dim=6,
        point_dim=8,
        backbone=backbone,
        backbone_output_dim=32,
        pretrained_loaded=True,
    ).to("cuda", dtype=torch.bfloat16)
    backbone.requires_grad_(trainable)
    if not trainable:
        backbone.eval()
    original_parameter_id = id(backbone.conv.weight)
    original_bias_id = id(backbone.conv.bias)
    observed = []
    original_implicit_gemm = spconv_ops.implicit_gemm

    def record_implicit_gemm(features, filters, *args, **kwargs):
        import inspect

        bound = inspect.signature(original_implicit_gemm).bind(features, filters, *args, **kwargs)
        bias = bound.arguments.get("bias")
        output = original_implicit_gemm(features, filters, *args, **kwargs)
        observed.append(
            (
                features.dtype,
                filters.dtype,
                bias.dtype if bias is not None else None,
                output[0].dtype,
            )
        )
        return output

    monkeypatch.setattr(spconv_ops, "implicit_gemm", record_implicit_gemm)
    optimizer = torch.optim.SGD([p for p in encoder.parameters() if p.requires_grad], lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(_points("cuda", torch.float32))
            loss = (tokens.float()[..., 0] - 0.5).square().mean()
        loss.backward()
        for parameter in encoder.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()

    expected_bias = None if trainable else torch.float16
    assert observed and all(
        row == (torch.float16, torch.float16, expected_bias, torch.float16) for row in observed
    )
    assert backbone.output_dtype == torch.float16
    assert tokens.dtype == torch.bfloat16
    assert torch.isfinite(tokens).all()
    assert id(backbone.conv.weight) == original_parameter_id
    assert id(backbone.conv.bias) == original_bias_id
    assert backbone.conv.weight.dtype == backbone.conv.bias.dtype == torch.bfloat16
    if trainable:
        assert backbone.conv.weight.grad is not None
        assert backbone.conv.bias.grad is not None
        assert torch.isfinite(backbone.conv.weight.grad).all()
        assert torch.isfinite(backbone.conv.bias.grad).all()
        assert backbone.conv.weight.grad.abs().sum() > 0
    else:
        assert backbone.conv.weight.grad is None and backbone.conv.bias.grad is None


class _SparseFeatures:
    def __init__(self, features):
        self.features = features

    def replace_feature(self, features):
        return _SparseFeatures(features)


class _StrictSparseConv(nn.Module):
    """CPU oracle for tensorview's unsupported BF16 bias/weight failure."""

    __module__ = "spconv.pytorch.conv"

    def __init__(self, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4) * 0.1)
        self.bias = nn.Parameter(torch.randn(4) * 0.1) if bias else None
        self.seen = []

    def _conv_forward(self, training, input, weight, bias, add_input=None):
        self.seen.append(
            (input.features.dtype, weight.dtype, bias.dtype if bias is not None else None)
        )
        if bias is not None and bias.dtype == torch.bfloat16:
            raise KeyError(torch.bfloat16)
        if weight.dtype == torch.bfloat16:
            raise KeyError(torch.bfloat16)
        features = torch.nn.functional.linear(input.features, weight, bias)
        if add_input is not None:
            features = features + add_input.features
        return input.replace_feature(features)

    def forward(self, input, add_input=None):
        return self._conv_forward(self.training, input, self.weight, self.bias, add_input)


class _SparseBackbone(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.stem = nn.Linear(9, 4)
        self.conv = _StrictSparseConv(bias)

    def forward(self, data):
        sparse = _SparseFeatures(self.stem(data["feat"]))
        output = self.conv(sparse)
        return SimpleNamespace(feat=output.features, offset=data["offset"])


def test_autocast_alone_does_not_cast_eval_bias():
    conv = _StrictSparseConv().to(torch.bfloat16).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(KeyError, match="bfloat16"):
            conv(_SparseFeatures(torch.randn(3, 4)))


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_sparse_boundary_two_steps_preserves_parameters_and_finite_gradients(
    trainable, bias, monkeypatch, caplog
):
    import logging

    backbone = _SparseBackbone(bias)
    encoder = (
        point_encoder.ConcertoPointEncoder(
            backbone=backbone,
            backbone_output_dim=4,
            point_dim=8,
            pretrained_loaded=True,
        )
        .to(torch.bfloat16)
        .train()
    )
    backbone.requires_grad_(trainable)
    if not trainable:
        backbone.eval()  # Production freezes the sensor by running it in eval mode.
    parameters = dict(encoder.named_parameters())
    identities = {name: id(p) for name, p in parameters.items()}
    initial = {name: p.detach().clone() for name, p in parameters.items()}
    keys = set(encoder.state_dict())
    optimizer = torch.optim.SGD([p for p in parameters.values() if p.requires_grad], lr=0.01)
    monkeypatch.setenv("GROOT_DEBUG_CONCERTO_PRECISION", "1")
    hook_calls = []
    handle = backbone.conv.register_forward_pre_hook(lambda *_: hook_calls.append("gather"))
    with caplog.at_level(logging.INFO):
        for step in range(2):
            optimizer.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                tokens = encoder(_points("cpu", torch.float32))
                loss = (tokens.float()[..., 0] - 0.5).square().mean()
            assert tokens.dtype == torch.bfloat16
            assert torch.isfinite(loss) and torch.isfinite(tokens).all()
            loss.backward()
            if step == 0:
                assert all(torch.equal(p, initial[name]) for name, p in parameters.items())
            for name, parameter in parameters.items():
                assert id(parameter) == identities[name]
                assert parameter.dtype == torch.bfloat16
                if parameter.requires_grad:
                    assert parameter.grad is not None, name
                    assert parameter.grad.dtype == parameter.dtype
                    assert torch.isfinite(parameter.grad).all(), name
                else:
                    assert parameter.grad is None
            optimizer.step()
    handle.remove()
    assert hook_calls == ["gather", "gather"]
    assert all(
        row == (torch.float32, torch.float32, torch.float32 if bias else None)
        for row in backbone.conv.seen
    )
    assert "sparse boundary [conv]" in caplog.text and "output=torch.float32" in caplog.text
    assert set(encoder.state_dict()) == keys
    if not trainable:
        assert all(
            torch.equal(parameter, initial[name])
            for name, parameter in parameters.items()
            if name.startswith("backbone.")
        )


def test_sparse_boundary_residual_and_idempotent_setup():
    backbone = _SparseBackbone()
    encoder = point_encoder.ConcertoPointEncoder(
        backbone=backbone,
        backbone_output_dim=4,
        point_dim=8,
        pretrained_loaded=True,
    ).to(torch.bfloat16)
    forward = backbone.conv._conv_forward
    encoder._configure_sparse_conv_precision()
    assert backbone.conv._conv_forward is forward
    features = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(features, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = backbone.conv(_SparseFeatures(features), _SparseFeatures(residual))
    assert result.features.dtype == torch.float32
    result.features.square().mean().backward()
    assert features.grad is not None and residual.grad is not None
    assert torch.isfinite(features.grad).all() and torch.isfinite(residual.grad).all()


def test_unknown_sparse_conv_api_fails_before_forward():
    class UnsupportedConv(nn.Module):
        __module__ = "spconv.pytorch.conv"

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4))

        def _conv_forward(self, changed_signature):
            raise AssertionError("must reject this API before execution")

    backbone = _SparseBackbone()
    backbone.conv = UnsupportedConv()
    with pytest.raises(RuntimeError, match="Unsupported spconv convolution API"):
        point_encoder.ConcertoPointEncoder(
            backbone=backbone,
            backbone_output_dim=4,
            point_dim=8,
            pretrained_loaded=True,
        )
