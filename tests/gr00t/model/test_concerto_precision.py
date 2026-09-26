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
def test_spconv_implicit_gemm_receives_fp16_and_backpropagates(monkeypatch) -> None:
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
    original_parameter_id = id(backbone.conv.weight)
    observed: list[tuple[torch.dtype, torch.dtype]] = []
    original_implicit_gemm = spconv_ops.implicit_gemm

    def record_implicit_gemm(features, filters, *args, **kwargs):
        observed.append((features.dtype, filters.dtype))
        return original_implicit_gemm(features, filters, *args, **kwargs)

    monkeypatch.setattr(spconv_ops, "implicit_gemm", record_implicit_gemm)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = encoder(_points("cuda", torch.bfloat16))
        loss = tokens.float().square().mean()
    loss.backward()

    assert observed and all(pair == (torch.float16, torch.float16) for pair in observed)
    assert backbone.output_dtype == torch.float16
    assert tokens.dtype == torch.bfloat16
    assert torch.isfinite(tokens).all()
    assert id(backbone.conv.weight) == original_parameter_id
    assert backbone.conv.weight.grad is not None
    assert torch.isfinite(backbone.conv.weight.grad).all()
    assert backbone.conv.weight.grad.abs().sum() > 0
