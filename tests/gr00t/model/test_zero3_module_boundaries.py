# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for containers visible to DeepSpeed ZeRO-3 hooks."""

from types import SimpleNamespace

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def test_qwen_module_returns_plain_dict_for_zero3_hooks() -> None:
    class FakeQwen:
        config = SimpleNamespace(image_token_id=7)

        def __call__(self, **inputs):
            assert inputs["output_hidden_states"] is True
            return SimpleNamespace(hidden_states=[torch.ones(1, 2, 4, requires_grad=True)])

    backbone = SimpleNamespace(
        model=FakeQwen(),
        set_frozen_modules_to_eval_mode=lambda: None,
        _dtype_diagnostics_logged=True,
    )
    output = Qwen3Backbone.forward(
        backbone,
        {
            "input_ids": torch.tensor([[7, 1]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        },
    )

    assert type(output) is dict
    assert output["backbone_features"].requires_grad


def test_deepspeed_finds_plain_dict_tensors_but_not_batchfeature() -> None:
    zero_utils = pytest.importorskip("deepspeed.runtime.zero.utils")
    tensor = torch.ones(2, requires_grad=True)
    touched = []

    def visit(value):
        touched.append(value)
        return value

    zero_utils.apply_to_tensors_only(visit, {"features": tensor})
    assert len(touched) == 1 and touched[0] is tensor

    touched.clear()
    zero_utils.apply_to_tensors_only(visit, BatchFeature(data={"features": tensor}))
    assert not touched
