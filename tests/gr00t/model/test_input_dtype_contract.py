# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw sensor precision remains independent of GR00T's dense model dtype."""

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.policy.gr00t_policy import _prepare_policy_model_inputs
import torch
from transformers.feature_extraction_utils import BatchFeature


class _IdentityInput:
    def prepare_input(self, batch):
        return BatchFeature(data=batch)


class _ActionInput(_IdentityInput):
    def __init__(self):
        self.state_encoder = SimpleNamespace(
            layer1=SimpleNamespace(W=torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16)))
        )
        self.action_encoder = SimpleNamespace(
            W1=SimpleNamespace(W=torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16)))
        )


def test_prepare_input_keeps_point_and_tactile_fp32() -> None:
    model = SimpleNamespace(
        backbone=_IdentityInput(),
        action_head=_ActionInput(),
        config=SimpleNamespace(
            point_encoder_cfg="concerto_small", tactile_encoder_cfg="sparsh_dino_base"
        ),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        _dtype_diagnostics_logged=True,
    )
    batch = {
        "pixel_values": torch.rand(1, 3, 4, 4),
        "state": torch.rand(1, 2, 4),
        "action": torch.rand(1, 3, 4),
        "points": torch.tensor([[[0.03125, 0.125, 0.5, 128, 128, 128]]]),
        "point_mask": torch.tensor([[True]]),
        "tactile": torch.rand(1, 2, 3, 8, 8),
    }

    backbone, action = Gr00tN1d7.prepare_input(model, batch)

    assert backbone["pixel_values"].dtype == torch.float32
    assert action["state"].dtype == torch.bfloat16
    assert action["action"].dtype == torch.bfloat16
    assert action["points"].dtype == torch.float32
    assert action["tactile"].dtype == torch.float32
    assert action["point_mask"].dtype == torch.bool
    torch.testing.assert_close(action["points"], batch["points"])


def test_prepare_input_keeps_legacy_sensor_model_dtype() -> None:
    model = SimpleNamespace(
        backbone=_IdentityInput(),
        action_head=_ActionInput(),
        config=SimpleNamespace(point_encoder_cfg="pointnet2", tactile_encoder_cfg="resnet18"),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        _dtype_diagnostics_logged=True,
    )
    _backbone, action = Gr00tN1d7.prepare_input(
        model,
        {"points": torch.rand(1, 8, 6), "tactile": torch.rand(1, 3, 8, 8)},
    )
    assert action["points"].dtype == torch.bfloat16
    assert action["tactile"].dtype == torch.bfloat16


def test_policy_preserves_n1d7_sensor_precision_but_retains_legacy_cast() -> None:
    raw = {"inputs": {"points": torch.rand(1, 8, 6), "tactile": torch.rand(1, 2, 3, 8, 8)}}
    n1d7 = SimpleNamespace(config=SimpleNamespace(model_type="Gr00tN1d7"))
    legacy = SimpleNamespace(config=SimpleNamespace(model_type="other"))

    prepared = _prepare_policy_model_inputs(raw, n1d7)
    legacy_prepared = _prepare_policy_model_inputs(raw, legacy)

    assert prepared is raw
    assert prepared["inputs"]["points"].dtype == torch.float32
    assert prepared["inputs"]["tactile"].dtype == torch.float32
    assert legacy_prepared["inputs"]["points"].dtype == torch.bfloat16
