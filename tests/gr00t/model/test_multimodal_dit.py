# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for point- and tactile-conditioned DiT behavior."""

from gr00t.model.extension.multimodal_dit import MultiModalConditionedDiT
from gr00t.model.extension.point_encoder import (
    PointNet2Encoder,
    PointTransformerEncoder,
    build_point_encoder,
)
from gr00t.model.extension.tactile_encoder import TactileEncoder
from gr00t.model.modules.dit import AlternateVLDiT
import torch


def _dit_config() -> dict:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "output_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "final_dropout": False,
        "positional_embeddings": None,
        "interleave_self_attention": True,
        "cross_attention_dim": 6,
    }


def _multimodal_dit() -> MultiModalConditionedDiT:
    return MultiModalConditionedDiT(**_dit_config()).eval()


def _vlm_masks(batch_size: int = 2, sequence_length: int = 6) -> dict[str, torch.Tensor]:
    image_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    image_mask[:, sequence_length // 2 :] = True
    return {
        "image_mask": image_mask,
        "backbone_attention_mask": torch.ones_like(image_mask),
    }


def test_zero_gates_preserve_original_alternate_vl_dit_output() -> None:
    """New modality branches must begin as exact no-ops for old checkpoints."""
    torch.manual_seed(7)
    original_dit = AlternateVLDiT(**_dit_config()).eval()
    multimodal_dit = _multimodal_dit()

    missing_keys, unexpected_keys = multimodal_dit.load_state_dict(
        original_dit.state_dict(),
        strict=False,
    )
    assert missing_keys
    assert not unexpected_keys

    hidden_states = torch.randn(2, 5, 8)
    vlm_hidden_states = torch.randn(2, 6, 6)
    timesteps = torch.tensor([3, 9])
    point_tokens = torch.randn(2, 7, 8)
    tactile_tokens = torch.randn(2, 3, 8)
    vlm_masks = _vlm_masks()

    with torch.no_grad():
        original_output = original_dit(
            hidden_states,
            vlm_hidden_states,
            timestep=timesteps,
            **vlm_masks,
        )
        multimodal_output = multimodal_dit(
            hidden_states,
            vlm_hidden_states,
            timestep=timesteps,
            point_tokens=point_tokens,
            tactile_tokens=tactile_tokens,
            **vlm_masks,
        )

    torch.testing.assert_close(multimodal_output, original_output, rtol=0, atol=0)


def test_modality_attention_updates_only_action_slice() -> None:
    """The sensor branches must not directly overwrite state-token outputs."""
    torch.manual_seed(11)
    model = MultiModalConditionedDiT(**(_dit_config() | {"num_layers": 1})).eval()
    with torch.no_grad():
        model.point_gates.fill_(1.0)
        model.tactile_gates.fill_(1.0)

    hidden_states = torch.randn(2, 6, 8)
    vlm_hidden_states = torch.randn(2, 4, 6)
    timesteps = torch.tensor([2, 5])

    with torch.no_grad():
        _, baseline_hidden_states = model(
            hidden_states,
            vlm_hidden_states,
            timestep=timesteps,
            return_all_hidden_states=True,
            num_state_tokens=2,
            **_vlm_masks(sequence_length=4),
        )
        _, conditioned_hidden_states = model(
            hidden_states,
            vlm_hidden_states,
            timestep=timesteps,
            return_all_hidden_states=True,
            point_tokens=torch.randn(2, 7, 8),
            tactile_tokens=torch.randn(2, 3, 8),
            num_state_tokens=2,
            **_vlm_masks(sequence_length=4),
        )

    baseline_state_tokens = baseline_hidden_states[-1][:, :2]
    conditioned_state_tokens = conditioned_hidden_states[-1][:, :2]
    baseline_action_tokens = baseline_hidden_states[-1][:, 2:]
    conditioned_action_tokens = conditioned_hidden_states[-1][:, 2:]

    torch.testing.assert_close(conditioned_state_tokens, baseline_state_tokens)
    assert not torch.allclose(conditioned_action_tokens, baseline_action_tokens)


def test_fully_masked_modality_rows_produce_finite_outputs_and_gradients() -> None:
    """A sample with a missing sensor must not introduce attention NaNs."""
    torch.manual_seed(19)
    model = _multimodal_dit()

    point_attention_mask = torch.tensor(
        [
            [True, True, True, False, False, False, False],
            [False, False, False, False, False, False, False],
        ]
    )
    tactile_attention_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

    output = model(
        hidden_states=torch.randn(2, 5, 8),
        encoder_hidden_states=torch.randn(2, 6, 6),
        timestep=torch.tensor([1, 4]),
        point_tokens=torch.randn(2, 7, 8),
        tactile_tokens=torch.randn(2, 3, 8),
        point_attention_mask=point_attention_mask,
        tactile_attention_mask=tactile_attention_mask,
        **_vlm_masks(),
    )
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert model.point_gates.grad is not None
    assert model.tactile_gates.grad is not None
    assert torch.isfinite(model.point_gates.grad).all()
    assert torch.isfinite(model.tactile_gates.grad).all()


def test_modality_encoder_shapes() -> None:
    """The standard 1024-point and 224-pixel inputs produce fusion-ready tokens."""
    torch.manual_seed(23)
    points = torch.randn(2, 1024, 6)
    tactile_images = torch.randn(2, 3, 224, 224)

    pointnet2 = PointNet2Encoder(point_dim=256).eval()
    point_transformer = PointTransformerEncoder(point_dim=256).eval()
    tactile_encoder = TactileEncoder(tactile_dim=256).eval()

    with torch.no_grad():
        pointnet2_tokens = pointnet2(points)
        point_transformer_tokens = point_transformer(points)
        tactile_tokens = tactile_encoder(tactile_images)

    # PointNet++ downsamples to 64 centroids; the transformer retains all points.
    assert pointnet2_tokens.shape == (2, 64, 256)
    assert point_transformer_tokens.shape == (2, 1024, 256)
    # ResNet-18 layer4 is 7 x 7 for a 224 x 224 input. No global pooling is used.
    assert tactile_tokens.shape == (2, 49, 256)


def test_point_encoder_factory_and_multimodal_integration() -> None:
    """Both point backends expose tokens compatible with the unchanged fusion path."""
    assert isinstance(build_point_encoder("pointnet2"), PointNet2Encoder)
    assert isinstance(build_point_encoder("point_transformer"), PointTransformerEncoder)

    for encoder_type in ("pointnet2", "point_transformer"):
        model = MultiModalConditionedDiT(
            **(_dit_config() | {"num_layers": 1}),
        ).eval()
        point_encoder = build_point_encoder(encoder_type, point_dim=8).eval()
        tactile_encoder = TactileEncoder(tactile_dim=8).eval()
        with torch.no_grad():
            point_tokens = point_encoder(torch.randn(2, 128, 6))
            tactile_tokens = tactile_encoder(torch.randn(2, 3, 64, 64))
            output = model(
                hidden_states=torch.randn(2, 5, 8),
                encoder_hidden_states=torch.randn(2, 6, 6),
                timestep=torch.tensor([1, 4]),
                point_tokens=point_tokens,
                tactile_tokens=tactile_tokens,
                **_vlm_masks(),
            )

        assert output.shape == (2, 5, 8)
