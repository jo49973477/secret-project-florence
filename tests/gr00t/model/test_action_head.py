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

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

import math
from unittest.mock import patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        dit_type="dit",
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    image_mask[:, seq_len // 2 :] = True
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": image_mask,
        }
    )


def _make_action_input(config, batch_size=2):
    data = {
        "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
        "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
    }
    if config.dit_type == "multimodal_conditioned_dit" and config.use_point_conditioning:
        data["points"] = torch.randn(batch_size, 16, config.point_input_dim)
    if config.dit_type == "multimodal_conditioned_dit" and config.use_tactile_conditioning:
        data["tactile"] = torch.randn(batch_size, config.tactile_input_channels, 32, 32)
    return BatchFeature(data=data)


def _small_multimodal_config(**overrides) -> Gr00tN1d7Config:
    return _small_config(
        dit_type="multimodal_conditioned_dit",
        point_encoder_cfg="point_transformer",
        point_input_dim=3,
        tactile_input_channels=3,
        **overrides,
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


class TestMultimodalActionHead:
    @pytest.mark.parametrize("dit_type", ["alternate_vl_dit", "dit"])
    def test_original_modes_do_not_instantiate_sensor_encoders(self, dit_type):
        head = Gr00tN1d7ActionHead(_small_config(dit_type=dit_type))

        assert head.point_encoder is None
        assert head.tactile_encoder is None

    @pytest.mark.parametrize(
        "use_points,use_tactile",
        [(True, False), (False, True), (True, True)],
    )
    def test_enabled_modality_combinations_run(self, use_points, use_tactile):
        config = _small_multimodal_config(
            use_point_conditioning=use_points,
            use_tactile_conditioning=use_tactile,
        )
        head = Gr00tN1d7ActionHead(config).eval()
        output = head(_make_backbone_output(config), _make_action_input(config))

        assert torch.isfinite(output["loss"])
        assert (head.point_encoder is not None) is use_points
        assert (head.tactile_encoder is not None) is use_tactile

    def test_multimodal_adapters_can_train_with_frozen_base_dit(self):
        config = _small_multimodal_config(
            tune_diffusion_model=False,
            tune_point_encoder=True,
            tune_tactile_encoder=True,
            tune_multimodal_adapter=True,
        )
        head = Gr00tN1d7ActionHead(config)

        frozen_base_modules = (
            head.model.timestep_encoder,
            head.model.transformer_blocks,
            head.model.proj_out_1,
            head.model.proj_out_2,
        )
        assert not any(
            parameter.requires_grad
            for module in frozen_base_modules
            for parameter in module.parameters()
        )
        assert all(parameter.requires_grad for parameter in head.point_encoder.parameters())
        assert all(parameter.requires_grad for parameter in head.tactile_encoder.parameters())
        assert all(
            parameter.requires_grad for parameter in head.model.point_cross_attention.parameters()
        )
        assert all(
            parameter.requires_grad for parameter in head.model.tactile_cross_attention.parameters()
        )
        assert head.model.point_gates.requires_grad
        assert head.model.tactile_gates.requires_grad

    def test_multimodal_adapters_can_freeze_with_trainable_base_dit(self):
        config = _small_multimodal_config(
            tune_diffusion_model=True,
            tune_multimodal_adapter=False,
        )
        head = Gr00tN1d7ActionHead(config)

        assert all(
            parameter.requires_grad for parameter in head.model.transformer_blocks.parameters()
        )
        assert not any(
            parameter.requires_grad for parameter in head.model.point_cross_attention.parameters()
        )
        assert not any(
            parameter.requires_grad for parameter in head.model.tactile_cross_attention.parameters()
        )
        assert not head.model.point_gates.requires_grad
        assert not head.model.tactile_gates.requires_grad

    def test_frozen_sensor_encoders_stay_in_eval_mode(self):
        config = _small_multimodal_config(
            tune_point_encoder=False,
            tune_tactile_encoder=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        head.set_frozen_modules_to_eval_mode()

        assert not head.point_encoder.training
        assert not head.tactile_encoder.training

    def test_inference_encodes_static_modalities_once(self):
        config = _small_multimodal_config(num_inference_timesteps=3)
        head = Gr00tN1d7ActionHead(config).eval()
        action_input = _make_action_input(config)
        del action_input["action"]

        with (
            patch.object(
                head.point_encoder, "forward", wraps=head.point_encoder.forward
            ) as point_forward,
            patch.object(
                head.tactile_encoder,
                "forward",
                wraps=head.tactile_encoder.forward,
            ) as tactile_forward,
        ):
            head.get_action(_make_backbone_output(config), action_input)

        assert point_forward.call_count == 1
        assert tactile_forward.call_count == 1


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


def _beta_time_moments(alpha: float, beta: float, noise_s: float) -> tuple[float, float]:
    """Closed-form mean/variance of sample_time's output, derived from config.

    sample_time draws ``b ~ Beta(alpha, beta)`` and returns ``s = (1 - b) * noise_s``.
    The moments are re-derived here from the config values (not hand-picked
    constants and not a fingerprint of whatever the code currently emits), so
    the assertions are an independent oracle for the sampler rather than an
    author-vs-author restatement.
    """
    mean_b = alpha / (alpha + beta)
    var_b = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    mean_s = (1.0 - mean_b) * noise_s
    var_s = (noise_s**2) * var_b
    return mean_s, var_s


class TestActionHeadTimeSamplingMetaSafe:
    """Oracle tests for sample_time under meta / no_init_weights construction.

    Regression guard: when the action head is built while the default device is
    meta (as happens inside a nested from_pretrained), sample_time must still
    produce a valid, correctly-distributed noise schedule on the requested
    device — not a crash or uninitialized values from a meta-backed Beta.
    """

    @pytest.mark.parametrize(
        "alpha,beta,noise_s",
        [(1.5, 1.0, 0.999), (2.0, 3.0, 0.95)],
    )
    def test_sample_time_under_meta_construction(self, alpha, beta, noise_s):
        config = _small_config(noise_beta_alpha=alpha, noise_beta_beta=beta, noise_s=noise_s)
        # Reproduce the production failure condition: the whole action head is
        # instantiated while the default device is meta.
        with torch.device("meta"):
            head = Gr00tN1d7ActionHead(config)

        torch.manual_seed(0)
        n = 200_000
        sample = head.sample_time(n, device="cpu", dtype=torch.float32)

        # Requested device/dtype honored even though the module was built on meta.
        assert sample.device.type == "cpu"
        assert sample.dtype == torch.float32
        # No uninitialized / nan / inf leakage from a meta-backed distribution.
        assert torch.isfinite(sample).all()
        # Transformed-Beta support: s = (1 - b) * noise_s with b in [0, 1].
        assert (sample >= 0).all()
        assert (sample <= noise_s + 1e-6).all()

        # Distribution matches Beta(alpha, beta) transformed; RHS from config.
        mean_s, var_s = _beta_time_moments(alpha, beta, noise_s)
        assert math.isclose(sample.mean().item(), mean_s, abs_tol=5e-3)
        assert math.isclose(sample.var(unbiased=False).item(), var_s, rel_tol=0.1)

    def test_sample_time_construction_device_invariant(self):
        """Construction device must not change the sampled noise stream."""
        config = _small_config()
        head_real = Gr00tN1d7ActionHead(config)
        with torch.device("meta"):
            head_meta = Gr00tN1d7ActionHead(config)

        n = 200_000
        torch.manual_seed(1234)
        s_real = head_real.sample_time(n, device="cpu", dtype=torch.float32)
        torch.manual_seed(1234)
        s_meta = head_meta.sample_time(n, device="cpu", dtype=torch.float32)

        # Same seed + config-derived sampler ⇒ byte-identical stream regardless
        # of whether the head was constructed on a real device or on meta.
        assert torch.equal(s_real, s_meta)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad
