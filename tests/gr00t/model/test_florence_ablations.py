"""Small behavioral contracts; actual checkpoint verification is a separate preflight."""

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, ModalityConfig
from gr00t.model.extension.contact_input_moe import ContactInputAdapter
from gr00t.model.extension.dit_lora import DiTLoRALinear
from gr00t.model.extension.point_encoder import ConcertoPointEncoder
from gr00t.model.extension.temporal_fusion import (
    LatentRingBuffer,
    TemporalFusion,
    masked_feature_norm_mean,
    masked_point_mean,
    stack_padded_point_frames,
)
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def test_lora_exact_start_and_optimizer_step():
    torch.manual_seed(9)
    original = torch.nn.Linear(8, 8)
    adapted = DiTLoRALinear(original, rank=2, alpha=4, dropout=0)
    x = torch.randn(3, 8, requires_grad=True)
    assert torch.equal(original(x), adapted(x))
    before = adapted.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in adapted.parameters() if p.requires_grad), lr=0.01)
    adapted(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert adapted.lora_B.weight.grad is not None
    optimizer.step()
    assert torch.equal(adapted.weight, before)
    assert not torch.equal(adapted.lora_B.weight, torch.zeros_like(adapted.lora_B.weight))


def test_full_weight_updates_from_exact_initialization():
    original = torch.nn.Linear(8, 8)
    full = torch.nn.Linear(8, 8)
    full.load_state_dict(original.state_dict(), strict=True)
    assert torch.equal(full.weight, original.weight)
    before = full.weight.detach().clone()
    optimizer = torch.optim.AdamW(full.parameters(), lr=0.01)
    full(torch.randn(3, 8)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(full.weight, before)


def test_temporal_causality_masks_and_gradients():
    torch.manual_seed(8)
    fusion = TemporalFusion(8, 3, 5)
    points = torch.randn(2, 5, 4, 8, requires_grad=True)
    tactile = torch.randn(2, 5, 3, 8, requires_grad=True)
    state = torch.randn(2, 5, 3, requires_grad=True)
    valid = torch.tensor([[False, False, True, True, True], [True] * 5])
    output, mask = fusion(points, tactile, state, valid)
    assert output.shape == (2, 5, 8)
    assert torch.equal(mask, valid)
    assert torch.equal(output[0, :2], torch.zeros_like(output[0, :2]))
    changed = tactile.detach().clone()
    changed[:, -1] += 100
    altered, _ = fusion(points, changed, state, valid)
    assert torch.equal(output[:, :-1], altered[:, :-1])
    output.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


@pytest.mark.parametrize(
    "counts",
    [
        [15, 15, 14, 16, 13],
        [11, 13, 9, 15, 12, 14, 10, 16, 8, 13],
        [15] * 5,
    ],
)
def test_variable_point_token_frames_pad_without_losing_masks(counts):
    frames = [torch.randn(2, count, 1536) for count in counts]
    masks = [torch.ones(2, count, dtype=torch.bool) for count in counts]
    masks[0][1, -1] = False
    history, history_mask = stack_padded_point_frames(frames, masks)
    assert history.shape == (2, len(counts), max(counts), 1536)
    assert history_mask.shape == (2, len(counts), max(counts))
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        assert torch.equal(history[:, index, : frame.shape[1]], frame)
        assert torch.equal(history_mask[:, index, : frame.shape[1]], mask)
        assert not history_mask[:, index, frame.shape[1] :].any()


def test_concerto_padding_is_per_frame_not_per_temporal_window():
    first, first_mask = ConcertoPointEncoder._pad_tokens(torch.randn(29, 8), [15, 14])
    second, second_mask = ConcertoPointEncoder._pad_tokens(torch.randn(27, 8), [13, 14])
    assert first.shape == (2, 15, 8)
    assert second.shape == (2, 14, 8)
    history, mask = stack_padded_point_frames([first, second], [first_mask, second_mask])
    assert history.shape == (2, 2, 15, 8)
    assert not mask[:, 1, -1].any()


def test_temporal_point_masks_exclude_padding_and_empty_frames_from_values_and_gradients():
    torch.manual_seed(54)
    frames = [torch.randn(2, count, 8, requires_grad=True) for count in [4, 2, 3, 5, 2]]
    masks = [torch.ones(2, count, dtype=torch.bool) for count in [4, 2, 3, 5, 2]]
    masks[1][0] = False
    masks[2][1, -1] = False
    history, history_mask = stack_padded_point_frames(frames, masks)
    valid = torch.tensor([[False, True, True, True, True], [True] * 5])
    fusion = TemporalFusion(8, 3, 5)
    tactile = torch.randn(2, 5, 2, 8)
    state = torch.randn(2, 5, 3)
    output, _ = fusion(history, tactile, state, valid, history_mask)
    altered = history.detach().clone()
    altered[~history_mask] = 1e6
    altered_output, _ = fusion(altered, tactile, state, valid, history_mask)
    torch.testing.assert_close(output, altered_output, rtol=0, atol=0)
    altered[~history_mask] = float("nan")
    nan_output, _ = fusion(altered, tactile, state, valid, history_mask)
    assert torch.isfinite(nan_output).all()
    assert masked_feature_norm_mean(altered, history_mask).isfinite()
    example = torch.tensor([[[[3.0, 4.0], [float("nan"), float("nan")]]]])
    assert masked_feature_norm_mean(example, torch.tensor([[[True, False]]])) == 5
    output.square().mean().backward()
    assert all(frame.grad is not None and torch.isfinite(frame.grad).all() for frame in frames)
    assert torch.equal(frames[1].grad[0], torch.zeros_like(frames[1].grad[0]))
    assert frames[3].grad[0].abs().sum() > 0


def test_variable_point_history_is_causal():
    torch.manual_seed(55)
    frames = [torch.randn(1, count, 8) for count in [3, 5, 2, 4, 6]]
    masks = [torch.ones(1, frame.shape[1], dtype=torch.bool) for frame in frames]
    history, history_mask = stack_padded_point_frames(frames, masks)
    fusion = TemporalFusion(8, 3, 5).eval()
    tactile = torch.randn(1, 5, 2, 8)
    state = torch.randn(1, 5, 3)
    valid = torch.ones(1, 5, dtype=torch.bool)
    original, _ = fusion(history, tactile, state, valid, history_mask)
    changed = history.clone()
    changed[:, -1] += 100
    later, _ = fusion(changed, tactile, state, valid, history_mask)
    torch.testing.assert_close(original[:, :-1], later[:, :-1], rtol=0, atol=0)


def test_variable_point_tokens_match_short_inference_ring_history():
    torch.manual_seed(18)
    fusion = TemporalFusion(8, 3, 5).eval()
    frames = [torch.randn(1, count, 8) for count in [3, 4, 2, 5, 3]]
    masks = [torch.ones(1, count, dtype=torch.bool) for count in [3, 4, 2, 5, 3]]
    valid = torch.tensor([[False, False, True, True, True]])
    history, history_mask = stack_padded_point_frames(frames, masks)
    tactile = torch.randn(1, 5, 2, 8)
    state = torch.randn(1, 5, 3)
    with torch.no_grad():
        direct, _ = fusion(history, tactile, state, valid, history_mask)
        ring = LatentRingBuffer(5)
        for index in range(5):
            if valid[0, index]:
                ring.push(
                    "episode-a",
                    torch.stack(
                        (
                            masked_point_mean(frames[index], masks[index])[0],
                            tactile[0, index].mean(0),
                            fusion.state_projection(state[0, index]),
                        )
                    ),
                )
        packed, ring_mask = ring.window("episode-a", torch.zeros(3, 8))
        streamed, _ = fusion.forward_pooled(
            packed[None, :, 0], packed[None, :, 1], packed[None, :, 2], ring_mask[None]
        )
        assert ring_mask.tolist() == [False, False, True, True, True]
        torch.testing.assert_close(direct, streamed, rtol=0, atol=1e-6)
        ring.reset("episode-a")
        assert not ring.window("episode-a", torch.zeros(3, 8))[1].any()


def test_small_temporal_sidecar_forward_backward_and_optimizer_step():
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from tests.gr00t.model.test_action_head import (
        _make_action_input,
        _make_backbone_output,
        _small_multimodal_config,
    )

    class VariablePointEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 64)
            self.counts = [15, 15, 14, 16, 13]
            self.cursor = 0

        def forward(self, points, *, return_mask=False):
            count = self.counts[self.cursor % len(self.counts)]
            self.cursor += 1
            tokens = self.projection(points[:, :count])
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            mask[1, -1] = False
            return (tokens, mask) if return_mask else tokens

    class SmallTactileEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, 64)

        def forward(self, images):
            return self.projection(images.mean(dim=(-1, -2, -3), keepdim=False)[..., None])

    cfg = _small_multimodal_config(
        dit_type="sidecar_conditioned_dit",
        temporal_window_size=5,
        tactile_fusion_mode="none",
        contact_prediction_loss_weight=0.0,
        contact_injection_enabled=False,
        sidecar_hidden_dim=16,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=16,
        contact_num_tokens=2,
        tune_diffusion_model=False,
        freeze_pretrained_dit=True,
    )
    head = Gr00tN1d7ActionHead(cfg).train()
    head.point_encoder = VariablePointEncoder()
    head.tactile_encoder = SmallTactileEncoder()
    action = _make_action_input(cfg)
    action["points"] = torch.randn(2, 5, 16, 3)
    action["tactile"] = torch.randn(2, 6, 3, 8, 8)
    action["temporal_state"] = torch.randn(2, 5, cfg.max_state_dim)
    action["temporal_valid"] = torch.tensor([[False, False, True, True, True], [True] * 5])
    backbone = _make_backbone_output(cfg)
    base_before = [p.detach().clone() for p in head.model.transformer_blocks.parameters()]
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=1e-2)

    first = head(backbone, action)["loss"]
    assert torch.isfinite(first)
    first.backward()
    assert any(
        stage.action_gate.grad is not None
        and torch.isfinite(stage.action_gate.grad).all()
        and stage.action_gate.grad.abs().sum() > 0
        for stage in head.model.sidecar_stages
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = head(backbone, action)["loss"]
    assert torch.isfinite(second)
    second.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in head.temporal_fusion.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in head.point_encoder.parameters()
    )
    assert all(p.grad is None for p in head.model.transformer_blocks.parameters())
    assert all(
        torch.equal(p, before)
        for p, before in zip(head.model.transformer_blocks.parameters(), base_before)
    )


@pytest.mark.parametrize(
    "mode,temporal,moe",
    [("lora", 5, False), ("full", 5, False), ("frozen", 0, True), ("lora", 0, True)],
)
def test_other_florence_modes_instantiate_small_action_heads(mode, temporal, moe):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from tests.gr00t.model.test_action_head import _small_multimodal_config

    cfg = _small_multimodal_config(
        dit_type="sidecar_conditioned_dit",
        temporal_window_size=temporal,
        contact_input_mode="moe_soft" if moe else "none",
        tactile_fusion_mode="none",
        contact_prediction_loss_weight=0.0,
        contact_injection_enabled=False,
        sidecar_hidden_dim=16,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=16,
        contact_num_tokens=2,
        dit_training_mode=mode,
        tune_diffusion_model=mode == "full",
        freeze_pretrained_dit=mode != "full",
    )
    head = Gr00tN1d7ActionHead(cfg)
    assert (head.temporal_fusion is not None) == bool(temporal)
    assert (head.model.input_contact_adapter is not None) == moe
    assert any(p.requires_grad for p in head.model.transformer_blocks.parameters()) == (
        mode != "frozen"
    )


def test_ring_wrap_reset_and_environment_isolation():
    ring = LatentRingBuffer(5)
    for step in range(7):
        ring.push("a", torch.tensor([float(step)]))
    ring.push("b", torch.tensor([99.0]))
    values, valid = ring.window("a", torch.zeros(1))
    assert valid.all()
    assert torch.equal(values[:, 0], torch.arange(2, 7).float())
    other, mask = ring.window("b", torch.zeros(1))
    assert mask.tolist() == [False, False, False, False, True]
    assert other[-1, 0] == 99
    ring.reset("a")
    _, mask = ring.window("a", torch.zeros(1))
    assert not mask.any()
    assert ring.window("b", torch.zeros(1))[1].any()


def test_ring_and_differentiable_window_have_equal_temporal_semantics():
    torch.manual_seed(21)
    fusion = TemporalFusion(8, 3, 5).eval()
    point = torch.randn(1, 5, 2, 8)
    tactile = torch.randn(1, 5, 2, 8)
    state = torch.randn(1, 5, 3)
    valid = torch.tensor([[False, False, True, True, True]])
    direct, _ = fusion(point, tactile, state, valid)
    ring = LatentRingBuffer(5)
    with torch.no_grad():
        for index in range(5):
            if valid[0, index]:
                packed = torch.stack(
                    (
                        point[0, index].mean(0),
                        tactile[0, index].mean(0),
                        fusion.state_projection(state[0, index]),
                    )
                )
                ring.push("env", packed)
        cached, cached_valid = ring.window("env", torch.zeros(3, 8))
        streamed, _ = fusion.forward_pooled(
            cached[None, :, 0], cached[None, :, 1], cached[None, :, 2], cached_valid[None]
        )
    assert torch.equal(cached_valid[None], valid)
    assert torch.allclose(direct, streamed, atol=1e-6, rtol=0)


def test_contact_moe_mask_and_zero_residual_parity():
    adapter = ContactInputAdapter(8, 3, mode="moe_soft", contact_dim=4, num_tokens=2)
    tactile = torch.randn(2, 3, 8)
    state = torch.randn(2, 1, 3)
    contact = adapter.prepare_contact(tactile, state, torch.tensor([[True] * 3, [False] * 3]))
    probabilities = adapter.routing_probabilities(state, contact.observed, contact.valid)
    assert probabilities[1, 0] == 1
    assert probabilities[1, 1] == 0
    action = torch.randn(2, 4, 8)
    assert torch.equal(adapter(action, contact, state), action)
    assert adapter.surprise_expert is None


def test_temporal_dataset_alignment_and_episode_boundary():
    offsets = [-4, -3, -2, -1, 0]
    configs = {
        "pointcloud": ModalityConfig(delta_indices=offsets, modality_keys=["scene"]),
        "tactile": ModalityConfig(delta_indices=[-5, *offsets], modality_keys=["rgb"]),
        "state": ModalityConfig(delta_indices=offsets, modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }

    def episode(start):
        return pd.DataFrame(
            {
                "pointcloud.scene": [
                    np.full((2, 6), start + i, dtype=np.float32) for i in range(8)
                ],
                "tactile.rgb": [np.full((2, 2, 3), start + i, dtype=np.uint8) for i in range(8)],
                "state.joint": [np.array([start + i], dtype=np.float32) for i in range(8)],
                "language.task": ["task"] * 8,
            }
        )

    first = extract_step_data(episode(0), 0, configs, EmbodimentTag.NEW_EMBODIMENT, True)
    assert first.metadata["temporal_valid"] == [False, False, False, False, True]
    assert first.pointclouds["scene"][:, 0, 0].tolist() == [0] * 5
    second = extract_step_data(episode(100), 5, configs, EmbodimentTag.NEW_EMBODIMENT, True)
    assert second.metadata["temporal_valid"] == [True] * 5
    assert second.pointclouds["scene"][:, 0, 0].tolist() == [101, 102, 103, 104, 105]
    assert [frame[0, 0, 0] for frame in second.tactile["rgb"]] == [100, 101, 102, 103, 104, 105]
