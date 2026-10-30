"""Small behavioral contracts; actual checkpoint verification is a separate preflight."""

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, ModalityConfig
from gr00t.model.extension.contact_input_moe import ContactInputAdapter
from gr00t.model.extension.dit_lora import DiTLoRALinear
from gr00t.model.extension.temporal_fusion import LatentRingBuffer, TemporalFusion
import numpy as np
import pandas as pd
import torch


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
