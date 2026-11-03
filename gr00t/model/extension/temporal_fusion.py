"""Causal, differentiable observation fusion and fixed-size inference history."""

import torch
from torch import nn
import torch.nn.functional as F


def stack_padded_point_frames(frames, masks):
    """Pad only the token axis; retain each encoder-provided validity mask."""
    if not frames or len(frames) != len(masks):
        raise ValueError("Point frames and masks must be nonempty and aligned")
    batch, _, width = frames[0].shape
    max_tokens = max(frame.shape[1] for frame in frames)
    for frame, mask in zip(frames, masks):
        if frame.ndim != 3 or frame.shape[0] != batch or frame.shape[2] != width:
            raise ValueError("Point frames must share batch size and embedding width")
        if mask.shape != frame.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Each point mask must be boolean [B,K]")
    return (
        torch.stack([F.pad(frame, (0, 0, 0, max_tokens - frame.shape[1])) for frame in frames], 1),
        torch.stack([F.pad(mask, (0, max_tokens - mask.shape[1])) for mask in masks], 1),
    )


def masked_point_mean(points, mask):
    """Pool valid point tokens; empty frames produce finite zero vectors."""
    safe_points = torch.where(mask[..., None], points, torch.zeros_like(points))
    return safe_points.sum(-2) / mask.sum(-1).clamp_min(1)[..., None]


def masked_feature_norm_mean(features, mask):
    """Exclude padded point tokens or missing frames from feature diagnostics."""
    while mask.ndim < features.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.expand(features.shape[:-1])
    safe_features = torch.where(mask[..., None], features.detach().float(), 0)
    return safe_features.norm(dim=-1).sum() / mask.sum().clamp_min(1)


class LatentRingBuffer:
    """Per-environment FIFO; entries are detached only for fixed-weight inference."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._entries = {}

    def reset(self, environment_id=None):
        if environment_id is None:
            self._entries.clear()
        else:
            self._entries.pop(environment_id, None)

    def has(self, environment_id):
        return environment_id in self._entries

    def push(self, environment_id, latent: torch.Tensor):
        if latent.requires_grad:
            raise ValueError(
                "Ring buffer is inference-only; training windows must remain differentiable"
            )
        entries, next_index = self._entries.get(environment_id, ([None] * self.window_size, 0))
        entries[next_index] = latent.detach()
        self._entries[environment_id] = (entries, (next_index + 1) % self.window_size)

    def window(self, environment_id, template: torch.Tensor):
        entries, next_index = self._entries.get(environment_id, ([None] * self.window_size, 0))
        chronological = entries[next_index:] + entries[:next_index]
        valid = torch.tensor([x is not None for x in chronological], device=template.device)
        return torch.stack(
            [x if x is not None else torch.zeros_like(template) for x in chronological]
        ), valid


class TemporalFusion(nn.Module):
    """Concat point, tactile and normalized state per timestamp; causal GRU."""

    def __init__(self, width: int, state_dim: int, window_size: int):
        super().__init__()
        if window_size not in (5, 10):
            raise ValueError("Temporal Fusion supports window_size 5 or 10")
        self.window_size = window_size
        self.state_projection = nn.Linear(state_dim, width)
        self.input_projection = nn.Linear(width * 3, width)
        self.time_embedding = nn.Embedding(window_size, width)
        self.gru_cell = nn.GRUCell(width, width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, points, tactile, state, valid, point_mask=None):
        if points.ndim != 4 or tactile.ndim != 4 or state.ndim != 3:
            raise ValueError("Expected point/tactile [B,T,N,D] and state [B,T,S]")
        batch, steps, _, width = points.shape
        if (
            steps != self.window_size
            or tactile.shape[:2] != (batch, steps)
            or state.shape[:2] != (batch, steps)
        ):
            raise ValueError("Modalities must share the configured temporal window")
        if valid.shape != (batch, steps) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,T] mask")
        if not valid[:, -1].all():
            raise ValueError("Current observation must be valid")
        if point_mask is None:
            point_features = points.mean(2)
        else:
            if point_mask.shape != points.shape[:3]:
                raise ValueError("point_mask must match [B,T,N]")
            point_features = masked_point_mean(points, point_mask)
        tactile_features = tactile.mean(2)
        state_features = self.state_projection(state.to(self.state_projection.weight.dtype))
        return self.forward_pooled(point_features, tactile_features, state_features, valid)

    def forward_pooled(self, point_features, tactile_features, state_features, valid):
        batch, steps, width = point_features.shape
        if (
            steps != self.window_size
            or tactile_features.shape != (batch, steps, width)
            or state_features.shape != (batch, steps, width)
        ):
            raise ValueError("Pooled modalities must share [B,T,D]")
        if valid.shape != (batch, steps) or valid.dtype != torch.bool or not valid[:, -1].all():
            raise ValueError("Temporal validity must be boolean [B,T] with a valid current frame")
        combined = torch.cat((point_features, tactile_features, state_features), -1)
        combined = self.input_projection(combined.to(self.input_projection.weight.dtype))
        positions = torch.arange(steps, device=point_features.device)
        combined = (combined + self.time_embedding(positions)) * valid[..., None]
        hidden = combined.new_zeros(batch, width)
        frames = []
        for index in range(steps):
            updated = self.gru_cell(combined[:, index], hidden)
            hidden = torch.where(valid[:, index, None], updated, hidden)
            frames.append(hidden)
        output = torch.stack(frames, 1)
        output = self.output_norm(output)
        return output * valid[..., None], valid
