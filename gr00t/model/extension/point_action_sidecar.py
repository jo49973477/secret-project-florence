"""Small independent bidirectional stages; masks use True for valid tokens."""

import math

import torch
from torch import nn

from gr00t.model.extension.multimodal_dit import ModalityCrossAttention


def stage_indices(depth, count=4, indices=None):
    if depth < 1 or count < 1:
        raise ValueError("depth and stage count must be positive")
    result = (
        list(indices)
        if indices is not None
        else [math.ceil((i + 1) * depth / min(count, depth)) - 1 for i in range(min(count, depth))]
    )
    if not result or result != sorted(set(result)) or result[0] < 0 or result[-1] >= depth:
        raise ValueError("stage indices must be unique, increasing, zero-based block indices")
    return result


class TokenPool(nn.Module):
    def __init__(self, input_dim, dim, count, heads):
        super().__init__()
        if count < 1:
            raise ValueError("token count must be positive")
        self.queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        self.projection = nn.Linear(input_dim, dim)
        self.attention = ModalityCrossAttention(dim, dim, heads)

    def forward(self, tokens, mask=None):
        tokens = self.projection(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        output = self.attention(queries, encoder_hidden_states=tokens, encoder_attention_mask=mask)
        valid = torch.ones(output.shape[:2], dtype=torch.bool, device=output.device)
        if mask is not None:
            valid = valid & mask.bool().any(-1, keepdim=True)
        return output, valid


class PointActionStage(nn.Module):
    def __init__(self, action_dim, dim, heads):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, dim)
        self.output_projection = nn.Linear(dim, action_dim)
        self.action_norm = nn.LayerNorm(dim)
        self.point_norm = nn.LayerNorm(dim)
        self.updated_point_norm = nn.LayerNorm(dim)
        self.point_reads_action = ModalityCrossAttention(dim, dim, heads)
        self.action_reads_point = ModalityCrossAttention(dim, dim, heads)
        self.point_gate = nn.Parameter(torch.ones(()))
        self.action_gate = nn.Parameter(torch.zeros(()))

    def forward(self, action, points, point_mask, action_mask=None, observer=None, stage=0):
        query = self.action_norm(self.action_projection(action))
        delta = self.point_reads_action(
            self.point_norm(points), encoder_hidden_states=query, encoder_attention_mask=action_mask
        )
        point_update = self.point_gate * delta * point_mask[..., None].to(points.dtype)
        if observer is not None:
            observer("sidecar_point", stage, points, point_update)
        points = (points + point_update) * point_mask[..., None].to(points.dtype)
        delta = self.action_reads_point(
            query,
            encoder_hidden_states=self.updated_point_norm(points),
            encoder_attention_mask=point_mask,
        )
        update = self.action_gate * self.output_projection(delta)
        update = update * point_mask.any(-1)[:, None, None].to(update.dtype)
        if observer is not None:
            observer("sidecar_action", stage, action, update)
        return action + update, points
