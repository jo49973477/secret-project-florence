"""Experimental contact latent slots, predicted exclusively from scene and state."""

import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.extension.multimodal_dit import ModalityCrossAttention
from gr00t.model.extension.point_action_sidecar import TokenPool


class ContactSurprise(nn.Module):
    def __init__(self, input_dim, dim=256, count=4, heads=4, freeze_teacher=True):
        super().__init__()
        self.projector = TokenPool(input_dim, dim, count, heads)
        self.scene_pool = TokenPool(input_dim, dim, count, heads)
        self.state_projection = nn.Linear(input_dim, dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.freeze_teacher = freeze_teacher
        self.projector.requires_grad_(not freeze_teacher)

    def forward(self, points, state, tactile, point_mask=None, tactile_mask=None):
        # Slot i predicts the observed teacher pool's fixed query slot i.
        observed, valid = self.projector(tactile, tactile_mask)
        if self.freeze_teacher:
            observed = observed.detach()
        scene, _ = self.scene_pool(points, point_mask)
        predicted = self.predictor(scene + self.state_projection(state).mean(1, keepdim=True))
        surprise = observed - predicted
        gate = self.gate(torch.cat((observed, predicted), -1))
        error = F.smooth_l1_loss(predicted.float(), observed.detach().float(), reduction="none")
        loss = (error * valid[..., None]).sum() / (valid.sum() * observed.shape[-1]).clamp_min(1)
        return observed, predicted, surprise, gate, valid, loss


class ContactInjection(nn.Module):
    def __init__(self, action_dim, dim, heads):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, dim))
        self.attention = ModalityCrossAttention(dim, dim, heads)
        self.output = nn.Linear(dim, action_dim)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, action, tokens, mask):
        delta = self.attention(
            self.query(action), encoder_hidden_states=tokens, encoder_attention_mask=mask
        )
        delta = self.output(delta) * mask.any(-1)[:, None, None].to(action.dtype)
        return action + self.scale * delta
