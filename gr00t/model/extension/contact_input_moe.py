# SPDX-License-Identifier: Apache-2.0
"""Observation-routed, zero-gated corrections before the unmodified Action DiT.

Mask convention: True means valid. Routing never accepts action embeddings.
The expert FiLM corrections do accept the current denoising action embedding.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn


@contextmanager
def _gathered_module(module, *, modify=False):
    """Startup-only full parameters for strict provenance/loading under ZeRO-3."""
    partitioned = [p for p in module.parameters() if hasattr(p, "ds_id")]
    if not partitioned:
        yield
        return
    from deepspeed import zero

    with zero.GatheredParameters(partitioned, modifier_rank=0 if modify else None):
        yield


def encoder_digest(encoder):
    """Startup-only identity check for the exact sensor representation."""
    digest = hashlib.sha256()
    with _gathered_module(encoder):
        for name, value in sorted(encoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


class ContactRepresentation(NamedTuple):
    """Sensor-independent adapter boundary; predicted slots must share alignment."""

    observed: torch.Tensor  # [B,K,C]
    valid: torch.Tensor  # [B], True means a usable observation
    predicted: torch.Tensor | None = None  # [B,K,C], only from a validated predictor


class ContactTokenProjector(nn.Module):
    """Learned masked pooling: B x N x D -> B x K x C, including empty rows."""

    def __init__(self, input_dim, contact_dim=256, num_tokens=4):
        super().__init__()
        if min(input_dim, contact_dim, num_tokens) <= 0:
            raise ValueError("Contact dimensions and token count must be positive")
        self.projection = nn.Linear(input_dim, contact_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, contact_dim) * 0.02)

    def forward(self, features, mask=None):
        b, n, _ = features.shape
        if n == 0:
            return features.new_zeros(b, *self.queries.shape), torch.zeros(
                b, device=features.device, dtype=torch.bool
            )
        if mask is None:
            mask = torch.ones(b, n, device=features.device, dtype=torch.bool)
        if mask.shape != (b, n):
            raise ValueError("Contact mask must match encoded sensor tokens [B,N]")
        mask = mask.bool()
        valid = mask.any(-1)
        # Remove invalid NaNs before any learned operation.
        values = self.projection(
            features.masked_fill(~mask[..., None], 0).to(self.projection.weight.dtype)
        )
        scores = torch.einsum("kc,bnc->bkn", self.queries.to(values.dtype), values)
        scores = scores.float() / self.queries.shape[-1] ** 0.5
        scores = scores.masked_fill(~mask[:, None], -torch.inf)
        scores = torch.where(valid[:, None, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).to(values.dtype)
        tokens = torch.einsum("bkn,bnc->bkc", weights, values)
        return tokens * valid[:, None, None], valid


class NoOpContactExpert(nn.Module):
    def forward(self, action_features, contact=None):
        return torch.zeros_like(action_features)


class DirectContactExpert(nn.Module):
    """Small FiLM correction; recomputed for every denoising action embedding."""

    def __init__(self, action_dim, contact_dim):
        super().__init__()
        self.norm = nn.LayerNorm(action_dim)
        self.film = nn.Linear(contact_dim, 2 * action_dim)

    def forward(self, action_features, contact):
        scale, shift = self.film(contact.mean(1)).chunk(2, -1)
        return self.norm(action_features) * scale[:, None].tanh() + shift[:, None]


class ContactSurpriseExpert(DirectContactExpert):
    """Same lightweight family, independent weights, aligned discrepancy input."""


class ContactRouter(nn.Module):
    def __init__(self, state_dim, contact_dim, num_experts=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + contact_dim, contact_dim),
            nn.SiLU(),
            nn.Linear(contact_dim, num_experts),
        )

    def forward(self, robot_state, observed_contact, available):
        logits = self.network(
            torch.cat((robot_state.flatten(1), observed_contact.mean(1)), -1).to(
                self.network[0].weight.dtype
            )
        )
        width = logits.shape[-1]
        probabilities = logits.float().masked_fill(~available[:, :width], -torch.inf).softmax(-1)
        # Stable diagnostics interface; an absent expert has no trainable logit
        # and contributes exactly zero probability, outside the softmax.
        return torch.nn.functional.pad(probabilities, (0, available.shape[-1] - width))


class ContactPredictor(nn.Module):
    """State-only predictor interface. No actions, target, timestep, or noise input."""

    def __init__(self, state_dim, contact_dim, num_tokens):
        super().__init__()
        self.num_tokens, self.contact_dim = num_tokens, contact_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, contact_dim),
            nn.SiLU(),
            nn.Linear(contact_dim, num_tokens * contact_dim),
        )

    def forward(self, robot_state):
        return self.network(robot_state.flatten(1).to(self.network[0].weight.dtype)).reshape(
            -1, self.num_tokens, self.contact_dim
        )


class ContactInputAdapter(nn.Module):
    def __init__(
        self,
        action_dim,
        state_dim,
        mode="moe_soft",
        contact_dim=256,
        num_tokens=4,
        use_surprise=False,
        diagnostics=False,
    ):
        super().__init__()
        if mode not in {"direct", "scalar_gate", "moe_soft"}:
            raise ValueError(f"Invalid adapter mode: {mode}")
        if use_surprise and mode != "moe_soft":
            raise ValueError("Surprise requires moe_soft")
        self.mode, self.use_surprise = mode, use_surprise
        self.diagnostics_enabled = diagnostics
        self.last_diagnostics = {}
        self.projector = ContactTokenProjector(action_dim, contact_dim, num_tokens)
        self.direct_expert = DirectContactExpert(action_dim, contact_dim)
        self.injection_scale = nn.Parameter(torch.zeros(1))
        self.router = (
            ContactRouter(state_dim, contact_dim, 3 if use_surprise else 2)
            if mode == "moe_soft"
            else None
        )
        self.scalar_gate = nn.Linear(state_dim + contact_dim, 1) if mode == "scalar_gate" else None
        self.surprise_expert = (
            ContactSurpriseExpert(action_dim, contact_dim) if use_surprise else None
        )
        self.predictor = (
            ContactPredictor(state_dim, contact_dim, num_tokens) if use_surprise else None
        )
        self.predictor_ready = False
        self.num_tokens, self.contact_dim = num_tokens, contact_dim
        if self.predictor is not None:
            self.predictor.requires_grad_(False)

    def load_predictor(self, path, encoder):
        """Load a provenance-bearing bundle, never an arbitrary Sidecar checkpoint.

        Bundle includes the trained projector and binds the *entire* sensor encoder
        by digest. This guards latent alignment, not empirical predictor quality.
        """
        bundle = torch.load(Path(path), map_location="cpu", weights_only=True)
        metadata = bundle["metadata"]
        expected = {
            "format": "florence-contact-state-predictor-v1",
            "num_tokens": self.num_tokens,
            "contact_dim": self.contact_dim,
            "encoder_sha256": encoder_digest(encoder),
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("Contact predictor representation/encoder mismatch")
        if not metadata.get("training_provenance") or not metadata.get("validation_report"):
            raise ValueError("Predictor requires training provenance and validation report")
        with _gathered_module(self.predictor, modify=True):
            self.predictor.load_state_dict(bundle["predictor"], strict=True)
        with _gathered_module(self.projector, modify=True):
            self.projector.load_state_dict(bundle["projector"], strict=True)
        self.predictor.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.predictor_ready = True

    def prepare_contact(self, tactile_features, robot_state, tactile_mask=None):
        contact, valid = self.projector(tactile_features, tactile_mask)
        predicted = None
        if self.use_surprise:
            if not self.predictor_ready:
                raise RuntimeError("Surprise requires a validated predictor bundle")
            predicted = self.predictor(robot_state)
        return ContactRepresentation(contact, valid, predicted)

    def routing_probabilities(self, robot_state, observed_contact, valid, predicted_contact=None):
        available = torch.stack(
            (
                torch.ones_like(valid),
                valid,
                valid & (predicted_contact is not None and self.use_surprise),
            ),
            -1,
        )
        return self.router(robot_state, observed_contact, available)

    def forward(self, action_features, contact, robot_state, action_mask=None):
        observed, valid, predicted = contact
        direct = self.direct_expert(action_features, observed)
        probs = None
        if self.mode == "direct":
            delta = direct
        elif self.mode == "scalar_gate":
            strength = self.scalar_gate(
                torch.cat((robot_state.flatten(1), observed.mean(1)), -1).to(
                    self.scalar_gate.weight.dtype
                )
            ).sigmoid()
            delta = strength[:, None] * direct
        else:
            probs = self.routing_probabilities(robot_state, observed, valid, predicted)
            delta = probs[:, 1, None, None].to(direct.dtype) * direct
            if self.use_surprise and predicted is not None:
                delta = delta + probs[:, 2, None, None].to(direct.dtype) * self.surprise_expert(
                    action_features, observed - predicted
                )
        delta = delta * valid[:, None, None]
        if action_mask is not None:
            delta = delta * action_mask[..., None].to(delta.dtype)
        delta = delta.to(action_features.dtype)
        residual = self.injection_scale.to(delta.dtype) * delta
        if self.diagnostics_enabled:
            # Gate value is measured from the actual forward tensor, while ZeRO
            # has gathered it. Never inspect its possibly partitioned storage later.
            metrics = {
                "contact_input_injection_scale": self.injection_scale.detach().float().mean(),
                "contact_input_residual_ratio": residual.detach().float().norm()
                / action_features.detach().float().norm().clamp_min(1e-8),
            }
            if probs is not None:
                metrics["contact_router_entropy"] = (
                    -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean().detach()
                )
                for i, name in enumerate(("noop", "direct", "surprise")):
                    if i == 2 and not self.use_surprise:
                        continue
                    metrics[f"contact_router_{name}_probability"] = probs[:, i].mean().detach()
                    metrics[f"contact_router_{name}_expert_usage"] = (
                        (probs.argmax(-1) == i).float().mean().detach()
                    )
            self.last_diagnostics = metrics
        return action_features + residual
