"""DiT-only LoRA with the original checkpoint's Linear weight keys intact."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiTLoRALinear(nn.Linear):
    def __init__(self, source: nn.Linear, rank: int, alpha: float, dropout: float):
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid DiT LoRA rank, alpha, or dropout")
        super().__init__(
            source.in_features,
            source.out_features,
            source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        if not source.weight.is_meta:
            with torch.no_grad():
                self.weight.copy_(source.weight)
                if source.bias is not None:
                    self.bias.copy_(source.bias)
        self.lora_A = nn.Linear(
            source.in_features,
            rank,
            bias=False,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.lora_B = nn.Linear(
            rank,
            source.out_features,
            bias=False,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        if not self.lora_B.weight.is_meta:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_scale = alpha / rank
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, value):
        base = F.linear(value, self.weight, self.bias)
        return base + self.lora_B(self.lora_A(self.lora_dropout(value))) * self.lora_scale


def insert_dit_lora(dit: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.0):
    """Target inspected diffusers Attention projections in original transformer blocks only."""
    count = 0
    for block in dit.transformer_blocks:
        for attention_name in ("attn1", "attn2"):
            attention = getattr(block, attention_name, None)
            if attention is None:
                continue
            for name in ("to_q", "to_k", "to_v"):
                layer = getattr(attention, name, None)
                if layer is None:
                    continue
                if type(layer) is not nn.Linear:
                    raise TypeError(
                        f"Expected Linear at {attention_name}.{name}, got {type(layer)}"
                    )
                setattr(attention, name, DiTLoRALinear(layer, rank, alpha, dropout))
                count += 1
            if not isinstance(attention.to_out, nn.ModuleList | nn.Sequential):
                raise TypeError("Expected diffusers Attention.to_out container")
            layer = attention.to_out[0]
            if type(layer) is not nn.Linear:
                raise TypeError("Expected Linear at Attention.to_out.0")
            attention.to_out[0] = DiTLoRALinear(layer, rank, alpha, dropout)
            count += 1
    if count < len(dit.transformer_blocks) * 4:
        raise RuntimeError(f"DiT LoRA found only {count} attention projections")
    return count


def original_dit_parameters(dit: nn.Module):
    """Exclude Sidecar, temporal, MoE, and LoRA parameters from base DiT accounting."""
    excluded = ("sidecar_", "contact_", "temporal_", "input_contact_adapter")
    return (
        (name, p)
        for name, p in dit.named_parameters()
        if not name.startswith(excluded) and ".lora_" not in name
    )
