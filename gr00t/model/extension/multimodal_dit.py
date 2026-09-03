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

"""GR00T DiT extension conditioned on point clouds and tactile images.

The original DiT blocks continue to process the joint ``[state; action]``
sequence and attend to VLM features. After each block, only the action tokens
query point-cloud and tactile tokens. Zero-initialized residual gates make the
new branches an identity operation when the module is first constructed.
"""

from typing import Optional

from diffusers.configuration_utils import register_to_config
import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.modules.dit import DiT, _sdpa_context

from .point_encoder import PointEncoder, build_point_encoder
from .tactile_encoder import TactileEncoder


def _validate_token_mask(
    token_mask: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    mask_name: str,
) -> torch.Tensor:
    """Validate a ``True means valid`` token mask and convert it to bool."""
    expected_shape = (batch_size, sequence_length)
    if token_mask.shape != expected_shape:
        raise ValueError(
            f"{mask_name} must have shape {expected_shape}, got {tuple(token_mask.shape)}."
        )

    return token_mask.to(dtype=torch.bool)


class ModalityCrossAttention(nn.Module):
    """Produce an action-token update from tokens belonging to one modality.

    Action tokens are the queries (Q), while modality tokens provide keys and
    values (K/V). The caller owns the residual connection and its gate.
    """

    def __init__(
        self,
        action_dim: int,
        modality_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim % num_heads != 0:
            raise ValueError(
                f"action_dim ({action_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.action_norm = nn.LayerNorm(action_dim)
        self.modality_norm = nn.LayerNorm(modality_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=action_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=modality_dim,
            vdim=modality_dim,
            batch_first=True,
        )

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, modality_sequence_length = encoder_hidden_states.shape[:2]
        key_padding_mask = None
        rows_with_valid_tokens = None

        if encoder_attention_mask is not None:
            valid_token_mask = _validate_token_mask(
                encoder_attention_mask,
                batch_size=batch_size,
                sequence_length=modality_sequence_length,
                mask_name="encoder_attention_mask",
            )
            valid_token_mask = valid_token_mask.to(device=encoder_hidden_states.device)

            # PyTorch uses True for tokens that must be ignored, the inverse of
            # the public mask contract used throughout this module.
            rows_with_valid_tokens = valid_token_mask.any(dim=1)
            safe_valid_token_mask = valid_token_mask.clone()

            # MultiheadAttention produces NaNs if every K/V token in one row is
            # masked. Temporarily expose a zero token and discard that row's
            # update below, allowing batches with a missing modality sample.
            rows_without_valid_tokens = ~rows_with_valid_tokens
            if rows_without_valid_tokens.any():
                encoder_hidden_states = encoder_hidden_states.clone()
                encoder_hidden_states[rows_without_valid_tokens, 0] = 0
                safe_valid_token_mask[rows_without_valid_tokens, 0] = True

            key_padding_mask = ~safe_valid_token_mask

        normalized_actions = self.action_norm(action_hidden_states)
        normalized_modality = self.modality_norm(encoder_hidden_states)

        # Q: action tokens. K/V: point or tactile tokens. This direction lets
        # each future action timestep retrieve the sensor evidence it needs.
        with _sdpa_context():
            action_update, _ = self.attention(
                query=normalized_actions,
                key=normalized_modality,
                value=normalized_modality,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        if rows_with_valid_tokens is not None:
            action_update = action_update * rows_with_valid_tokens[:, None, None].to(
                dtype=action_update.dtype
            )

        return action_update


class MultiModalConditionedDiT(DiT):
    """Extend GR00T DiT with point-cloud and tactile action conditioning.

    The model accepts the same joint state/action and VLM inputs as
    :class:`~gr00t.model.modules.dit.DiT`, plus raw point clouds and tactile
    images. At each transformer layer, the original GR00T block runs first.
    Only the action-token suffix then cross-attends to encoded point and
    tactile tokens before state and action tokens are recombined.

    Inputs:
        hidden_states: Joint ``[state; action]`` tokens, shape
            ``[B, N_state + N_action, D_dit]``.
        encoder_hidden_states: VLM tokens, shape ``[B, N_vlm, D_vlm]``.
        point_cloud: Optional point features, shape
            ``[B, N_point, point_input_dim]``.
        tactile_images: Optional images, shape ``[B, C_tactile, H, W]``.
        point_attention_mask: Optional valid-token mask, shape
            ``[B, N_point]``. ``True`` means the point may be attended to.
        tactile_attention_mask: Optional valid-token mask, shape
            ``[B, N_tactile]``. ``True`` means the patch may be attended to.
        num_state_tokens: Length of the state-token prefix in ``hidden_states``.

    Output:
        Projected state/action tokens with shape
        ``[B, N_state + N_action, output_dim]``. When
        ``return_all_hidden_states=True``, the second return value contains the
        initial hidden states followed by the conditioned states from each
        transformer layer.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
        point_input_dim: int = 6,
        point_encoder_hidden_dim: Optional[int] = None,
        point_encoder_type: str = "pointnet2",
        point_num_samples: tuple[int, int] = (256, 64),
        point_num_neighbors: tuple[int, int] = (32, 32),
        point_transformer_layers: int = 2,
        tactile_input_channels: int = 3,
        tactile_patch_size: int = 16,
        tactile_pretrained: bool = False,
        modality_attention_heads: Optional[int] = None,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            max_num_positional_embeddings=max_num_positional_embeddings,
            compute_dtype=compute_dtype,
            final_dropout=final_dropout,
            positional_embeddings=positional_embeddings,
            interleave_self_attention=interleave_self_attention,
            cross_attention_dim=cross_attention_dim,
        )

        # ConfigMixin writes constructor values to JSON. Preserve the dtype as
        # a readable string because torch.dtype itself is not JSON serializable.
        if isinstance(compute_dtype, torch.dtype):
            self.register_to_config(compute_dtype=str(compute_dtype))

        modality_attention_heads = modality_attention_heads or num_attention_heads
        point_encoder_kwargs = {
            "input_dim": point_input_dim,
            "point_dim": self.inner_dim,
            "dropout": modality_dropout,
        }
        if point_encoder_type.lower().replace("-", "_") in {"pointnet2", "pointnet++"}:
            point_encoder_kwargs.update(
                {
                    "hidden_dim": point_encoder_hidden_dim,
                    "num_samples": point_num_samples,
                    "num_neighbors": point_num_neighbors,
                }
            )
        else:
            point_encoder_kwargs.update(
                {
                    "num_layers": point_transformer_layers,
                    "num_heads": modality_attention_heads,
                }
            )
        self.point_encoder = build_point_encoder(point_encoder_type, **point_encoder_kwargs)
        self.tactile_encoder = TactileEncoder(
            input_channels=tactile_input_channels,
            token_dim=self.inner_dim,
            patch_size=tactile_patch_size,
            dropout=modality_dropout,
            pretrained=tactile_pretrained,
        )

        self.point_cross_attention = nn.ModuleList(
            [
                ModalityCrossAttention(
                    action_dim=self.inner_dim,
                    modality_dim=self.inner_dim,
                    num_heads=modality_attention_heads,
                    dropout=modality_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.tactile_cross_attention = nn.ModuleList(
            [
                ModalityCrossAttention(
                    action_dim=self.inner_dim,
                    modality_dim=self.inner_dim,
                    num_heads=modality_attention_heads,
                    dropout=modality_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Each modality starts as an exact no-op. The pretrained GR00T path is
        # therefore preserved while training learns how strongly to use the
        # newly initialized residual branches at each transformer layer.
        self.point_gates = nn.Parameter(torch.zeros(num_layers))
        self.tactile_gates = nn.Parameter(torch.zeros(num_layers))

    def _run_original_gr00t_block(
        self,
        block_index: int,
        hidden_states: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        vlm_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one unmodified self- or VLM-cross-attention DiT block."""
        transformer_block = self.transformer_blocks[block_index]
        is_self_attention_block = block_index % 2 == 1 and self.config.interleave_self_attention

        if is_self_attention_block:
            return transformer_block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=timestep_embedding,
            )

        return transformer_block(
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=vlm_hidden_states,
            encoder_attention_mask=vlm_attention_mask,
            temb=timestep_embedding,
        )

    def _apply_point_conditioning(
        self,
        block_index: int,
        action_hidden_states: torch.Tensor,
        point_tokens: Optional[torch.Tensor],
        point_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if point_tokens is None:
            return action_hidden_states

        # action_hidden_states: [B, N_action, D_dit] (Q)
        # point_tokens:         [B, N_point, D_dit]  (K/V)
        point_update = self.point_cross_attention[block_index](
            action_hidden_states,
            encoder_hidden_states=point_tokens,
            encoder_attention_mask=point_attention_mask,
        )

        # Zero-initialized gated residual: new checkpoints begin with exactly
        # the pretrained GR00T behavior, then learn the conditioning strength.
        gated_point_update = self.point_gates[block_index] * point_update
        action_hidden_states = action_hidden_states + gated_point_update
        return action_hidden_states

    def _apply_tactile_conditioning(
        self,
        block_index: int,
        action_hidden_states: torch.Tensor,
        tactile_tokens: Optional[torch.Tensor],
        tactile_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tactile_tokens is None:
            return action_hidden_states

        # action_hidden_states: [B, N_action, D_dit]   (Q)
        # tactile_tokens:       [B, N_tactile, D_dit] (K/V)
        tactile_update = self.tactile_cross_attention[block_index](
            action_hidden_states,
            encoder_hidden_states=tactile_tokens,
            encoder_attention_mask=tactile_attention_mask,
        )

        # This gate independently controls how much tactile evidence enters
        # the action stream at the current layer.
        gated_tactile_update = self.tactile_gates[block_index] * tactile_update
        action_hidden_states = action_hidden_states + gated_tactile_update
        return action_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        *,
        point_cloud: Optional[torch.Tensor] = None,
        tactile_images: Optional[torch.Tensor] = None,
        point_attention_mask: Optional[torch.Tensor] = None,
        tactile_attention_mask: Optional[torch.Tensor] = None,
        num_state_tokens: int = 1,
    ):
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [B, N_state + N_action, D_dit], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch_size = hidden_states.shape[0]
        if hidden_states.shape[-1] != self.inner_dim:
            raise ValueError(
                f"Expected hidden_states width {self.inner_dim}, got {hidden_states.shape[-1]}."
            )
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape [B, N_vlm, D_vlm], "
                f"got {tuple(encoder_hidden_states.shape)}."
            )
        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("hidden_states and encoder_hidden_states must share a batch size.")
        if not 0 <= num_state_tokens < hidden_states.shape[1]:
            raise ValueError(
                "num_state_tokens must leave at least one action token. "
                f"Got {num_state_tokens} for a sequence of length {hidden_states.shape[1]}."
            )
        if point_cloud is None and point_attention_mask is not None:
            raise ValueError("point_attention_mask was provided without point_cloud.")
        if tactile_images is None and tactile_attention_mask is not None:
            raise ValueError("tactile_attention_mask was provided without tactile_images.")

        # Encode each sensor once, then reuse its K/V tokens at every DiT layer.
        # point_tokens:   [B, N_point_out, D_dit] or None
        # tactile_tokens: [B, N_tactile, D_dit] or None
        if point_cloud is not None:
            point_tokens, point_attention_mask = self.point_encoder(
                point_cloud,
                point_mask=point_attention_mask,
                return_mask=True,
            )
        else:
            point_tokens = None
        tactile_tokens = (
            self.tactile_encoder(tactile_images) if tactile_images is not None else None
        )
        if point_tokens is not None and point_tokens.shape[0] != batch_size:
            raise ValueError("hidden_states and point_cloud must share a batch size.")
        if tactile_tokens is not None and tactile_tokens.shape[0] != batch_size:
            raise ValueError("hidden_states and tactile_images must share a batch size.")

        timestep_embedding = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        vlm_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        for block_index in range(len(self.transformer_blocks)):
            # 1. Original GR00T block computation, including the established
            #    self-attention/VLM-cross-attention schedule.
            hidden_states = self._run_original_gr00t_block(
                block_index,
                hidden_states,
                vlm_hidden_states,
                timestep_embedding,
                encoder_attention_mask,
            )

            # 2. State is the prefix and action is the suffix. Sensor branches
            #    intentionally query with action tokens only.
            # state_hidden_states:  [B, N_state, D_dit]
            # action_hidden_states: [B, N_action, D_dit]
            state_hidden_states = hidden_states[:, :num_state_tokens]
            action_hidden_states = hidden_states[:, num_state_tokens:]

            # 3. Action <- point-cloud cross-attention.
            action_hidden_states = self._apply_point_conditioning(
                block_index,
                action_hidden_states,
                point_tokens,
                point_attention_mask,
            )

            # 4. Action <- tactile-image cross-attention.
            action_hidden_states = self._apply_tactile_conditioning(
                block_index,
                action_hidden_states,
                tactile_tokens,
                tactile_attention_mask,
            )

            # 5. Rebuild [state; action] for the next original GR00T block.
            hidden_states = torch.cat((state_hidden_states, action_hidden_states), dim=1)
            all_hidden_states.append(hidden_states)

        # Preserve the original DiT adaptive output normalization and projection.
        shift, scale = self.proj_out_1(F.silu(timestep_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)

        if return_all_hidden_states:
            return output, all_hidden_states
        return output


__all__ = ["ModalityCrossAttention", "MultiModalConditionedDiT", "PointEncoder", "TactileEncoder"]
