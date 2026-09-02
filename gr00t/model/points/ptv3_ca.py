"""Encoder-only PTv3 with joint point/action attention and context cross-attention."""

from huggingface_hub import PyTorchModelHubMixin
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .ptv3_utonia import Point3DRoPE
from .utils import MLP, RPE, DropPath, Embedding, GridPooling, LayerScale, Point, offset2bincount


try:
    import flash_attn
except ImportError:
    flash_attn = None


class PointActionAttention(nn.Module):
    """Serialized self-attention over point patches and per-cloud action tokens."""

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash

        if enable_flash:
            assert not enable_rpe, "Set enable_rpe to False when enabling Flash Attention"
            assert not upcast_attention, (
                "Set upcast_attention to False when enabling Flash Attention"
            )
            assert not upcast_softmax, "Set upcast_softmax to False when enabling Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None
        self.rope = Point3DRoPE(head_dim=channels // num_heads, base=rope_base)
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        cache_key = f"rel_pos_{self.order_index}"
        if cache_key not in point:
            grid_coord = point.grid_coord[order].reshape(-1, self.patch_size, 3)
            point[cache_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[cache_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        if not {"pad", "unpad", "cu_seqlens_key"}.issubset(point):
            offset = point.offset
            counts = offset2bincount(offset)
            padded_counts = (
                torch.div(
                    counts + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            needs_padding = counts > self.patch_size
            padded_counts = ~needs_padding * counts + needs_padding * padded_counts

            offset_start = nn.functional.pad(offset, (1, 0))
            padded_offset = nn.functional.pad(torch.cumsum(padded_counts, dim=0), (1, 0))
            pad = torch.arange(padded_offset[-1], device=offset.device)
            unpad = torch.arange(offset_start[-1], device=offset.device)
            sequence_starts = []
            for batch_index in range(len(offset)):
                unpad[offset_start[batch_index] : offset_start[batch_index + 1]] += (
                    padded_offset[batch_index] - offset_start[batch_index]
                )
                if counts[batch_index] != padded_counts[batch_index]:
                    remainder = counts[batch_index] % self.patch_size
                    pad[
                        padded_offset[batch_index + 1]
                        - self.patch_size
                        + remainder : padded_offset[batch_index + 1]
                    ] = pad[
                        padded_offset[batch_index + 1]
                        - 2 * self.patch_size
                        + remainder : padded_offset[batch_index + 1] - self.patch_size
                    ]
                pad[padded_offset[batch_index] : padded_offset[batch_index + 1]] -= (
                    padded_offset[batch_index] - offset_start[batch_index]
                )
                sequence_starts.append(
                    torch.arange(
                        padded_offset[batch_index],
                        padded_offset[batch_index + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point.pad = pad
            point.unpad = unpad
            point.cu_seqlens_key = nn.functional.pad(
                torch.cat(sequence_starts), (0, 1), value=padded_offset[-1]
            )
        return point.pad, point.unpad, point.cu_seqlens_key

    def forward(self, point: Point):
        point_counts = offset2bincount(point.offset)
        if not self.enable_flash:
            self.patch_size = min(point_counts.min().item(), self.patch_size_max)

        num_heads = self.num_heads
        patch_size = self.patch_size
        channels = self.channels
        pad, unpad, cumulative_lengths = self.get_padding_and_inverse(point)
        patch_lengths = torch.diff(cumulative_lengths)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        point_qkv = self.qkv(point.feat)[order]
        point_qkv_dtype = point_qkv.dtype
        point_qkv = point_qkv.reshape(-1, 3, num_heads, channels // num_heads)
        q, k, v = point_qkv.unbind(dim=1)
        q, k = self.rope(q, k, point.coord[order].clone())
        point_qkv = torch.stack([q, k, v], dim=1)

        num_actions = point.action_feat.shape[1]
        action_qkv = self.qkv(point.action_feat).reshape(
            -1, num_actions, 3, num_heads, channels // num_heads
        )
        repeats = torch.div(
            point_counts + patch_size - 1,
            patch_size,
            rounding_mode="trunc",
        )
        action_qkv = action_qkv.repeat_interleave(repeats, dim=0)

        if not self.enable_flash:
            point_qkv = point_qkv.reshape(-1, patch_size, 3, num_heads, channels // num_heads)
            qkv = torch.cat([action_qkv, point_qkv], dim=1)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attention = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                relative_position = self.rpe(self.get_rel_pos(point, order))
                attention[:, :, num_actions:, num_actions:] += relative_position
            if self.upcast_softmax:
                attention = attention.float()
            attention = self.attn_drop(self.softmax(attention)).to(qkv.dtype)
            feat = (attention @ v).transpose(1, 2)

            action_feat = torch.split(feat[:, :num_actions], repeats.cpu().tolist(), dim=0)
            action_feat = torch.stack(
                [action_patch.mean(dim=0) for action_patch in action_feat], dim=0
            ).reshape(-1, num_actions, channels)
            feat = feat[:, num_actions:].reshape(-1, channels)
        else:
            point_qkv = torch.split(point_qkv, patch_lengths.cpu().tolist(), dim=0)
            qkv = torch.cat(
                [
                    torch.cat([action_patch, point_patch], dim=0)
                    for point_patch, action_patch in zip(point_qkv, action_qkv)
                ],
                dim=0,
            )
            patch_lengths = patch_lengths + num_actions
            cumulative_lengths = torch.cumsum(patch_lengths, dim=0).int()
            cumulative_lengths = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=cumulative_lengths.device),
                    cumulative_lengths,
                ]
            )
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16),
                cumulative_lengths,
                max_seqlen=patch_size + num_actions,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, channels)

            feat = torch.split(feat, patch_lengths.cpu().tolist(), dim=0)
            action_feat = torch.stack([patch[:num_actions] for patch in feat], dim=0)
            feat = torch.cat([patch[num_actions:] for patch in feat], dim=0).to(point_qkv_dtype)
            action_feat = torch.split(action_feat, repeats.cpu().tolist(), dim=0)
            action_feat = torch.stack(
                [action_patch.mean(dim=0) for action_patch in action_feat], dim=0
            ).reshape(-1, num_actions, channels)
            action_feat = action_feat.to(point_qkv_dtype)

        point.feat = self.proj_drop(self.proj(feat[inverse]))
        point.action_feat = self.proj_drop(self.proj(action_feat))
        return point


class PointActionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # Numeric children retain the existing checkpoint paths.
        self.cpe = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            ]
        )
        self.norm1 = nn.Sequential(norm_layer(channels))
        self.ls1 = nn.Sequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.attn = PointActionAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
        )
        self.norm2 = nn.Sequential(norm_layer(channels))
        self.ls2 = nn.Sequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = nn.Sequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = nn.Sequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

        self.action_proj = nn.Linear(channels, channels)
        self.action_norm0 = norm_layer(channels)
        self.action_norm1 = norm_layer(channels)
        self.action_norm2 = norm_layer(channels)
        self.action_mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, point):
        shortcut = point.feat
        point.sparse_conv_feat = self.cpe[0](point.sparse_conv_feat)
        point.feat = point.sparse_conv_feat.features
        point.feat = self.cpe[1](point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = self.cpe[2](point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point.feat = self.norm1(point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        action_feat = point.action_feat
        point.action_feat = action_feat + self.action_norm0(self.action_proj(action_feat))
        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm1(action_feat)

        point = self.attn(point)
        point.feat = self.ls1(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = self.drop_path(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point.feat = self.norm1(point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        shortcut = point.feat
        if self.pre_norm:
            point.feat = self.norm2(point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = self.mlp(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = self.ls2(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = self.drop_path(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point.feat = self.norm2(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)
        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        point.action_feat = self.drop_path(self.action_mlp(point.action_feat))
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        return point


class CABlock(nn.Module):
    """Cross-attend point and action tokens to per-cloud context."""

    def __init__(
        self,
        channels,
        num_heads,
        kv_channels,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qk_norm=True,
        pre_norm=True,
        enable_flash=True,
        apply_point_ca=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels**-0.5
        self.pre_norm = pre_norm
        self.enable_flash = enable_flash
        self.apply_point_ca = apply_point_ca

        self.norm1 = norm_layer(channels)
        # ModuleDict keeps the original attn.* projection and normalization paths.
        self.attn = nn.ModuleDict(
            {
                "q": nn.Linear(channels, channels),
                "kv": nn.Linear(kv_channels, channels * 2),
                "q_norm": norm_layer(self.head_channels) if qk_norm else nn.Identity(),
                "k_norm": norm_layer(self.head_channels) if qk_norm else nn.Identity(),
                "attn_drop": nn.Dropout(attn_drop),
                "proj": nn.Linear(channels, channels),
                "proj_drop": nn.Dropout(proj_drop),
            }
        )
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.action_norm1 = norm_layer(channels)
        self.action_norm2 = norm_layer(channels)
        self.action_mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def _cross_attention(self, query, context, query_offset, context_offset):
        query_dtype = query.dtype
        q = self.attn["q"](query).reshape(-1, self.num_heads, self.head_channels)
        k, v = (
            self.attn["kv"](context)
            .reshape(-1, 2, self.num_heads, self.head_channels)
            .unbind(dim=1)
        )
        q = self.attn["q_norm"](q)
        k = self.attn["k_norm"](k)

        if self.enable_flash:
            assert flash_attn is not None, "Make sure flash_attn is installed."
            query_lengths = offset2bincount(query_offset)
            context_lengths = offset2bincount(context_offset)
            query_cu = nn.functional.pad(query_offset, (1, 0)).int()
            context_cu = nn.functional.pad(context_offset, (1, 0)).int()
            feat = flash_attn.flash_attn_varlen_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                query_cu,
                context_cu,
                max_seqlen_q=query_lengths.max().item(),
                max_seqlen_k=context_lengths.max().item(),
                dropout_p=self.attn["attn_drop"].p if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, self.channels)
            feat = feat.to(query_dtype)
        else:
            query_start = nn.functional.pad(query_offset, (1, 0))
            context_start = nn.functional.pad(context_offset, (1, 0))
            outputs = []
            for batch_index in range(len(query_offset)):
                q_batch = q[query_start[batch_index] : query_start[batch_index + 1]].transpose(0, 1)
                k_batch = k[context_start[batch_index] : context_start[batch_index + 1]].transpose(
                    0, 1
                )
                v_batch = v[context_start[batch_index] : context_start[batch_index + 1]].transpose(
                    0, 1
                )
                attention = self.attn["attn_drop"](
                    torch.softmax((q_batch * self.scale) @ k_batch.transpose(-2, -1), dim=-1)
                )
                outputs.append((attention @ v_batch).transpose(0, 1).reshape(-1, self.channels))
            feat = torch.cat(outputs, dim=0)

        return self.attn["proj_drop"](self.attn["proj"](feat))

    def forward(self, point):
        if self.apply_point_ca:
            shortcut = point.feat
            if self.pre_norm:
                point.feat = self.norm1(point.feat)
            point.feat = self._cross_attention(
                point.feat,
                point.context,
                point.offset,
                point.context_offset,
            )
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point.feat = self.norm1(point.feat)

            shortcut = point.feat
            if self.pre_norm:
                point.feat = self.norm2(point.feat)
            point.feat = shortcut + self.mlp(point.feat)
            if not self.pre_norm:
                point.feat = self.norm2(point.feat)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)
        batch_size, num_actions, _ = point.action_feat.shape
        action_offset = (
            torch.arange(
                1,
                batch_size + 1,
                dtype=torch.long,
                device=point.action_feat.device,
            )
            * num_actions
        )
        point.action_feat = self._cross_attention(
            point.action_feat.reshape(batch_size * num_actions, -1),
            point.context,
            action_offset,
            point.context_offset,
        ).reshape(batch_size, num_actions, -1)
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)

        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        point.action_feat = action_shortcut + self.action_mlp(point.action_feat)
        if not self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        return point


class PointTransformerV3CA(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        mask_token=False,
        apply_point_ca=True,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=norm_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        drop_path_rates = [rate.item() for rate in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = nn.Sequential()
        for stage_index in range(self.num_stages):
            stage = nn.Sequential()
            if stage_index > 0:
                down = GridPooling(
                    in_channels=enc_channels[stage_index - 1],
                    out_channels=enc_channels[stage_index],
                    stride=stride[stage_index - 1],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                # Action tokens follow the point channel schedule but are not spatially pooled.
                down.action_proj = nn.Linear(
                    enc_channels[stage_index - 1], enc_channels[stage_index]
                )
                down.action_norm = nn.Sequential(norm_layer(enc_channels[stage_index]))
                stage.add_module("down", down)

            stage_start = sum(enc_depths[:stage_index])
            for block_index in range(enc_depths[stage_index]):
                stage.add_module(
                    f"block{block_index}",
                    PointActionBlock(
                        channels=enc_channels[stage_index],
                        num_heads=enc_num_head[stage_index],
                        patch_size=enc_patch_size[stage_index],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path_rates[stage_start + block_index],
                        layer_scale=layer_scale,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=block_index % len(self.order),
                        cpe_indice_key=f"stage{stage_index}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        rope_base=rope_base,
                        shift_coords=shift_coords,
                        jitter_coords=jitter_coords,
                        rescale_coords=rescale_coords,
                    ),
                )
                stage.add_module(
                    f"ca_block{block_index}",
                    CABlock(
                        channels=enc_channels[stage_index],
                        num_heads=enc_num_head[stage_index],
                        kv_channels=ctx_channels,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        qk_norm=qk_norm,
                        pre_norm=pre_norm,
                        enable_flash=enable_flash,
                        apply_point_ca=apply_point_ca,
                    ),
                )
            self.enc.add_module(f"enc{stage_index}", stage)

        if freeze_encoder:
            for parameter in self.embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.enc.parameters():
                parameter.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _downsample(stage, point):
        down = stage.down
        action_feat = down.action_proj(point.action_feat)
        action_feat = down.action_norm(action_feat)
        if down.act is not None:
            action_feat = down.act(action_feat)
        context = point.context
        context_offset = point.context_offset
        point = down(point)
        point.action_feat = action_feat
        point.context = context
        point.context_offset = context_offset
        return point

    def forward(self, data_dict, return_layer_outputs=False):
        point = self.embedding(Point(data_dict))
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        layer_outputs = []
        for stage_index, stage in enumerate(self.enc):
            if stage_index > 0:
                point = self._downsample(stage, point)
            for name, module in stage.named_children():
                if name != "down":
                    point = module(point)
            if return_layer_outputs:
                layer_outputs.append(point)
        return layer_outputs if return_layer_outputs else point


if __name__ == "__main__":
    model = PointTransformerV3CA(in_channels=6)
    num_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Model params: {num_parameters / 1e6:.2f}M")
