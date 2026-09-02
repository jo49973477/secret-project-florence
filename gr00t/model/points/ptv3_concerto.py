"""
Point Transformer - V3 Mode2 - Sonata & Concerto
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from huggingface_hub import PyTorchModelHubMixin
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .utils import MLP, RPE, DropPath, Embedding, GridPooling, LayerScale, Point, offset2bincount


try:
    import flash_attn
except ImportError:
    flash_attn = None


class SerializedAttention(nn.Module):
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

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().item(), self.patch_size_max)

        num_heads = self.num_heads
        patch_size = self.patch_size
        channels = self.channels
        pad, unpad, cumulative_lengths = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, patch_size, 3, num_heads, channels // num_heads)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attention = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attention = attention + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attention = attention.float()
            attention = self.attn_drop(self.softmax(attention)).to(qkv.dtype)
            feat = (attention @ v).transpose(1, 2).reshape(-1, channels)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, num_heads, channels // num_heads),
                cumulative_lengths,
                max_seqlen=patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, channels)
            feat = feat.to(qkv.dtype)

        point.feat = self.proj_drop(self.proj(feat[inverse]))
        return point


class Block(nn.Module):
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
        attn_class=SerializedAttention,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # Keep numeric children to preserve existing encoder checkpoint keys.
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
        self.attn = attn_class(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
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
        return point


class PointTransformerV3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
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
        freeze_encoder=False,
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
                stage.add_module(
                    "down",
                    GridPooling(
                        in_channels=enc_channels[stage_index - 1],
                        out_channels=enc_channels[stage_index],
                        stride=stride[stage_index - 1],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                )

            stage_start = sum(enc_depths[:stage_index])
            for block_index in range(enc_depths[stage_index]):
                stage.add_module(
                    f"block{block_index}",
                    Block(
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

    def forward(self, data_dict, return_layer_outputs=False):
        point = self.embedding(Point(data_dict))
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        if not return_layer_outputs:
            return self.enc(point)

        layer_outputs = []
        for stage in self.enc:
            point = stage(point)
            layer_outputs.append(point)
        return layer_outputs


if __name__ == "__main__":
    model = PointTransformerV3(in_channels=6)
    num_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Model params: {num_parameters / 1e6:.2f}M")
