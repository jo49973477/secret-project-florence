from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset):
    counts = offset2bincount(offset)
    return torch.arange(len(counts), device=offset.device).repeat_interleave(counts)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def _morton_encode(grid_coord, depth):
    x, y, z = grid_coord.long().unbind(dim=-1)
    code = torch.zeros_like(x)
    for bit in range(depth):
        mask = 1 << bit
        code |= (x & mask) << (2 * bit + 2)
        code |= (y & mask) << (2 * bit + 1)
        code |= (z & mask) << (2 * bit)
    return code


def _hilbert_encode(grid_coord, depth):
    # John Skilling's transpose algorithm maps 3-D coordinates to a Hilbert key.
    coord = grid_coord.long().clone()
    highest_bit = 1 << (depth - 1)

    bit = highest_bit
    while bit > 1:
        lower_bits = bit - 1
        for dim in range(3):
            active = (coord[:, dim] & bit) != 0
            coord[active, 0] ^= lower_bits
            inactive = ~active
            exchange = (coord[inactive, 0] ^ coord[inactive, dim]) & lower_bits
            coord[inactive, 0] ^= exchange
            coord[inactive, dim] ^= exchange
        bit >>= 1

    coord[:, 1:] ^= coord[:, :-1]
    gray_decode = torch.zeros_like(coord[:, 0])
    bit = highest_bit
    while bit > 1:
        gray_decode ^= torch.where((coord[:, 2] & bit) != 0, bit - 1, 0)
        bit >>= 1
    coord ^= gray_decode.unsqueeze(-1)

    code = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
    for bit in range(depth - 1, -1, -1):
        for dim in range(3):
            code = (code << 1) | ((coord[:, dim] >> bit) & 1)
    return code


@torch.inference_mode()
def encode(grid_coord, batch, depth, order):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order.endswith("-trans"):
        grid_coord = grid_coord[:, [1, 0, 2]]
    if order.startswith("z"):
        code = _morton_encode(grid_coord, depth)
    else:
        code = _hilbert_encode(grid_coord, depth)
    return batch.long() << (depth * 3) | code


class Point(dict):
    """Minimal mutable state used while encoding a batched point cloud."""

    def __init__(self, data):
        super().__init__(data)
        if "batch" not in self and "offset" in self:
            self.batch = offset2batch(self.offset)
        elif "offset" not in self and "batch" in self:
            self.offset = batch2offset(self.batch)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self
        if "grid_coord" not in self:
            assert {"grid_size", "coord"}.issubset(self)
            self.grid_coord = torch.div(
                self.coord - self.coord.min(0).values,
                self.grid_size,
                rounding_mode="trunc",
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self.serialized_depth = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        codes = torch.stack([encode(self.grid_coord, self.batch, depth, curve) for curve in order])
        serialized_order = torch.argsort(codes)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(codes.shape[1], device=serialized_order.device).repeat(
                codes.shape[0], 1
            ),
        )

        if shuffle_orders:
            permutation = torch.randperm(codes.shape[0])
            codes = codes[permutation]
            serialized_order = serialized_order[permutation]
            serialized_inverse = serialized_inverse[permutation]

        self.order = order
        self.serialized_code = codes
        self.serialized_order = serialized_order
        self.serialized_inverse = serialized_inverse

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self)
        if "grid_coord" not in self:
            assert {"grid_size", "coord"}.issubset(self)
            self.grid_coord = torch.div(
                self.coord - self.coord.min(0).values,
                self.grid_size,
                rounding_mode="trunc",
            ).int()

        sparse_shape = self.get("sparse_shape", (self.grid_coord.max(dim=0).values + pad).tolist())
        self.sparse_shape = sparse_shape
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].item() + 1,
        )


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        index = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        rpe = self.rpe_table.index_select(0, index.reshape(-1))
        rpe = rpe.view(index.shape + (-1,)).sum(3)
        return rpe.permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class GridPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.Sequential(norm_layer(out_channels)) if norm_layer else None
        self.act = nn.Sequential(act_layer()) if act_layer else None

    def forward(self, point):
        if "grid_coord" in point:
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point):
            grid_coord = torch.div(
                point.coord - point.coord.min(0).values,
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError("Point requires grid_coord or coord and grid_size")

        pooled_grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        batched_grid_coord = pooled_grid_coord | (point.batch.view(-1, 1) << 48)
        batched_grid_coord, cluster, counts = torch.unique(
            batched_grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        pooled_grid_coord = batched_grid_coord & ((1 << 48) - 1)

        _, sorted_indices = torch.sort(cluster)
        index_pointer = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        cluster_heads = sorted_indices[index_pointer[:-1]]
        pooled_feat = torch_scatter.segment_csr(
            self.proj(point.feat)[sorted_indices],
            index_pointer,
            reduce=self.reduce,
        )
        if self.norm is not None:
            pooled_feat = self.norm(pooled_feat)
        if self.act is not None:
            pooled_feat = self.act(pooled_feat)

        data = {
            "feat": pooled_feat,
            "coord": torch_scatter.segment_csr(
                point.coord[sorted_indices], index_pointer, reduce="mean"
            ),
            "grid_coord": pooled_grid_coord,
            "batch": point.batch[cluster_heads],
        }
        if "grid_size" in point:
            data["grid_size"] = point.grid_size * self.stride

        pooled_point = Point(data)
        pooled_point.serialization(order=point.order, shuffle_orders=self.shuffle_orders)
        pooled_point.sparsify()
        return pooled_point


class Embedding(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        layers = [("linear", nn.Linear(in_channels, embed_channels))]
        if norm_layer is not None:
            layers.append(("norm", norm_layer(embed_channels)))
        if act_layer is not None:
            layers.append(("act", act_layer()))
        self.stem = nn.Sequential(OrderedDict(layers))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_channels)) if mask_token else None

    def forward(self, point):
        point.feat = self.stem(point.feat)
        if "mask" in point:
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
        return point
