# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from examples.UniVTAC.scene_pointcloud import (
    ScenePreprocessConfig,
    backproject_rgbd,
    build_scene_frame,
    merge_scene_clouds,
    transform_xyzrgb,
)
from gr00t.data.types import ModalityConfig
from gr00t.model.extension.point_encoder import ConcertoPointEncoder
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
import pytest
import torch
from torch import nn


def test_rgbd_backprojection_known_intrinsics() -> None:
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    depth = np.array([[2.0, 4.0]], dtype=np.float32)
    intrinsics = np.array([[2.0, 0, 0.0], [0, 2.0, 0.0], [0, 0, 1.0]])

    cloud = backproject_rgbd(rgb, depth, intrinsics, depth_scale=0.5)

    np.testing.assert_allclose(
        cloud,
        [[0, 0, 1, 10, 20, 30], [1, 0, 2, 40, 50, 60]],
    )
    assert cloud.dtype == np.float32


def test_two_camera_transforms_land_in_same_common_frame() -> None:
    head = np.array([[0, 0, 1, 255, 0, 0]], dtype=np.float32)
    wrist = np.array([[0, 0, 1, 0, 255, 0]], dtype=np.float32)
    transform_base_head = np.eye(4)
    transform_base_wrist = np.eye(4)
    transform_base_wrist[0, 3] = -1
    wrist[:, 0] = 1

    head_base = transform_xyzrgb(head, transform_base_head, source="head")
    wrist_base = transform_xyzrgb(wrist, transform_base_wrist, source="wrist")

    np.testing.assert_allclose(head_base[0, :3], wrist_base[0, :3])
    np.testing.assert_array_equal(head_base[0, 3:], [255, 0, 0])
    np.testing.assert_array_equal(wrist_base[0, 3:], [0, 255, 0])


def test_camera_clouds_concatenate_over_point_axis() -> None:
    head = np.ones((3, 6), dtype=np.float32)
    wrist = np.full((5, 6), 2, dtype=np.float32)

    scene = merge_scene_clouds((head, wrist))

    assert scene.shape == (8, 6)
    np.testing.assert_array_equal(scene[:3], head)
    np.testing.assert_array_equal(scene[3:], wrist)


def test_scene_frame_preserves_camera_provenance_counts() -> None:
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = np.eye(3)
    config = ScenePreprocessConfig(voxel_size=None, max_points=8, seed=7)

    scene, metadata = build_scene_frame(
        head_rgb=rgb,
        head_depth=depth,
        head_intrinsics=intrinsics,
        head_depth_scale=1.0,
        transform_base_head=np.eye(4),
        wrist_rgb=rgb,
        wrist_depth=depth,
        wrist_intrinsics=intrinsics,
        wrist_depth_scale=1.0,
        transform_base_wrist=np.eye(4),
        config=config,
        frame_index=0,
    )

    assert scene.shape == (8, 6)
    assert metadata["head_points_before_merge"] == 4
    assert metadata["wrist_points_before_merge"] == 4


def test_processor_keeps_one_scene_field_without_axis_change() -> None:
    content = SimpleNamespace(
        pointclouds={"scene": np.arange(1 * 7 * 6, dtype=np.float32).reshape(1, 7, 6)}
    )
    modality = {"pointcloud": ModalityConfig(delta_indices=[0], modality_keys=["scene"])}

    points = Gr00tN1d7Processor._training_pointcloud(content, modality)

    assert tuple(points.shape) == (7, 6)
    np.testing.assert_array_equal(points.numpy(), content.pointclouds["scene"][0])


def test_processor_rejects_feature_concat_with_semantic_scene() -> None:
    content = SimpleNamespace(
        pointclouds={
            "scene": np.zeros((1, 7, 6), dtype=np.float32),
            "other": np.zeros((1, 7, 3), dtype=np.float32),
        }
    )
    modality = {"pointcloud": ModalityConfig(delta_indices=[0], modality_keys=["scene", "other"])}

    with pytest.raises(ValueError, match="point-axis merge"):
        Gr00tN1d7Processor._training_pointcloud(content, modality)


class _FakeConcertoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(9, 12)
        nn.init.constant_(self.linear.weight, 0.25)

    def forward(self, data):
        assert data["feat"].shape[-1] == 9
        assert data["grid_coord"].shape[-1] == 3
        return SimpleNamespace(feat=self.linear(data["feat"]), offset=data["offset"])


def test_concerto_wrapper_shape_mask_projection_and_rgb_normalization() -> None:
    backbone = _FakeConcertoBackbone()
    encoder = ConcertoPointEncoder(
        input_dim=6,
        point_dim=16,
        model_name="concerto_small",
        backbone=backbone,
        backbone_output_dim=12,
        pretrained_loaded=True,
    )
    points = torch.zeros(2, 5, 6)
    points[..., :3] = torch.randn(2, 5, 3)
    points[..., 3:] = 255
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])

    tokens, output_mask = encoder(points, point_mask=mask, return_mask=True)

    assert tokens.shape == (2, 5, 16)
    assert output_mask.shape == (2, 5)
    assert output_mask.sum().item() == 8
    assert torch.isfinite(tokens).all()
    assert encoder.pretrained_loaded is True
    assert torch.count_nonzero(backbone.linear.weight).item() > 0


def test_concerto_refuses_unverified_random_backbone() -> None:
    with pytest.raises(RuntimeError, match="without verified pretrained weights"):
        ConcertoPointEncoder(
            input_dim=6,
            point_dim=16,
            model_name="concerto_small",
            backbone=_FakeConcertoBackbone(),
            backbone_output_dim=12,
            pretrained_loaded=False,
        )
