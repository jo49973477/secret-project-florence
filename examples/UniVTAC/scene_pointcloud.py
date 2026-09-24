#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Calibration-strict RGB-D scene-cloud utilities for UniVTAC conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class ScenePointCloudError(RuntimeError):
    """Raised when a scene cloud cannot be constructed without guessing calibration."""


@dataclass(frozen=True)
class ScenePreprocessConfig:
    """Geometry-preserving preprocessing applied after common-frame fusion."""

    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    voxel_size: float | None = 0.01
    max_points: int = 8192
    seed: int = 0

    def __post_init__(self) -> None:
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must be provided together")
        if self.workspace_min is not None and np.any(
            np.asarray(self.workspace_min) >= np.asarray(self.workspace_max)
        ):
            raise ValueError("Every workspace minimum must be smaller than its maximum")
        if self.voxel_size is not None and self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive or None, got {self.voxel_size}")
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")


def validate_intrinsics(intrinsics: np.ndarray, *, source: str) -> np.ndarray:
    """Validate and return a standard 3x3 pinhole intrinsic matrix."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape == (4,):
        fx, fy, cx, cy = matrix
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ScenePointCloudError(
            f"Intrinsics from {source} must have shape [3, 3] or [fx, fy, cx, cy], "
            f"got {matrix.shape}."
        )
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ScenePointCloudError(f"Invalid pinhole intrinsics from {source}: {matrix}")
    if not np.allclose(matrix[2], [0, 0, 1]):
        raise ScenePointCloudError(f"Intrinsics from {source} must have final row [0, 0, 1].")
    return matrix


def validate_transform(transform: np.ndarray, *, source: str) -> np.ndarray:
    """Validate a rigid homogeneous transform without repairing it silently."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ScenePointCloudError(f"Transform from {source} must be [4, 4], got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ScenePointCloudError(f"Transform from {source} contains NaN or Inf.")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ScenePointCloudError(f"Transform from {source} has an invalid homogeneous row.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ScenePointCloudError(f"Transform from {source} does not contain a rigid rotation.")
    return matrix


def pose_wxyz_to_matrix(pose: np.ndarray, *, source: str) -> np.ndarray:
    """Convert UniVTAC's ``[x,y,z,qw,qx,qy,qz]`` base-frame EE pose."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ScenePointCloudError(f"Pose from {source} must be finite [7], got {value.shape}.")
    quaternion = value[3:]
    norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ScenePointCloudError(
            f"Quaternion from {source} must already be normalized; norm={norm:.8g}."
        )
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = value[:3]
    return matrix


def backproject_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_scale: float,
) -> np.ndarray:
    """Backproject optical-frame RGB-D into finite XYZRGB with RGB in [0, 255]."""
    color = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ScenePointCloudError(f"RGB must have shape [H, W, 3], got {color.shape}.")
    if depth_array.ndim != 2 or depth_array.shape != color.shape[:2]:
        raise ScenePointCloudError(
            f"Depth must have shape {color.shape[:2]} to align with RGB, got {depth_array.shape}."
        )
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ScenePointCloudError(f"depth_scale must be positive and finite, got {depth_scale}.")
    matrix = validate_intrinsics(intrinsics, source="RGB-D calibration")
    metric_depth = depth_array.astype(np.float64) * depth_scale
    valid = np.isfinite(metric_depth) & (metric_depth > 0)
    if not valid.any():
        raise ScenePointCloudError("RGB-D frame has no positive finite depth pixels.")
    rows, columns = np.nonzero(valid)
    z = metric_depth[rows, columns]
    x = (columns.astype(np.float64) - matrix[0, 2]) * z / matrix[0, 0]
    y = (rows.astype(np.float64) - matrix[1, 2]) * z / matrix[1, 1]
    xyz = np.stack((x, y, z), axis=-1)
    rgb_values = color[rows, columns].astype(np.float64)
    return np.ascontiguousarray(np.concatenate((xyz, rgb_values), axis=-1), dtype=np.float32)


def transform_xyzrgb(cloud: np.ndarray, transform: np.ndarray, *, source: str) -> np.ndarray:
    """Transform XYZ only, preserving each point's RGB features."""
    points = np.asarray(cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ScenePointCloudError(f"{source} cloud must have shape [N, 6], got {points.shape}.")
    matrix = validate_transform(transform, source=source)
    xyz = points[:, :3].astype(np.float64)
    transformed_xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    return np.ascontiguousarray(
        np.concatenate((transformed_xyz, points[:, 3:6]), axis=-1), dtype=np.float32
    )


def merge_scene_clouds(clouds: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate calibrated camera clouds strictly over their point axis."""
    arrays = [np.asarray(cloud, dtype=np.float32) for cloud in clouds]
    if not arrays:
        raise ScenePointCloudError("At least one calibrated camera cloud is required.")
    for index, array in enumerate(arrays):
        if array.ndim != 2 or array.shape[1] != 6:
            raise ScenePointCloudError(
                f"Camera cloud {index} must have shape [N, 6], got {array.shape}."
            )
    return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)


def crop_workspace(
    cloud: np.ndarray,
    workspace_min: tuple[float, float, float] | None,
    workspace_max: tuple[float, float, float] | None,
) -> np.ndarray:
    if workspace_min is None and workspace_max is None:
        return cloud
    if workspace_min is None or workspace_max is None:
        raise ValueError("workspace_min and workspace_max must be provided together")
    minimum = np.asarray(workspace_min, dtype=np.float32)
    maximum = np.asarray(workspace_max, dtype=np.float32)
    mask = np.all((cloud[:, :3] >= minimum) & (cloud[:, :3] <= maximum), axis=1)
    return cloud[mask]


def voxel_downsample(cloud: np.ndarray, voxel_size: float | None) -> np.ndarray:
    """Average XYZRGB within a common-frame voxel grid anchored at the origin."""
    if voxel_size is None:
        return cloud
    grid = np.floor(cloud[:, :3] / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 6), dtype=np.float64)
    np.add.at(sums, inverse, cloud)
    return np.ascontiguousarray(sums / counts[:, None], dtype=np.float32)


def sample_fixed_scene(
    cloud: np.ndarray, *, max_points: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Randomly cap points after voxelization; repeat only when dense storage needs padding."""
    valid_count = int(len(cloud))
    if valid_count == 0:
        raise ScenePointCloudError("Scene cloud is empty after filtering and voxel downsampling.")
    if valid_count >= max_points:
        indices = rng.choice(valid_count, size=max_points, replace=False)
    else:
        # Dense LeRobot tensors require a fixed N. Keep every geometric point once,
        # then fill remaining rows by unbiased repeats rather than zero sentinels.
        extra = rng.choice(valid_count, size=max_points - valid_count, replace=True)
        indices = np.concatenate((np.arange(valid_count), extra))
        rng.shuffle(indices)
    return np.ascontiguousarray(cloud[indices], dtype=np.float32), valid_count


def build_scene_frame(
    *,
    head_rgb: np.ndarray,
    head_depth: np.ndarray,
    head_intrinsics: np.ndarray,
    head_depth_scale: float,
    transform_base_head: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_depth: np.ndarray,
    wrist_intrinsics: np.ndarray,
    wrist_depth_scale: float,
    transform_base_wrist: np.ndarray,
    config: ScenePreprocessConfig,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Construct one head+wrist common-frame scene cloud."""
    head_camera = backproject_rgbd(
        head_rgb, head_depth, head_intrinsics, depth_scale=head_depth_scale
    )
    wrist_camera = backproject_rgbd(
        wrist_rgb, wrist_depth, wrist_intrinsics, depth_scale=wrist_depth_scale
    )
    head_base = transform_xyzrgb(head_camera, transform_base_head, source="head T_base_camera")
    wrist_base = transform_xyzrgb(
        wrist_camera, transform_base_wrist, source="wrist T_base_camera(t)"
    )
    merged = merge_scene_clouds((head_base, wrist_base))
    cropped = crop_workspace(merged, config.workspace_min, config.workspace_max)
    voxelized = voxel_downsample(cropped, config.voxel_size)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, frame_index]))
    sampled, valid_count = sample_fixed_scene(voxelized, max_points=config.max_points, rng=rng)
    return sampled, {
        "head_points_before_merge": int(len(head_base)),
        "wrist_points_before_merge": int(len(wrist_base)),
        "points_after_crop": int(len(cropped)),
        "points_after_voxel": int(len(voxelized)),
        "valid_points_before_dense_fill": valid_count,
    }
