# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_stats
from gr00t.data.types import ModalityConfig, VLAStepData
import numpy as np
import pandas as pd
import pytest


_CONVERTER_PATH = (
    Path(__file__).resolve().parents[3] / "examples/UniVTAC/convert_univtac_to_lerobot.py"
)
_CONVERTER_SPEC = importlib.util.spec_from_file_location("test_univtac_converter", _CONVERTER_PATH)
assert _CONVERTER_SPEC is not None and _CONVERTER_SPEC.loader is not None
_CONVERTER_MODULE = importlib.util.module_from_spec(_CONVERTER_SPEC)
sys.modules[_CONVERTER_SPEC.name] = _CONVERTER_MODULE
_CONVERTER_SPEC.loader.exec_module(_CONVERTER_MODULE)
reconstruct_fixed_pointcloud = _CONVERTER_MODULE.reconstruct_fixed_pointcloud


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def _make_external_modality_dataset(root: Path, *, pointcloud_frames: int = 3) -> None:
    original_tactile_keys = {"rgb": "observation.tactile.rgb"}
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        **{key: {"dtype": "video", "shape": [4, 6, 3]} for key in original_tactile_keys.values()},
        "observation.pointcloud.xyz": {"dtype": "float32", "shape": [4, 3]},
    }
    _write_json(
        root / "meta/info.json",
        {
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "pointcloud_path": "pointclouds/chunk-{episode_chunk:03d}/{pointcloud_key}/episode_{episode_index:06d}.npz",
            "chunks_size": 1000,
            "features": features,
        },
    )
    _write_json(
        root / "meta/modality.json",
        {
            "state": {"joint": {"start": 0, "end": 2}},
            "action": {"joint": {"start": 0, "end": 2}},
            "tactile": {
                tactile_key: {"original_key": original_key}
                for tactile_key, original_key in original_tactile_keys.items()
            },
            "pointcloud": {
                "xyz": {
                    "original_key": "observation.pointcloud.xyz",
                    "array_key": "xyz",
                }
            },
        },
    )
    _write_jsonl(
        root / "meta/episodes.jsonl", [{"episode_index": 0, "length": 3, "tasks": ["task"]}]
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "task"}])
    stats = {
        key: {name: [0.0, 1.0] for name in ("mean", "std", "min", "max", "q01", "q99")}
        for key in ("observation.state", "action")
    }
    _write_json(root / "meta/stats.json", stats)

    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": [np.array([i, i + 1], dtype=np.float32) for i in range(3)],
            "action": [np.array([i + 1, i + 2], dtype=np.float32) for i in range(3)],
        }
    ).to_parquet(parquet_path)

    pointcloud_path = root / "pointclouds/chunk-000/observation.pointcloud.xyz/episode_000000.npz"
    pointcloud_path.parent.mkdir(parents=True)
    xyz = np.arange(pointcloud_frames * 4 * 3, dtype=np.float32).reshape(pointcloud_frames, 4, 3)
    np.savez_compressed(pointcloud_path, xyz=xyz)


def test_pointcloud_reconstruction_is_fixed_size_and_deterministic() -> None:
    depth = np.array([[1.0, np.nan, 2.0], [0.0, 3.0, np.inf]], dtype=np.float32)
    intrinsics = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]])

    first = reconstruct_fixed_pointcloud(depth, intrinsics, num_points=5, depth_scale=0.1)
    second = reconstruct_fixed_pointcloud(depth, intrinsics, num_points=5, depth_scale=0.1)

    assert first.shape == (5, 3)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(
        first,
        [
            [-0.05, -0.0125, 0.1],
            [0.1, -0.025, 0.2],
            [0.0, 0.0375, 0.3],
            [-0.05, -0.0125, 0.1],
            [0.1, -0.025, 0.2],
        ],
    )


def test_vla_step_data_preserves_existing_positional_constructor() -> None:
    sample = VLAStepData(
        {},
        {},
        {},
        None,
        "task",
        EmbodimentTag.NEW_EMBODIMENT,
        True,
        {"source": "legacy"},
    )

    assert sample.text == "task"
    assert sample.is_demonstration is True
    assert sample.metadata == {"source": "legacy"}
    assert sample.tactile is None
    assert sample.pointclouds is None


def test_loader_and_extract_step_preserve_external_modalities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_external_modality_dataset(tmp_path)
    tactile_frames = np.arange(3 * 4 * 6 * 3, dtype=np.uint8).reshape(3, 4, 6, 3)

    def fake_video_decode(_path: str, indices: np.ndarray, decoder_kwargs: dict) -> np.ndarray:
        assert decoder_kwargs == {}
        return tactile_frames[indices]

    monkeypatch.setattr(
        "gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices",
        fake_video_decode,
    )
    configs = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        "tactile": ModalityConfig(delta_indices=[-1, 0], modality_keys=["rgb"]),
        "pointcloud": ModalityConfig(delta_indices=[-1, 0], modality_keys=["xyz"]),
    }
    loader = LeRobotEpisodeLoader(tmp_path, configs)
    episode = loader[0]
    sample = extract_step_data(
        episode,
        step_index=1,
        modality_configs=configs,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )

    assert sample.tactile is not None
    assert list(sample.tactile) == ["rgb"]
    assert len(sample.tactile["rgb"]) == 2
    assert sample.tactile["rgb"][0].dtype == np.uint8
    assert sample.pointclouds is not None
    assert sample.pointclouds["xyz"].shape == (2, 4, 3)
    assert sample.pointclouds["xyz"].dtype == np.float32
    np.testing.assert_array_equal(sample.tactile["rgb"][0], tactile_frames[0])
    np.testing.assert_array_equal(sample.pointclouds["xyz"][1], episode["pointcloud.xyz"].iloc[1])


def test_loader_rejects_pointcloud_episode_length_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_external_modality_dataset(tmp_path, pointcloud_frames=2)
    monkeypatch.setattr(
        "gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices",
        lambda _path, indices, decoder_kwargs: np.zeros((len(indices), 4, 6, 3), dtype=np.uint8),
    )
    configs = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "pointcloud": ModalityConfig(delta_indices=[0], modality_keys=["xyz"]),
    }
    loader = LeRobotEpisodeLoader(tmp_path, configs)

    with pytest.raises(ValueError, match="has 2 frames; episode dataframe has 3 rows"):
        loader[0]


def test_stats_skip_external_float32_pointcloud(tmp_path: Path) -> None:
    _make_external_modality_dataset(tmp_path)
    (tmp_path / "meta/stats.json").unlink()

    generate_stats(tmp_path)

    stats = json.loads((tmp_path / "meta/stats.json").read_text(encoding="utf-8"))
    assert "observation.state" in stats
    assert "action" in stats
    assert "observation.pointcloud.xyz" not in stats
