#!/usr/bin/env python3

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

"""Inspect a converted UniVTAC dataset and compare it with its source HDF5."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from convert_univtac_to_lerobot import (
    ACTION_COLUMN,
    ANNOTATION_COLUMN,
    CAMERA_DATASETS,
    CHUNK_SIZE,
    JOINT_DATASET,
    JOINT_DIMENSION,
    STATE_COLUMN,
    ConversionError,
    inspect_video,
)
import h5py
import numpy as np
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _episode_record(episode_rows: list[dict[str, Any]], episode_index: int) -> dict[str, Any]:
    matches = [row for row in episode_rows if row["episode_index"] == episode_index]
    if len(matches) != 1:
        raise ConversionError(
            f"Expected one episodes.jsonl entry for episode {episode_index}, got {len(matches)}."
        )
    return matches[0]


def _numeric_summary(name: str, values: np.ndarray) -> None:
    print(f"{name} shape: {values.shape}")
    print(f"{name} min: {values.min(axis=0)}")
    print(f"{name} max: {values.max(axis=0)}")
    print(f"{name} contains NaN: {np.isnan(values).any()}")
    print(f"{name} contains Inf: {np.isinf(values).any()}")


def _video_path(
    dataset_root: Path,
    info: dict[str, Any],
    *,
    episode_index: int,
    camera_name: str,
) -> Path:
    video_pattern = info["video_path"]
    relative_path = video_pattern.format(
        episode_chunk=episode_index // info["chunks_size"],
        video_key=f"observation.images.{camera_name}",
        episode_index=episode_index,
    )
    return dataset_root / relative_path


def compare_with_source(
    source_path: Path,
    converted_state: np.ndarray,
    converted_action: np.ndarray,
) -> None:
    """Verify the first q_t/q_(t+1) pair against the raw UniVTAC episode."""
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source HDF5 does not exist: {source_path}. "
            "Pass --source-hdf5 or --skip-source-comparison."
        )

    with h5py.File(source_path, "r") as hdf5_file:
        if JOINT_DATASET not in hdf5_file:
            raise ConversionError(f"{source_path} does not contain {JOINT_DATASET!r}.")
        source_joint_pair = np.asarray(
            hdf5_file[JOINT_DATASET][:2, :JOINT_DIMENSION],
            dtype=np.float32,
        )

    state_matches = np.allclose(converted_state[0], source_joint_pair[0])
    action_matches = np.allclose(converted_action[0], source_joint_pair[1])
    print(f"Source HDF5: {source_path}")
    print(f"converted state[0] == original joint[0, :8]: {state_matches}")
    print(f"converted action[0] == original joint[1, :8]: {action_matches}")
    if not state_matches or not action_matches:
        raise ConversionError("Converted state/action values do not match the source HDF5.")


def check_gr00t_loader(
    dataset_root: Path,
    modality_config_path: Path,
    *,
    episode_index: int,
) -> None:
    """Initialize the current GR00T loader and extract one unprocessed VLA sample."""
    if not modality_config_path.is_file():
        raise FileNotFoundError(f"Modality config does not exist: {modality_config_path}")

    spec = importlib.util.spec_from_file_location(
        "univtac_loader_check_config", modality_config_path
    )
    if spec is None or spec.loader is None:
        raise ConversionError(f"Could not import modality config: {modality_config_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config_module
    spec.loader.exec_module(config_module)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    modality_configs = MODALITY_CONFIGS[embodiment_tag.value]
    dataset = ShardedSingleStepDataset(
        dataset_path=dataset_root,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_configs,
        episode_sampling_rate=1.0,
        seed=0,
    )
    episode_data = dataset.episode_loader[episode_index]
    sample = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
    )

    state_shape = sample.states["joint"].shape
    action_shape = sample.actions["joint"].shape
    if state_shape != (1, JOINT_DIMENSION):
        raise ConversionError(f"GR00T loader returned unexpected state shape: {state_shape}")
    if action_shape[1] != JOINT_DIMENSION:
        raise ConversionError(f"GR00T loader returned unexpected action shape: {action_shape}")
    if set(sample.images) != set(CAMERA_DATASETS):
        raise ConversionError(
            f"GR00T loader returned unexpected image keys: {sample.images.keys()}"
        )

    print("GR00T dataset initialization: PASS")
    print(f"GR00T sample state shape: {state_shape}")
    print(f"GR00T sample action shape: {action_shape}")
    print(f"GR00T sample image keys: {sorted(sample.images)}")
    print(f"GR00T sample language: {sample.text!r}")


def inspect_dataset(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    modality = read_json(dataset_root / "meta" / "modality.json")
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episode = _episode_record(episodes, args.episode_index)

    chunk_index = args.episode_index // info.get("chunks_size", CHUNK_SIZE)
    parquet_path = (
        dataset_root
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{args.episode_index:06d}.parquet"
    )
    table = pq.read_table(parquet_path)
    converted_state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
    converted_action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)

    print("info.json:")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("modality.json:")
    print(json.dumps(modality, indent=2, ensure_ascii=False))
    print(f"Episode count: {len(episodes)}")
    print(f"Tasks ({len(tasks)}): {[row['task'] for row in tasks]}")
    print(f"Selected episode: {json.dumps(episode, ensure_ascii=False)}")
    print(f"Parquet: {parquet_path}")
    print(f"Parquet columns: {table.column_names}")
    print("Parquet schema:")
    print(table.schema)
    print(f"First state: {converted_state[0]}")
    print(f"First action: {converted_action[0]}")
    _numeric_summary("State", converted_state)
    _numeric_summary("Action", converted_action)

    if converted_state.shape != (episode["length"], JOINT_DIMENSION):
        raise ConversionError(f"Unexpected converted state shape: {converted_state.shape}")
    if converted_action.shape != (episode["length"], JOINT_DIMENSION):
        raise ConversionError(f"Unexpected converted action shape: {converted_action.shape}")
    if np.isnan(converted_state).any() or np.isinf(converted_state).any():
        raise ConversionError("Converted state contains NaN or Inf.")
    if np.isnan(converted_action).any() or np.isinf(converted_action).any():
        raise ConversionError("Converted action contains NaN or Inf.")

    annotation_indices = table[ANNOTATION_COLUMN].to_numpy(zero_copy_only=False)
    task_indices = table["task_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(annotation_indices, task_indices):
        raise ConversionError(f"{ANNOTATION_COLUMN} does not match task_index.")

    for camera_name in CAMERA_DATASETS:
        video_path = _video_path(
            dataset_root,
            info,
            episode_index=args.episode_index,
            camera_name=camera_name,
        )
        video_metadata = inspect_video(video_path, decode_all_frames=True)
        print(f"{camera_name.capitalize()} video: {video_path}")
        print(f"  frames: {video_metadata.frame_count}")
        print(f"  resolution: {video_metadata.width}x{video_metadata.height}")
        print(f"  fps: {video_metadata.fps:g}")
        if video_metadata.frame_count != episode["length"]:
            raise ConversionError(
                f"{camera_name} video has {video_metadata.frame_count} frames; "
                f"parquet has {episode['length']} rows."
            )

    if args.skip_source_comparison:
        print("Source comparison: skipped by request")
    else:
        source_path = args.source_hdf5
        if source_path is None:
            source_value = episode.get("source_path")
            if source_value is None:
                raise ConversionError(
                    "episodes.jsonl has no source_path; pass --source-hdf5 explicitly."
                )
            source_path = Path(source_value)
        compare_with_source(source_path.expanduser().resolve(), converted_state, converted_action)

    if args.check_gr00t_loader:
        check_gr00t_loader(
            dataset_root,
            args.modality_config.expanduser().resolve(),
            episode_index=args.episode_index,
        )

    print("Inspection result: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a converted UniVTAC GR00T dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Converted dataset root.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to inspect (default: 0).",
    )
    parser.add_argument(
        "--source-hdf5",
        type=Path,
        help="Raw source episode. Defaults to source_path recorded in episodes.jsonl.",
    )
    parser.add_argument(
        "--skip-source-comparison",
        action="store_true",
        help="Inspect only converted files when the source HDF5 is unavailable.",
    )
    parser.add_argument(
        "--check-gr00t-loader",
        action="store_true",
        help="Load one sample through GR00T (requires generated stats files).",
    )
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=Path(__file__).with_name("univtac_config.py"),
        help="Config used by --check-gr00t-loader (default: adjacent univtac_config.py).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        inspect_dataset(parse_args())
    except (ConversionError, FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
