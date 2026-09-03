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

"""Convert raw UniVTAC HDF5 episodes to GR00T's LeRobot v2 flavor.

The mapping follows UniVTAC's own ``HDF5Handler.batch_gather_hdf5`` and ACT
preprocessing code:

===============================  ==============================================
UniVTAC HDF5                     GR00T LeRobot v2
===============================  ==============================================
``embodiment/joint[t, :8]``      ``observation.state[t]``
``embodiment/joint[t + 1, :8]``  ``action[t]`` (absolute joint target)
``observation/head/rgb[t]``      ``observation.images.head`` video frame
``observation/wrist/rgb[t]``     ``observation.images.wrist`` video frame
task directory / ``--task``      task and annotation indices
===============================  ==============================================

T joint samples therefore produce T-1 aligned rows and T-1 frames per video.
The converter intentionally excludes tactile, depth, point-cloud, and
end-effector data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterator

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


try:
    import h5py
except ImportError as exc:  # pragma: no cover - depends on the user's conversion environment
    raise ImportError(
        "UniVTAC conversion requires h5py. Install it with `uv pip install h5py` "
        "in the active GR00T environment."
    ) from exc


LOGGER = logging.getLogger("convert_univtac_to_lerobot")

JOINT_DATASET = "embodiment/joint"
HEAD_RGB_DATASET = "observation/head/rgb"
WRIST_RGB_DATASET = "observation/wrist/rgb"
CAMERA_DATASETS = {
    "head": HEAD_RGB_DATASET,
    "wrist": WRIST_RGB_DATASET,
}

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
ANNOTATION_COLUMN = "annotation.human.task_description"
CHUNK_SIZE = 1000
JOINT_DIMENSION = 8
JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
]

TASK_NAME_OVERRIDES = {
    "grasp_classify": "grasp and classify the object",
    "insert_card": "insert the card",
    "insert_hdmi": "insert the HDMI connector",
    "insert_hole": "insert the object into the hole",
    "insert_lean": "insert the leaning object",
    "insert_tube": "insert the tube",
    "lift_bottle": "lift the bottle",
    "lift_can": "lift the can",
    "pull_out_key": "pull out the key",
    "put_bottle_in_shelf": "put the bottle in the shelf",
}
GENERIC_DIRECTORY_NAMES = {
    "clean",
    "contact",
    "data",
    "dataset",
    "demo",
    "episodes",
    "h5",
    "hdf5",
    "raw",
    "univtac",
}


class ConversionError(RuntimeError):
    """Raised when an input episode cannot be converted without data loss."""


@dataclass(frozen=True)
class VideoMetadata:
    """Properties verified from one encoded episode video."""

    frame_count: int
    height: int
    width: int
    fps: float


@dataclass(frozen=True)
class PreparedEpisode:
    """Aligned low-dimensional data and validated output-video metadata."""

    state: np.ndarray
    action: np.ndarray
    videos: dict[str, VideoMetadata]

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


@dataclass(frozen=True)
class ConvertedEpisode:
    """Metadata for one successfully converted episode."""

    episode_index: int
    task: str
    task_index: int
    length: int
    source_path: Path


def _natural_sort_key(path: Path) -> tuple[tuple[int, str | int], ...]:
    """Sort numeric episode stems naturally while remaining deterministic."""
    parts = re.split(r"(\d+)", path.as_posix().lower())
    # Tag every component so unlike paths never require comparing str with int.
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts)


def discover_hdf5_files(input_path: Path, max_episodes: int | None) -> list[Path]:
    """Return one file or all HDF5 files below a directory in stable order."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Input file must end in .h5 or .hdf5: {input_path}")
        hdf5_files = [input_path]
    else:
        hdf5_files = sorted(
            (
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
            ),
            key=_natural_sort_key,
        )

    if not hdf5_files:
        raise FileNotFoundError(f"No .h5 or .hdf5 episodes found below {input_path}")

    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {max_episodes}")
        hdf5_files = hdf5_files[:max_episodes]

    return hdf5_files


def normalize_task_name(raw_name: str) -> str:
    """Turn a UniVTAC task directory name into a short English instruction."""
    normalized_key = re.sub(r"[-\s]+", "_", raw_name.strip()).strip("_").lower()
    if normalized_key in TASK_NAME_OVERRIDES:
        return TASK_NAME_OVERRIDES[normalized_key]

    words = normalized_key.replace("_", " ").strip()
    if not words:
        raise ValueError(f"Could not normalize an empty task name from {raw_name!r}")
    return words


def infer_task_name(source_path: Path, input_root: Path) -> str:
    """Infer a task from known UniVTAC task ancestors, then use a safe fallback."""
    candidate_directories: list[str] = []
    stop_path = input_root if input_root.is_dir() else input_root.parent

    for parent in source_path.parents:
        candidate_directories.append(parent.name)
        if parent == stop_path:
            break

    # Prefer known task modules even when an arbitrary config directory sits
    # between the task directory and its HDF5 files.
    for candidate in candidate_directories:
        normalized_key = candidate.lower().replace("-", "_")
        if normalized_key in TASK_NAME_OVERRIDES:
            return TASK_NAME_OVERRIDES[normalized_key]

    for candidate in candidate_directories:
        normalized_key = candidate.lower().replace("-", "_")
        if normalized_key and normalized_key not in GENERIC_DIRECTORY_NAMES:
            return normalize_task_name(candidate)

    raise ConversionError(f"Could not infer a task name for {source_path}. Pass --task explicitly.")


def _encoded_image_bytes(encoded_frame: Any) -> np.ndarray:
    """Normalize fixed-length strings or uint8 arrays into an OpenCV buffer."""
    if isinstance(encoded_frame, (bytes, bytearray, np.bytes_)):
        return np.frombuffer(bytes(encoded_frame), dtype=np.uint8)

    encoded_array = np.asarray(encoded_frame)
    if encoded_array.dtype != np.uint8:
        raise ConversionError(
            "Compressed RGB frames must be byte strings or uint8 arrays; "
            f"received dtype {encoded_array.dtype}."
        )
    return encoded_array.reshape(-1)


def decode_univtac_rgb(encoded_frame: Any, *, source: str) -> np.ndarray:
    """Decode one UniVTAC JPEG frame into the simulator's original RGB order.

    UniVTAC writes Isaac Lab RGB arrays directly with ``cv2.imencode``. OpenCV
    interprets those bytes as BGR while encoding, and ``cv2.imdecode`` reverses
    that same interpretation. The decoded numeric channels consequently match
    the original Isaac Lab RGB array. Do not apply an additional BGR/RGB swap.
    """
    encoded_buffer = _encoded_image_bytes(encoded_frame)
    decoded_rgb = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
    if decoded_rgb is None:
        raise ConversionError(f"OpenCV could not decode JPEG frame {source}")
    if decoded_rgb.ndim != 3 or decoded_rgb.shape[2] != 3:
        raise ConversionError(
            f"Decoded frame {source} must have shape [H, W, 3], got {decoded_rgb.shape}."
        )
    return np.ascontiguousarray(decoded_rgb, dtype=np.uint8)


def _ffmpeg_validation_error(ffmpeg_executable: str) -> str | None:
    """Return why an FFmpeg candidate is unsuitable, or None when usable."""
    try:
        version_result = subprocess.run(
            [ffmpeg_executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return str(exc)

    if version_result.returncode != 0:
        return version_result.stderr.strip() or version_result.stdout.strip()

    encoder_result = subprocess.run(
        [ffmpeg_executable, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    if encoder_result.returncode != 0:
        return encoder_result.stderr.strip() or "could not list encoders"
    if "libx264" not in encoder_result.stdout:
        return "the libx264 encoder is unavailable"
    return None


def resolve_ffmpeg_executable(requested_executable: str) -> str:
    """Locate a working FFmpeg with libx264 support.

    When the default system FFmpeg exists but is unusable (for example, due to
    a missing shared library), also try imageio-ffmpeg's self-contained binary.
    An explicit --ffmpeg path is never silently replaced.
    """
    requested_path = Path(requested_executable).expanduser()
    if requested_path.parent != Path(".") or requested_path.is_absolute():
        candidates = [str(requested_path)]
    else:
        resolved_executable = shutil.which(requested_executable)
        candidates = [resolved_executable] if resolved_executable else []

    if requested_executable == "ffmpeg":
        try:
            import imageio_ffmpeg

            bundled_executable = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled_executable not in candidates:
                candidates.append(bundled_executable)
        except ImportError:
            pass

    if not candidates:
        raise FileNotFoundError(
            f"Could not find FFmpeg executable {requested_executable!r}. "
            "Install FFmpeg with libx264 or pass --ffmpeg /path/to/ffmpeg."
        )

    validation_errors = []
    for ffmpeg_executable in candidates:
        validation_error = _ffmpeg_validation_error(ffmpeg_executable)
        if validation_error is None:
            return ffmpeg_executable
        validation_errors.append(f"{ffmpeg_executable}: {validation_error}")

    details = "; ".join(validation_errors)
    raise ConversionError(f"No usable FFmpeg with libx264 was found ({details}).")


def _ffmpeg_command(
    ffmpeg_executable: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    """Build the deterministic raw-RGB to H.264 encoding command."""
    return [
        ffmpeg_executable,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(output_path),
    ]


def encode_hdf5_video(
    image_dataset: h5py.Dataset,
    output_path: Path,
    *,
    frame_count: int,
    fps: float,
    ffmpeg_executable: str,
    source_label: str,
) -> VideoMetadata:
    """Stream JPEG frames from HDF5 through FFmpeg without materializing a video."""
    if frame_count <= 0:
        raise ConversionError(f"Cannot encode an empty video for {source_label}")

    first_frame = decode_univtac_rgb(image_dataset[0], source=f"{source_label}[0]")
    height, width = first_frame.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ConversionError(
            f"H.264 yuv420p requires even dimensions, but {source_label} is {width}x{height}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = _ffmpeg_command(
        ffmpeg_executable,
        output_path,
        width=width,
        height=height,
        fps=fps,
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None or process.stderr is None:  # pragma: no cover - subprocess invariant
        process.kill()
        raise ConversionError("Failed to open FFmpeg stdin/stderr pipes.")

    try:
        for frame_index in range(frame_count):
            frame = (
                first_frame
                if frame_index == 0
                else decode_univtac_rgb(
                    image_dataset[frame_index],
                    source=f"{source_label}[{frame_index}]",
                )
            )
            if frame.shape != first_frame.shape:
                raise ConversionError(
                    f"Frame shape changed in {source_label}: frame 0 is {first_frame.shape}, "
                    f"frame {frame_index} is {frame.shape}."
                )
            process.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError) as exc:
        process.stdin.close()
        error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.wait()
        raise ConversionError(f"FFmpeg failed while encoding {source_label}: {error_text}") from exc
    except BaseException:
        process.stdin.close()
        process.kill()
        process.wait()
        raise

    process.stdin.close()
    error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise ConversionError(
            f"FFmpeg exited with code {return_code} while encoding {source_label}: {error_text}"
        )

    video_metadata = inspect_video(output_path, decode_all_frames=True)
    if video_metadata.frame_count != frame_count:
        raise ConversionError(
            f"Encoded {source_label} has {video_metadata.frame_count} frames; expected {frame_count}."
        )
    if video_metadata.height != height or video_metadata.width != width:
        raise ConversionError(
            f"Encoded {source_label} resolution changed from {width}x{height} to "
            f"{video_metadata.width}x{video_metadata.height}."
        )
    if not np.isclose(video_metadata.fps, fps, rtol=0.0, atol=1e-3):
        raise ConversionError(
            f"Encoded {source_label} FPS is {video_metadata.fps}; expected {fps}."
        )
    return video_metadata


def inspect_video(video_path: Path, *, decode_all_frames: bool) -> VideoMetadata:
    """Read video metadata and optionally count every decodable frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ConversionError(f"OpenCV could not open encoded video: {video_path}")

    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if decode_all_frames:
            frame_count = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame is None or frame.shape[:2] != (height, width):
                    raise ConversionError(f"Invalid decoded frame {frame_count} in {video_path}")
                frame_count += 1
        else:
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ConversionError(
            f"Invalid video metadata for {video_path}: "
            f"frames={frame_count}, resolution={width}x{height}, fps={fps}."
        )
    return VideoMetadata(frame_count=frame_count, height=height, width=width, fps=fps)


def _require_hdf5_dataset(hdf5_file: h5py.File, key: str) -> h5py.Dataset:
    if key not in hdf5_file:
        raise ConversionError(
            f"Required UniVTAC dataset {key!r} is missing from {hdf5_file.filename}"
        )
    dataset = hdf5_file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ConversionError(f"Expected {key!r} to be an HDF5 dataset in {hdf5_file.filename}")
    return dataset


def prepare_episode(
    source_path: Path,
    output_root: Path,
    *,
    episode_index: int,
    fps: float,
    ffmpeg_executable: str,
) -> PreparedEpisode:
    """Validate one source episode, align arrays, and encode its two videos."""
    try:
        hdf5_file = h5py.File(source_path, "r")
    except (OSError, ValueError) as exc:
        raise ConversionError(f"Could not open HDF5 episode {source_path}: {exc}") from exc

    with hdf5_file:
        joint_dataset = _require_hdf5_dataset(hdf5_file, JOINT_DATASET)
        head_dataset = _require_hdf5_dataset(hdf5_file, HEAD_RGB_DATASET)
        wrist_dataset = _require_hdf5_dataset(hdf5_file, WRIST_RGB_DATASET)

        if joint_dataset.ndim != 2 or joint_dataset.shape[1] < JOINT_DIMENSION:
            raise ConversionError(
                f"{source_path}:{JOINT_DATASET} must have shape [T, >=8], "
                f"got {joint_dataset.shape}."
            )
        joint_sample_count = int(joint_dataset.shape[0])
        if joint_sample_count < 2:
            raise ConversionError(
                f"{source_path}:{JOINT_DATASET} needs at least 2 samples, got {joint_sample_count}."
            )

        for camera_name, image_dataset in (("head", head_dataset), ("wrist", wrist_dataset)):
            if image_dataset.ndim != 1:
                raise ConversionError(
                    f"{source_path}:{CAMERA_DATASETS[camera_name]} must be a 1D JPEG stream, "
                    f"got shape {image_dataset.shape}."
                )
            if len(image_dataset) != joint_sample_count:
                raise ConversionError(
                    f"Length mismatch in {source_path}: {JOINT_DATASET} has "
                    f"{joint_sample_count} samples but {CAMERA_DATASETS[camera_name]} has "
                    f"{len(image_dataset)}."
                )

        # This is the exact state/action convention used by UniVTAC's ACT converter.
        joint_values = np.asarray(joint_dataset[:, :JOINT_DIMENSION], dtype=np.float32)
        state = np.ascontiguousarray(joint_values[:-1])
        action = np.ascontiguousarray(joint_values[1:])
        frame_count = joint_sample_count - 1

        if state.shape != (frame_count, JOINT_DIMENSION):
            raise ConversionError(f"Unexpected aligned state shape in {source_path}: {state.shape}")
        if action.shape != (frame_count, JOINT_DIMENSION):
            raise ConversionError(
                f"Unexpected aligned action shape in {source_path}: {action.shape}"
            )
        if not np.isfinite(state).all():
            raise ConversionError(f"NaN or Inf found in joint state for {source_path}")
        if not np.isfinite(action).all():
            raise ConversionError(f"NaN or Inf found in joint action for {source_path}")

        episode_chunk = episode_index // CHUNK_SIZE
        videos: dict[str, VideoMetadata] = {}
        for camera_name, image_dataset in (("head", head_dataset), ("wrist", wrist_dataset)):
            video_path = (
                output_root
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / f"observation.images.{camera_name}"
                / f"episode_{episode_index:06d}.mp4"
            )
            videos[camera_name] = encode_hdf5_video(
                image_dataset,
                video_path,
                frame_count=frame_count,
                fps=fps,
                ffmpeg_executable=ffmpeg_executable,
                source_label=f"{source_path}:{CAMERA_DATASETS[camera_name]}",
            )

    return PreparedEpisode(state=state, action=action, videos=videos)


def _fixed_size_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    """Create an Arrow fixed-size float32 list without per-row Python objects."""
    contiguous_values = np.ascontiguousarray(values, dtype=np.float32)
    flat_values = pa.array(contiguous_values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat_values, JOINT_DIMENSION)


def write_episode_parquet(
    prepared_episode: PreparedEpisode,
    output_root: Path,
    *,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    fps: float,
) -> Path:
    """Write one episode using explicit Arrow dtypes expected by GR00T."""
    episode_length = prepared_episode.length
    frame_indices = np.arange(episode_length, dtype=np.int64)
    global_indices = np.arange(
        global_index_start,
        global_index_start + episode_length,
        dtype=np.int64,
    )
    timestamps = frame_indices.astype(np.float32) / np.float32(fps)
    task_indices = np.full(episode_length, task_index, dtype=np.int64)
    episode_indices = np.full(episode_length, episode_index, dtype=np.int64)
    next_done = np.zeros(episode_length, dtype=np.bool_)
    next_done[-1] = True

    table = pa.Table.from_arrays(
        [
            _fixed_size_float_list(prepared_episode.action),
            _fixed_size_float_list(prepared_episode.state),
            pa.array(timestamps, type=pa.float32()),
            pa.array(frame_indices, type=pa.int64()),
            pa.array(episode_indices, type=pa.int64()),
            pa.array(global_indices, type=pa.int64()),
            pa.array(task_indices, type=pa.int64()),
            pa.array(task_indices, type=pa.int64()),
            pa.array(np.zeros(episode_length, dtype=np.float32), type=pa.float32()),
            pa.array(next_done, type=pa.bool_()),
        ],
        names=[
            ACTION_COLUMN,
            STATE_COLUMN,
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            ANNOTATION_COLUMN,
            "next.reward",
            "next.done",
        ],
    )

    episode_chunk = episode_index // CHUNK_SIZE
    parquet_path = (
        output_root / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd")
    return parquet_path


def _scalar_feature(dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


def _video_feature(metadata: VideoMetadata, fps: float) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [metadata.height, metadata.width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": metadata.height,
            "video.width": metadata.width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def build_info_metadata(
    converted_episodes: list[ConvertedEpisode],
    video_metadata: dict[str, VideoMetadata],
    *,
    fps: float,
    task_count: int,
) -> dict[str, Any]:
    """Build current GR00T/LeRobot v2 dataset-level metadata."""
    total_frames = sum(episode.length for episode in converted_episodes)
    total_episodes = len(converted_episodes)
    return {
        "codebase_version": "v2.1",
        "robot_type": "franka_panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": task_count,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            ACTION_COLUMN: {
                "dtype": "float32",
                "shape": [JOINT_DIMENSION],
                "names": JOINT_NAMES,
            },
            STATE_COLUMN: {
                "dtype": "float32",
                "shape": [JOINT_DIMENSION],
                "names": JOINT_NAMES,
            },
            "observation.images.head": _video_feature(video_metadata["head"], fps),
            "observation.images.wrist": _video_feature(video_metadata["wrist"], fps),
            "timestamp": _scalar_feature("float32"),
            "frame_index": _scalar_feature("int64"),
            "episode_index": _scalar_feature("int64"),
            "index": _scalar_feature("int64"),
            "task_index": _scalar_feature("int64"),
            ANNOTATION_COLUMN: _scalar_feature("int64"),
            "next.reward": _scalar_feature("float32"),
            "next.done": _scalar_feature("bool"),
        },
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "total_videos": total_episodes * len(CAMERA_DATASETS),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4, ensure_ascii=False)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def write_metadata(
    output_root: Path,
    converted_episodes: list[ConvertedEpisode],
    tasks_by_index: list[str],
    video_metadata: dict[str, VideoMetadata],
    *,
    fps: float,
) -> None:
    """Write all metadata files required by the current GR00T loader."""
    info = build_info_metadata(
        converted_episodes,
        video_metadata,
        fps=fps,
        task_count=len(tasks_by_index),
    )
    modality = {
        "state": {"joint": {"start": 0, "end": JOINT_DIMENSION}},
        "action": {"joint": {"start": 0, "end": JOINT_DIMENSION}},
        "video": {
            "head": {"original_key": "observation.images.head"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            # Current GR00T guidance requires a dedicated column for custom
            # annotations, even when its values equal the default task_index.
            "human.task_description": {"original_key": ANNOTATION_COLUMN},
        },
    }

    write_json(output_root / "meta" / "info.json", info)
    write_json(output_root / "meta" / "modality.json", modality)
    write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        (
            {"task_index": task_index, "task": task}
            for task_index, task in enumerate(tasks_by_index)
        ),
    )
    write_jsonl(
        output_root / "meta" / "episodes.jsonl",
        (
            {
                "episode_index": episode.episode_index,
                "tasks": [episode.task],
                "length": episode.length,
                "source_path": str(episode.source_path.resolve()),
            }
            for episode in converted_episodes
        ),
    )


def validate_converted_dataset(
    output_root: Path,
    converted_episodes: list[ConvertedEpisode],
    *,
    fps: float,
) -> None:
    """Re-read parquet files and enforce cross-episode index invariants."""
    expected_global_index = 0
    required_columns = {
        ACTION_COLUMN,
        STATE_COLUMN,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        ANNOTATION_COLUMN,
        "next.reward",
        "next.done",
    }

    for episode in converted_episodes:
        chunk_index = episode.episode_index // CHUNK_SIZE
        parquet_path = (
            output_root
            / "data"
            / f"chunk-{chunk_index:03d}"
            / f"episode_{episode.episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        if set(table.column_names) != required_columns:
            raise ConversionError(
                f"Unexpected parquet columns in {parquet_path}: {table.column_names}"
            )
        if table.num_rows != episode.length:
            raise ConversionError(
                f"{parquet_path} has {table.num_rows} rows; expected {episode.length}."
            )

        frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
        global_indices = table["index"].to_numpy(zero_copy_only=False)
        timestamps = table["timestamp"].to_numpy(zero_copy_only=False)
        expected_frames = np.arange(episode.length, dtype=np.int64)
        expected_globals = np.arange(
            expected_global_index,
            expected_global_index + episode.length,
            dtype=np.int64,
        )
        if not np.array_equal(frame_indices, expected_frames):
            raise ConversionError(f"Non-contiguous frame_index values in {parquet_path}")
        if not np.array_equal(global_indices, expected_globals):
            raise ConversionError(f"Non-contiguous global index values in {parquet_path}")
        if timestamps.shape[0] > 1 and not np.all(np.diff(timestamps) > 0):
            raise ConversionError(f"Timestamps are not strictly increasing in {parquet_path}")
        expected_timestamps = expected_frames.astype(np.float32) / np.float32(fps)
        if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-6):
            raise ConversionError(f"Incorrect timestamps in {parquet_path}")

        state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
        action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
        if state.shape != (episode.length, JOINT_DIMENSION):
            raise ConversionError(f"Incorrect state shape in {parquet_path}: {state.shape}")
        if action.shape != (episode.length, JOINT_DIMENSION):
            raise ConversionError(f"Incorrect action shape in {parquet_path}: {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ConversionError(f"NaN or Inf found in {parquet_path}")

        expected_global_index += episode.length


def _remove_episode_outputs(output_root: Path, episode_index: int) -> None:
    """Remove only files created for one failed episode in the private staging tree."""
    chunk_index = episode_index // CHUNK_SIZE
    candidates = [
        output_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet",
        *(
            output_root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / f"observation.images.{camera_name}"
            / f"episode_{episode_index:06d}.mp4"
            for camera_name in CAMERA_DATASETS
        ),
    ]
    for candidate in candidates:
        candidate.unlink(missing_ok=True)


def _validate_video_schema(
    reference_metadata: dict[str, VideoMetadata] | None,
    episode_metadata: dict[str, VideoMetadata],
    source_path: Path,
) -> dict[str, VideoMetadata]:
    if reference_metadata is None:
        return episode_metadata

    for camera_name in CAMERA_DATASETS:
        reference = reference_metadata[camera_name]
        current = episode_metadata[camera_name]
        if (current.height, current.width) != (reference.height, reference.width):
            raise ConversionError(
                f"{camera_name} resolution changed in {source_path}: expected "
                f"{reference.width}x{reference.height}, got {current.width}x{current.height}."
            )
    return reference_metadata


def _safe_replace_output(staging_root: Path, output_root: Path, *, overwrite: bool) -> None:
    """Publish a completed staging tree without risking broad deletion targets."""
    resolved_output = output_root.resolve()
    dangerous_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved_output in dangerous_targets:
        raise ValueError(f"Refusing to replace unsafe output path: {resolved_output}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --overwrite to replace it."
            )
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()

    staging_root.replace(output_root)


def convert_dataset(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    hdf5_files = discover_hdf5_files(input_path, args.max_episodes)
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Pass --overwrite to replace it."
        )

    ffmpeg_executable = resolve_ffmpeg_executable(args.ffmpeg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )

    converted_episodes: list[ConvertedEpisode] = []
    task_to_index: dict[str, int] = {}
    tasks_by_index: list[str] = []
    reference_video_metadata: dict[str, VideoMetadata] | None = None
    global_index = 0

    try:
        for source_path in hdf5_files:
            episode_index = len(converted_episodes)
            task = args.task or infer_task_name(source_path, input_path)
            try:
                prepared_episode = prepare_episode(
                    source_path,
                    staging_root,
                    episode_index=episode_index,
                    fps=args.fps,
                    ffmpeg_executable=ffmpeg_executable,
                )
                validated_video_metadata = _validate_video_schema(
                    reference_video_metadata,
                    prepared_episode.videos,
                    source_path,
                )

                task_is_new = task not in task_to_index
                task_index = task_to_index.get(task, len(tasks_by_index))

                write_episode_parquet(
                    prepared_episode,
                    staging_root,
                    episode_index=episode_index,
                    task_index=task_index,
                    global_index_start=global_index,
                    fps=args.fps,
                )

                # Commit metadata only after all episode files were written.
                # This keeps --skip-invalid from leaving unused tasks or a
                # reference video schema from an episode that was skipped.
                reference_video_metadata = validated_video_metadata
                if task_is_new:
                    task_to_index[task] = task_index
                    tasks_by_index.append(task)
            except Exception as exc:
                _remove_episode_outputs(staging_root, episode_index)
                if args.skip_invalid:
                    LOGGER.warning("Skipping invalid episode %s: %s", source_path, exc)
                    continue
                raise ConversionError(f"Failed to convert {source_path}: {exc}") from exc

            converted_episodes.append(
                ConvertedEpisode(
                    episode_index=episode_index,
                    task=task,
                    task_index=task_index,
                    length=prepared_episode.length,
                    source_path=source_path,
                )
            )
            global_index += prepared_episode.length
            LOGGER.info(
                "Converted episode %06d: %s (%d frames, task=%r)",
                episode_index,
                source_path,
                prepared_episode.length,
                task,
            )

        if not converted_episodes or reference_video_metadata is None:
            raise ConversionError("No valid episodes were converted.")

        write_metadata(
            staging_root,
            converted_episodes,
            tasks_by_index,
            reference_video_metadata,
            fps=args.fps,
        )
        validate_converted_dataset(staging_root, converted_episodes, fps=args.fps)
        _safe_replace_output(staging_root, output_root, overwrite=args.overwrite)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    head_video = reference_video_metadata["head"]
    wrist_video = reference_video_metadata["wrist"]
    print(f"Converted episodes: {len(converted_episodes)}")
    print(f"Total frames: {global_index}")
    print(f"State shape: ({JOINT_DIMENSION},)")
    print(f"Action shape: ({JOINT_DIMENSION},)")
    print(f"Tasks: {len(tasks_by_index)}")
    print(f"Head video FPS: {head_video.fps:g}")
    print(f"Wrist video FPS: {wrist_video.fps:g}")
    print(f"Output: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC HDF5 episodes to GR00T-flavored LeRobot v2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One UniVTAC .h5/.hdf5 file or a directory tree containing episodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination dataset root. Must not already exist unless --overwrite is passed.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS (default: 10).")
    parser.add_argument(
        "--task",
        help="Task-language override. By default the task is inferred from parent directories.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Convert only the first N episodes after deterministic natural sorting.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable or path (default: ffmpeg). Must provide libx264.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Warn and skip corrupt/missing episodes instead of failing the conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after the new dataset validates successfully.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one progress line per converted episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        convert_dataset(args)
    except (ConversionError, FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
