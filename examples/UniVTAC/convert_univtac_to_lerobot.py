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
selected tactile RGB ``[t]``     ``observation.tactile.rgb`` video frame
selected depth stream ``[t]``    ``observation.pointcloud.xyz`` NPZ array
task directory / ``--task``      task and annotation indices
===============================  ==============================================

T joint samples therefore produce T-1 aligned rows and T-1 frames per video.
Tactile and point-cloud conversion are opt-in so existing RGB-only workflows
remain unchanged. End-effector data is intentionally excluded.
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
TACTILE_OUTPUT_KEYS = {"rgb": "observation.tactile.rgb"}
POINTCLOUD_OUTPUT_KEY = "observation.pointcloud.xyz"
POINTCLOUD_ARRAY_KEY = "xyz"

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
    tactile_videos: dict[str, VideoMetadata] | None = None
    tactile_source_key: str | None = None
    pointcloud: PointCloudMetadata | None = None

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


@dataclass(frozen=True)
class PointCloudMetadata:
    """Properties used to reconstruct one episode's camera-frame point clouds."""

    num_points: int
    intrinsics: tuple[float, float, float, float]
    depth_scale: float
    depth_key: str
    coordinate_frame: str


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


def _dataset_paths(hdf5_file: h5py.File) -> list[str]:
    paths: list[str] = []

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            paths.append(name)

    hdf5_file.visititems(collect)
    return paths


def resolve_tactile_dataset(
    hdf5_file: h5py.File,
    requested_key: str | None,
    depth_key: str | None,
) -> tuple[str, h5py.Dataset]:
    """Resolve one unmarked tactile RGB stream without inventing sensor-side semantics."""
    if requested_key is not None:
        return requested_key, _require_hdf5_dataset(hdf5_file, requested_key)

    if depth_key is not None:
        sibling_rgb_key = f"{depth_key.rsplit('/', 1)[0]}/rgb"
        if sibling_rgb_key in hdf5_file:
            return sibling_rgb_key, _require_hdf5_dataset(hdf5_file, sibling_rgb_key)

    matches = sorted(
        key
        for key in _dataset_paths(hdf5_file)
        if key.startswith("tactile/") and key.endswith("/rgb") and not key.endswith("/rgb_marker")
    )
    if len(matches) != 1:
        raise ConversionError(
            f"Could not uniquely resolve one tactile RGB stream in {hdf5_file.filename}; "
            f"matches={matches}. Pass --tactile-rgb-key when multiple streams exist."
        )
    return matches[0], _require_hdf5_dataset(hdf5_file, matches[0])


def resolve_depth_dataset(
    hdf5_file: h5py.File, requested_key: str | None
) -> tuple[str, h5py.Dataset]:
    """Resolve an explicit depth stream, or a unique depth stream when unambiguous."""
    if requested_key is not None:
        return requested_key, _require_hdf5_dataset(hdf5_file, requested_key)

    matches = [key for key in _dataset_paths(hdf5_file) if key.lower().endswith("/depth")]
    if len(matches) != 1:
        raise ConversionError(
            "Point-cloud conversion needs --pointcloud-depth-key when the source does not "
            f"contain exactly one depth stream; found {matches} in {hdf5_file.filename}."
        )
    return matches[0], _require_hdf5_dataset(hdf5_file, matches[0])


def _coerce_intrinsics(value: Any, *, source: str) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float64)
    if intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if intrinsics.shape != (3, 3):
        raise ConversionError(
            f"Camera intrinsics from {source} must have shape [3, 3] or [4], got "
            f"{intrinsics.shape}."
        )
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ConversionError(f"Camera intrinsics from {source} are invalid: {intrinsics}")
    if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0]):
        raise ConversionError(
            f"Camera intrinsics from {source} must use a standard pinhole last row [0, 0, 1]."
        )
    return intrinsics


def resolve_camera_intrinsics(
    hdf5_file: h5py.File,
    depth_key: str,
    override: list[float] | tuple[float, ...] | None,
) -> np.ndarray:
    """Read pinhole intrinsics near the depth stream, falling back to an explicit CLI value."""
    if override is not None:
        return _coerce_intrinsics(override, source="--pointcloud-intrinsics")

    depth_dataset = _require_hdf5_dataset(hdf5_file, depth_key)
    parent_key = depth_key.rsplit("/", 1)[0]
    attribute_names = ("intrinsics", "intrinsic", "camera_intrinsics", "intrinsic_matrix")
    for owner_name, owner in (
        (depth_key, depth_dataset),
        (parent_key, hdf5_file[parent_key]),
        ("/", hdf5_file),
    ):
        for attribute_name in attribute_names:
            if attribute_name in owner.attrs:
                return _coerce_intrinsics(
                    owner.attrs[attribute_name], source=f"{owner_name}.attrs[{attribute_name!r}]"
                )

    candidate_keys = [
        f"{parent_key}/{name}"
        for name in ("intrinsics", "intrinsic", "camera_intrinsics", "intrinsic_matrix")
    ]
    candidate_keys.extend(["camera_intrinsics", "intrinsics"])
    for candidate_key in candidate_keys:
        if candidate_key in hdf5_file and isinstance(hdf5_file[candidate_key], h5py.Dataset):
            return _coerce_intrinsics(
                hdf5_file[candidate_key][()], source=f"{hdf5_file.filename}:{candidate_key}"
            )

    raise ConversionError(
        f"No pinhole intrinsics metadata was found for {depth_key!r} in {hdf5_file.filename}. "
        "Pass --pointcloud-intrinsics FX FY CX CY from the source camera calibration."
    )


def resolve_depth_scale(
    depth_dataset: h5py.Dataset,
    override: float | None,
) -> float:
    """Resolve the multiplier that converts source depth values to metres."""
    if override is not None:
        scale = float(override)
    elif "depth_scale" in depth_dataset.attrs:
        scale = float(depth_dataset.attrs["depth_scale"])
    else:
        unit = depth_dataset.attrs.get("unit", depth_dataset.attrs.get("units"))
        if isinstance(unit, bytes):
            unit = unit.decode("utf-8")
        normalized_unit = str(unit).strip().lower() if unit is not None else ""
        unit_scales = {"m": 1.0, "meter": 1.0, "metre": 1.0, "mm": 1e-3, "millimeter": 1e-3}
        if normalized_unit not in unit_scales:
            raise ConversionError(
                f"No depth scale/unit metadata was found on {depth_dataset.name!r}. "
                "Pass --pointcloud-depth-scale (for example 0.001 for millimetres)."
            )
        scale = unit_scales[normalized_unit]
    if not np.isfinite(scale) or scale <= 0:
        raise ConversionError(f"Point-cloud depth scale must be positive and finite, got {scale}.")
    return scale


def reconstruct_fixed_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    num_points: int,
    depth_scale: float,
) -> np.ndarray:
    """Unproject valid depth pixels and deterministically return exactly ``num_points`` points."""
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 2:
        raise ConversionError(
            f"Depth frame must have shape [H, W] or [H, W, 1], got {depth_array.shape}."
        )
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    scaled_depth = np.asarray(depth_array, dtype=np.float64) * depth_scale
    valid = np.isfinite(scaled_depth) & (scaled_depth > 0.0)
    valid_flat_indices = np.flatnonzero(valid.reshape(-1))
    if valid_flat_indices.size == 0:
        raise ConversionError("Depth frame contains no finite positive values.")

    if valid_flat_indices.size >= num_points:
        sample_positions = np.linspace(
            0, valid_flat_indices.size - 1, num=num_points, dtype=np.int64
        )
        selected_flat_indices = valid_flat_indices[sample_positions]
    else:
        selected_flat_indices = np.resize(valid_flat_indices, num_points)

    height, width = scaled_depth.shape
    pixel_v, pixel_u = np.unravel_index(selected_flat_indices, (height, width))
    z = scaled_depth[pixel_v, pixel_u]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xyz = np.column_stack(
        (
            (pixel_u.astype(np.float64) - cx) * z / fx,
            (pixel_v.astype(np.float64) - cy) * z / fy,
            z,
        )
    )
    return np.ascontiguousarray(xyz, dtype=np.float32)


def write_pointcloud_episode(
    depth_dataset: h5py.Dataset,
    output_path: Path,
    *,
    frame_count: int,
    intrinsics: np.ndarray,
    num_points: int,
    depth_scale: float,
) -> None:
    """Reconstruct and write one fixed-shape point-cloud array without loading all depth first."""
    xyz = np.empty((frame_count, num_points, 3), dtype=np.float32)
    for frame_index in range(frame_count):
        xyz[frame_index] = reconstruct_fixed_pointcloud(
            depth_dataset[frame_index],
            intrinsics,
            num_points=num_points,
            depth_scale=depth_scale,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{POINTCLOUD_ARRAY_KEY: xyz})


def prepare_episode(
    source_path: Path,
    output_root: Path,
    *,
    episode_index: int,
    fps: float,
    ffmpeg_executable: str,
    include_tactile: bool,
    tactile_rgb_key: str | None,
    include_pointcloud: bool,
    pointcloud_depth_key: str | None,
    pointcloud_intrinsics: list[float] | None,
    pointcloud_depth_scale: float | None,
    pointcloud_num_points: int,
) -> PreparedEpisode:
    """Validate one source episode and write all requested aligned modalities."""
    try:
        hdf5_file = h5py.File(source_path, "r")
    except (OSError, ValueError) as exc:
        raise ConversionError(f"Could not open HDF5 episode {source_path}: {exc}") from exc

    with hdf5_file:
        joint_dataset = _require_hdf5_dataset(hdf5_file, JOINT_DATASET)
        head_dataset = _require_hdf5_dataset(hdf5_file, HEAD_RGB_DATASET)
        wrist_dataset = _require_hdf5_dataset(hdf5_file, WRIST_RGB_DATASET)
        depth_key: str | None = None
        depth_dataset: h5py.Dataset | None = None
        intrinsics: np.ndarray | None = None
        depth_scale: float | None = None
        if include_pointcloud:
            depth_key, depth_dataset = resolve_depth_dataset(hdf5_file, pointcloud_depth_key)
            intrinsics = resolve_camera_intrinsics(hdf5_file, depth_key, pointcloud_intrinsics)
            depth_scale = resolve_depth_scale(depth_dataset, pointcloud_depth_scale)
        tactile_data = (
            resolve_tactile_dataset(hdf5_file, tactile_rgb_key, depth_key)
            if include_tactile
            else None
        )

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

        if tactile_data is not None:
            tactile_key, tactile_dataset = tactile_data
            if tactile_dataset.ndim != 1:
                raise ConversionError(
                    f"{source_path}:{tactile_key} must be a 1D JPEG stream, got "
                    f"shape {tactile_dataset.shape}."
                )
            if len(tactile_dataset) != joint_sample_count:
                raise ConversionError(
                    f"Length mismatch in {source_path}: {JOINT_DATASET} has "
                    f"{joint_sample_count} samples but tactile RGB has "
                    f"{len(tactile_dataset)}."
                )

        if depth_dataset is not None:
            if depth_dataset.ndim not in {3, 4}:
                raise ConversionError(
                    f"{source_path}:{depth_key} must have shape [T, H, W] or [T, H, W, 1], "
                    f"got {depth_dataset.shape}."
                )
            if depth_dataset.ndim == 4 and depth_dataset.shape[-1] != 1:
                raise ConversionError(
                    f"{source_path}:{depth_key} must have a singleton final channel, got "
                    f"{depth_dataset.shape}."
                )
            if len(depth_dataset) != joint_sample_count:
                raise ConversionError(
                    f"Length mismatch in {source_path}: {JOINT_DATASET} has "
                    f"{joint_sample_count} samples but {depth_key} has {len(depth_dataset)}."
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

        tactile_videos: dict[str, VideoMetadata] | None = None
        if tactile_data is not None:
            tactile_key, tactile_dataset = tactile_data
            video_path = (
                output_root
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / TACTILE_OUTPUT_KEYS["rgb"]
                / f"episode_{episode_index:06d}.mp4"
            )
            tactile_videos = {
                "rgb": encode_hdf5_video(
                    tactile_dataset,
                    video_path,
                    frame_count=frame_count,
                    fps=fps,
                    ffmpeg_executable=ffmpeg_executable,
                    source_label=f"{source_path}:{tactile_key}",
                )
            }

        pointcloud_metadata: PointCloudMetadata | None = None
        if depth_dataset is not None:
            assert depth_key is not None and intrinsics is not None and depth_scale is not None
            pointcloud_path = (
                output_root
                / "pointclouds"
                / f"chunk-{episode_chunk:03d}"
                / POINTCLOUD_OUTPUT_KEY
                / f"episode_{episode_index:06d}.npz"
            )
            write_pointcloud_episode(
                depth_dataset,
                pointcloud_path,
                frame_count=frame_count,
                intrinsics=intrinsics,
                num_points=pointcloud_num_points,
                depth_scale=depth_scale,
            )
            pointcloud_metadata = PointCloudMetadata(
                num_points=pointcloud_num_points,
                intrinsics=(
                    float(intrinsics[0, 0]),
                    float(intrinsics[1, 1]),
                    float(intrinsics[0, 2]),
                    float(intrinsics[1, 2]),
                ),
                depth_scale=depth_scale,
                depth_key=depth_key,
                coordinate_frame=f"{depth_key.rsplit('/', 1)[0]} camera (OpenCV)",
            )

    return PreparedEpisode(
        state=state,
        action=action,
        videos=videos,
        tactile_videos=tactile_videos,
        tactile_source_key=tactile_data[0] if tactile_data is not None else None,
        pointcloud=pointcloud_metadata,
    )


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
    tactile_video_metadata: dict[str, VideoMetadata] | None,
    tactile_source_key: str | None,
    pointcloud_metadata: PointCloudMetadata | None,
    *,
    fps: float,
    task_count: int,
) -> dict[str, Any]:
    """Build current GR00T/LeRobot v2 dataset-level metadata."""
    total_frames = sum(episode.length for episode in converted_episodes)
    total_episodes = len(converted_episodes)
    features = {
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
    }
    if tactile_video_metadata is not None:
        for tactile_key, metadata in tactile_video_metadata.items():
            tactile_feature = _video_feature(metadata, fps)
            tactile_feature["info"]["source_key"] = tactile_source_key
            features[TACTILE_OUTPUT_KEYS[tactile_key]] = tactile_feature
    if pointcloud_metadata is not None:
        features[POINTCLOUD_OUTPUT_KEY] = {
            "dtype": "float32",
            "shape": [pointcloud_metadata.num_points, 3],
            "names": ["point", "xyz"],
            "info": {
                "array_key": POINTCLOUD_ARRAY_KEY,
                "coordinate_frame": pointcloud_metadata.coordinate_frame,
                "depth_scale_to_meters": pointcloud_metadata.depth_scale,
                "intrinsics_fx_fy_cx_cy": list(pointcloud_metadata.intrinsics),
            },
        }

    info = {
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
        "features": features,
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "total_videos": total_episodes
        * (len(CAMERA_DATASETS) + (len(tactile_video_metadata or {}))),
    }
    if pointcloud_metadata is not None:
        info["pointcloud_path"] = (
            "pointclouds/chunk-{episode_chunk:03d}/{pointcloud_key}/episode_{episode_index:06d}.npz"
        )
    return info


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
    tactile_video_metadata: dict[str, VideoMetadata] | None,
    tactile_source_key: str | None,
    pointcloud_metadata: PointCloudMetadata | None,
    *,
    fps: float,
) -> None:
    """Write all metadata files required by the current GR00T loader."""
    info = build_info_metadata(
        converted_episodes,
        video_metadata,
        tactile_video_metadata,
        tactile_source_key,
        pointcloud_metadata,
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
    if tactile_video_metadata is not None:
        modality["tactile"] = {
            tactile_key: {
                "original_key": original_key,
                "source_key": tactile_source_key,
            }
            for tactile_key, original_key in TACTILE_OUTPUT_KEYS.items()
        }
    if pointcloud_metadata is not None:
        modality["pointcloud"] = {
            "xyz": {
                "original_key": POINTCLOUD_OUTPUT_KEY,
                "array_key": POINTCLOUD_ARRAY_KEY,
                "coordinate_frame": pointcloud_metadata.coordinate_frame,
                "depth_source": pointcloud_metadata.depth_key,
                "depth_scale_to_meters": pointcloud_metadata.depth_scale,
                "intrinsics_fx_fy_cx_cy": list(pointcloud_metadata.intrinsics),
                "num_points": pointcloud_metadata.num_points,
            }
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
    pointcloud_metadata: PointCloudMetadata | None,
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

        if pointcloud_metadata is not None:
            pointcloud_path = (
                output_root
                / "pointclouds"
                / f"chunk-{chunk_index:03d}"
                / POINTCLOUD_OUTPUT_KEY
                / f"episode_{episode.episode_index:06d}.npz"
            )
            with np.load(pointcloud_path, allow_pickle=False) as archive:
                if POINTCLOUD_ARRAY_KEY not in archive:
                    raise ConversionError(
                        f"{pointcloud_path} does not contain {POINTCLOUD_ARRAY_KEY!r}."
                    )
                xyz = archive[POINTCLOUD_ARRAY_KEY]
                expected_shape = (episode.length, pointcloud_metadata.num_points, 3)
                if xyz.shape != expected_shape:
                    raise ConversionError(
                        f"Incorrect point-cloud shape in {pointcloud_path}: {xyz.shape}; "
                        f"expected {expected_shape}."
                    )
                if xyz.dtype != np.float32:
                    raise ConversionError(
                        f"Incorrect point-cloud dtype in {pointcloud_path}: {xyz.dtype}."
                    )
                if not np.isfinite(xyz).all():
                    raise ConversionError(f"NaN or Inf found in {pointcloud_path}")

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
        *(
            output_root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / original_key
            / f"episode_{episode_index:06d}.mp4"
            for original_key in TACTILE_OUTPUT_KEYS.values()
        ),
        output_root
        / "pointclouds"
        / f"chunk-{chunk_index:03d}"
        / POINTCLOUD_OUTPUT_KEY
        / f"episode_{episode_index:06d}.npz",
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

    if episode_metadata.keys() != reference_metadata.keys():
        raise ConversionError(
            f"Video keys changed in {source_path}: expected {list(reference_metadata)}, "
            f"got {list(episode_metadata)}."
        )
    for video_name in reference_metadata:
        reference = reference_metadata[video_name]
        current = episode_metadata[video_name]
        if (current.height, current.width) != (reference.height, reference.width):
            raise ConversionError(
                f"{video_name} resolution changed in {source_path}: expected "
                f"{reference.width}x{reference.height}, got {current.width}x{current.height}."
            )
    return reference_metadata


def _validate_pointcloud_schema(
    reference_metadata: PointCloudMetadata | None,
    episode_metadata: PointCloudMetadata | None,
    source_path: Path,
) -> PointCloudMetadata | None:
    if reference_metadata is None:
        return episode_metadata
    if episode_metadata is None:
        raise ConversionError(f"Point cloud is missing from {source_path}")
    if episode_metadata != reference_metadata:
        raise ConversionError(
            f"Point-cloud schema/calibration changed in {source_path}: expected "
            f"{reference_metadata}, got {episode_metadata}."
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
    if args.pointcloud_num_points <= 0:
        raise ValueError(
            f"pointcloud-num-points must be positive, got {args.pointcloud_num_points}"
        )
    if args.tactile_rgb_key is not None and not args.include_tactile:
        raise ValueError("--tactile-rgb-key requires --include-tactile")
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
    reference_tactile_video_metadata: dict[str, VideoMetadata] | None = None
    reference_tactile_source_key: str | None = None
    reference_pointcloud_metadata: PointCloudMetadata | None = None
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
                    include_tactile=args.include_tactile,
                    tactile_rgb_key=args.tactile_rgb_key,
                    include_pointcloud=args.include_pointcloud,
                    pointcloud_depth_key=args.pointcloud_depth_key,
                    pointcloud_intrinsics=args.pointcloud_intrinsics,
                    pointcloud_depth_scale=args.pointcloud_depth_scale,
                    pointcloud_num_points=args.pointcloud_num_points,
                )
                validated_video_metadata = _validate_video_schema(
                    reference_video_metadata,
                    prepared_episode.videos,
                    source_path,
                )
                validated_tactile_video_metadata = (
                    _validate_video_schema(
                        reference_tactile_video_metadata,
                        prepared_episode.tactile_videos,
                        source_path,
                    )
                    if prepared_episode.tactile_videos is not None
                    else None
                )
                if (
                    reference_tactile_source_key is not None
                    and prepared_episode.tactile_source_key != reference_tactile_source_key
                ):
                    raise ConversionError(
                        "Tactile RGB source key changed across episodes: "
                        f"{reference_tactile_source_key!r} != "
                        f"{prepared_episode.tactile_source_key!r} in {source_path}"
                    )
                validated_pointcloud_metadata = _validate_pointcloud_schema(
                    reference_pointcloud_metadata,
                    prepared_episode.pointcloud,
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
                reference_tactile_video_metadata = validated_tactile_video_metadata
                reference_tactile_source_key = prepared_episode.tactile_source_key
                reference_pointcloud_metadata = validated_pointcloud_metadata
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
            reference_tactile_video_metadata,
            reference_tactile_source_key,
            reference_pointcloud_metadata,
            fps=args.fps,
        )
        validate_converted_dataset(
            staging_root,
            converted_episodes,
            reference_pointcloud_metadata,
            fps=args.fps,
        )
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
    if reference_tactile_video_metadata is not None:
        for tactile_key, metadata in reference_tactile_video_metadata.items():
            print(
                f"Tactile {tactile_key}: ({metadata.height}, {metadata.width}, 3) uint8, "
                f"video FPS {metadata.fps:g}"
            )
    if reference_pointcloud_metadata is not None:
        print(
            f"Point cloud: ({reference_pointcloud_metadata.num_points}, 3) float32 "
            f"in {reference_pointcloud_metadata.coordinate_frame}"
        )
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
        "--include-tactile",
        action="store_true",
        help="Encode one unmarked tactile RGB stream as observation.tactile.rgb video.",
    )
    parser.add_argument(
        "--tactile-rgb-key",
        help=(
            "HDF5 tactile RGB dataset to encode. If omitted, use the RGB sibling of "
            "--pointcloud-depth-key or require a unique tactile */rgb stream."
        ),
    )
    parser.add_argument(
        "--include-pointcloud",
        action="store_true",
        help="Reconstruct and store fixed-size camera-frame point clouds from depth.",
    )
    parser.add_argument(
        "--pointcloud-depth-key",
        help=(
            "HDF5 depth dataset used for point clouds (for example "
            "tactile/<sensor>/depth). Required when multiple depth streams exist."
        ),
    )
    parser.add_argument(
        "--pointcloud-intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Pinhole intrinsics override. Otherwise intrinsics are read from HDF5 metadata.",
    )
    parser.add_argument(
        "--pointcloud-depth-scale",
        type=float,
        help=(
            "Multiplier from stored depth units to metres (for example 0.001 for mm). "
            "Otherwise depth_scale/unit HDF5 metadata is required."
        ),
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        default=1024,
        help="Fixed number of deterministic points per frame (default: 1024).",
    )
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
