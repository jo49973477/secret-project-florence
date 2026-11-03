#!/usr/bin/env python3
"""Validate a UniVTAC -> LeRobot/GR00T conversion from actual values.

This validator is intentionally independent of the training stack (and therefore
does not import torch).  It implements the data contract used by
``examples/UniVTAC/convert_univtac_to_lerobot.py`` and
``examples/UniVTAC/univtac_config.py``:

* stored state[t]  = raw embodiment/joint[t, :8]
* stored action[t] = raw embodiment/joint[t + 1, :8] (absolute target)
* GR00T training samples action[t:t+H], converts every target to
  action[t+h] - state[t], and percentile-min/max normalizes it.

The source may be the tactile-enabled LeRobot ``univtac_full`` dataset or a
directory of raw HDF5 episodes.  Inputs are opened read-only.  Reports and plots
are the only files written.
"""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence

import h5py
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


JOINT_DATASET = "embodiment/joint"
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
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
JOINT_UNITS = ["rad"] * 7 + ["m"]
EXPECTED_SEMANTICS = "absolute_next_state"
SEMANTICS_LABELS = {
    "state_t": "action[t] ~= state[t]",
    "state_t_plus_1": "action[t] ~= state[t+1]",
    "next_minus_current": "action[t] ~= state[t+1] - state[t]",
    "current_minus_previous": "action[t] ~= state[t] - state[t-1]",
}
TASK_OVERRIDES = {
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
SEVERITY = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class Finding:
    severity: str
    check: str
    message: str
    episode: int | None = None


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    source_path: Path | None
    raw: dict[str, Any]


@dataclass
class EpisodeData:
    meta: EpisodeMeta
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    episode_index: np.ndarray
    state_dtype: str
    action_dtype: str
    raw_joint: np.ndarray | None = None


@dataclass
class DatasetView:
    root: Path
    kind: str
    info: dict[str, Any]
    modality: dict[str, Any]
    episodes: list[EpisodeMeta]
    stats: dict[str, Any] | None
    relative_stats: dict[str, Any] | None
    parquet_schema: Any | None = None


@dataclass
class EpisodeResult:
    target_episode: int
    source_episode: int
    mapping_key: str
    frames: int
    metrics: dict[str, Any]
    semantics_mae: dict[str, float]
    lag_mae: dict[int, float]
    plots: list[Path] = field(default_factory=list)
    modality: dict[str, Any] = field(default_factory=dict)


class Findings:
    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, severity: str, check: str, message: str, episode: int | None = None) -> None:
        self.items.append(Finding(severity, check, message, episode))

    def fail(self, check: str, message: str, episode: int | None = None) -> None:
        self.add("FAIL", check, message, episode)

    def warn(self, check: str, message: str, episode: int | None = None) -> None:
        self.add("WARN", check, message, episode)

    def passed(self, check: str, message: str, episode: int | None = None) -> None:
        self.add("PASS", check, message, episode)

    @property
    def verdict(self) -> str:
        return (
            max(self.items, key=lambda x: SEVERITY[x.severity]).severity if self.items else "FAIL"
        )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def natural_key(path: Path) -> tuple[tuple[int, str | int], ...]:
    parts = re.split(r"(\d+)", path.as_posix().lower())
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts)


def normalize_task(text: str) -> str:
    key = re.sub(r"[-\s]+", "_", text.strip()).strip("_").lower()
    return TASK_OVERRIDES.get(key, key.replace("_", " "))


def infer_raw_task(path: Path) -> str:
    for parent in path.parents:
        candidate = parent.name.lower().replace("-", "_")
        if candidate in TASK_OVERRIDES:
            return TASK_OVERRIDES[candidate]
    return normalize_task(
        path.parent.parent.name if path.parent.name == "clean" else path.parent.name
    )


def load_dataset(path: Path) -> DatasetView:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    info_path = path / "meta" / "info.json"
    episodes_path = path / "meta" / "episodes.jsonl"
    if info_path.is_file() and episodes_path.is_file():
        info = read_json(info_path)
        modality_path = path / "meta" / "modality.json"
        modality = read_json(modality_path) if modality_path.is_file() else {}
        episodes = []
        for row in read_jsonl(episodes_path):
            source_path = row.get("source_path")
            episodes.append(
                EpisodeMeta(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    tasks=tuple(normalize_task(x) for x in row.get("tasks", [])),
                    source_path=Path(source_path).expanduser() if source_path else None,
                    raw=row,
                )
            )
        stats_path = path / "meta" / "stats.json"
        rel_path = path / "meta" / "relative_stats.json"
        return DatasetView(
            root=path,
            kind="lerobot",
            info=info,
            modality=modality,
            episodes=episodes,
            stats=read_json(stats_path) if stats_path.is_file() else None,
            relative_stats=read_json(rel_path) if rel_path.is_file() else None,
        )

    files = sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}],
        key=natural_key,
    )
    if not files:
        raise ValueError(f"Neither a LeRobot dataset nor raw HDF5 directory: {path}")
    episodes = []
    for idx, raw_path in enumerate(files):
        with h5py.File(raw_path, "r") as handle:
            length = int(handle[JOINT_DATASET].shape[0]) - 1
        task = infer_raw_task(raw_path)
        episodes.append(EpisodeMeta(idx, length, (task,), raw_path, {"source_path": str(raw_path)}))
    return DatasetView(path, "hdf5", {"total_episodes": len(episodes)}, {}, episodes, None, None)


def episode_identity(meta: EpisodeMeta) -> tuple[tuple[str, ...], str] | None:
    if meta.source_path is None:
        return None
    return meta.tasks, meta.source_path.name.lower()


def format_identity(key: tuple[tuple[str, ...], str]) -> str:
    return f"{' | '.join(key[0])} :: {key[1]}"


def build_mapping(
    source: DatasetView, target: DatasetView, findings: Findings
) -> dict[int, tuple[EpisodeMeta, str]]:
    src_by_key: dict[tuple[tuple[str, ...], str], list[EpisodeMeta]] = {}
    for meta in source.episodes:
        key = episode_identity(meta)
        if key is not None:
            src_by_key.setdefault(key, []).append(meta)
    mapping = {}
    ambiguous = 0
    missing = 0
    missing_labels: list[str] = []
    for target_meta in target.episodes:
        key = episode_identity(target_meta)
        matches = src_by_key.get(key, []) if key is not None else []
        if len(matches) == 1 and key is not None:
            mapping[target_meta.episode_index] = (matches[0], format_identity(key))
        elif len(matches) > 1:
            ambiguous += 1
        else:
            missing += 1
            missing_labels.append(
                f"target {target_meta.episode_index}: "
                f"{format_identity(key) if key is not None else 'no source identity'}"
            )
    if ambiguous or missing:
        findings.fail(
            "episode_mapping",
            f"Mapping is incomplete: mapped={len(mapping)}, missing={missing}, ambiguous={ambiguous}. "
            f"Unmatched: {missing_labels[:20]}. No episode-index fallback was used.",
        )
    else:
        findings.passed(
            "episode_mapping", f"All {len(mapping)} targets map uniquely by task + HDF5 filename."
        )
    return mapping


def parquet_path(dataset: DatasetView, episode_index: int) -> Path:
    chunk_size = int(dataset.info.get("chunks_size", dataset.info.get("chunk_size", 1000)))
    template = dataset.info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return dataset.root / template.format(
        episode_chunk=episode_index // chunk_size, episode_index=episode_index
    )


def raw_joint_from_path(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as handle:
            if JOINT_DATASET not in handle:
                return None
            return np.asarray(handle[JOINT_DATASET][:, :JOINT_DIMENSION], dtype=np.float64)
    except OSError:
        return None


def load_episode(dataset: DatasetView, meta: EpisodeMeta, *, load_raw: bool = True) -> EpisodeData:
    raw_joint = (
        raw_joint_from_path(meta.source_path) if load_raw or dataset.kind == "hdf5" else None
    )
    if dataset.kind == "hdf5":
        if raw_joint is None:
            raise ValueError(f"Could not load {meta.source_path}:{JOINT_DATASET}")
        length = len(raw_joint) - 1
        fps = float(dataset.info.get("fps", 10.0))
        return EpisodeData(
            meta,
            raw_joint[:-1].astype(np.float32),
            raw_joint[1:].astype(np.float32),
            np.arange(length, dtype=np.float64) / fps,
            np.arange(length, dtype=np.int64),
            np.full(length, meta.episode_index, dtype=np.int64),
            "float32",
            "float32",
            raw_joint,
        )
    path = parquet_path(dataset, meta.episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    table = pq.read_table(path)
    required = [STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index", "episode_index"]
    missing = [key for key in required if key not in table.column_names]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float64)
    action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
    return EpisodeData(
        meta,
        state,
        action,
        np.asarray(table["timestamp"].to_pylist(), dtype=np.float64),
        np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        np.asarray(table["episode_index"].to_pylist(), dtype=np.int64),
        str(table.schema.field(STATE_COLUMN).type),
        str(table.schema.field(ACTION_COLUMN).type),
        raw_joint,
    )


def names_for(dataset: DatasetView, feature: str) -> list[str]:
    names = dataset.info.get("features", {}).get(feature, {}).get("names")
    return [str(x) for x in names] if isinstance(names, list) else []


def finite_mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if np.any(mask) else math.inf


def finite_max(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask] - b[mask]))) if np.any(mask) else math.inf


def semantics_candidates(action: np.ndarray, state: np.ndarray) -> dict[str, float]:
    candidates: dict[str, float] = {}
    n = min(len(action), len(state))
    candidates["state_t"] = finite_mae(action[:n], state[:n])
    n_next = min(len(action), max(0, len(state) - 1))
    candidates["state_t_plus_1"] = finite_mae(action[:n_next], state[1 : 1 + n_next])
    candidates["next_minus_current"] = finite_mae(
        action[:n_next], state[1 : 1 + n_next] - state[:n_next]
    )
    n_prev = min(max(0, len(action) - 1), max(0, len(state) - 1))
    candidates["current_minus_previous"] = finite_mae(
        action[1 : 1 + n_prev], state[1 : 1 + n_prev] - state[:n_prev]
    )
    return candidates


def lag_analysis(
    source_action: np.ndarray, target_action: np.ndarray, lags: Iterable[int]
) -> dict[int, float]:
    result = {}
    for lag in lags:
        if lag >= 0:
            n = min(len(target_action), len(source_action) - lag)
            aa, bb = target_action[: max(0, n)], source_action[lag : lag + max(0, n)]
        else:
            n = min(len(target_action) + lag, len(source_action))
            aa, bb = target_action[-lag : -lag + max(0, n)], source_action[: max(0, n)]
        result[int(lag)] = finite_mae(aa, bb) if n > 0 else math.inf
    return result


def validate_indices(
    data: EpisodeData, dataset_name: str, fps: float, tolerance: float, findings: Findings
) -> None:
    ep = data.meta.episode_index
    n = len(data.state)
    if len(data.action) != n:
        findings.fail(
            "frame_count", f"{dataset_name} state/action rows differ: {n}/{len(data.action)}", ep
        )
    if data.meta.length != n:
        findings.fail(
            "metadata_length", f"{dataset_name} metadata={data.meta.length}, parquet={n}", ep
        )
    if data.state.ndim != 2 or data.action.ndim != 2:
        findings.fail("dimension", f"{dataset_name} state/action must be rank 2", ep)
    elif data.state.shape[1] != data.action.shape[1]:
        findings.fail(
            "dimension",
            f"{dataset_name} state/action dims differ: {data.state.shape}/{data.action.shape}",
            ep,
        )
    if not np.isfinite(data.state).all() or not np.isfinite(data.action).all():
        findings.fail("finite", f"{dataset_name} contains NaN/Inf", ep)
    if len(data.frame_index) != n or not np.array_equal(data.frame_index, np.arange(n)):
        findings.fail("frame_index", f"{dataset_name} frame_index is not contiguous 0..{n - 1}", ep)
    if len(data.episode_index) != n or np.any(data.episode_index != ep):
        findings.fail(
            "episode_index", f"{dataset_name} episode_index column does not equal {ep}", ep
        )
    if len(data.timestamp) != n or (n > 1 and not np.all(np.diff(data.timestamp) > 0)):
        findings.fail("timestamp", f"{dataset_name} timestamp is not strictly increasing", ep)
    expected_ts = np.arange(n, dtype=np.float64) / fps
    if len(data.timestamp) == n and finite_max(data.timestamp, expected_ts) > tolerance:
        findings.fail("timestamp", f"{dataset_name} timestamp != frame_index/fps", ep)
    # Arrow spells float32 list elements as ``float``.
    state_dtype_ok = "float" in data.state_dtype and "double" not in data.state_dtype
    action_dtype_ok = "float" in data.action_dtype and "double" not in data.action_dtype
    if not state_dtype_ok or not action_dtype_ok:
        findings.fail(
            "dtype",
            f"{dataset_name} schema state/action dtype={data.state_dtype}/{data.action_dtype}, expected float32",
            ep,
        )


def modality_keys(dataset: DatasetView, category: str) -> dict[str, dict[str, Any]]:
    value = dataset.modality.get(category, {})
    return value if isinstance(value, dict) else {}


def raw_modality_details(raw_path: Path | None) -> dict[str, Any]:
    details: dict[str, Any] = {"raw_path": str(raw_path) if raw_path else None, "streams": {}}
    if raw_path is None or not raw_path.is_file():
        details["raw_available"] = False
        return details
    details["raw_available"] = True
    with h5py.File(raw_path, "r") as handle:

        def visitor(name: str, value: Any) -> None:
            if not isinstance(value, h5py.Dataset):
                return
            lower = name.lower()
            if (
                name == JOINT_DATASET
                or lower.endswith("/rgb")
                or "timestamp" in lower
                or lower.endswith("/depth")
            ):
                entry = {
                    "length": int(value.shape[0]) if value.ndim else 1,
                    "shape": list(value.shape),
                }
                if lower.startswith("tactile/") and lower.endswith("/rgb") and value.ndim == 1:
                    # Exact byte duplicates are diagnostic only: a stationary tactile
                    # image can be legitimate, while duplicate timestamps are fatal.
                    fingerprints = []
                    for payload in value:
                        if isinstance(payload, np.ndarray):
                            payload = payload.tobytes()
                        elif hasattr(payload, "tobytes"):
                            payload = payload.tobytes()
                        fingerprints.append(hash(bytes(payload)))
                    entry["duplicate_payloads"] = int(len(fingerprints) - len(set(fingerprints)))
                if "timestamp" in lower and value.ndim == 1:
                    arr = np.asarray(value[:], dtype=np.float64)
                    entry["strictly_increasing"] = bool(len(arr) < 2 or np.all(np.diff(arr) > 0))
                    entry["duplicates"] = int(len(arr) - len(np.unique(arr)))
                    entry["first"] = float(arr[0]) if len(arr) else None
                    entry["last"] = float(arr[-1]) if len(arr) else None
                details["streams"][name] = entry

        handle.visititems(visitor)
    return details


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        stream = json.loads(proc.stdout).get("streams", [{}])[0]
        count = stream.get("nb_read_frames") or stream.get("nb_frames")
        rate = stream.get("r_frame_rate", "0/1")
        numerator, denominator = (float(x) for x in rate.split("/"))
        return {
            "exists": True,
            "frames": int(count) if count not in {None, "N/A"} else None,
            "fps": numerator / denominator if denominator else None,
            "width": stream.get("width"),
            "height": stream.get("height"),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        ffprobe_error = str(exc)
    try:
        import cv2  # Existing project dependency; optional for this standalone script.

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError("cv2.VideoCapture could not open the file")
        result = {
            "exists": True,
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "probe_backend": "opencv",
            "ffprobe_error": ffprobe_error,
        }
        capture.release()
        return result
    except Exception as cv_error:
        return {
            "exists": True,
            "probe_error": f"ffprobe={ffprobe_error}; opencv={cv_error}",
        }


def video_path(dataset: DatasetView, episode: int, original_key: str) -> Path:
    chunk_size = int(dataset.info.get("chunks_size", 1000))
    template = dataset.info.get(
        "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return dataset.root / template.format(
        episode_chunk=episode // chunk_size, episode_index=episode, video_key=original_key
    )


def validate_modalities(
    dataset: DatasetView,
    episode: EpisodeData,
    fps: float,
    findings: Findings,
    probe_videos: bool,
    report_episode: int | None = None,
) -> dict[str, Any]:
    report_episode = episode.meta.episode_index if report_episode is None else report_episode
    raw = raw_modality_details(episode.meta.source_path)
    result: dict[str, Any] = {"raw": raw, "videos": {}}
    joint_samples = (
        len(episode.raw_joint) if episode.raw_joint is not None else len(episode.state) + 1
    )
    for name, entry in raw.get("streams", {}).items():
        lower = name.lower()
        if (lower.endswith("/rgb") or lower.endswith("/depth")) and entry[
            "length"
        ] != joint_samples:
            findings.fail(
                "modality_alignment",
                f"Raw stream {name} length={entry['length']} but joint length={joint_samples}",
                report_episode,
            )
        if "timestamp" in lower and not entry.get("strictly_increasing", True):
            findings.fail(
                "modality_timestamp", f"Raw {name} is not strictly increasing", report_episode
            )
        if entry.get("duplicate_payloads", 0):
            findings.warn(
                "tactile_duplicates",
                f"Raw tactile stream {name} has {entry['duplicate_payloads']} byte-identical frames; "
                "indices are still unique and length-aligned.",
                report_episode,
            )
    timestamp_streams = [k for k in raw.get("streams", {}) if "timestamp" in k.lower()]
    if raw.get("raw_available") and not timestamp_streams:
        findings.warn(
            "modality_timestamp",
            "Raw HDF5 has no explicit timestamps; alignment is verifiable by shared sample index and 10 Hz metadata only.",
            report_episode,
        )
    if probe_videos and dataset.kind == "lerobot":
        for category in ("video", "tactile"):
            for key, cfg in modality_keys(dataset, category).items():
                original = cfg.get("original_key", f"observation.{category}.{key}")
                probe = probe_video(video_path(dataset, episode.meta.episode_index, original))
                result["videos"][original] = probe
                if not probe.get("exists"):
                    findings.fail(
                        "modality_video", f"Missing {category} video {original}", report_episode
                    )
                elif probe.get("frames") is not None and probe["frames"] != len(episode.state):
                    findings.fail(
                        "modality_alignment",
                        f"{original} video frames={probe['frames']}, rows={len(episode.state)}",
                        report_episode,
                    )
                if probe.get("fps") is not None and abs(float(probe["fps"]) - fps) > 1e-6:
                    findings.fail(
                        "modality_alignment",
                        f"{original} video fps={probe['fps']}, metadata fps={fps}",
                        report_episode,
                    )
                if probe.get("probe_error"):
                    findings.warn(
                        "modality_video",
                        f"Could not probe {original}: {probe['probe_error']}",
                        report_episode,
                    )
    return result


def select_plot_indices(state: np.ndarray, action: np.ndarray) -> list[int]:
    n = len(state)
    chosen = set(range(min(20, n)))
    chosen.update(range(max(0, n // 2 - 10), min(n, n // 2 + 10)))
    chosen.update(range(max(0, n - 20), n))
    for values in (state, action):
        if len(values) > 1:
            change = np.linalg.norm(np.diff(values, axis=0), axis=1)
            peak = int(np.argmax(change)) + 1
            chosen.update(range(max(0, peak - 10), min(n, peak + 10)))
    return sorted(chosen)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlays(
    output: Path,
    names: Sequence[str],
    source_values: np.ndarray,
    target_values: np.ndarray,
    title: str,
    stem: str,
    max_joints: int,
) -> list[Path]:
    paths = []
    for page, start in enumerate(range(0, len(names), max_joints)):
        end = min(len(names), start + max_joints)
        fig, axes = plt.subplots(
            end - start, 1, figsize=(13, max(3, 2.1 * (end - start))), sharex=True
        )
        axes = np.atleast_1d(axes)
        for ax, joint in zip(axes, range(start, end)):
            ax.plot(source_values[:, joint], label="univtac_full", linewidth=1.4)
            ax.plot(target_values[:, joint], label="univtac_gr00t", linewidth=1, linestyle="--")
            ax.set_ylabel(names[joint])
            ax.grid(alpha=0.2)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("frame")
        fig.suptitle(title)
        suffix = "" if len(names) <= max_joints else f"_{page + 1:02d}"
        path = output / f"{stem}{suffix}.png"
        save_figure(fig, path)
        paths.append(path)
    return paths


def plot_heatmap(
    output: Path, errors: np.ndarray, names: Sequence[str], title: str, stem: str
) -> Path:
    fig, ax = plt.subplots(figsize=(13, max(3, 0.55 * len(names))))
    image = ax.imshow(errors.T, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("frame")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="absolute error")
    path = output / f"{stem}.png"
    save_figure(fig, path)
    return path


def plot_lags(output: Path, lags: dict[int, float], expected: int, best: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = sorted(lags)
    ax.plot(xs, [lags[x] for x in xs], marker="o")
    ax.axvline(expected, color="green", linestyle="--", label=f"expected lag={expected}")
    ax.axvline(best, color="red", linestyle=":", label=f"best lag={best}")
    ax.set(
        xlabel="lag (target action[t] vs source action[t+lag])",
        ylabel="MAE",
        title="Action lag analysis",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    path = output / "lag_analysis.png"
    save_figure(fig, path)
    return path


def plot_distributions(output: Path, source: EpisodeData, target: EpisodeData) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, attr, title in zip(
        axes, ("state", "action"), ("State distribution", "Action distribution")
    ):
        source_values = getattr(source, attr).ravel()
        target_values = getattr(target, attr).ravel()
        ax.hist(source_values, bins=80, density=True, alpha=0.45, label="univtac_full")
        ax.hist(target_values, bins=80, density=True, alpha=0.45, label="univtac_gr00t")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
    path = output / "value_distribution.png"
    save_figure(fig, path)
    return path


def plot_timeline(output: Path, episode: EpisodeData, modality: dict[str, Any], fps: float) -> Path:
    streams: list[tuple[str, int]] = [
        ("state", len(episode.state)),
        ("action", len(episode.action)),
    ]
    for name, entry in modality.get("raw", {}).get("streams", {}).items():
        if name == JOINT_DATASET or "timestamp" in name.lower():
            continue
        streams.append((f"raw:{name}", int(entry["length"])))
    for name, probe in modality.get("videos", {}).items():
        if probe.get("frames") is not None:
            streams.append((f"converted:{name}", int(probe["frames"])))
    fig, ax = plt.subplots(figsize=(13, max(3.5, 0.7 * len(streams))))
    expected = len(episode.state)
    for row, (name, count) in enumerate(streams):
        frames = np.arange(count)
        color = np.where(frames < expected, "tab:blue", "tab:red")
        ax.scatter(frames / fps, np.full(count, row), c=color, s=8)
        if count < expected:
            ax.scatter(
                np.arange(count, expected) / fps,
                np.full(expected - count, row),
                marker="x",
                c="red",
                s=18,
            )
    ax.set_yticks(range(len(streams)), labels=[x[0] for x in streams])
    ax.set_xlabel("synthetic timestamp (s, frame/fps)")
    ax.set_title("Modality timeline (red x = missing; red dots = extra vs converted rows)")
    ax.grid(axis="x", alpha=0.2)
    path = output / "modality_timeline.png"
    save_figure(fig, path)
    return path


def decode_hdf5_jpeg(raw_path: Path, key: str, index: int) -> Image.Image | None:
    try:
        with h5py.File(raw_path, "r") as handle:
            payload = handle[key][index]
            if isinstance(payload, np.ndarray):
                if payload.ndim == 3:
                    return Image.fromarray(payload.astype(np.uint8)).convert("RGB")
                payload = payload.tobytes()
            elif hasattr(payload, "tobytes"):
                payload = payload.tobytes()
            return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        return None


def plot_frame_samples(
    output: Path,
    episode: EpisodeData,
    source_next: np.ndarray,
    predicted_next: np.ndarray,
) -> Path | None:
    raw_path = episode.meta.source_path
    if raw_path is None or not raw_path.is_file():
        return None
    frames = sorted({0, len(episode.state) // 2, max(0, len(episode.state) - 1)})
    decoded = [decode_hdf5_jpeg(raw_path, "observation/head/rgb", i) for i in frames]
    if not any(image is not None for image in decoded):
        return None
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 4.5 * len(frames)))
    axes = np.atleast_1d(axes)
    tactile_keys = [
        name
        for name in raw_modality_details(raw_path).get("streams", {})
        if name.startswith("tactile/") and name.endswith("/rgb")
    ]
    for ax, frame, image in zip(axes, frames, decoded):
        if image is not None:
            ax.imshow(image)
        ax.axis("off")
        err = np.abs(source_next[frame] - predicted_next[frame])
        text = (
            f"episode/frame: {episode.meta.episode_index}/{frame} | tactile frame: {frame if tactile_keys else 'N/A'}\n"
            f"state[t]: {np.array2string(episode.state[frame], precision=4)}\n"
            f"action[t]: {np.array2string(episode.action[frame], precision=4)}\n"
            f"state[t+1]: {np.array2string(source_next[frame], precision=4)}\n"
            f"predicted state[t+1]: {np.array2string(predicted_next[frame], precision=4)}\n"
            f"abs error: {np.array2string(err, precision=3)}"
        )
        ax.text(
            0.01,
            0.01,
            text,
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
    path = output / "frame_samples.png"
    save_figure(fig, path)
    return path


def write_rows(
    writer: csv.DictWriter,
    target_episode: int,
    source: EpisodeData,
    target: EpisodeData,
    names: Sequence[str],
    source_next: np.ndarray,
    predicted_next: np.ndarray,
) -> list[dict[str, Any]]:
    worst = []
    n = min(len(source.state), len(target.state), len(source.action), len(target.action))
    for frame in range(n):
        for joint, name in enumerate(names):
            row = {
                "episode": target_episode,
                "source_episode": source.meta.episode_index,
                "frame": frame,
                "timestamp": float(target.timestamp[frame]),
                "joint_name": name,
                "source_state": float(source.state[frame, joint]),
                "lerobot_state": float(target.state[frame, joint]),
                "state_abs_error": float(
                    abs(source.state[frame, joint] - target.state[frame, joint])
                ),
                "source_action": float(source.action[frame, joint]),
                "lerobot_action": float(target.action[frame, joint]),
                "action_abs_error": float(
                    abs(source.action[frame, joint] - target.action[frame, joint])
                ),
                "source_next_state": float(source_next[frame, joint]),
                "reconstructed_next_state": float(predicted_next[frame, joint]),
                "reconstruction_error": float(
                    abs(source_next[frame, joint] - predicted_next[frame, joint])
                ),
            }
            writer.writerow(row)
            worst.append(row)
    return worst


def actual_next_state(source: EpisodeData) -> np.ndarray:
    if source.raw_joint is not None and len(source.raw_joint) >= len(source.state) + 1:
        return source.raw_joint[1 : len(source.state) + 1]
    result = np.full_like(source.state, np.nan, dtype=np.float64)
    if len(source.state) > 1:
        result[:-1] = source.state[1:]
    return result


def compare_one_episode(
    source_dataset: DatasetView,
    target_dataset: DatasetView,
    source_meta: EpisodeMeta,
    target_meta: EpisodeMeta,
    mapping_key: str,
    output: Path,
    csv_writer: csv.DictWriter,
    tolerance: float,
    expected_lag: int,
    lag_min: int,
    lag_max: int,
    action_horizon: int,
    max_plot_joints: int,
    findings: Findings,
    probe_videos: bool,
) -> tuple[EpisodeResult, list[dict[str, Any]]]:
    source = load_episode(source_dataset, source_meta)
    target = load_episode(target_dataset, target_meta)
    fps_source = float(source_dataset.info.get("fps", 10.0))
    fps_target = float(target_dataset.info.get("fps", 10.0))
    validate_indices(source, "source", fps_source, tolerance, findings)
    validate_indices(target, "target", fps_target, tolerance, findings)
    ep = target_meta.episode_index
    if fps_source != fps_target:
        findings.fail("fps", f"Source fps={fps_source}, target fps={fps_target}", ep)
    if source.state.shape != target.state.shape or source.action.shape != target.action.shape:
        findings.fail(
            "dimension",
            f"Source/target shapes differ: state {source.state.shape}/{target.state.shape}, "
            f"action {source.action.shape}/{target.action.shape}",
            ep,
        )
    n = min(len(source.state), len(target.state), len(source.action), len(target.action))
    dim = min(
        source.state.shape[1], target.state.shape[1], source.action.shape[1], target.action.shape[1]
    )
    src_names = names_for(source_dataset, STATE_COLUMN) or JOINT_NAMES[:dim]
    dst_names = names_for(target_dataset, STATE_COLUMN) or JOINT_NAMES[:dim]
    if len(src_names) != len(set(src_names)) or len(dst_names) != len(set(dst_names)):
        findings.fail("joint_names", "Duplicate joint names found", ep)
    missing = sorted(set(src_names) ^ set(dst_names))
    if missing:
        findings.fail("joint_names", f"Joint name sets differ: {missing}", ep)
    reordered_state_error = math.inf
    reordered_action_error = math.inf
    if src_names != dst_names:
        findings.fail(
            "joint_order", f"Stored joint order differs: source={src_names}, target={dst_names}", ep
        )
        if not missing and len(src_names) == len(dst_names):
            order = [dst_names.index(name) for name in src_names]
            reordered_state_error = finite_mae(source.state[:n, :dim], target.state[:n, order])
            reordered_action_error = finite_mae(source.action[:n, :dim], target.action[:n, order])
    else:
        reordered_state_error = finite_mae(source.state[:n, :dim], target.state[:n, :dim])
        reordered_action_error = finite_mae(source.action[:n, :dim], target.action[:n, :dim])
    names = src_names[:dim]
    state_mae = finite_mae(source.state[:n, :dim], target.state[:n, :dim])
    state_max = finite_max(source.state[:n, :dim], target.state[:n, :dim])
    action_mae = finite_mae(source.action[:n, :dim], target.action[:n, :dim])
    action_max = finite_max(source.action[:n, :dim], target.action[:n, :dim])
    timestamp_max = finite_max(source.timestamp[:n], target.timestamp[:n])
    if state_max > tolerance:
        findings.fail(
            "state_values", f"State max error {state_max:.8g} > tolerance {tolerance:g}", ep
        )
    if action_max > tolerance:
        findings.fail(
            "action_values", f"Action max error {action_max:.8g} > tolerance {tolerance:g}", ep
        )
    if timestamp_max > tolerance:
        findings.fail("timestamp", f"Source/target timestamp max error {timestamp_max:.8g}", ep)

    source_next = actual_next_state(source)[:n, :dim]
    predicted_next = target.action[:n, :dim].copy()  # absolute target semantics
    reconstruction_mae = finite_mae(source_next, predicted_next)
    reconstruction_max = finite_max(source_next, predicted_next)
    if reconstruction_max > tolerance:
        findings.fail(
            "action_semantics",
            f"Absolute action does not reconstruct q[t+1]: max error={reconstruction_max:.8g}",
            ep,
        )
    semantic_state = (
        source.raw_joint[:, :dim] if source.raw_joint is not None else source.state[:, :dim]
    )
    semantics = semantics_candidates(target.action[:n, :dim], semantic_state)
    observed_key = min(semantics, key=semantics.get)
    if observed_key != "state_t_plus_1":
        findings.fail(
            "action_semantics",
            f"Expected {SEMANTICS_LABELS['state_t_plus_1']}, best candidate is {SEMANTICS_LABELS[observed_key]}",
            ep,
        )
    elif semantics[observed_key] > tolerance:
        findings.fail(
            "action_semantics", f"Best expected semantics MAE={semantics[observed_key]:.8g}", ep
        )

    lags = lag_analysis(
        source.action[:n, :dim], target.action[:n, :dim], range(lag_min, lag_max + 1)
    )
    best_lag = min(lags, key=lags.get)
    if best_lag != expected_lag:
        findings.fail("action_lag", f"Expected lag={expected_lag}, best lag={best_lag}", ep)
    if n < action_horizon:
        findings.fail(
            "action_horizon",
            f"Episode has {n} rows, shorter than action horizon {action_horizon}",
            ep,
        )
    else:
        final_start = n - action_horizon
        final_end = final_start + action_horizon - 1
        if final_end != n - 1:
            findings.fail(
                "episode_boundary", "Last action chunk boundary calculation is inconsistent", ep
            )
        else:
            findings.passed(
                "episode_boundary",
                f"H={action_horizon}; valid starts 0..{final_start}; final chunk ends at {final_end}. "
                "The training loader drops later starts (allow_padding=False), so no padding mask is required.",
                ep,
            )
    joint_std = np.std(target.action[:n, :dim], axis=0)
    if np.all(np.abs(target.action[:n, :dim]) <= tolerance):
        findings.fail("constant_action", "All target actions are zero", ep)
    elif np.max(joint_std) <= tolerance:
        findings.warn("constant_action", "All target action joints are nearly constant", ep)

    modality = validate_modalities(
        source_dataset, source, fps_source, findings, probe_videos, report_episode=ep
    )
    target_modality = validate_modalities(
        target_dataset, target, fps_target, findings, probe_videos, report_episode=ep
    )
    modality["target"] = target_modality
    source_has_tactile = bool(modality_keys(source_dataset, "tactile"))
    target_has_tactile = bool(modality_keys(target_dataset, "tactile"))
    if source_has_tactile and not target_has_tactile:
        findings.passed(
            "tactile_conversion",
            "Target omits tactile; converter makes tactile opt-in and RGB-only UniVTAC training config does not request it.",
            ep,
        )

    episode_output = output / f"episode_{ep:06d}"
    episode_output.mkdir(parents=True, exist_ok=True)
    state_error = np.abs(source.state[:n, :dim] - target.state[:n, :dim])
    action_error = np.abs(source.action[:n, :dim] - target.action[:n, :dim])
    plots = []
    plots += plot_overlays(
        episode_output,
        names,
        source.state[:n, :dim],
        target.state[:n, :dim],
        f"Episode {ep}: state",
        "state_overlay",
        max_plot_joints,
    )
    plots += plot_overlays(
        episode_output,
        names,
        source.action[:n, :dim],
        target.action[:n, :dim],
        f"Episode {ep}: absolute action",
        "action_overlay",
        max_plot_joints,
    )
    plots += plot_overlays(
        episode_output,
        names,
        source_next,
        predicted_next,
        f"Episode {ep}: q[t+1] vs reconstruction from absolute action",
        "action_reconstruction",
        max_plot_joints,
    )
    plots.append(
        plot_heatmap(
            episode_output, state_error, names, "State absolute error", "state_error_heatmap"
        )
    )
    plots.append(
        plot_heatmap(
            episode_output, action_error, names, "Action absolute error", "action_error_heatmap"
        )
    )
    plots.append(plot_lags(episode_output, lags, expected_lag, best_lag))
    plots.append(plot_distributions(episode_output, source, target))
    plots.append(plot_timeline(episode_output, source, modality, fps_source))
    sample_plot = plot_frame_samples(episode_output, source, source_next, predicted_next)
    if sample_plot is not None:
        plots.append(sample_plot)

    worst = write_rows(csv_writer, ep, source, target, names, source_next, predicted_next)
    metrics: dict[str, Any] = {
        "state_mae": state_mae,
        "state_max": state_max,
        "action_mae": action_mae,
        "action_max": action_max,
        "reconstruction_mae": reconstruction_mae,
        "reconstruction_max": reconstruction_max,
        "timestamp_max": timestamp_max,
        "stored_order_state_mae": state_mae,
        "stored_order_action_mae": action_mae,
        "name_reordered_state_mae": reordered_state_error,
        "name_reordered_action_mae": reordered_action_error,
        "observed_semantics": SEMANTICS_LABELS[observed_key],
        "best_lag": best_lag,
        "action_std_min": float(np.min(joint_std)),
        "action_std_max": float(np.max(joint_std)),
        "joint_statistics": [
            {
                "joint": names[joint],
                "unit": JOINT_UNITS[joint] if joint < len(JOINT_UNITS) else "unknown",
                "source_state_min": float(np.min(source.state[:n, joint])),
                "source_state_max": float(np.max(source.state[:n, joint])),
                "source_state_mean": float(np.mean(source.state[:n, joint])),
                "source_state_std": float(np.std(source.state[:n, joint])),
                "target_state_min": float(np.min(target.state[:n, joint])),
                "target_state_max": float(np.max(target.state[:n, joint])),
                "target_state_mean": float(np.mean(target.state[:n, joint])),
                "target_state_std": float(np.std(target.state[:n, joint])),
                "source_action_min": float(np.min(source.action[:n, joint])),
                "source_action_max": float(np.max(source.action[:n, joint])),
                "source_action_mean": float(np.mean(source.action[:n, joint])),
                "source_action_std": float(np.std(source.action[:n, joint])),
                "target_action_min": float(np.min(target.action[:n, joint])),
                "target_action_max": float(np.max(target.action[:n, joint])),
                "target_action_mean": float(np.mean(target.action[:n, joint])),
                "target_action_std": float(np.std(target.action[:n, joint])),
            }
            for joint in range(dim)
        ],
    }
    return EpisodeResult(
        ep, source_meta.episode_index, mapping_key, n, metrics, semantics, lags, plots, modality
    ), worst


def compute_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    # Match gr00t/data/stats.py exactly: its pandas rows are explicitly cast to
    # float32 before np.mean/np.std.  Float64 accumulation differs by ~3e-4 on
    # this 152k-frame dataset and would create a false stale-stats diagnosis.
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def validate_stat_block(
    label: str,
    cached: dict[str, Any],
    actual: dict[str, np.ndarray],
    tolerance: float,
    findings: Findings,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, values in actual.items():
        if key not in cached:
            findings.fail("normalization_stats", f"{label} missing {key}")
            continue
        expected = np.asarray(cached[key], dtype=np.float64)
        error = finite_max(expected, values)
        result[f"{key}_max_error"] = error
        # Quantiles can vary slightly across numpy/arrow implementations.
        allowed = max(tolerance, 2e-5 if key in {"q01", "q99"} else tolerance)
        if expected.shape != values.shape or error > allowed:
            findings.fail(
                "normalization_stats",
                f"{label}.{key} cached shape/value mismatch; shapes={expected.shape}/{values.shape}, max error={error:.8g}",
            )
    return result


def minmax_roundtrip(
    values: np.ndarray,
    stats: dict[str, Any],
    use_percentiles: bool,
    top_count: int = 20,
) -> tuple[float, int, float, list[tuple[tuple[int, ...], float]]]:
    low_key, high_key = ("q01", "q99") if use_percentiles else ("min", "max")
    low = np.asarray(stats[low_key], dtype=np.float64)
    high = np.asarray(stats[high_key], dtype=np.float64)
    scale = high - low
    safe = np.where(np.abs(scale) > 1e-8, scale, 1.0)
    normalized = 2.0 * (values - low) / safe - 1.0
    restored = (normalized + 1.0) * 0.5 * safe + low
    clipped_count = int(np.count_nonzero((normalized < -1.0) | (normalized > 1.0)))
    clipped_restored = (np.clip(normalized, -1.0, 1.0) + 1.0) * 0.5 * safe + low
    clipped_error = np.abs(values - clipped_restored)
    count = min(top_count, clipped_error.size)
    flat = clipped_error.ravel()
    top_indices = np.argpartition(flat, -count)[-count:] if count else np.array([], dtype=int)
    top = sorted(
        [
            (
                tuple(int(i) for i in np.unravel_index(index, clipped_error.shape)),
                float(flat[index]),
            )
            for index in top_indices
            if flat[index] > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return finite_max(values, restored), clipped_count, float(np.max(clipped_error)), top


def validate_normalization(
    target: DatasetView,
    action_horizon: int,
    tolerance: float,
    findings: Findings,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stats_path": str(target.root / "meta" / "stats.json"),
        "relative_stats_path": str(target.root / "meta" / "relative_stats.json"),
        "training_mode": "q01/q99 min-max to [-1,1], clip=True, absolute chunk -> relative to state[t]",
    }
    if target.stats is None:
        findings.fail("normalization", "Target meta/stats.json is missing")
        return result
    all_state = []
    all_action = []
    relative_chunks: list[np.ndarray] = []
    row_locations: list[tuple[int, int]] = []
    relative_locations: list[tuple[int, int]] = []
    for meta in target.episodes:
        data = load_episode(target, meta, load_raw=False)
        all_state.append(data.state)
        all_action.append(data.action)
        row_locations.extend((meta.episode_index, frame) for frame in range(len(data.state)))
        usable = len(data.action) - action_horizon + 1
        if usable > 0:
            starts = np.arange(usable)[:, None]
            offsets = np.arange(action_horizon)[None, :]
            relative_chunks.append(data.action[starts + offsets] - data.state[:usable, None, :])
            relative_locations.extend((meta.episode_index, start) for start in range(usable))
    state = np.concatenate(all_state, axis=0)
    action = np.concatenate(all_action, axis=0)
    expected_total = int(target.info.get("total_frames", len(state)))
    if len(state) != expected_total:
        findings.fail(
            "normalization_subset", f"Read {len(state)} frames but info.json says {expected_total}"
        )
    for feature, values in ((STATE_COLUMN, state), (ACTION_COLUMN, action)):
        cached = target.stats.get(feature)
        if not isinstance(cached, dict):
            findings.fail("normalization_stats", f"stats.json missing {feature}")
            continue
        result[feature] = validate_stat_block(
            feature, cached, compute_stats(values), tolerance, findings
        )
        std = np.asarray(cached.get("std", []), dtype=np.float64)
        scale = np.asarray(cached.get("q99", []), dtype=np.float64) - np.asarray(
            cached.get("q01", []), dtype=np.float64
        )
        result[feature]["min_std"] = float(np.min(std)) if len(std) else None
        result[feature]["min_percentile_scale"] = float(np.min(scale)) if len(scale) else None
        if np.any(std <= 1e-8) or np.any(np.abs(scale) <= 1e-8):
            findings.fail(
                "normalization_scale", f"{feature} has zero/near-zero std or percentile scale"
            )
        elif np.any(std < 1e-5) or np.any(np.abs(scale) < 1e-5):
            findings.warn("normalization_scale", f"{feature} has an extremely small std/scale")
        roundtrip, clipped, clipped_roundtrip, clipped_top = minmax_roundtrip(
            values, cached, use_percentiles=True
        )
        result[feature]["unclipped_roundtrip_max_error"] = roundtrip
        result[feature]["training_clip_count"] = clipped
        result[feature]["training_clipped_roundtrip_max_error"] = clipped_roundtrip
        result[feature]["top_clipped_roundtrip_errors"] = [
            {
                "episode": row_locations[row][0],
                "frame": row_locations[row][1],
                "joint": JOINT_NAMES[joint] if joint < len(JOINT_NAMES) else joint,
                "error": error,
            }
            for (row, joint), error in clipped_top
        ]
        if roundtrip > tolerance:
            findings.fail(
                "normalization_roundtrip", f"{feature} round-trip max error={roundtrip:.8g}"
            )
        if clipped_roundtrip > tolerance:
            findings.fail(
                "normalization_roundtrip",
                f"{feature} actual clipped training round-trip max error={clipped_roundtrip:.8g}; "
                f"{clipped} values lie outside q01/q99 and are irreversibly clipped",
            )
    rel_stats = target.relative_stats
    if rel_stats is None or "joint" not in rel_stats:
        findings.fail(
            "normalization",
            "relative_stats.json with joint stats is required by UniVTAC relative training",
        )
        return result
    if not relative_chunks:
        findings.fail(
            "normalization_horizon", "No episode is long enough for relative action chunks"
        )
        return result
    relative = np.concatenate(relative_chunks, axis=0)
    cached_rel = rel_stats["joint"]
    result["relative_action"] = validate_stat_block(
        "relative_action.joint", cached_rel, compute_stats(relative), tolerance, findings
    )
    result["relative_action"]["mean_shape"] = list(np.asarray(cached_rel.get("mean", [])).shape)
    cached_shape = np.asarray(cached_rel.get("mean", [])).shape
    if cached_shape != (action_horizon, action.shape[1]):
        findings.fail(
            "normalization_horizon",
            f"relative_stats shape={cached_shape}, expected {(action_horizon, action.shape[1])}",
        )
    rel_roundtrip, rel_clipped, rel_clipped_roundtrip, rel_clipped_top = minmax_roundtrip(
        relative, cached_rel, use_percentiles=True
    )
    result["relative_action"]["unclipped_roundtrip_max_error"] = rel_roundtrip
    result["relative_action"]["training_clip_count"] = rel_clipped
    result["relative_action"]["training_clipped_roundtrip_max_error"] = rel_clipped_roundtrip
    result["relative_action"]["top_clipped_roundtrip_errors"] = [
        {
            "episode": relative_locations[sample][0],
            "frame": relative_locations[sample][1],
            "horizon_index": horizon,
            "joint": JOINT_NAMES[joint] if joint < len(JOINT_NAMES) else joint,
            "error": error,
        }
        for (sample, horizon, joint), error in rel_clipped_top
    ]
    std = np.asarray(cached_rel.get("std", []), dtype=np.float64)
    result["relative_action"]["min_std"] = float(np.min(std))
    if np.any(std <= 1e-8):
        findings.fail("normalization_scale", "relative action has zero/near-zero std")
    if rel_roundtrip > tolerance:
        findings.fail(
            "normalization_roundtrip", f"Relative action round-trip max error={rel_roundtrip:.8g}"
        )
    if rel_clipped_roundtrip > tolerance:
        findings.fail(
            "normalization_roundtrip",
            f"Relative action actual clipped training round-trip max error={rel_clipped_roundtrip:.8g}; "
            f"{rel_clipped} values lie outside q01/q99 and are irreversibly clipped",
        )
    if "__fingerprints__" not in target.stats or "__fingerprints__" not in rel_stats:
        findings.warn("normalization_subset", "Stats cache fingerprint metadata is missing")
    else:
        result["fingerprints_present"] = True
    return result


def aggregate_metrics(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {}
    metric_names = [
        "state_mae",
        "state_max",
        "action_mae",
        "action_max",
        "reconstruction_mae",
        "reconstruction_max",
        "timestamp_max",
    ]
    aggregate = {}
    for name in metric_names:
        values = [float(r.metrics[name]) for r in results]
        aggregate[name] = max(values) if name.endswith("max") else float(np.mean(values))
    aggregate["observed_semantics"] = max(
        (str(r.metrics["observed_semantics"]) for r in results),
        key=lambda label: sum(str(x.metrics["observed_semantics"]) == label for x in results),
    )
    available_lags = sorted(set.intersection(*(set(result.lag_mae) for result in results)))
    aggregate["best_lag"] = int(
        min(
            available_lags,
            key=lambda lag: np.mean([r.lag_mae[lag] for r in results]),
        )
    )
    return aggregate


def image_data_uri(path: Path) -> str:
    mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_html(
    path: Path,
    source: DatasetView,
    target: DatasetView,
    selected: Sequence[int],
    mapping: dict[int, tuple[EpisodeMeta, str]],
    results: Sequence[EpisodeResult],
    findings: Findings,
    aggregate: dict[str, Any],
    normalization: dict[str, Any],
    worst: Sequence[dict[str, Any]],
    command: str,
    tolerance: float,
    expected_lag: int,
) -> None:
    colors = {"PASS": "#147d3f", "WARN": "#a76700", "FAIL": "#b42318"}
    verdict = findings.verdict
    finding_rows = "".join(
        f"<tr><td class='{f.severity}'>{f.severity}</td><td>{escape(f.check)}</td>"
        f"<td>{'' if f.episode is None else f.episode}</td><td>{escape(f.message)}</td></tr>"
        for f in sorted(
            findings.items, key=lambda x: (-SEVERITY[x.severity], x.check, x.episode or -1)
        )
    )
    selected_set = set(selected)
    mapping_rows = "".join(
        f"<tr><td>{ep}</td><td>{mapping[ep][0].episode_index}</td>"
        f"<td>{'yes' if ep in selected_set else ''}</td><td>{escape(mapping[ep][1])}</td></tr>"
        for ep in sorted(mapping)
    )
    worst_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(row[key]))}</td>"
            for key in (
                "episode",
                "frame",
                "joint_name",
                "state_abs_error",
                "action_abs_error",
                "reconstruction_error",
            )
        )
        + "</tr>"
        for row in worst[:20]
    )
    episode_sections = []
    for result in results:
        images = "".join(
            f"<figure><img src='{image_data_uri(plot)}' alt='{escape(plot.name)}'><figcaption>{escape(plot.name)}</figcaption></figure>"
            for plot in result.plots
        )
        semantic_rows = "".join(
            f"<tr><td>{escape(SEMANTICS_LABELS[key])}</td><td>{value:.9g}</td></tr>"
            for key, value in result.semantics_mae.items()
        )
        episode_sections.append(
            f"<details open><summary>Episode {result.target_episode} &larr; source {result.source_episode}</summary>"
            f"<p><code>{escape(result.mapping_key)}</code></p>"
            f"<pre>{escape(json.dumps(json_safe(result.metrics), indent=2))}</pre>"
            f"<h4>Action semantics candidates</h4><table><tr><th>Candidate</th><th>MAE</th></tr>{semantic_rows}</table>"
            f"<div class='gallery'>{images}</div></details>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>UniVTAC conversion validation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#202124}}
h1{{color:{colors[verdict]}}}.badge{{background:{colors[verdict]};color:white;padding:6px 12px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}}
th{{background:#f4f5f7}}td.FAIL{{color:#b42318;font-weight:bold}}td.WARN{{color:#a76700;font-weight:bold}}td.PASS{{color:#147d3f}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.card{{border:1px solid #ddd;border-radius:8px;padding:12px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:12px}}figure{{margin:0}}img{{max-width:100%;border:1px solid #ddd}}figcaption{{text-align:center;color:#555}}
code,pre{{background:#f6f8fa;padding:3px;border-radius:4px;overflow:auto}}summary{{font-size:1.25rem;font-weight:600;margin:16px 0;cursor:pointer}}
</style></head><body>
<h1>UniVTAC &rarr; LeRobot/GR00T Conversion Validation <span class="badge">{verdict}</span></h1>
<p>Generated {escape(datetime.now(timezone.utc).isoformat())}; tolerance={tolerance:g}</p>
<div class="summary">
<div class="card"><b>Source</b><br><code>{escape(str(source.root))}</code><br>{len(source.episodes)} episodes</div>
<div class="card"><b>Converted target</b><br><code>{escape(str(target.root))}</code><br>{len(target.episodes)} episodes</div>
<div class="card"><b>Expected semantics</b><br>stored action[t] = absolute state[t+1]<br>training representation = action[t+h] - state[t]</div>
<div class="card"><b>Observed semantics</b><br>{escape(str(aggregate.get("observed_semantics", "N/A")))}</div>
<div class="card"><b>Lag</b><br>expected={expected_lag}, best={aggregate.get("best_lag", "N/A")}</div>
<div class="card"><b>Errors</b><br>state MAE/max={aggregate.get("state_mae", math.nan):.6g}/{aggregate.get("state_max", math.nan):.6g}<br>
action MAE/max={aggregate.get("action_mae", math.nan):.6g}/{aggregate.get("action_max", math.nan):.6g}<br>
reconstruction MAE/max={aggregate.get("reconstruction_mae", math.nan):.6g}/{aggregate.get("reconstruction_max", math.nan):.6g}</div>
<div class="card"><b>Units</b><br>arm joints: radians; finger: meters<br>converter performs no unit transform; units are not encoded in LeRobot metadata</div>
<div class="card"><b>Normalization</b><br>q01/q99 min-max; relative joint actions; H={escape(str(normalization.get("relative_action", {}).get("mean_shape", 16)))}</div>
</div>
<h2>Episode mapping ({len(mapping)} unique matches)</h2><details><summary>Show complete mapping</summary>
<table><tr><th>Target</th><th>Source</th><th>Sampled</th><th>Unique key</th></tr>{mapping_rows}</table></details>
<h2>Findings</h2><table><tr><th>Status</th><th>Check</th><th>Episode</th><th>Message</th></tr>{finding_rows}</table>
<h2>Normalization details</h2><pre>{escape(json.dumps(json_safe(normalization), indent=2))}</pre>
<h2>Top 20 failing frame/joints</h2><table><tr><th>Episode</th><th>Frame</th><th>Joint</th><th>State error</th><th>Action error</th><th>Reconstruction error</th></tr>{worst_rows}</table>
<h2>Episode plots</h2>{"".join(episode_sections)}
<h2>Reproduction</h2><pre>{escape(command)}</pre>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--lerobot-dataset", type=Path, required=True)
    parser.add_argument(
        "--episodes", type=int, nargs="+", help="Converted target episode indices (default: 0 1 2)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conversion_validation"))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--full-scan", action="store_true", help="Compare every mapped episode instead of a sample"
    )
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--expected-lag", type=int, default=0)
    parser.add_argument("--lag-min", type=int, default=-5)
    parser.add_argument("--lag-max", type=int, default=5)
    parser.add_argument("--max-joints-per-plot", type=int, default=8)
    parser.add_argument(
        "--skip-video-probe", action="store_true", help="Skip ffprobe frame-count checks"
    )
    parser.add_argument(
        "--skip-normalization-scan",
        action="store_true",
        help="Skip all-episode stats recomputation",
    )
    args = parser.parse_args(argv)
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.action_horizon <= 0:
        parser.error("--action-horizon must be positive")
    if args.lag_min > args.lag_max:
        parser.error("--lag-min must be <= --lag-max")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = load_dataset(args.source_dataset)
    target = load_dataset(args.lerobot_dataset)
    if target.kind != "lerobot":
        raise ValueError(f"--lerobot-dataset is not LeRobot format: {target.root}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    findings = Findings()
    if len(source.episodes) != len(target.episodes):
        findings.fail(
            "episode_count",
            f"Source has {len(source.episodes)} episodes; target has {len(target.episodes)} episodes",
        )
    else:
        findings.passed("episode_count", f"Both datasets have {len(source.episodes)} episodes")
    mapping = build_mapping(source, target, findings)
    target_by_index = {meta.episode_index: meta for meta in target.episodes}
    if args.full_scan:
        selected = sorted(mapping)
    else:
        selected = args.episodes if args.episodes is not None else [0, 1, 2]
        if len(selected) < 3:
            findings.warn(
                "episode_sampling", "Fewer than the recommended three episodes were selected"
            )
    if len(selected) != len(set(selected)):
        raise ValueError("--episodes contains duplicate indices")
    for episode in selected:
        if episode not in target_by_index:
            findings.fail("episode_selection", f"Target episode {episode} does not exist")
        elif episode not in mapping:
            findings.fail(
                "episode_mapping",
                f"Target episode {episode} has no unambiguous source mapping",
                episode,
            )

    source_state_names = names_for(source, STATE_COLUMN)
    target_state_names = names_for(target, STATE_COLUMN)
    source_action_names = names_for(source, ACTION_COLUMN)
    target_action_names = names_for(target, ACTION_COLUMN)
    if source_state_names and source_action_names and source_state_names != source_action_names:
        findings.fail("joint_order", "Source state and action joint orders differ")
    if target_state_names and target_action_names and target_state_names != target_action_names:
        findings.fail("joint_order", "Target state and action joint orders differ")
    if target_state_names != JOINT_NAMES:
        findings.fail(
            "joint_order",
            f"Target joint order differs from converter contract: {target_state_names}",
        )
    else:
        findings.passed(
            "joint_order", f"Joint order matches converter contract: {target_state_names}"
        )
    findings.warn(
        "units",
        "Converter performs no numeric unit conversion. Values are Franka joint positions (arm radians, finger meters), "
        "but info.json has no explicit per-joint unit field.",
    )

    csv_path = output / "frame_joint_comparison.csv"
    fields = [
        "episode",
        "source_episode",
        "frame",
        "timestamp",
        "joint_name",
        "source_state",
        "lerobot_state",
        "state_abs_error",
        "source_action",
        "lerobot_action",
        "action_abs_error",
        "source_next_state",
        "reconstructed_next_state",
        "reconstruction_error",
    ]
    results: list[EpisodeResult] = []
    worst: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ep in selected:
            if ep not in mapping or ep not in target_by_index:
                continue
            source_meta, key = mapping[ep]
            try:
                result, episode_rows = compare_one_episode(
                    source,
                    target,
                    source_meta,
                    target_by_index[ep],
                    key,
                    output,
                    writer,
                    args.tolerance,
                    args.expected_lag,
                    args.lag_min,
                    args.lag_max,
                    args.action_horizon,
                    args.max_joints_per_plot,
                    findings,
                    not args.skip_video_probe,
                )
                results.append(result)
                worst.extend(episode_rows)
            except Exception as exc:
                findings.fail(
                    "episode_validation",
                    f"Unhandled validation error: {type(exc).__name__}: {exc}",
                    ep,
                )
    worst = [
        row
        for row in worst
        if max(row["state_abs_error"], row["action_abs_error"], row["reconstruction_error"])
        > args.tolerance
    ]
    worst.sort(
        key=lambda row: max(
            row["state_abs_error"], row["action_abs_error"], row["reconstruction_error"]
        ),
        reverse=True,
    )
    normalization: dict[str, Any]
    if args.skip_normalization_scan:
        normalization = {"skipped": True}
        findings.warn("normalization", "Normalization scan was skipped by CLI option")
    else:
        normalization = validate_normalization(
            target, args.action_horizon, args.tolerance, findings
        )
    aggregate = aggregate_metrics(results)
    report_data = {
        "verdict": findings.verdict,
        "source": str(source.root),
        "target": str(target.root),
        "source_episode_count": len(source.episodes),
        "target_episode_count": len(target.episodes),
        "selected_episodes": selected,
        "expected_semantics": "stored action[t] = absolute joint position state[t+1]; training uses action[t+h]-state[t]",
        "observed_semantics": aggregate.get("observed_semantics"),
        "expected_lag": args.expected_lag,
        "best_lag": aggregate.get("best_lag"),
        "joint_names": target_state_names,
        "joint_units": dict(zip(target_state_names, JOINT_UNITS)),
        "aggregate": aggregate,
        "normalization": normalization,
        "mapping": {
            str(ep): {"source_episode": meta.episode_index, "key": key}
            for ep, (meta, key) in mapping.items()
        },
        "episodes": [json_safe(result.__dict__) for result in results],
        "findings": [finding.__dict__ for finding in findings.items],
        "top_20_errors": worst[:20],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report_data), indent=2), encoding="utf-8"
    )
    command = " ".join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ]
    )
    write_html(
        output / "report.html",
        source,
        target,
        selected,
        mapping,
        results,
        findings,
        aggregate,
        normalization,
        worst,
        command,
        args.tolerance,
        args.expected_lag,
    )
    print(f"Verdict: {findings.verdict}")
    print(f"Source: {source.root}")
    print(f"Converted target: {target.root}")
    print(f"Episodes: {selected}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {output / 'report.html'}")
    print(f"JSON: {output / 'report.json'}")
    return 0 if findings.verdict == "PASS" else (1 if findings.verdict == "WARN" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
