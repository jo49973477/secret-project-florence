# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, dependency-free process-memory diagnostics for model loading."""

from contextlib import contextmanager
import logging
import os


def model_load_memory_kib() -> dict[str, int]:
    """Read current RSS and process high-water RSS without extra dependencies."""
    values = {}
    try:
        with open("/proc/self/status") as status:
            for line in status:
                key, separator, value = line.partition(":")
                if separator and key in {"VmRSS", "VmHWM"}:
                    values[{"VmRSS": "rss_kib", "VmHWM": "peak_kib"}[key]] = int(value.split()[0])
    except OSError:
        values = {}
    return values


def log_model_load_memory(label: str) -> None:
    """Log Linux RSS/high-water memory when GROOT_DEBUG_MODEL_LOAD_MEMORY is enabled."""
    if model_load_io_enabled():
        _emit_model_load_event({**model_load_snapshot(), "phase": label, "event": "point"})
    if os.environ.get("GROOT_DEBUG_MODEL_LOAD_MEMORY", "").lower() not in {"1", "true", "yes"}:
        return
    values = model_load_memory_kib()
    logger = logging.getLogger("gr00t.model_load")
    logger.setLevel(logging.INFO)
    logger.info(
        "Model-load memory [%s] rank=%s pid=%d RSS=%s KiB peak=%s KiB",
        label,
        os.environ.get("RANK", "0"),
        os.getpid(),
        values.get("rss_kib", "unavailable"),
        values.get("peak_kib", "unavailable"),
    )


def model_load_io_enabled() -> bool:
    return os.environ.get("GROOT_DEBUG_MODEL_LOAD_IO", "").lower() in {"1", "true", "yes"}


def model_load_snapshot() -> dict:
    """Small Linux process counters; read_bytes measures physical reads, not cache hits."""
    from datetime import datetime, timezone
    import time

    values = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
        "rank": os.environ.get("RANK", "0"),
        "pid": os.getpid(),
        **model_load_memory_kib(),
    }
    try:
        with open("/proc/self/io") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in {"read_bytes", "rchar"}:
                    values[key] = int(value)
    except OSError:
        pass
    return values


def _emit_model_load_event(event: dict) -> None:
    import json
    from pathlib import Path

    # Rank>0's root logger is WARNING in experiment.run. Use an explicit logger
    # level so opted-in diagnostics from those ranks remain visible.
    logger = logging.getLogger("gr00t.model_load")
    logger.setLevel(logging.INFO)
    payload = json.dumps(event, sort_keys=True)
    logger.info("Model-load trace %s", payload)
    directory = os.environ.get("GROOT_MODEL_LOAD_TRACE_DIR")
    if directory:
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"rank-{event['rank']}-pid-{event['pid']}.jsonl").open("a") as stream:
                stream.write(payload + "\n")
        except OSError as exc:
            logger.warning("Cannot write model-load trace: %s", exc)


@contextmanager
def model_load_phase(label: str, *, checkpoint=None, **details):
    """Time a stage including exceptions, without keeping tensors or changing loading."""
    if not model_load_io_enabled():
        yield
        return
    from pathlib import Path

    if checkpoint is not None:
        details["checkpoint"] = str(checkpoint)
        try:
            path = Path(checkpoint)
            details["file_size_bytes"] = path.stat().st_size if path.is_file() else None
        except (OSError, TypeError):
            details["file_size_bytes"] = None
    start = model_load_snapshot()
    _emit_model_load_event({**details, **start, "phase": label, "event": "enter"})
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        end = model_load_snapshot()
        delta = {
            f"{key}_delta": end[key] - start[key]
            for key in ("read_bytes", "rchar")
            if key in start and key in end
        }
        _emit_model_load_event(
            {
                **details,
                **end,
                **delta,
                "phase": label,
                "event": "exit",
                "elapsed_seconds": end["monotonic"] - start["monotonic"],
                "error": error,
            }
        )


@contextmanager
def trace_checkpoint_loading():
    """Opt-in scoped observations of existing loaders; never replace their implementation.

    TF load_shard_file includes materialization/page faults, unlike its meta-only
    load_state_dict header pass. Nested records intentionally distinguish these.
    Restore every function on exit, including when a loader raises.
    """
    if not model_load_io_enabled():
        yield
        return
    from functools import wraps

    import safetensors.torch
    import torch
    import transformers.modeling_utils as modeling_utils

    targets = [
        (torch, "load", "torch.load"),
        (safetensors.torch, "load_file", "safetensors.load_file"),
        (modeling_utils, "load_state_dict", "transformers.load_state_dict"),
        (modeling_utils, "load_shard_file", "transformers.load_shard_file"),
    ]
    originals = []
    try:
        for module, name, label in targets:
            original = getattr(module, name)
            originals.append((module, name, original))

            @wraps(original)
            def observed(*args, _fn=original, _label=label, **kwargs):
                path = (
                    args[0]
                    if args
                    else kwargs.get("f", kwargs.get("checkpoint_file", kwargs.get("filename")))
                )
                if _label == "transformers.load_shard_file":
                    path = path[0]
                if not isinstance(path, (str, os.PathLike)):
                    path = getattr(path, "name", None)
                with model_load_phase(_label, checkpoint=path):
                    return _fn(*args, **kwargs)

            setattr(module, name, observed)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
