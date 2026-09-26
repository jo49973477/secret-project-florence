# SPDX-License-Identifier: Apache-2.0
"""Low overhead host memory diagnostics for checkpoint operations."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import threading


logger = logging.getLogger(__name__)


def _cgroup_files() -> tuple[Path | None, Path | None, Path | None]:
    """Return memory.current, memory.peak and memory.events for this process."""
    try:
        entry = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        relative = entry.split("::", 1)[1].lstrip("/")
        mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
        for line in mountinfo:
            before, after = line.split(" - ", 1)
            fields = before.split()
            if after.split()[0] != "cgroup2":
                continue
            root = Path(fields[3])
            mount = Path(fields[4])
            rel = Path(relative)
            if root != Path("/"):
                rel = rel.relative_to(root)
            directory = mount / rel
            current, peak, events = (directory / name for name in
                                     ("memory.current", "memory.peak", "memory.events"))
            if current.exists():
                return current, peak, events
    except (OSError, StopIteration, ValueError):
        pass
    return None, None, None


def _rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _snapshot() -> dict[str, object]:
    current, peak, events = _cgroup_files()
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    event_values = {}
    if events is not None:
        try:
            event_values = {key: int(value) for key, value in
                            (line.split() for line in events.read_text().splitlines())}
        except (OSError, ValueError):
            pass
    current_value = None
    peak_value = None
    try:
        current_value = int(current.read_text()) if current else None
        peak_value = int(peak.read_text()) if peak and peak.exists() else None
    except (OSError, ValueError):
        pass
    return {
        "rank": os.environ.get("RANK", "0"),
        "rss_bytes": _rss_bytes(),
        "system_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_current_bytes": current_value,
        "cgroup_peak_bytes": peak_value,
        "memory_events": event_values,
    }


@contextmanager
def monitor_checkpoint_memory(operation: str, interval_seconds: float = 0.2):
    """Log per-rank RSS and cgroup/system memory before, peak, and after an operation."""
    samples = [_snapshot()]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(interval_seconds):
            try:
                snapshot = _snapshot()
                samples.append(snapshot)
                logger.info(
                    "checkpoint-memory phase=%s point=during metrics=%s",
                    operation,
                    snapshot,
                )
            except Exception:
                logger.exception("Checkpoint memory sampling failed")

    worker = threading.Thread(target=sample, name="checkpoint-memory", daemon=True)
    logger.info("checkpoint-memory phase=%s point=before metrics=%s", operation, samples[0])
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2)
        try:
            samples.append(_snapshot())
        except Exception:
            logger.exception("Checkpoint memory final sampling failed")
        peak_current = max(
            (sample.get("cgroup_current_bytes") or 0 for sample in samples), default=0
        )
        peak_rss = max((sample.get("rss_bytes") or 0 for sample in samples), default=0)
        peak_available = min(
            (sample.get("system_available_bytes") for sample in samples
             if sample.get("system_available_bytes") is not None), default=None
        )
        before_events = samples[0].get("memory_events", {})
        after_events = samples[-1].get("memory_events", {})
        event_delta = {
            key: after_events.get(key, 0) - before_events.get(key, 0)
            for key in set(before_events) | set(after_events)
        }
        logger.info(
            "checkpoint-memory phase=%s point=after samples=%d baseline_rss_bytes=%s "
            "sampled_peak_rss_bytes=%d sampled_peak_rss_delta_bytes=%s "
            "sampled_peak_cgroup_current_bytes=%d minimum_system_available_bytes=%s "
            "memory_events_delta=%s metrics=%s",
            operation,
            len(samples),
            samples[0].get("rss_bytes"),
            peak_rss,
            max(0, peak_rss - samples[0]["rss_bytes"])
            if samples[0].get("rss_bytes") is not None
            else None,
            peak_current,
            peak_available,
            event_delta,
            samples[-1],
        )
