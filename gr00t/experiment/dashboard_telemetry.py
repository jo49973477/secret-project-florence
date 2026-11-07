"""Rank-zero JSONL telemetry and a single parent-owned Rich dashboard.

This is display telemetry only: no counters, checkpoint writes, tracker database
access, notifications, or process management belong here.
"""

from __future__ import annotations

import json
import logging
import math
from numbers import Real
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from transformers import TrainerCallback

from gr00t.experiment.training_dashboard import TrainingDashboardCallback


logger = logging.getLogger(__name__)
TELEMETRY_ENV = "GR00T_DASHBOARD_TELEMETRY"


def scalars(values):
    """Keep existing CPU measurements without calling tensor.item()."""
    result = {}
    for name, value in (values or {}).items():
        if not isinstance(name, str):
            continue
        if value is None:
            result[name] = None
        elif isinstance(value, Real) and not isinstance(value, bool):
            try:
                if math.isfinite(value):
                    result[name] = int(value) if isinstance(value, int) else float(value)
            except (ValueError, OverflowError):
                pass
    return result


class DashboardTelemetryCallback(TrainerCallback):
    """Replace worker rendering with bounded, best-effort rank-zero observations."""

    def __init__(self, path, refresh_seconds=1.0):
        self.path = Path(path)
        self.refresh_seconds = refresh_seconds
        self._last_step = 0.0
        self._warned = False

    def _send(self, event, args, state, control=None, **payload):
        if (
            not getattr(state, "is_world_process_zero", False)
            or getattr(args, "process_index", 0) != 0
            or os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) != "0"
        ):
            return
        try:
            record = {
                "event": event,
                "rank": 0,
                "step": state.global_step,
                "total": state.max_steps,
                "saving": bool(getattr(control, "should_save", False)),
                **payload,
            }
            # One append per observation. The parent tolerates incomplete final lines.
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError):
            if not self._warned:
                logger.warning("Dashboard telemetry unavailable; training will continue")
                self._warned = True

    def on_train_begin(self, args, state, control, **kwargs):
        deepspeed = getattr(args, "deepspeed", None)
        stage = (
            deepspeed.get("zero_optimization", {}).get("stage")
            if isinstance(deepspeed, dict)
            else None
        )
        self._send(
            "begin",
            args,
            state,
            control,
            arguments={
                "process_index": 0,
                "world_size": getattr(args, "world_size", 1),
                "bf16": getattr(args, "bf16", False),
                "fp16": getattr(args, "fp16", False),
                "max_grad_norm": getattr(args, "max_grad_norm", None),
                "deepspeed": {"zero_optimization": {"stage": stage}} if stage else None,
            },
        )

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        if now - self._last_step >= self.refresh_seconds or getattr(control, "should_save", False):
            self._last_step = now
            self._send("step", args, state, control)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._send("log", args, state, control, metrics=scalars(logs))

    def on_save(self, args, state, control, **kwargs):
        self._send("save", args, state, control)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._send("log", args, state, control, metrics=scalars(metrics))

    def on_train_end(self, args, state, control, **kwargs):
        self._send("finalizing", args, state, control)


class ParentDashboard:
    """Replay display observations through WAH's existing Rich formatting."""

    def __init__(self, cfg, config, output, env):
        self.env = env
        self.enabled = (
            cfg.trainctl.enabled
            and sys.stdout.isatty()
            and config.training.enable_training_dashboard is not False
        )
        self.renderer = None
        self.result = None
        self.stop = threading.Event()
        self.thread = None
        self.path = Path(output) / "logs" / f"dashboard-{uuid.uuid4().hex}.jsonl"
        self.cfg, self.config = cfg, config
        self._warned = False

    def __enter__(self):
        if not self.enabled:
            self.env.pop(TELEMETRY_ENV, None)
            return self
        self.path.touch()
        self.env[TELEMETRY_ENV] = str(self.path.resolve())
        self.renderer = TrainingDashboardCallback(
            True,
            self.config.training.dashboard_refresh_seconds,
            diagnostics_enabled=self.config.training.enable_multimodal_diagnostics,
            experiment_name=f"{self.cfg.name}-{self.cfg.run_id}",
            screen=False,
        )
        try:
            args = SimpleNamespace(process_index=0, world_size=self.config.training.num_gpus)
            state = SimpleNamespace(global_step=0, max_steps=self.config.training.max_steps)
            self.renderer.on_train_begin(args, state, None)
            self.renderer._status = "Starting workers"
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            self.renderer._stop_live()
            raise
        return self

    def apply(self, record):
        if record.get("rank") != 0:
            return
        if type(record.get("step")) is not int or type(record.get("total")) is not int:
            return
        state = SimpleNamespace(global_step=record["step"], max_steps=record["total"])
        args = SimpleNamespace(process_index=0)
        control = SimpleNamespace(should_save=record.get("saving", False))
        event = record.get("event")
        if event == "begin":
            self.renderer.on_train_begin(SimpleNamespace(**record["arguments"]), state, control)
        elif event == "log":
            self.renderer.on_log(args, state, control, logs=record.get("metrics"))
            self.renderer.on_step_end(args, state, control)
        elif event == "step":
            self.renderer.on_step_begin(args, state, control)
            self.renderer.on_step_end(args, state, control)
        elif event == "save":
            self.renderer.on_save(args, state, control)
        elif event == "finalizing":
            # HF train_end precedes explicit final saving and cleanup.
            self.renderer._status = "Finalizing (saving and cleanup)"
            self.renderer._refresh(force=True)

    def _run(self):
        pending = ""
        with self.path.open(encoding="utf-8") as stream:
            while True:
                stopping = self.stop.is_set()
                pending += stream.read()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    try:
                        self.apply(json.loads(line))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        if not self._warned:
                            logger.warning("Ignored malformed dashboard telemetry")
                            self._warned = True
                if stopping:
                    break
                self.stop.wait(0.05)

    def __exit__(self, exc_type, exc, tb):
        if self.renderer is not None:
            try:
                self.stop.set()
                if self.thread is not None:
                    self.thread.join()
                self.renderer._status = (
                    "Completed" if exc_type is None and self.result == 0 else "Failed / interrupted"
                )
                self.renderer._refresh(force=True)
            finally:
                self.renderer._stop_live()
        return False
