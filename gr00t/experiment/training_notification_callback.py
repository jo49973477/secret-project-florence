# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral training notification callback and message formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
import socket
import time
from typing import Any, Protocol

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint


logger = logging.getLogger(__name__)

_MAX_EXCEPTION_MESSAGE_LENGTH = 300


class NotificationSender(Protocol):
    """Transport interface required by :class:`TrainingNotificationCallback`."""

    def send(self, message: str) -> bool: ...


@dataclass(frozen=True)
class TrainingNotificationContext:
    """Non-secret metadata used to format training notifications."""

    experiment_name: str
    output_dir: str
    model_path: str | None = None
    dataset_path: str | None = None
    num_gpus: int | None = None
    global_batch_size: int | None = None
    learning_rate: float | None = None
    deepspeed_stage: int | None = None
    hostname: str = ""

    def __post_init__(self) -> None:
        if not self.hostname:
            object.__setattr__(self, "hostname", socket.gethostname())


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_metric(log_history: list[dict[str, Any]], *keys: str) -> float | None:
    for entry in reversed(log_history):
        for key in keys:
            value = _finite_number(entry.get(key))
            if value is not None:
                return value
    return None


def _best_loss(log_history: list[dict[str, Any]]) -> float | None:
    losses = [
        value for entry in log_history if (value := _finite_number(entry.get("loss"))) is not None
    ]
    return min(losses) if losses else None


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _sanitize_exception(exc: BaseException) -> str:
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", str(exc))
    message = re.sub(r"\s+", " ", message).strip() or "No exception message"
    if len(message) > _MAX_EXCEPTION_MESSAGE_LENGTH:
        message = message[: _MAX_EXCEPTION_MESSAGE_LENGTH - 1].rstrip() + "…"
    return f"{type(exc).__name__}: {message}"


class TrainingNotificationCallback(TrainerCallback):
    """Send rank-zero-only training events through one notification transport."""

    def __init__(
        self,
        notifier: NotificationSender,
        context: TrainingNotificationContext,
        *,
        notify_start: bool = True,
        notify_save: bool = True,
        notify_finish: bool = True,
        notify_error: bool = True,
        defer_finish_until_final_save: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.notifier = notifier
        self.context = context
        self.notify_start_enabled = notify_start
        self.notify_save_enabled = notify_save
        self.notify_finish_enabled = notify_finish
        self.notify_error_enabled = notify_error
        self.defer_finish_until_final_save = defer_finish_until_final_save
        self._monotonic = monotonic
        self._created_at = monotonic()
        self._started_at: float | None = None
        self._start_sent = False
        self._finish_sent = False
        self._failure_sent = False
        self._saved_steps: set[int] = set()

    @staticmethod
    def _is_world_process_zero(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", False))

    def _send(self, message: str) -> None:
        try:
            self.notifier.send(message)
        except Exception as exc:
            # Third-party/custom transports must not be able to interrupt training.
            logger.warning(
                "Training notification callback failed (%s); training will continue.",
                type(exc).__name__,
            )

    def _elapsed(self) -> str:
        start = self._started_at if self._started_at is not None else self._created_at
        return _format_duration(self._monotonic() - start)

    @staticmethod
    def _max_steps(args: Any, state: Any) -> int | None:
        state_max = getattr(state, "max_steps", None)
        args_max = getattr(args, "max_steps", None)
        value = state_max if state_max not in (None, 0) else args_max
        try:
            return int(value) if value is not None and int(value) > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _step_lines(args: Any, state: Any, *, label: str = "Step") -> list[str]:
        step = int(getattr(state, "global_step", 0))
        max_steps = TrainingNotificationCallback._max_steps(args, state)
        if max_steps is None:
            return [f"{label}: {step}"]
        progress = min(100.0, max(0.0, step / max_steps * 100.0))
        return [f"{label}: {step} / {max_steps}", f"Progress: {progress:.1f}%"]

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_world_process_zero(state) or self._start_sent:
            return
        self._started_at = self._monotonic()
        self._start_sent = True
        if not self.notify_start_enabled:
            return

        lines = [
            "🚀 GR00T Training Started",
            "",
            f"Experiment: {self.context.experiment_name}",
            f"Host: {self.context.hostname}",
            f"Output: {self.context.output_dir}",
        ]
        optional_fields = (
            ("Model", self.context.model_path),
            ("Dataset", self.context.dataset_path),
            ("GPUs", self.context.num_gpus),
        )
        for label, value in optional_fields:
            if value not in (None, ""):
                lines.append(f"{label}: {value}")

        lines.append("")
        max_steps = self._max_steps(args, state)
        if max_steps is not None:
            lines.append(f"Max steps: {max_steps}")
        if self.context.global_batch_size is not None:
            lines.append(f"Global batch size: {self.context.global_batch_size}")
        if self.context.learning_rate is not None:
            lines.append(f"Learning rate: {self.context.learning_rate:g}")
        if self.context.deepspeed_stage is not None:
            lines.append(f"DeepSpeed: Stage {self.context.deepspeed_stage}")
        self._send("\n".join(lines))

    def on_save(self, args, state, control, **kwargs):
        if not self._is_world_process_zero(state) or not self.notify_save_enabled:
            return
        step = int(getattr(state, "global_step", 0))
        if step in self._saved_steps:
            return
        self._saved_steps.add(step)

        checkpoint = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{step}"
        lines = [
            "💾 GR00T Checkpoint Saved",
            "",
            f"Experiment: {self.context.experiment_name}",
            "",
            *self._step_lines(args, state),
        ]
        history = getattr(state, "log_history", []) or []
        loss = _latest_metric(history, "loss")
        learning_rate = _latest_metric(history, "learning_rate")
        if loss is not None or learning_rate is not None:
            lines.append("")
        if loss is not None:
            lines.append(f"Loss: {loss:.4f}")
        if learning_rate is not None:
            lines.append(f"Learning rate: {learning_rate:.4g}")
        lines.extend(["", "Checkpoint:", str(checkpoint), "", f"Elapsed: {self._elapsed()}"])
        self._send("\n".join(lines))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.defer_finish_until_final_save:
            self.notify_finish(args=args, state=state)

    def notify_finish(self, *, args: Any, state: Any) -> None:
        """Send success after the caller's final model save has completed."""
        if (
            not self._is_world_process_zero(state)
            or not self.notify_finish_enabled
            or self._finish_sent
            or self._failure_sent
        ):
            return
        self._finish_sent = True

        history = getattr(state, "log_history", []) or []
        final_loss = _latest_metric(history, "loss", "train_loss")
        best_loss = _best_loss(history)
        lines = [
            "✅ GR00T Training Finished",
            "",
            f"Experiment: {self.context.experiment_name}",
            "",
            *self._step_lines(args, state, label="Final step"),
        ]
        if final_loss is not None:
            lines.append(f"Final loss: {final_loss:.4f}")
        if best_loss is not None:
            lines.append(f"Best observed loss: {best_loss:.4f}")
        lines.extend(["", f"Duration: {self._elapsed()}", "", "Output:", self.context.output_dir])
        self._send("\n".join(lines))

    def notify_failure(self, exc: BaseException, *, args: Any, state: Any) -> None:
        """Send a short failure summary; the caller remains responsible for re-raising."""
        if (
            not self._is_world_process_zero(state)
            or not self.notify_error_enabled
            or self._failure_sent
            or self._finish_sent
        ):
            return
        self._failure_sent = True

        output_dir = str(getattr(args, "output_dir", self.context.output_dir))
        try:
            last_checkpoint = get_last_checkpoint(output_dir) or "none"
        except Exception:
            last_checkpoint = "none"
        lines = [
            "🚨 GR00T Training Failed",
            "",
            f"Experiment: {self.context.experiment_name}",
            f"Host: {self.context.hostname}",
            "",
            *self._step_lines(args, state),
            "",
            f"Last checkpoint: {last_checkpoint}",
            "",
            "Exception:",
            _sanitize_exception(exc),
            "",
            f"Duration: {self._elapsed()}",
        ]
        self._send("\n".join(lines))
