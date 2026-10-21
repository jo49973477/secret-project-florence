"""Rank-zero Trainer rendering, with a coordinated Rich display or clean text."""

from __future__ import annotations

import atexit
import logging
import math
import sys
import time
from typing import Any

from transformers.trainer_callback import ProgressCallback, TrainerCallback


class QuietProgressCallback(ProgressCallback):
    """Keep HF's tqdm train/eval bars while removing dictionary printing at source."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control

    def close(self):
        for name in ("training_bar", "prediction_bar"):
            bar = getattr(self, name, None)
            if bar is not None:
                bar.close()
                setattr(self, name, None)


class TrainingDashboardCallback(TrainerCallback):
    """Display already logged CPU scalars; never touch model tensors."""

    TRAINING = {
        "loss": "Loss",
        "grad_norm": "Gradient Norm",
        "learning_rate": "Learning Rate",
        "predicted_velocity_std": "Pred. Velocity Std",
        "target_velocity_std": "Target Velocity Std",
        "zero_prediction_mse": "Zero Prediction MSE",
        "valid_action_ratio": "Valid Action Ratio",
    }
    DIAGNOSTICS = {
        "sidecar_action_gate_mean_abs": "Sidecar Action Gate",
        "sidecar_point_gate_mean_abs": "Sidecar Point Gate",
        "sidecar_grad_norm": "Sidecar Gradient",
        "contact_prediction_loss": "Contact Prediction Loss",
        "observed_contact_variance": "Observed Contact Variance",
        "predicted_contact_variance": "Predicted Contact Variance",
        "contact_surprise_norm": "Contact Surprise Norm",
        "contact_gate_mean": "Contact Gate",
        "contact_predictor_grad_norm": "Contact Predictor Gradient",
        "point_gate_mean_abs": "Point Gate Mean |·|",
        "tactile_gate_mean_abs": "Tactile Gate Mean |·|",
        "point_gate_grad_norm": "Point Gate Grad",
        "tactile_gate_grad_norm": "Tactile Gate Grad",
        "point_encoder_grad_norm": "Point Encoder Grad",
        "tactile_encoder_grad_norm": "Tactile Encoder Grad",
        "point_cross_attention_grad_norm": "Point Attention Grad",
        "tactile_cross_attention_grad_norm": "Tactile Attention Grad",
        "point_residual_ratio_mean": "Point Residual Ratio",
        "tactile_residual_ratio_mean": "Tactile Residual Ratio",
    }

    def __init__(
        self,
        enabled: bool | None = None,
        refresh_seconds: float = 1.0,
        diagnostics_enabled: bool = False,
    ):
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be finite and positive")
        self.enabled = sys.stdout.isatty() and enabled is not False
        self.diagnostics_enabled = diagnostics_enabled
        self.refresh_seconds = refresh_seconds
        self._live = None
        self._progress = None
        self._task = None
        self._eval_task = None
        self._metrics: dict[str, Any] = {}
        self._started = 0.0
        self._last_render = 0.0
        self._initial_step = 0
        self._training = False
        self._redirected_handlers = []

    @staticmethod
    def _is_main(args: Any) -> bool:
        return getattr(args, "process_index", 0) == 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._training = True
        self._started = time.monotonic()
        self._initial_step = state.global_step
        self._metrics.clear()
        if not self._is_main(args) or not self.enabled:
            return control
        self._start_live()
        total = max(int(getattr(state, "max_steps", 0) or 0), 1)
        self._task = self._progress.add_task(
            "Training", total=total, completed=state.global_step, speed="0.00 step/s"
        )
        self._refresh(force=True)
        return control

    def _start_live(self):
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.table import Table

        self._Table, self._Panel, self._Group = Table, Panel, Group
        self._console = Console(file=sys.stdout)
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[speed]}", style="dim"),
            console=self._console,
            auto_refresh=False,
        )
        self._live = Live(
            self._progress,
            console=self._console,
            refresh_per_second=1 / self.refresh_seconds,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        old_streams = (sys.stdout, sys.stderr)
        self._live.start()
        # Existing StreamHandlers keep a reference to the old stream; simply
        # replacing sys.stderr would let them write through the live display.
        loggers = [logging.getLogger()] + [
            logger
            for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ]
        seen = set()
        for logger in loggers:
            for handler in logger.handlers:
                if id(handler) in seen or not isinstance(handler, logging.StreamHandler):
                    continue
                seen.add(id(handler))
                for old, redirected in zip(old_streams, (sys.stdout, sys.stderr)):
                    if handler.stream is old:
                        handler.setStream(redirected)
                        self._redirected_handlers.append((handler, old, redirected))
                        break
        atexit.register(self._stop_live)

    def _stop_live(self):
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                for handler, original, redirected in self._redirected_handlers:
                    if handler.stream is redirected:
                        handler.setStream(original)
                self._redirected_handlers.clear()
                self._live = None
                self._progress = None
                self._task = self._eval_task = None
                atexit.unregister(self._stop_live)

    @staticmethod
    def _format(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return "N/A"
        return f"{value:.6g}"

    def _panel(self, title, labels):
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(justify="right")
        for key, label in labels.items():
            table.add_row(label, self._format(self._metrics.get(key)))
        return self._Panel(table, title=title)

    def _render(self):
        panels = [self._progress]
        if self._training:
            panels.append(self._panel("Training Metrics", self.TRAINING))
            if self.diagnostics_enabled:
                panels.append(self._panel("Multimodal Diagnostics", self.DIAGNOSTICS))
        evaluation = {
            key: key.removeprefix("eval_").replace("_", " ").title()
            for key in self._metrics
            if key.startswith("eval_")
        }
        if evaluation:
            panels.append(self._panel("Evaluation Metrics", evaluation))
        return self._Group(*panels)

    def _refresh(self, force=False):
        if self._live is not None and (
            force or time.monotonic() - self._last_render >= self.refresh_seconds
        ):
            self._live.update(self._render(), refresh=True)
            self._last_render = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self._progress is not None and self._task is not None:
            elapsed = max(time.monotonic() - self._started, 0.001)
            self._progress.update(
                self._task,
                completed=state.global_step,
                speed=f"{(state.global_step - self._initial_step) / elapsed:.2f} step/s",
            )
            self._refresh()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if not self._is_main(args):
            return control
        self._metrics.update(
            {k: v for k, v in logs.items() if isinstance(v, (float, int)) or v is None}
        )
        if self._live is not None:
            self._refresh()
        else:
            # Trainer already decides logging cadence. Keep eval/final/diagnostic
            # events even when several occur at the same global step.
            clean = " ".join(
                f"{k}={self._format(v)}"
                for k, v in logs.items()
                if (isinstance(v, (float, int)) or v is None) and k != "total_flos"
            )
            if clean:
                print(f"Step {state.global_step}: {clean}", flush=True)
        return control

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not self._is_main(args) or not self.enabled:
            return control
        if self._live is None:
            self._start_live()
        if self._eval_task is None:
            try:
                total = len(eval_dataloader)
            except TypeError:
                total = 0
            self._eval_task = self._progress.add_task("Evaluation", total=max(total, 1), speed="")
        self._progress.advance(self._eval_task)
        self._refresh()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._live is not None:
            if metrics:
                self._metrics.update(metrics)
            if self._eval_task is not None:
                self._progress.remove_task(self._eval_task)
                self._eval_task = None
            self._refresh(force=True)
            if not self._training:
                self._stop_live()
        return control

    def on_predict(self, args, state, control, metrics=None, **kwargs):
        return self.on_evaluate(args, state, control, metrics=metrics, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self._refresh(force=True)
        self._stop_live()
        self._training = False
        return control
