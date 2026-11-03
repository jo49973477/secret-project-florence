"""Best-effort, rank-zero Hugging Face adapter for trainctl's public tracker SDK.

Process lifecycle belongs exclusively to ``trainctl run``. No callback sends
start/finish/failure annotations, and no tensors are moved or synchronized here.
"""

from __future__ import annotations

import logging
import math
from numbers import Real
import os
from pathlib import Path

from transformers import TrainerCallback


logger = logging.getLogger(__name__)

# Prefer the canonical diagnostic names already emitted by Gr00tTrainer.log.
ALIASES = {
    "grad_norm": "train/grad_norm_global",
    "loss/action": "train/action_loss",
    "loss/total": "train/total_loss",
    "loss/zero_prediction_mse": "train/zero_prediction_mse",
    "loss/action_to_zero_ratio": "train/action_to_zero_ratio",
    **{
        f"grad_norm/{part}": f"train/grad_norm_{part}"
        for part in ("dit", "lora", "sidecar", "temporal", "moe")
    },
}


def numeric_metrics(logs):
    result = {}
    for key, value in (logs or {}).items():
        # Deliberately do not call .item(): CUDA tensors can synchronize here.
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, Real):
            continue
        try:
            if not math.isfinite(value):
                continue
        except (OverflowError, ValueError, TypeError):
            continue
        target = ALIASES.get(key, key)
        if target.startswith("eval_"):
            target = "eval/" + target[5:]
        elif "/" not in target:
            target = "train/" + target
        if target not in result or key == target:
            result[target] = int(value) if isinstance(value, int) else float(value)
    return result


class TrainctlCallback(TrainerCallback):
    def __init__(self, *, name, total_steps, sample_exposure=None, notify_checkpoints=True):
        # Import only when enabled; the parent performs the mandatory dependency preflight.
        from trainctl import tracker

        self.tracker = tracker
        self.name = name
        self.total_steps = total_steps
        self.sample_exposure = sample_exposure
        self.notify_checkpoints = notify_checkpoints
        self.latest_metrics = {}
        self.saved_paths = set()
        self._warned = False

    @staticmethod
    def _rank_zero(state):
        return (
            bool(getattr(state, "is_world_process_zero", False))
            and os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"
        )

    def _call(self, method, **kwargs):
        try:
            getattr(self.tracker, method)(**kwargs)
        except Exception:
            # Never include transport exception text: it may contain credentials.
            if not self._warned:
                logger.warning("trainctl tracker unavailable; training will continue")
                self._warned = True

    def on_train_begin(self, args, state, control, **kwargs):
        if self._rank_zero(state):
            self._call("init", name=self.name, total_steps=self.total_steps)
            self._call(
                "define_metrics",
                definitions={
                    "train/loss": {"goal": "min"},
                    "train/learning_rate": {},
                    "train/grad_norm_global": {},
                    "train/samples_seen": {},
                    "train/effective_epoch": {},
                    "eval/loss": {"goal": "min"},
                },
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._rank_zero(state):
            return
        metrics = numeric_metrics(logs)
        if metrics:
            self.latest_metrics.update(metrics)
            self._call("log", step=state.global_step, metrics=metrics)

    def on_save(self, args, state, control, **kwargs):
        # HF calls on_save only after _save_checkpoint has returned successfully.
        self.register_checkpoint(
            Path(args.output_dir) / f"checkpoint-{state.global_step}", state=state
        )

    def register_checkpoint(self, path, *, state):
        if not self.notify_checkpoints or not self._rank_zero(state):
            return
        path = Path(path).resolve()
        if path in self.saved_paths or not path.is_dir():
            return
        metrics = dict(self.latest_metrics)
        # Reuse WAH's existing implementation at the actual checkpoint step,
        # including its suppression of ambiguous resumed sample exposure.
        if self.sample_exposure is not None:
            metrics.update(numeric_metrics(self.sample_exposure.metrics(state.global_step)))
        self.saved_paths.add(path)
        self._call("checkpoint", path=str(path), step=state.global_step, metrics=metrics)
