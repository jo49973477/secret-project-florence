# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.experiment.training_notification_callback import (
    TrainingNotificationCallback,
    TrainingNotificationContext,
)
import pytest


def _args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path), max_steps=2000)


def _state(*, rank_zero=True, step=500):
    return SimpleNamespace(
        is_world_process_zero=rank_zero,
        global_step=step,
        max_steps=2000,
        log_history=[
            {"loss": 0.82314, "learning_rate": 8.4e-5},
            {"loss": 0.40124},
            {"loss": 0.43214},
        ],
    )


def _callback(tmp_path, notifier=None, **kwargs):
    notifier = notifier or Mock()
    callback = TrainingNotificationCallback(
        notifier=notifier,
        context=TrainingNotificationContext(
            experiment_name="univtac_multimodal",
            hostname="spl",
            output_dir=str(tmp_path),
            model_path="nvidia/GR00T-N1.7-3B",
            dataset_path="/data/univtac",
            num_gpus=2,
            global_batch_size=2,
            learning_rate=1e-4,
            deepspeed_stage=3,
        ),
        **kwargs,
    )
    return callback, notifier


def test_start_message_and_rank_zero_only(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)

    callback.on_train_begin(args, _state(rank_zero=False), None)
    notifier.send.assert_not_called()
    callback.on_train_begin(args, _state(), None)

    message = notifier.send.call_args.args[0]
    assert "🚀 GR00T Training Started" in message
    assert "Experiment: univtac_multimodal" in message
    assert "GPUs: 2" in message
    assert "Max steps: 2000" in message
    assert "Global batch size: 2" in message
    assert "Learning rate: 0.0001" in message
    assert "DeepSpeed: Stage 3" in message


def test_nonzero_rank_suppresses_every_event(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)
    state = _state(rank_zero=False)

    callback.on_train_begin(args, state, None)
    callback.on_save(args, state, None)
    callback.notify_finish(args=args, state=state)
    callback.notify_failure(RuntimeError("boom"), args=args, state=state)

    notifier.send.assert_not_called()


def test_checkpoint_format_and_duplicate_suppression(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)
    state = _state(step=500)

    callback.on_save(args, state, None)
    callback.on_save(args, state, None)

    notifier.send.assert_called_once()
    message = notifier.send.call_args.args[0]
    assert "Step: 500 / 2000" in message
    assert "Progress: 25.0%" in message
    assert "Loss: 0.4321" in message
    assert "Learning rate: 8.4e-05" in message
    assert str(tmp_path / "checkpoint-500") in message


def test_finish_includes_final_and_best_loss(tmp_path):
    callback, notifier = _callback(tmp_path)
    callback.notify_finish(args=_args(tmp_path), state=_state(step=2000))

    message = notifier.send.call_args.args[0]
    assert "✅ GR00T Training Finished" in message
    assert "Final step: 2000 / 2000" in message
    assert "Progress: 100.0%" in message
    assert "Final loss: 0.4321" in message
    assert "Best observed loss: 0.4012" in message


def test_failure_is_sanitized_and_sent_once(tmp_path):
    (tmp_path / "checkpoint-500").mkdir()
    callback, notifier = _callback(tmp_path)
    state = _state(step=837)

    callback.notify_failure(
        RuntimeError("CUDA\nout of memory\x00"), args=_args(tmp_path), state=state
    )
    callback.notify_failure(RuntimeError("again"), args=_args(tmp_path), state=state)

    notifier.send.assert_called_once()
    message = notifier.send.call_args.args[0]
    assert "🚨 GR00T Training Failed" in message
    assert "Step: 837 / 2000" in message
    assert "Progress: 41.9%" in message
    assert str(tmp_path / "checkpoint-500") in message
    assert "RuntimeError: CUDA out of memory" in message
    assert "\x00" not in message


@pytest.mark.parametrize(
    "flag,event",
    [
        ("notify_start", "start"),
        ("notify_save", "save"),
        ("notify_finish", "finish"),
        ("notify_error", "error"),
    ],
)
def test_individual_notify_flags(tmp_path, flag, event):
    callback, notifier = _callback(tmp_path, **{flag: False})
    args = _args(tmp_path)
    state = _state()

    if event == "start":
        callback.on_train_begin(args, state, None)
    elif event == "save":
        callback.on_save(args, state, None)
    elif event == "finish":
        callback.notify_finish(args=args, state=state)
    else:
        callback.notify_failure(RuntimeError("boom"), args=args, state=state)

    notifier.send.assert_not_called()


def test_notifier_exception_does_not_escape_callback(tmp_path):
    notifier = Mock()
    notifier.send.side_effect = RuntimeError("transport failed")
    callback, _ = _callback(tmp_path, notifier=notifier)

    callback.on_train_begin(_args(tmp_path), _state(), None)


def test_one_transport_failure_does_not_prevent_another(tmp_path):
    failing_notifier = Mock()
    failing_notifier.send.side_effect = RuntimeError("transport failed")
    working_notifier = Mock()
    failing_callback, _ = _callback(tmp_path, notifier=failing_notifier)
    working_callback, _ = _callback(tmp_path, notifier=working_notifier)

    for callback in (failing_callback, working_callback):
        callback.on_train_begin(_args(tmp_path), _state(), None)

    failing_notifier.send.assert_called_once()
    working_notifier.send.assert_called_once()
