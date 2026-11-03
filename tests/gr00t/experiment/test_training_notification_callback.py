# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.experiment.discord_notifier import KOYORI_ERROR, KOYORI_FINISH, KOYORI_START
from gr00t.experiment.sample_exposure import SampleExposure
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


def _exposure():
    class Shards:
        dataset_path = "/data/univtac"

        def __len__(self):
            return 2

        def get_shard_length(self, index):
            return [20, 25][index]

    shard = Shards()
    mixture = SimpleNamespace(datasets=[shard], weights=[1.0])
    args = SimpleNamespace(
        per_device_train_batch_size=2, world_size=4, gradient_accumulation_steps=4
    )
    return SampleExposure(mixture, args)


def test_koyori_copy_matches_requested_korean_text():
    assert KOYORI_START == (
        "콘코요~! 조수군, 새로운 모델 학습 실험 스타트야! holoX의 두뇌인 코요리가 옆에서 지켜보고 있으니까 "
        "이번 Loss는 무조건 완벽하게 수렴할 거라구~! 🧪✨"
    )
    assert KOYORI_FINISH == (
        "조수군, 실험 대성공이야~! 마지막 에폭까지 가중치가 아주 예쁘게 농축됐어! 역시 코요리의 완벽한 "
        "조력 덕분이지? 얼른 결과 보러 와~! 🐾💖"
    )
    assert KOYORI_ERROR == (
        "와아아앗 조수군 큰일이야!! 실험실 플라스크가… 아니, 그래디언트가 폭발해서 학습이 멈췄어!! "
        "코요리는 아무것도 안 건드렸다구, 빨리 와서 로그 좀 봐줘어어~! 💥😭"
    )


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

    (tmp_path / "checkpoint-500").mkdir()

    callback.on_save(args, state, None)
    callback.on_save(args, state, None)

    notifier.send.assert_called_once()
    message = notifier.send.call_args.args[0]
    assert "Step: 500 / 2000" in message
    assert "Progress: 25.0%" in message
    assert "Loss: 0.4321" in message
    assert "Learning rate: 8.4e-05" in message
    assert str(tmp_path / "checkpoint-500") in message


def test_checkpoint_is_not_notified_before_directory_exists(tmp_path):
    callback, notifier = _callback(tmp_path)
    args, state = _args(tmp_path), _state(step=500)
    callback.on_save(args, state, None)
    notifier.send.assert_not_called()
    (tmp_path / "checkpoint-500").mkdir()
    callback.on_save(args, state, None)
    notifier.send.assert_called_once()


def test_discord_koyori_messages_and_exposure(tmp_path, monkeypatch):
    exposure = _exposure()
    notifier = Mock()
    context = TrainingNotificationContext(
        experiment_name="florence-a",
        output_dir=str(tmp_path),
        hostname="spl",
        model_path="nvidia/GR00T-N1.7-3B",
        dataset_path="/data/univtac",
        learning_rate=1e-4,
        deepspeed_stage=3,
        sample_exposure=exposure,
    )
    callback = TrainingNotificationCallback(notifier, context, provider="discord")
    args = SimpleNamespace(output_dir=str(tmp_path), max_steps=2000, world_size=4)
    state = _state(step=500)
    callback.on_train_begin(args, state, None)
    start = notifier.send.call_args.args[0]
    assert start.startswith("🧪 Training Started\n\n" + KOYORI_START)
    assert "Effective Global Batch Size: 32" in start
    assert "Gradient Accumulation Steps: 4" in start
    assert "Total Training Samples: 45" in start
    assert "Estimated Effective Epochs at Max Steps: 1422.2222" in start

    (tmp_path / "checkpoint-500").mkdir()
    callback.on_save(args, state, None)
    callback.on_save(args, state, None)
    save = notifier.send.call_args.args[0]
    assert save.startswith("💾 GR00T Checkpoint Saved")
    assert "Global Step: 500" in save
    assert "Samples Seen: 16,000" in save
    assert "Effective Epoch: 355.5556" in save
    assert "Training Loss: 0.4321" in save
    assert "Learning Rate: 8.4e-05" in save

    callback.notify_finish(
        args=args, state=state, final_checkpoint=str(tmp_path / "checkpoint-500")
    )
    callback.notify_finish(args=args, state=state)
    finish = notifier.send.call_args.args[0]
    assert finish.startswith("✅ Training Finished\n\n" + KOYORI_FINISH)
    assert "Final Global Step: 500" in finish
    assert "Final Loss: 0.4321" in finish
    assert "Best Observed Loss: 0.4012" in finish
    assert "Final Checkpoint: " in finish
    assert notifier.send.call_count == 3

    failed = TrainingNotificationCallback(notifier, context, provider="discord")
    monkeypatch.setenv("GR00T_TRAIN_LOG", str(tmp_path / "train.log"))
    failed.notify_failure(
        RuntimeError("token=abc https://discord.com/api/webhooks/id/secret"),
        args=args,
        state=state,
    )
    failed.notify_failure(RuntimeError("again"), args=args, state=state)
    error = notifier.send.call_args.args[0]
    assert error.startswith("🚨 Training Failed\n\n" + KOYORI_ERROR)
    assert "Exception Type: RuntimeError" in error
    assert "Failed Global Step: 500" in error
    assert "Error Message: token=[redacted] [webhook redacted]" in error
    assert "Log Location: " in error
    assert "token=abc" not in error
    assert notifier.send.call_count == 4


def test_discord_nonzero_rank_and_delivery_failure(tmp_path):
    notifier = Mock()
    notifier.send.side_effect = RuntimeError("webhook offline")
    callback = TrainingNotificationCallback(
        notifier,
        TrainingNotificationContext("florence-a", str(tmp_path), sample_exposure=_exposure()),
        provider="discord",
    )
    args = _args(tmp_path)
    state = _state(rank_zero=False)
    callback.on_train_begin(args, state, None)
    callback.notify_failure(RuntimeError("training failed"), args=args, state=state)
    notifier.send.assert_not_called()
    state.is_world_process_zero = True
    callback.notify_failure(RuntimeError("training failed"), args=args, state=state)
    notifier.send.assert_called_once()


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


@pytest.mark.parametrize("result", [False, None])
def test_failed_failure_delivery_is_not_marked_sent(tmp_path, result):
    notifier = Mock()
    notifier.send.return_value = result
    callback, _ = _callback(tmp_path, notifier=notifier)
    callback.notify_failure(RuntimeError("boom"), args=_args(tmp_path), state=_state())
    assert callback._failure_sent is False
    notifier.send.return_value = True
    callback.notify_failure(RuntimeError("boom"), args=_args(tmp_path), state=_state())
    assert callback._failure_sent is True
    assert notifier.send.call_count == 2


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


def test_failure_delivery_marker_written_atomically_and_sent_once(tmp_path):
    marker = tmp_path / "notifications/delivered"
    notifier = Mock()
    notifier.send.return_value = True
    callback, _ = _callback(tmp_path, notifier=notifier, failure_delivery_marker=str(marker))
    callback.notify_failure(KeyError("bfloat16"), args=_args(tmp_path), state=_state())
    assert marker.read_text() == "delivered\n"
    assert not list(marker.parent.glob("*.tmp"))
    callback.notify_failure(RuntimeError("second failure"), args=_args(tmp_path), state=_state())
    notifier.send.assert_called_once()


def test_failure_marker_write_error_does_not_hide_original_failure(tmp_path, monkeypatch, caplog):
    from gr00t.experiment import training_notification_callback as notifications

    def failed_replace(*args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(notifications.os, "replace", failed_replace)
    notifier = Mock()
    notifier.send.return_value = True
    callback, _ = _callback(
        tmp_path, notifier=notifier, failure_delivery_marker=str(tmp_path / "delivered")
    )
    callback.notify_failure(KeyError("bfloat16"), args=_args(tmp_path), state=_state())
    assert callback._failure_sent
    assert "Could not record failure notification delivery" in caplog.text
    assert not (tmp_path / "delivered").exists()
