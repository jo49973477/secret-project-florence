# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.parse

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.telegram_callback import TelegramTrainingCallback, TrainingNotificationContext
from gr00t.experiment.telegram_notifier import TelegramNotifier, validate_telegram_configuration
import pytest
import tyro


ROOT = Path(__file__).resolve().parents[3]


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _state(*, rank_zero: bool = True, step: int = 25):
    return SimpleNamespace(
        is_world_process_zero=rank_zero,
        global_step=step,
        max_steps=100,
        log_history=[{"loss": 0.7421, "learning_rate": 9.2e-5}],
    )


def _args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path), max_steps=100)


def _callback(tmp_path, notifier=None, **kwargs):
    notifier = notifier or Mock(spec=TelegramNotifier)
    return TelegramTrainingCallback(
        notifier=notifier,
        context=TrainingNotificationContext(
            experiment_name="test_run",
            output_dir=str(tmp_path),
        ),
        **kwargs,
    ), notifier


def test_successful_send_constructs_endpoint_and_payload():
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return _Response()

    notifier = TelegramNotifier(
        token="test-token",
        chat_id="123456789",
        timeout=1.5,
        opener=opener,
    )

    assert notifier.send("hello GR00T") is True
    request, timeout = calls[0]
    assert request.full_url == "https://api.telegram.org/bottest-token/sendMessage"
    assert request.method == "POST"
    assert timeout == 1.5
    assert urllib.parse.parse_qs(request.data.decode()) == {
        "chat_id": ["123456789"],
        "text": ["hello GR00T"],
    }


def test_transport_error_log_never_contains_token(caplog):
    token = "super-secret-bot-token"

    def opener(request, *, timeout):
        raise TimeoutError(f"request failed for {request.full_url}")

    notifier = TelegramNotifier(token=token, chat_id="123", opener=opener)
    with caplog.at_level(logging.WARNING):
        assert notifier.send("message") is False

    assert token not in caplog.text
    assert "api.telegram.org" not in caplog.text
    assert "TimeoutError" in caplog.text


def test_disabled_notifier_does_not_make_http_request():
    opener = Mock()
    notifier = TelegramNotifier(enabled=False, opener=opener)

    assert notifier.send("do not send") is False
    opener.assert_not_called()


def test_callback_sends_only_from_world_process_zero(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)

    callback.on_train_begin(args, _state(rank_zero=False), None)
    notifier.send.assert_not_called()

    callback.on_train_begin(args, _state(rank_zero=True), None)
    notifier.send.assert_called_once()


def test_nonzero_rank_suppresses_every_event(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)
    state = _state(rank_zero=False)

    callback.on_train_begin(args, state, None)
    callback.on_save(args, state, None)
    callback.on_train_end(args, state, None)
    callback.notify_failure(RuntimeError("boom"), args=args, state=state)

    notifier.send.assert_not_called()


def test_on_save_contains_step_and_checkpoint_path(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)

    callback.on_save(args, _state(step=25), None)

    message = notifier.send.call_args.args[0]
    assert "Step: 25 / 100" in message
    assert str(tmp_path / "checkpoint-25") in message
    assert "Loss: 0.7421" in message


def test_on_train_end_emits_finish_once(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)
    state = _state(step=100)

    callback.on_train_end(args, state, None)
    callback.on_train_end(args, state, None)

    notifier.send.assert_called_once()
    assert "✅ GR00T Training Finished" in notifier.send.call_args.args[0]


def test_production_finish_is_deferred_until_explicit_finalization(tmp_path):
    callback, notifier = _callback(tmp_path, defer_finish_until_final_save=True)
    args = _args(tmp_path)
    state = _state(step=100)

    callback.on_train_end(args, state, None)
    notifier.send.assert_not_called()

    callback.notify_finish(args=args, state=state)
    callback.notify_finish(args=args, state=state)
    notifier.send.assert_called_once()


def test_callback_does_not_raise_on_network_failure(tmp_path):
    def opener(request, *, timeout):
        raise TimeoutError("offline")

    notifier = TelegramNotifier(token="secret", chat_id="123", opener=opener)
    callback, _ = _callback(tmp_path, notifier=notifier)

    callback.on_train_begin(_args(tmp_path), _state(), None)
    callback.on_save(_args(tmp_path), _state(), None)
    callback.on_train_end(_args(tmp_path), _state(), None)


def test_missing_configuration_fails_early(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is not set"):
        FinetuneConfig(
            base_model_path="model",
            dataset_path="dataset",
            embodiment_tag="NEW_EMBODIMENT",
            telegram_on=True,
        )

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    with pytest.raises(ValueError, match="no chat ID was provided"):
        validate_telegram_configuration(enabled=True)


def test_cli_chat_id_takes_priority_over_environment():
    assert validate_telegram_configuration(
        enabled=True,
        chat_id="cli-chat",
        environ={"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_CHAT_ID": "env-chat"},
    ) == ("secret", "cli-chat")


def test_tyro_accepts_documented_telegram_flags(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    config = tyro.cli(
        FinetuneConfig,
        args=[
            "--base-model-path",
            "model",
            "--dataset-path",
            "dataset",
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--telegram-on",
            "--telegram-notify-start",
            "--no-telegram-notify-save",
            "--telegram-notify-finish",
            "--telegram-notify-error",
        ],
    )

    assert config.telegram_on is True
    assert config.telegram_notify_start is True
    assert config.telegram_notify_save is False
    assert config.telegram_notify_finish is True
    assert config.telegram_notify_error is True


def test_finetune_shell_forwards_telegram_extra_arguments(tmp_path):
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        NUM_GPUS="1",
        USE_WANDB="0",
        RESUME="0",
        RESUME_FROM_CHECKPOINT="",
        SAVE_ONLY_MODEL="0",
    )
    result = subprocess.run(
        [
            "bash",
            "examples/finetune.sh",
            "--base-model-path",
            "model",
            "--dataset-path",
            "dataset",
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--output-dir",
            str(tmp_path / "output"),
            "--",
            "--telegram-on",
            "--no-telegram-notify-save",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    forwarded = result.stdout.splitlines()
    assert "--telegram-on" in forwarded
    assert "--no-telegram-notify-save" in forwarded


def test_bot_token_is_not_part_of_serializable_finetune_config(monkeypatch):
    token = "must-not-be-serialized"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    config = FinetuneConfig(
        base_model_path="model",
        dataset_path="dataset",
        embodiment_tag="NEW_EMBODIMENT",
        telegram_on=True,
    )

    assert token not in json.dumps(asdict(config))


def test_failure_exception_is_typed_sanitized_and_truncated(tmp_path):
    callback, notifier = _callback(tmp_path)
    args = _args(tmp_path)
    long_message = "CUDA\nout of memory\x00 " + "x" * 500

    callback.notify_failure(RuntimeError(long_message), args=args, state=_state(step=68))

    message = notifier.send.call_args.args[0]
    exception_summary = message.split("Exception:\n", 1)[1].split("\n\nDuration:", 1)[0]
    assert exception_summary.startswith("RuntimeError: CUDA out of memory ")
    assert "\x00" not in exception_summary
    assert "\n" not in exception_summary
    assert exception_summary.endswith("…")
    assert len(exception_summary) <= 314  # 300-char message plus type prefix


def test_failure_reports_latest_existing_checkpoint(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-30").mkdir()
    callback, notifier = _callback(tmp_path)

    callback.notify_failure(
        RuntimeError("boom"),
        args=_args(tmp_path),
        state=_state(step=42),
    )

    assert str(tmp_path / "checkpoint-30") in notifier.send.call_args.args[0]
