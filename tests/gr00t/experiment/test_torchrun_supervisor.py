# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from unittest.mock import Mock

from gr00t.experiment.discord_notifier import KOYORI_ERROR, DiscordNotifier
from gr00t.experiment.torchrun_supervisor import main


def _run(tmp_path, monkeypatch, code):
    marker = tmp_path / "delivered"
    monkeypatch.setenv("GR00T_SUPERVISOR_NOTIFY", "1")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/id/secret")
    notifier = Mock()
    monkeypatch.setattr(
        DiscordNotifier,
        "from_environment",
        classmethod(lambda cls, **kwargs: notifier),
    )
    program = "import os,sys; " + code
    result = main(
        [
            "--marker",
            str(marker),
            "--output-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            program,
        ]
    )
    return result, notifier


def test_child_python_exception_causes_one_fallback(tmp_path, monkeypatch):
    result, notifier = _run(tmp_path, monkeypatch, "raise RuntimeError('worker failure')")
    assert result != 0
    notifier.send.assert_called_once()
    message = notifier.send.call_args.args[0]
    assert message.startswith("🚨 Training Failed\n\n" + KOYORI_ERROR)
    assert "torchrun supervisor" in message


def test_rank_worker_nonzero_exit_causes_fallback(tmp_path, monkeypatch):
    result, notifier = _run(tmp_path, monkeypatch, "sys.exit(7)")
    assert result == 7
    notifier.send.assert_called_once()


def test_sigkill_worker_causes_fallback(tmp_path, monkeypatch):
    result, notifier = _run(tmp_path, monkeypatch, "os.kill(os.getpid(), 9)")
    assert result == 137
    message = notifier.send.call_args.args[0]
    assert "SIGKILL may be OOM Killer or external kill" in message


def test_success_exit_does_not_send_failure(tmp_path, monkeypatch):
    result, notifier = _run(tmp_path, monkeypatch, "sys.exit(0)")
    assert result == 0
    notifier.send.assert_not_called()


def test_internal_delivery_marker_written_by_child_suppresses_fallback(tmp_path, monkeypatch):
    result, notifier = _run(
        tmp_path,
        monkeypatch,
        "open(os.environ['GR00T_FAILURE_NOTIFICATION_MARKER'],'w').write('delivered'); sys.exit(9)",
    )
    assert result == 9
    notifier.send.assert_not_called()


def test_supervisor_reports_checkpoint_exposure_without_leaking_secrets(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-1001"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1001}))
    (tmp_path / "sample_exposure.json").write_text(
        json.dumps(
            {
                "per_device_train_batch_size": 2,
                "world_size": 4,
                "gradient_accumulation_steps": 4,
                "total_training_samples": 45,
            }
        )
    )
    logfile = tmp_path / "train.log"
    logfile.write_text("password=hidden https://discord.com/api/webhooks/123/secret\n")
    monkeypatch.setenv("GR00T_TRAIN_LOG", str(logfile))
    result, notifier = _run(tmp_path, monkeypatch, "sys.exit(7)")
    assert result == 7
    message = notifier.send.call_args.args[0]
    assert "Last Checkpoint Global Step: 1001" in message
    assert "Samples Seen (at Last Checkpoint): 32,032" in message
    assert "Effective Epoch (at Last Checkpoint): 711.8222" in message
    assert "Failed Global Step:" not in message
    assert "Last Checkpoint: " in message
    assert "password=hidden" not in message
    assert "api/webhooks/123/secret" not in message
    assert "Log Location: " in message


def test_supervisor_omits_ambiguous_exposure_and_preserves_exit_on_webhook_failure(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 12}))
    (tmp_path / "sample_exposure_unavailable").touch()
    result, notifier = _run(tmp_path, monkeypatch, "sys.exit(9)")
    assert result == 9
    assert "Samples Seen" not in notifier.send.call_args.args[0]

    def failed_from_environment(cls, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(DiscordNotifier, "from_environment", classmethod(failed_from_environment))
    result = main(
        [
            "--marker",
            str(tmp_path / "other-marker"),
            "--output-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(11)",
        ]
    )
    assert result == 11
