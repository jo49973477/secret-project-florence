# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import Mock

from gr00t.experiment.discord_notifier import DiscordNotifier
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
    if marker_written:
        marker.write_text("delivered\n")
    program = "import os,sys; " + code
    result = main(["--marker", str(marker), "--output-dir", str(tmp_path), "--", sys.executable, "-c", program])
    return result, notifier


def test_child_python_exception_causes_one_fallback(tmp_path, monkeypatch):
    result, notifier = _run(tmp_path, monkeypatch, "raise RuntimeError('worker failure')")
    assert result != 0
    notifier.send.assert_called_once()
    assert "torchrun supervisor" in notifier.send.call_args.args[0]


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
