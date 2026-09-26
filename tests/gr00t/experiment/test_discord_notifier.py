# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.error

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.experiment.discord_notifier import DiscordNotifier, _main, validate_discord_configuration
import pytest
import tyro


ROOT = Path(__file__).resolve().parents[3]
WEBHOOK_URL = "https://discord.com/api/webhooks/123/super-secret-token"


class _Response:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_missing_webhook_url(monkeypatch, capsys):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    with pytest.raises(ValueError, match="DISCORD_WEBHOOK_URL is not set"):
        DiscordNotifier.from_environment()

    assert _main(["--test"]) == 2
    assert "DISCORD_WEBHOOK_URL is not set" in capsys.readouterr().err


@pytest.mark.parametrize("status", [200, 204])
def test_successful_send_has_json_body_and_required_headers(status):
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return _Response(status)

    notifier = DiscordNotifier(webhook_url=WEBHOOK_URL, timeout=1.5, opener=opener)

    assert notifier.send("hello GR00T") is True
    request, timeout = calls[0]
    assert request.full_url == WEBHOOK_URL
    assert request.method == "POST"
    assert timeout == 1.5
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("User-agent") == "secret-project-florence/1.0"
    assert json.loads(request.data.decode("utf-8")) == {"content": "hello GR00T"}


def test_http_error_reports_safe_status_without_leaking_webhook(caplog):
    def opener(request, *, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    notifier = DiscordNotifier(webhook_url=WEBHOOK_URL, opener=opener)
    with caplog.at_level(logging.WARNING):
        assert notifier.send("message") is False

    assert "Discord notification failed (HTTP 403); training will continue." in caplog.text
    assert WEBHOOK_URL not in caplog.text
    assert "super-secret-token" not in caplog.text


def test_network_failure_returns_false_without_leaking_webhook(caplog):
    def opener(request, *, timeout):
        raise urllib.error.URLError(f"request failed for {request.full_url}")

    notifier = DiscordNotifier(webhook_url=WEBHOOK_URL, opener=opener)
    with caplog.at_level(logging.WARNING):
        assert notifier.send("message") is False

    assert WEBHOOK_URL not in caplog.text
    assert "super-secret-token" not in caplog.text
    assert "Discord notification failed (URLError); training will continue." in caplog.text


def test_oversized_message_is_safely_truncated():
    requests = []

    def opener(request, *, timeout):
        requests.append(request)
        return _Response(204)

    notifier = DiscordNotifier(webhook_url=WEBHOOK_URL, opener=opener)
    assert notifier.send("start-" + "x" * 3000 + "-useful-tail") is True

    content = json.loads(requests[0].data.decode("utf-8"))["content"]
    assert len(content) == 2000
    assert content.startswith("start-")
    assert content.endswith("-useful-tail")
    assert "message truncated" in content


@pytest.mark.parametrize("delivered,expected", [(True, 0), (False, 1)])
def test_cli_exit_status(monkeypatch, delivered, expected):
    notifier = Mock()
    notifier.send.return_value = delivered
    monkeypatch.setattr(
        DiscordNotifier,
        "from_environment",
        classmethod(lambda cls: notifier),
    )

    assert _main(["--test"]) == expected
    notifier.send.assert_called_once_with("✅ GR00T Discord notification test successful.")


def test_configuration_is_environment_only(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK_URL)
    config = FinetuneConfig(
        base_model_path="model",
        dataset_path="dataset",
        embodiment_tag="NEW_EMBODIMENT",
        discord_on=True,
    )

    assert WEBHOOK_URL not in json.dumps(asdict(config))
    assert "discord_webhook_url" not in FinetuneConfig.__annotations__
    assert "discord_webhook_url" not in TrainingConfig.__annotations__
    assert validate_discord_configuration(enabled=False, environ={}) is None


def test_tyro_accepts_documented_discord_flags(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK_URL)
    config = tyro.cli(
        FinetuneConfig,
        args=[
            "--base-model-path",
            "model",
            "--dataset-path",
            "dataset",
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--discord-on",
            "--no-discord-notify-start",
            "--no-discord-notify-save",
            "--no-discord-notify-finish",
            "--no-discord-notify-error",
        ],
    )

    assert config.discord_on is True
    assert config.discord_notify_start is False
    assert config.discord_notify_save is False
    assert config.discord_notify_finish is False
    assert config.discord_notify_error is False


def test_launcher_propagates_discord_options(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK_URL)
    ft_config = FinetuneConfig(
        base_model_path="model",
        dataset_path="dataset",
        embodiment_tag="NEW_EMBODIMENT",
        discord_on=True,
        discord_notify_start=False,
        discord_notify_save=False,
        discord_notify_finish=False,
        discord_notify_error=False,
    )
    config = SimpleNamespace(training=TrainingConfig())
    path = ROOT / "gr00t/experiment/launch_finetune.py"
    tree = ast.parse(path.read_text())
    names = {
        "discord_on",
        "discord_notify_start",
        "discord_notify_save",
        "discord_notify_finish",
        "discord_notify_error",
    }
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(target) == f"config.training.{name}"
            for target in node.targets
            for name in names
        )
    ]

    assert len(assignments) == len(names)
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"),
        {"config": config, "ft_config": ft_config},
    )
    assert config.training.discord_on is True
    assert config.training.discord_notify_start is False
    assert config.training.discord_notify_save is False
    assert config.training.discord_notify_finish is False
    assert config.training.discord_notify_error is False


def test_finetune_shell_forwards_discord_extra_arguments(tmp_path):
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
            "--discord-on",
            "--no-discord-notify-save",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    forwarded = result.stdout.splitlines()
    assert "--discord-on" in forwarded
    assert "--no-discord-notify-save" in forwarded
