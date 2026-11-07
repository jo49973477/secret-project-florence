"""Offline configuration diagnostics and actual trainctl events over mocked HTTP."""

import asyncio
import json
from pathlib import Path
import sys

from gr00t.experiment import trainctl_launch
from gr00t.experiment.trainctl_launch import discord_environment, trainctl_run
import httpx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
from typer.testing import CliRunner


WEBHOOK = "https://discord.com/api/webhooks/123/fake-test-credential"
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "values,selected",
    [
        ({"DISCORD_WEBHOOK": WEBHOOK}, "DISCORD_WEBHOOK"),
        ({"DISCORD_WEBHOOK": "old", "DISCORD_WEBHOOK_URL": WEBHOOK}, "DISCORD_WEBHOOK_URL"),
        (
            {
                "DISCORD_WEBHOOK": "old",
                "DISCORD_WEBHOOK_URL": "other",
                "TRAINCTL_DISCORD_WEBHOOK_URL": WEBHOOK,
            },
            "TRAINCTL_DISCORD_WEBHOOK_URL",
        ),
    ],
)
def test_webhook_precedence_without_secret_output(values, selected, caplog):
    original = {**values, "UNRELATED": "untouched"}
    normalized, status = discord_environment(original, {})
    assert normalized["TRAINCTL_DISCORD_WEBHOOK_URL"] == values[selected]
    assert normalized["UNRELATED"] == "untouched"
    assert original == {**values, "UNRELATED": "untouched"}
    assert status.startswith("configured")
    assert "fake-test-credential" not in caplog.text
    if len(values) > 1:
        assert "precedence" in caplog.text


def test_same_alias_values_do_not_warn(caplog):
    discord_environment({name: WEBHOOK for name in trainctl_launch.WEBHOOK_VARIABLES}, {})
    assert "Conflicting" not in caplog.text


@pytest.mark.parametrize(
    "settings",
    [
        {"notifications": {"discord": {"enabled": False}}},
        {"discord": {"enabled": False}},
    ],
)
def test_explicit_disable_is_respected(settings, caplog):
    env, status = discord_environment({"DISCORD_WEBHOOK": WEBHOOK}, settings)
    assert env["TRAINCTL_DISCORD_WEBHOOK_URL"] == WEBHOOK
    assert status.startswith("disabled")
    assert "enabled=false" in caplog.text
    assert "fake-test-credential" not in caplog.text


@pytest.mark.parametrize("webhook,status", [(None, "missing"), ("invalid-secret-value", "invalid")])
def test_missing_or_invalid_diagnostics(webhook, status, caplog):
    _, result = discord_environment({"DISCORD_WEBHOOK": webhook} if webhook else {}, {})
    assert result == status
    assert "TRAINCTL_DISCORD_WEBHOOK_URL" in caplog.text or "valid HTTPS" in caplog.text
    assert "invalid-secret-value" not in caplog.text


def test_missing_httpx_is_actionable(monkeypatch, caplog):
    monkeypatch.setattr(trainctl_launch, "find_spec", lambda name: None)
    _, status = discord_environment({"DISCORD_WEBHOOK": WEBHOOK}, {})
    assert "httpx missing" in status
    assert "trainctl[discord]==0.2.0" in caplog.text


def test_sopia_default_and_persona_override():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="experiment/univtac_overfit5")
        override = compose(
            config_name="experiment/univtac_overfit5", overrides=["trainctl.message_mode=koyori"]
        )
    assert cfg.trainctl.message_mode == "sopia"
    assert override.trainctl.message_mode == "koyori"
    assert trainctl_launch.trainctl_command(cfg, ROOT, ["worker"])[11] == "sopia"
    from gr00t.configs.hydra_config import Trainctl

    assert Trainctl().message_mode == "sopia"


def isolate(monkeypatch, tmp_path, disabled=False):
    for key in list(__import__("os").environ):
        if key.startswith(("TRAINCTL_", "DISCORD_")):
            monkeypatch.delenv(key)
    source = tmp_path / "trainctl.toml"
    source.write_text(
        "[notifications.discord]\nenabled=" + ("false" if disabled else "true") + "\n"
    )
    monkeypatch.setenv("TRAINCTL_CONFIG", str(source))
    monkeypatch.setenv("TRAINCTL_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("DISCORD_WEBHOOK", WEBHOOK)


@pytest.mark.parametrize(
    "ending", ["success", "exception", "nonzero", "oom", "disabled", "transport-failure"]
)
def test_actual_cli_event_delivery_once(monkeypatch, tmp_path, ending):
    import os

    from trainctl.cli import app

    isolate(monkeypatch, tmp_path, disabled=ending == "disabled")
    deliveries = []
    real_client = httpx.AsyncClient

    def receive(request):
        deliveries.append(json.loads(request.content))
        return httpx.Response(403 if ending == "transport-failure" else 204)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(receive), **kwargs),
    )
    worker = tmp_path / "tiny_worker.py"
    worker.write_text(
        """
from pathlib import Path
import os
from types import SimpleNamespace as S
from gr00t.experiment.trainctl_callback import TrainctlCallback
state = S(global_step=2, max_steps=3, is_world_process_zero=True)
args = S(output_dir=os.environ['CHECKPOINT_OUTPUT'])
cb = TrainctlCallback(name='tiny', total_steps=3)
cb.on_train_begin(args, state, None)
cb.on_log(args, state, None, logs={'loss': 0.5, 'samples_seen': 64, 'effective_epoch': 0.1})
p = Path(args.output_dir) / 'checkpoint-2'
p.mkdir()
(p / 'trainer_state.json').write_text('{"global_step":2}')
cb.on_save(args, state, None)
cb.register_checkpoint(p, state=state)
cb.on_train_end(args, state, None)
"""
        + {
            "success": "",
            "disabled": "",
            "transport-failure": "",
            "exception": "raise RuntimeError('worker failed')\n",
            "nonzero": "raise SystemExit(7)\n",
            "oom": "raise RuntimeError('CUDA out of memory')\n",
        }[ending]
    )
    cfg = OmegaConf.create(
        {
            "name": "tiny",
            "run_id": "id",
            "training": {"max_steps": 3},
            "trainctl": {"enabled": True, "message_mode": "sopia", "notify_checkpoints": True},
        }
    )
    env = dict(os.environ, CHECKPOINT_OUTPUT=str(tmp_path), PYTHONPATH=str(ROOT))
    with trainctl_run(cfg, tmp_path, [sys.executable, str(worker)], env) as (cmd, child_env):
        for key, value in child_env.items():
            monkeypatch.setenv(key, value)
        result = CliRunner().invoke(app, cmd[3:])
    assert result.exit_code == (
        7 if ending == "nonzero" else 1 if ending in ("exception", "oom") else 0
    ), result.output
    if ending == "disabled":
        assert not deliveries
        return
    titles = [payload["embeds"][0]["title"] for payload in deliveries]
    assert sum("Training Started" in t for t in titles) == 1
    assert sum("Checkpoint Saved" in t for t in titles) == 1
    final = (
        "Training Completed" if ending in ("success", "transport-failure") else "Training Failed"
    )
    assert sum(final in t for t in titles) == 1
    assert len(deliveries) == 3
    assert "fake-test-credential" not in json.dumps(deliveries)
    if ending == "oom":
        assert "CUDA Out of Memory" in json.dumps(deliveries)
    # Persona is chosen by trainctl's own pack, including the first event.
    from trainctl.messages.registry import PersonaRegistry

    persona = PersonaRegistry().load("sopia").display_name
    assert persona in titles[0] and persona in titles[-1]


@pytest.mark.parametrize("status", [204, 401, 403, 404, 429, 500, "network"])
def test_transport_outcomes_are_isolated_and_redacted(status, caplog):
    from trainctl.integrations.discord import DiscordWebhookAdapter

    def receive(request):
        if status == "network":
            raise httpx.ConnectError("fake-test-credential", request=request)
        return httpx.Response(status, headers={"Retry-After": "0"})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            adapter = DiscordWebhookAdapter(WEBHOOK, client=client, retries=0)
            await adapter.send_message("offline smoke")

    asyncio.run(exercise())
    assert "fake-test-credential" not in caplog.text
    if status != 204:
        assert "Discord webhook" in caplog.text
