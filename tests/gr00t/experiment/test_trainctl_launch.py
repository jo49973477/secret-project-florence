"""Public CLI and configuration integration without GPU initialization."""

import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import tomllib

from gr00t.experiment import trainctl_launch
from gr00t.experiment.trainctl_launch import trainctl_command, trainctl_run
from omegaconf import OmegaConf
import pytest


def config():
    return OmegaConf.create(
        {
            "name": "temporal",
            "run_id": "original-id",
            "training": {"max_steps": 2000},
            "trainctl": {
                "enabled": True,
                "message_mode": "koyori",
                "notify_checkpoints": True,
            },
        }
    )


def test_command_mapping_preserves_argv(tmp_path):
    cfg = config()
    worker = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        "worker.py",
        "a b",
    ]
    command = trainctl_command(cfg, tmp_path, worker)
    assert command[:4] == [sys.executable, "-m", "gr00t.experiment.trainctl_cli", "run"]
    assert command[4:12] == [
        "--name",
        "temporal-original-id",
        "--checkpoint-dir",
        str(tmp_path),
        "--total-steps",
        "2000",
        "--message-mode",
        "koyori",
    ]
    assert command[12:] == ["--", *worker]


def test_private_config_overlay_preserves_other_settings(tmp_path):
    source = tmp_path / "user.toml"
    source.write_text(
        '[notifications.messages]\nvariant="first"\n[storage]\ndatabase="custom.db"\n[checkpoints]\npoll_interval_seconds=7\n'
    )
    env = {
        "TRAINCTL_CONFIG": str(source),
        "TRAINCTL_DISCORD_WEBHOOK_URL": "private",
        "OTHER": "value",
    }
    with trainctl_run(config(), tmp_path, ["python", "worker.py"], env) as (command, child_env):
        path = Path(child_env["TRAINCTL_CONFIG"])
        settings = tomllib.loads(path.read_text())
        assert settings["checkpoints"]["pattern"] == "(?!)"
        assert settings["checkpoints"]["poll_interval_seconds"] == 7
        assert settings["storage"]["database"] == "custom.db"
        assert settings["notifications"]["messages"]["variant"] == "first"
        assert child_env["TRAINCTL_DISCORD_WEBHOOK_URL"] == "private"
        assert "private" not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert child_env["GR00T_TRAINCTL_ACTIVE"] == "1"
    assert not path.exists()
    assert env["TRAINCTL_CONFIG"] == str(source)


def test_disabled_does_not_wrap_or_attach(tmp_path):
    cfg = config()
    cfg.trainctl.enabled = False
    with trainctl_run(
        cfg, tmp_path, ["worker"], {"TRAINCTL_SOCKET": "socket", "TRAINCTL_RUN_ID": "outer"}
    ) as (cmd, env):
        assert cmd == ["worker"]
        assert "TRAINCTL_SOCKET" not in env
        assert "TRAINCTL_RUN_ID" not in env


def test_nested_supervisor_rejected(tmp_path):
    with pytest.raises(ValueError, match="nested"):
        with trainctl_run(config(), tmp_path, ["worker"], {"TRAINCTL_RUN_ID": "outer"}):
            pass


def test_missing_dependency(monkeypatch):
    monkeypatch.setattr(trainctl_launch, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="uv sync --locked"):
        trainctl_launch.check_trainctl()


def clean_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRAINCTL_", "DISCORD_"))}
    env["TRAINCTL_HOME"] = str(tmp_path / "home")
    source = tmp_path / "notifications-off.toml"
    source.write_text(
        "[notifications.discord]\nenabled=false\n[notifications.telegram]\nenabled=false\n"
    )
    env["TRAINCTL_CONFIG"] = str(source)
    return env


def rows(tmp_path, table):
    with sqlite3.connect(tmp_path / "home/trainctl.db") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"SELECT * FROM {table}")]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_cli_exit_and_persistence(tmp_path, exit_code):
    with trainctl_run(
        config(),
        tmp_path,
        [
            sys.executable,
            "-c",
            f"print('original output', flush=True); raise SystemExit({exit_code})",
        ],
        clean_env(tmp_path),
    ) as (cmd, env):
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == exit_code, result.stderr
        logs = list(Path(env["TRAINCTL_LOG_DIR"]).glob("*.log"))
        assert "original output" in logs[0].read_text()
    run = rows(tmp_path, "runs")[0]
    assert run["status"] == ("completed" if exit_code == 0 else "failed")
    assert run["exit_code"] == exit_code


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_reaps_separate_worker_group(tmp_path, sig):
    pidfile = tmp_path / "worker.pid"
    rankfile = tmp_path / "rank.pid"
    rank_code = f"import os,time; from pathlib import Path; Path({str(rankfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    code = f"""
import os, signal, subprocess, sys, time
from pathlib import Path
rank = subprocess.Popen([sys.executable, '-c', {rank_code!r}])
def terminate(*_):
    rank.wait(timeout=5)
    raise SystemExit(143)
signal.signal(signal.SIGTERM, terminate)
Path({str(pidfile)!r}).write_text(str(os.getpid()))
time.sleep(60)
"""
    with trainctl_run(config(), tmp_path, [sys.executable, "-c", code], clean_env(tmp_path)) as (
        cmd,
        env,
    ):
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert pidfile.exists()
            while not rankfile.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert rankfile.exists()
            rank_pid = int(rankfile.read_text())
            worker_pid = int(pidfile.read_text())
            os.killpg(process.pid, sig)
            output, _ = process.communicate(timeout=15)
            assert process.returncode != 0, output
            with pytest.raises(ProcessLookupError):
                os.kill(worker_pid, 0)
            with pytest.raises(ProcessLookupError):
                os.kill(rank_pid, 0)
            assert rows(tmp_path, "runs")[0]["status"] == "failed"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if pidfile.exists():
                try:
                    os.killpg(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_real_sdk_metrics_completed_checkpoint_and_exception(tmp_path):
    checkpoint = tmp_path / "checkpoint-123"
    # Exercise actual IPC, not stdout parsing. An unfinished directory must not be polled.
    code = f"""
from pathlib import Path
from trainctl import tracker
p = Path({str(checkpoint)!r})
p.mkdir()
tracker.init(name='temporal-original-id', total_steps=2000)
tracker.define_metric('train/loss', goal='min')
tracker.log(step=123, metrics={{'train/loss': 0.5, 'train/samples_seen': 3936, 'train/effective_epoch': 3.936}})
(p / 'trainer_state.json').write_text('{{"global_step":123}}')
tracker.checkpoint(p, step=123, metrics={{'train/loss': 0.5}})
tracker.checkpoint(p, step=123, metrics={{'train/loss': 0.5}})
raise RuntimeError('failure after final saving')
"""
    with trainctl_run(config(), tmp_path, [sys.executable, "-c", code], clean_env(tmp_path)) as (
        cmd,
        env,
    ):
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stderr
    assert rows(tmp_path, "runs")[0]["status"] == "failed"
    metrics = rows(tmp_path, "metrics")
    assert {(m["name"], m["step"], m["value"]) for m in metrics} >= {
        ("train/loss", 123, 0.5),
        ("train/samples_seen", 123, 3936),
        ("train/effective_epoch", 123, 3.936),
    }, result.stderr
    checkpoints = rows(tmp_path, "checkpoints")
    assert len(checkpoints) == 1
    assert checkpoints[0]["path"] == str(checkpoint)
    assert checkpoints[0]["source"] == "sdk"
    assert checkpoints[0]["step"] == 123


def test_log_relay_preserves_worker_stdout_and_tracebacks(tmp_path, capsys):
    import io

    log = io.StringIO()
    relay = trainctl_launch.LogRelay(tmp_path, log)
    relay.start()
    (tmp_path / "worker.log").write_text("worker stdout\nTraceback: original failure\n")
    relay.close()
    expected = "worker stdout\nTraceback: original failure\n"
    assert capsys.readouterr().out == expected
    assert log.getvalue() == expected
