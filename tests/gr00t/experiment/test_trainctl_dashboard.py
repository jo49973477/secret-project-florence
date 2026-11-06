"""Display telemetry and real parent/trainctl/PTY integration with tiny CPU stubs."""

import errno
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
from types import SimpleNamespace as S

from gr00t.experiment.dashboard_telemetry import DashboardTelemetryCallback, ParentDashboard
from gr00t.experiment.trainctl_callback import TrainctlCallback
import pytest


ROOT = Path(__file__).resolve().parents[3]


def args_state(rank=0):
    return S(
        process_index=rank, world_size=2, bf16=True, deepspeed={"zero_optimization": {"stage": 3}}
    ), S(global_step=100, max_steps=10000, is_world_process_zero=rank == 0)


def test_two_ranks_publish_only_zero(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import trainctl

    output = tmp_path / "telemetry.jsonl"
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    tracker = Mock()
    monkeypatch.setattr(trainctl, "tracker", tracker)
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        args, state = args_state(rank)
        telemetry = DashboardTelemetryCallback(output)
        callback = TrainctlCallback(name="two-rank", total_steps=10000)
        telemetry.on_train_begin(args, state, None)
        telemetry.on_log(
            args,
            state,
            None,
            logs={"loss/total": 0.4, "samples_seen": 3200, "effective_epoch": 0.03},
        )
        telemetry.on_save(args, state, None)
        callback.on_train_begin(args, state, None)
        callback.on_log(args, state, None, logs={"loss": 0.4, "samples_seen": 3200})
        callback.register_checkpoint(checkpoint, state=state)
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [e["event"] for e in events] == ["begin", "log", "save"]
    assert all(e["rank"] == 0 and e["step"] == 100 for e in events)
    assert events[1]["metrics"]["samples_seen"] == 3200
    tracker.log.assert_called_once()
    tracker.checkpoint.assert_called_once()


def test_parent_receives_metrics_and_restores_on_exception(tmp_path, monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", output)
    (tmp_path / "logs").mkdir()
    cfg = S(name="test", run_id="original", trainctl=S(enabled=True))
    config = S(
        training=S(
            enable_training_dashboard=True,
            dashboard_refresh_seconds=0.01,
            enable_multimodal_diagnostics=True,
            num_gpus=2,
            max_steps=10000,
        )
    )
    env = {}
    with pytest.raises(RuntimeError, match="failure after saving"):
        with ParentDashboard(cfg, config, tmp_path, env) as parent:
            args, state = args_state()
            publisher = DashboardTelemetryCallback(env["GR00T_DASHBOARD_TELEMETRY"])
            publisher.on_train_begin(args, state, None)
            publisher.on_log(
                args,
                state,
                None,
                logs={
                    "loss/total": 0.4,
                    "samples_seen": 3200,
                    "effective_epoch": 0.03,
                    "grad_norm/lora": 0.2,
                },
            )
            publisher.on_save(args, state, None)
            publisher.on_train_end(args, state, None)
            # __exit__ must drain even observations published just before process exit.
            raise RuntimeError("failure after saving")
    assert parent.renderer._metrics["samples_seen"] == 3200
    assert parent.renderer._metrics["grad_norm/lora"] == 0.2
    assert parent.renderer._checkpoint == "100"
    assert parent.renderer._status == "Failed / interrupted"
    assert parent.renderer._live is None
    assert sys.stdout is output
    assert "\x1b[?25h" in output.getvalue()
    assert "Step 100/10000" not in output.getvalue()


def fixture_scripts(tmp_path, mode):
    worker = tmp_path / "worker.py"
    worker.write_text("""
import os, sys, time
from pathlib import Path
from types import SimpleNamespace as S
from gr00t.experiment.dashboard_telemetry import DashboardTelemetryCallback
from gr00t.experiment.training_dashboard import TrainingDashboardCallback
from gr00t.experiment.trainctl_callback import TrainctlCallback
args = S(process_index=0, world_size=2, bf16=True, max_grad_norm=1.0,
         deepspeed={"zero_optimization":{"stage":3}}, output_dir=os.environ['GR00T_OUTPUT_DIR'])
state = S(global_step=0, max_steps=10, is_world_process_zero=True)
path = os.environ.get('GR00T_DASHBOARD_TELEMETRY')
ui = DashboardTelemetryCallback(path, .01) if path else TrainingDashboardCallback(True)
tracker = TrainctlCallback(name='dashboard-fixture', total_steps=10)
ui.on_train_begin(args,state,None)
tracker.on_train_begin(args,state,None)
for step in range(1,4):
    state.global_step=step
    logs={'loss/total': .4, 'loss/action':.3, 'samples_seen':3200,
          'effective_epoch':.03, 'learning_rate':.0001, 'grad_norm/pre_clip':1.2,
          'system/gpu_memory_allocated':2.5, 'system/steps_per_second':1.0}
    ui.on_step_end(args,state,S(should_save=False))
    ui.on_log(args,state,None,logs=logs)
    tracker.on_log(args,state,None,logs=logs)
print('WARNING visible exactly once', flush=True)
Path(os.environ['WORKER_PID_FILE']).write_text(str(os.getpid()))
print('READY_PARENT_TEST', flush=True)
while not Path(os.environ['CONTINUE_FILE']).exists():
    time.sleep(.05)
time.sleep(.15)
if os.environ['TEST_ENDING'] == 'failure':
    raise RuntimeError('fixture worker exception')
p = Path(args.output_dir) / 'checkpoint-3'
p.mkdir()
(p/'trainer_state.json').write_text('{"global_step":3}')
ui.on_save(args,state,None)
tracker.on_save(args,state,None)
ui.on_train_end(args,state,None)
""")
    parent = tmp_path / "parent.py"
    parent.write_text(f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from pathlib import Path
from hydra import compose, initialize_config_dir
from gr00t.experiment import hydra_launch
with initialize_config_dir(config_dir={str(ROOT / "configs")!r}, version_base='1.3'):
    cfg=compose(config_name='experiment/univtac_overfit5',overrides=[
        'name=dashboard-fixture', 'run_id=original', 'training.dashboard_refresh_seconds=0.05',
        'training.enable_training_dashboard=true', 'output_dir={tmp_path / "run"}',
        'hardware.cuda_visible_devices="0,1"'])
hydra_launch.check_sensor_dependencies=lambda *_:None
hydra_launch.validate_dataset_inputs=lambda *_:None
hydra_launch.worker_command=lambda *_:[sys.executable,{str(worker)!r}]
original=sys.stdout
code=hydra_launch.launch(cfg)
assert sys.stdout is original
print('PARENT_TERMINAL_RESTORED', flush=True)
raise SystemExit(code)
""")
    source = tmp_path / "no-notifications.toml"
    source.write_text(
        "[notifications.discord]\nenabled=false\n[notifications.telegram]\nenabled=false\n"
    )
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("TRAINCTL_", "DISCORD_", "GR00T_"))
    }
    env.update(
        PYTHONPATH=str(ROOT),
        TRAINCTL_CONFIG=str(source),
        TRAINCTL_HOME=str(tmp_path / "trainctl"),
        TERM="xterm-256color",
        TEST_ENDING=mode,
        WORKER_PID_FILE=str(tmp_path / "worker.pid"),
        CONTINUE_FILE=str(tmp_path / "continue"),
    )
    return parent, env


@pytest.mark.parametrize("ending", ["success", "failure", "sigint", "sigterm"])
def test_real_trainctl_parent_rich_pty_resize_and_cleanup(tmp_path, ending):
    import fcntl
    import struct
    import termios

    parent, env = fixture_scripts(tmp_path, ending)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 110, 0, 0))
    process = subprocess.Popen([sys.executable, str(parent)], stdout=slave, stderr=slave, env=env)
    os.close(slave)
    chunks = []
    triggered = False
    try:
        while True:
            assert select.select([master], [], [], 45)[0], "PTY fixture stalled"
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
            if not triggered and b"READY_PARENT_TEST" in b"".join(chunks):
                triggered = True
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 70, 0, 0))
                os.kill(process.pid, signal.SIGWINCH)
                if ending.startswith("sig"):
                    os.kill(process.pid, signal.SIGINT if ending == "sigint" else signal.SIGTERM)
                else:
                    (tmp_path / "continue").touch()
        assert triggered
        result = process.wait(timeout=15)
        stream = b"".join(chunks).decode(errors="replace")
        assert (result == 0) == (ending == "success"), stream[-3000:]
        assert "PARENT_TERMINAL_RESTORED" in stream
        assert "\x1b[?25l" in stream and "\x1b[?25h" in stream
        assert stream.rfind("\x1b[?25h") < stream.rfind("PARENT_TERMINAL_RESTORED")
        assert "Samples Seen" in stream and "3,200" in stream
        log = (tmp_path / "run/logs/train.log").read_text()
        assert "\x1b[" not in log
        assert log.count("WARNING visible exactly once") == 1
        assert "Step 1/10" not in log
        raw_logs = list((tmp_path / "run/logs/trainctl").rglob("*.log"))
        assert len(raw_logs) == 1
        assert raw_logs[0].read_text().count("WARNING visible exactly once") == 1
        if ending == "failure":
            assert "fixture worker exception" in log and "fixture worker exception" in stream
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "worker.pid").read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def test_real_trainctl_redirected_text_has_no_ansi(tmp_path):
    parent, env = fixture_scripts(tmp_path, "success")
    (tmp_path / "continue").touch()
    result = subprocess.run(
        [sys.executable, str(parent)], env=env, capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stderr
    assert "\x1b[" not in result.stdout + result.stderr
    assert "Dashboard: text fallback" in result.stdout
    assert "Step 3/10" in result.stdout
    assert result.stdout.count("WARNING visible exactly once") == 1
    assert not list((tmp_path / "run/logs").glob("dashboard-*.jsonl"))


def test_gr00t_trainer_selects_telemetry_without_hooks(tmp_path, monkeypatch):
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.experiment.training_dashboard import TrainingDashboardCallback
    import torch
    from transformers import TrainingArguments

    monkeypatch.setenv("GR00T_DASHBOARD_TELEMETRY", str(tmp_path / "display.jsonl"))
    model = torch.nn.Linear(2, 2)
    trainer = Gr00tTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        enable_training_dashboard=True,
    )
    callbacks = trainer.callback_handler.callbacks
    assert sum(isinstance(cb, DashboardTelemetryCallback) for cb in callbacks) == 1
    assert not any(isinstance(cb, TrainingDashboardCallback) for cb in callbacks)
    assert not any(module._forward_hooks or module._backward_hooks for module in model.modules())


def test_offline_smoke_helper_never_launches_a_run(monkeypatch, capsys):
    import importlib.util

    path = ROOT / "scripts/smoke_trainctl_notifications.py"
    spec = importlib.util.spec_from_file_location("wah_notification_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "discord_environment", lambda *_: ({}, "missing"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline inspection must not launch a supervised process")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    assert module.main([]) == 0
    assert "no notifications sent" in capsys.readouterr().out
