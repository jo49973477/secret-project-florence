"""CPU-only instrumentation tests; no sensor/checkpoint downloads."""

import io
import json
from types import SimpleNamespace

from gr00t.experiment.multimodal_diagnostics import (
    MultimodalDiagnostics,
    full_value,
    gradient_stats,
    residual_ratio,
)
from gr00t.experiment.training_dashboard import TrainingDashboardCallback
import pytest
import torch
from transformers import TrainingArguments


class TinyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.point_gates = torch.nn.Parameter(torch.zeros(2))
        self.tactile_gates = torch.nn.Parameter(torch.zeros(2))
        self.point_cross_attention = torch.nn.Linear(2, 2, bias=False)
        self.tactile_cross_attention = torch.nn.Linear(2, 2, bias=False)


def fixture(tmp_path, interval=1):
    dit = TinyBranch()
    head = torch.nn.Module()
    head.model = dit
    head.point_encoder = torch.nn.Linear(2, 2, bias=False)
    head.tactile_encoder = torch.nn.Linear(2, 2, bias=False)
    model = torch.nn.Module()
    model.action_head = head
    trainer = SimpleNamespace(
        model=model,
        state=SimpleNamespace(global_step=0),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        is_deepspeed_enabled=False,
        is_world_process_zero=lambda: True,
        log=lambda logs: None,
    )
    return MultimodalDiagnostics(trainer, interval, "per_layer"), trainer, dit


def test_ratio_and_gradient_states():
    assert residual_ratio(
        torch.tensor([3.0, 4.0]), torch.tensor([0.0, 1.0])
    ).item() == pytest.approx(0.2)
    p = torch.nn.Parameter(torch.ones(2))
    assert gradient_stats([p])["grad_norm"] is None
    p.grad = torch.zeros(2)
    assert gradient_stats([p])["grad_norm"] == 0
    p.grad[0] = float("nan")
    assert gradient_stats([p])["grad_finite"] is False
    assert gradient_stats([p])["grad_norm"] is None


def test_zero_gate_gradient_then_branch_gradient(tmp_path):
    monitor, trainer, dit = fixture(tmp_path)
    head = trainer.model.action_head
    x = torch.tensor([[1.0, 2.0]])

    def backward():
        branch = dit.point_cross_attention(head.point_encoder(x))
        (x + dit.point_gates[0] * branch).square().sum().backward()

    backward()
    assert dit.point_gates.grad.norm() > 0
    assert dit.point_cross_attention.weight.grad.norm() == 0
    assert head.point_encoder.weight.grad.norm() == 0
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    backward()
    assert dit.point_cross_attention.weight.grad.norm() > 0
    assert head.point_encoder.weight.grad.norm() > 0


def test_sampled_optimizer_delta_accumulation_json(tmp_path):
    monitor, trainer, dit = fixture(tmp_path, interval=2)
    assert not monitor.sampled()
    trainer.state.global_step = 1
    assert monitor.sampled()
    for _ in range(3):  # microbatches accumulate into one optimizer update
        dit.point_gates.sum().backward()
        monitor.observe("point", 0, torch.ones(2), torch.ones(2) * 0.25)
    monitor.pre_step()
    monitor.pre_step()  # duplicate callbacks must not double count the sample
    assert monitor.record["point_gate_grad_norm"] == pytest.approx(18**0.5)
    assert monitor.record["point_gate_mean_abs"] == 0
    assert monitor.record["point_residual_ratio_mean"] == pytest.approx(0.25)
    trainer.optimizer.step()
    monitor.post_step()
    assert monitor.record["point_gate_update_norm"] == pytest.approx(0.3 * 2**0.5)
    trainer.state.global_step += 1
    monitor.on_step_end(SimpleNamespace(output_dir=tmp_path), trainer.state, None)
    monitor.on_step_end(SimpleNamespace(output_dir=tmp_path), trainer.state, None)
    records = (tmp_path / "multimodal_diagnostics.jsonl").read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["global_step"] == 2
    assert record["layers"][0]["point_gate_delta"] == pytest.approx(-0.3)
    assert record["tactile_gate_grad_present"] is False


def test_zero_safe_access_uses_supported_utilities(monkeypatch):
    import deepspeed.utils

    p = torch.nn.Parameter(torch.empty(0))
    p._z3_optimizer = object()
    p.ds_id = 123
    p.ds_numel = 2
    monkeypatch.setattr(
        deepspeed.utils, "safe_get_full_fp32_param", lambda p: torch.tensor([2.0, 4.0])
    )
    monkeypatch.setattr(deepspeed.utils, "safe_get_full_grad", lambda p: torch.tensor([3.0, 4.0]))
    from contextlib import contextmanager

    from deepspeed import zero

    @contextmanager
    def gathered(parameters, modifier_rank=None):
        assert parameters == [p] and modifier_rank is None
        old = p.data
        p.data = torch.tensor([2.0, 4.0])
        try:
            yield
        finally:
            p.data = old

    monkeypatch.setattr(zero, "GatheredParameters", gathered)
    assert full_value(p).tolist() == [2.0, 4.0]
    monkeypatch.setattr(deepspeed.utils, "safe_get_local_grad", lambda p: torch.tensor([3.0, 4.0]))
    assert gradient_stats([p])["grad_norm"] == 5.0
    assert gradient_stats([p])["grad_present"] is None


def test_deepspeed_wraps_real_step_and_restores(tmp_path):
    monitor, trainer, dit = fixture(tmp_path)
    trainer.is_deepspeed_enabled = True
    original = trainer.optimizer.step
    engine = SimpleNamespace(optimizer=trainer.optimizer)
    monitor.install_deepspeed(engine)
    dit.point_gates.sum().backward()
    engine.optimizer.step()
    assert monitor.record["point_gate_update_norm"] > 0
    monitor.close()
    assert engine.optimizer.step == original


def test_non_tty_and_rank_safety(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TrainingDashboardCallback(True)
    assert not dashboard.enabled
    args = SimpleNamespace(process_index=1, logging_steps=1)
    state = SimpleNamespace(global_step=1)
    dashboard.on_log(args, state, None, logs={"loss": 1.2})
    assert capsys.readouterr().out == ""
    args.process_index = 0
    dashboard.on_log(args, state, None, logs={"loss": 1.2})
    output = capsys.readouterr().out
    assert "loss=1.2" in output and "\x1b" not in output and "{'loss'" not in output


def test_rich_render_smoke_and_cleanup(monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr("sys.stdout", output)
    dashboard = TrainingDashboardCallback(True, 0.1)
    args = SimpleNamespace(process_index=0, logging_steps=1)
    state = SimpleNamespace(global_step=0, max_steps=2)
    dashboard.on_train_begin(args, state, None)
    try:
        dashboard.on_log(args, state, None, logs={"loss": 1.2, "point_gate_mean_abs": 0.01})
        state.global_step = 1
        dashboard.on_step_end(args, state, None)
    finally:
        dashboard._stop_live()
    assert dashboard._live is None
    assert "LOSS" in output.getvalue()


def test_trainer_off_no_hooks_and_callback_replacement(tmp_path):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    model = torch.nn.Linear(2, 2)
    hooks_before = sum(len(m._forward_hooks) + len(m._backward_hooks) for m in model.modules())
    trainer = Gr00tTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        enable_multimodal_diagnostics=False,
        enable_training_dashboard=False,
    )
    assert trainer.multimodal_monitor is None
    assert not any(
        isinstance(cb, (PrinterCallback, ProgressCallback))
        for cb in trainer.callback_handler.callbacks
    )
    assert hooks_before == sum(
        len(m._forward_hooks) + len(m._backward_hooks) for m in model.modules()
    )


def test_zero3_local_norm_reduces_scalars_once(monkeypatch):
    import deepspeed.utils

    p = torch.nn.Parameter(torch.empty(0))
    p.ds_id, p.ds_numel = 1, 4
    p._z3_optimizer = object()
    monkeypatch.setattr(deepspeed.utils, "safe_get_local_grad", lambda p: torch.tensor([3.0, 4.0]))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    calls = []

    def reduce(value, group=None):
        calls.append(value.numel())
        value.mul_(2)  # same local shard squared norm from a second rank

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    result = gradient_stats([p])
    assert result["grad_norm"] == pytest.approx(50**0.5)
    assert calls == [3]
    assert result["grad_present"] is None
    assert result["grad_status"] == "nonzero_buffer"


def test_rich_evaluation_uses_same_display(monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr("sys.stdout", output)
    dashboard = TrainingDashboardCallback(True)
    args = SimpleNamespace(process_index=0)
    state = SimpleNamespace(global_step=0, max_steps=2)
    dashboard.on_train_begin(args, state, None)
    try:
        live = dashboard._live
        dashboard.on_prediction_step(args, state, None, eval_dataloader=[1, 2])
        assert len(dashboard._progress.tasks) == 2
        assert live is dashboard._live
        dashboard.on_evaluate(args, state, None, metrics={"eval_loss": 0.25})
        assert len(dashboard._progress.tasks) == 1
        assert dashboard._metrics["eval_loss"] == 0.25
    finally:
        dashboard.on_train_end(args, state, None)


def test_hf_trainer_accumulation_smoke_and_unchanged_weights(tmp_path):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import Trainer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = TinyBranch()
            self.action_head.point_encoder = torch.nn.Linear(2, 2)
            self.action_head.tactile_encoder = torch.nn.Linear(2, 2)

        def forward(self, x, target):
            head = self.action_head
            dit = head.model
            hidden = x
            for layer in range(2):
                for modality in ("point", "tactile"):
                    tokens = getattr(head, f"{modality}_encoder")(x)
                    update = getattr(dit, f"{modality}_gates")[layer] * getattr(
                        dit, f"{modality}_cross_attention"
                    )(tokens)
                    observer = getattr(dit, "_diagnostic_observer", None)
                    if observer is not None:
                        observer(modality, layer, hidden, update)
                    hidden = hidden + update
            return {
                "loss": (hidden - target).square().mean(),
                "diagnostics": {"predicted_velocity_std": hidden.detach().std()},
            }

    class SmokeTrainer(Gr00tTrainer):
        # Use HF's generic dataloader for this tiny fixture, not the robot dataset.
        get_train_dataloader = Trainer.get_train_dataloader

    torch.manual_seed(19)
    initial = Model().state_dict()
    dataset = [{"x": torch.tensor([1.0, 2.0]), "target": torch.tensor([2.0, 1.0])}] * 8
    outputs = []
    for enabled in (False, True):
        model = Model()
        model.load_state_dict(initial)
        path = tmp_path / str(enabled)
        args = TrainingArguments(
            output_dir=str(path),
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            logging_steps=2,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            disable_tqdm=True,
        )
        trainer = SmokeTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            enable_training_dashboard=False,
            enable_multimodal_diagnostics=enabled,
            multimodal_diagnostics_interval=2,
            multimodal_diagnostics_level="per_layer",
        )
        trainer.train()
        outputs.append({k: v.clone() for k, v in model.state_dict().items()})
        if enabled:
            rows = [
                json.loads(row)
                for row in (path / "multimodal_diagnostics.jsonl").read_text().splitlines()
            ]
            assert len(rows) == 1 and rows[0]["global_step"] == 2
            assert rows[0]["point_gate_grad_norm"] > 0
            assert len(rows[0]["layers"]) == 2
        else:
            assert not (path / "multimodal_diagnostics.jsonl").exists()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)


def test_frozen_encoder_not_reported_in_optimizer(tmp_path):
    monitor, trainer, dit = fixture(tmp_path)
    trainer.model.action_head.point_encoder.requires_grad_(False)
    trainer.optimizer = torch.optim.SGD(
        [p for p in trainer.model.parameters() if p.requires_grad], lr=0.1
    )
    dit.point_gates.sum().backward()
    monitor.pre_step()
    assert monitor.record["point_encoder_trainable"] is False
    assert monitor.record["point_encoder_in_optimizer"] is False
    assert monitor.record["point_encoder_grad_norm"] is None


def test_exception_closes_display_and_propagates(tmp_path, monkeypatch):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import Trainer

    trainer = Gr00tTrainer(
        model=torch.nn.Linear(2, 2),
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        enable_training_dashboard=False,
    )
    closed = []
    monkeypatch.setattr(trainer, "_close_training_display", lambda: closed.append(True))

    def fail(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(Trainer, "train", fail)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert closed == [True]


def test_existing_logging_handler_coordinated_and_restored(monkeypatch):
    import logging

    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr("sys.stdout", output)
    logger = logging.getLogger("instrumentation_handler_test")
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    dashboard = TrainingDashboardCallback(True)
    try:
        dashboard.on_train_begin(
            SimpleNamespace(process_index=0), SimpleNamespace(global_step=0, max_steps=2), None
        )
        assert handler.stream is not output
        logger.warning("checkpoint warning preserved")
        dashboard._stop_live()
        assert handler.stream is output
        assert "checkpoint warning preserved" in output.getvalue()
    finally:
        dashboard._stop_live()
        logger.removeHandler(handler)


@pytest.mark.parametrize("warmup", [False, True])
def test_loss_decomposition_mask_and_no_gradient_changes(warmup):
    from gr00t.experiment.action_diagnostics import action_diagnostics, loss_ratios

    p = torch.nn.Parameter(torch.ones(1, 40, 132))
    mask = torch.zeros_like(p)
    mask[..., :8] = 1
    target = torch.full_like(p, 2)
    action = ((p - target).square() * mask).sum() / (mask.sum() + 1e-6)
    contact = p.mean().square()
    loss = 0.1 * contact if warmup else action + 0.1 * contact
    stats = action_diagnostics(p, target, mask, loss, action, contact, 0.1, warmup)
    assert stats["loss/total"] == stats["loss/action"] + stats["loss/contact_weighted"]
    assert stats["action/valid_dimension_count"] == 8
    assert stats["action/valid_timestep_ratio"] == 1
    assert stats["action/valid_action_count"] == 320
    assert stats["action/valid_action_ratio"] == pytest.approx(8 / 132)
    assert not any(value.requires_grad for value in stats.values())
    expected = torch.autograd.grad(loss, p, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    reduced = loss_ratios({key: float(value) for key, value in stats.items()})
    assert reduced["loss/contact_fraction"] == pytest.approx(float(0.1 * contact / loss))


def test_zero_padding_excluded_and_scale(monkeypatch):
    import deepspeed.utils

    p = torch.nn.Parameter(torch.empty(0))
    p.ds_id, p.ds_numel, p._z3_optimizer = 1, 3, object()
    monkeypatch.setattr(
        deepspeed.utils, "safe_get_local_grad", lambda p: torch.tensor([8.0, 999.0])
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")

    def reduce(values, group=None):
        values[0] += 9  # other rank contains 3,0 after unscaling

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert gradient_stats([p], scale=2)["grad_norm"] == 5


def test_console_filter_preserves_details_and_errors():
    from gr00t.experiment.console_logging import DetailedConsoleStream

    console, detail = io.StringIO(), io.StringIO()
    stream = DetailedConsoleStream(console, detail)
    stream.write("Rank 2, Worker 0: Caching shard...\n")
    stream.write("INFO checkpoint-memory phase=save point=during metrics={}\n")
    stream.write("WARNING Rank 2, Worker 0: Caching shard failed!\n")
    stream.flush()
    assert "Caching shard..." not in console.getvalue()
    assert "WARNING" in console.getvalue()
    assert "checkpoint-memory" not in console.getvalue()
    assert "checkpoint-memory" in detail.getvalue()
    assert "Caching shard..." in detail.getvalue()


def test_dashboard_checkpoint_and_rank_lifecycle(monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr("sys.stdout", output)
    dashboard = TrainingDashboardCallback(True)
    args = SimpleNamespace(process_index=1)
    state = SimpleNamespace(global_step=5, max_steps=10)
    control = SimpleNamespace(should_save=True)
    dashboard.on_train_begin(args, state, control)
    assert dashboard._live is None
    args.process_index = 0
    dashboard.on_train_begin(args, state, control)
    try:
        dashboard.on_step_end(args, state, control)
        assert dashboard._status == "Saving checkpoint"
        dashboard.on_save(args, state, control)
        assert dashboard._checkpoint == "5" and dashboard._status == "Running"
    finally:
        dashboard.on_train_end(args, state, control)
    assert dashboard._live is None


def test_window_reduction_and_canonical_wandb(tmp_path, monkeypatch):
    import sys
    from unittest.mock import Mock

    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import Trainer

    trainer = Gr00tTrainer(
        model=torch.nn.Linear(2, 2),
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        enable_training_dashboard=False,
    )
    trainer._forward_diagnostics = {
        "loss/total": torch.tensor(12.0),
        "loss/action": torch.tensor(8.0),
        "loss/contact_weighted": torch.tensor(4.0),
        "loss/zero_prediction_mse": torch.tensor(4.0),
    }
    trainer._forward_diagnostics_count = 4  # two steps, two accumulation microbatches
    trainer.args.report_to = ["wandb"]
    wandb = SimpleNamespace(run=object(), log=Mock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    logged = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, **kw: logged.append(logs))
    trainer.log({"loss": 3.0, "grad_norm": 5.0})
    assert logged[0]["loss/total"] == 3
    assert logged[0]["loss/action"] == 2
    assert logged[0]["loss/contact_fraction"] == pytest.approx(1 / 3)
    assert logged[0]["loss/action_to_zero_ratio"] == 2
    assert "grad_norm/post_clip" not in logged[0]
    assert trainer._forward_diagnostics_count == 0
    assert wandb.log.call_args.kwargs == {"commit": False}
    assert wandb.log.call_args.args[0]["loss/total"] == 3
    assert "train/global_step" in wandb.log.call_args.args[0]


def test_worker_stdout_can_be_tty_without_forcing_redirected_console():
    import os
    import pty
    import subprocess
    import sys

    master, slave = pty.openpty()
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys.stdout.isatty(), flush=True)"],
            stdout=slave,
            check=True,
        )
        assert result.returncode == 0
        assert b"True" in os.read(master, 100)
    finally:
        os.close(master)
        os.close(slave)


def test_horizon_mask_distinguished_from_dimension_padding():
    from gr00t.experiment.action_diagnostics import action_diagnostics

    prediction = torch.ones(2, 40, 132)
    mask = torch.zeros_like(prediction)
    mask[:, :30, :8] = 1
    zero = torch.tensor(0.0)
    stats = action_diagnostics(prediction, prediction, mask, zero, zero, zero, 0.1)
    assert stats["action/valid_dimension_count"] == 8
    assert stats["action/valid_timestep_ratio"] == 0.75
    assert stats["action/valid_action_ratio"] == pytest.approx(0.75 * 8 / 132)
    assert stats["action/valid_action_count"] == 2 * 30 * 8
