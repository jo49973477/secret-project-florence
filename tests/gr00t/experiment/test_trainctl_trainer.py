"""Tiny CPU Trainer saves verify callback timing without GR00T weights or GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.experiment.trainctl_callback import TrainctlCallback
import pytest
import torch
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, labels=None):
        output = x * self.weight
        return {"loss": ((output - labels) ** 2).mean(), "logits": output}


def trainer(tmp_path, callback, max_steps=2):
    return Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            max_steps=max_steps,
            save_steps=1,
            logging_steps=1,
            per_device_train_batch_size=1,
            save_total_limit=1,
            disable_tqdm=True,
            dataloader_pin_memory=False,
        ),
        train_dataset=[{"x": torch.ones(1), "labels": torch.zeros(1)}] * 4,
        callbacks=[callback],
    )


def test_successful_save_checkpoint_rotation_and_resume(tmp_path, monkeypatch):
    import trainctl

    tracker = Mock()
    monkeypatch.setattr(trainctl, "tracker", tracker)
    monkeypatch.setenv("RANK", "0")

    def after_save(*, path, step, metrics):
        checkpoint = Path(path)
        # Saving all native state must precede registration.
        assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == step
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            assert (checkpoint / name).is_file()

    tracker.checkpoint.side_effect = after_save
    cb = TrainctlCallback(name="tiny-original", total_steps=2)
    first = trainer(tmp_path, cb)
    first.train()
    assert first.state.global_step == 2
    assert [call.kwargs["step"] for call in tracker.checkpoint.call_args_list] == [1, 2]
    assert not (tmp_path / "checkpoint-1").exists()  # Original HF rotation.
    tracker.finish.assert_not_called()
    first.save_model()
    cb.register_checkpoint(tmp_path / "checkpoint-2", state=first.state)
    assert tracker.checkpoint.call_count == 2
    tracker.reset_mock()
    cb = TrainctlCallback(name="tiny-original", total_steps=3)
    resumed = trainer(tmp_path, cb, max_steps=3)
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert resumed.state.global_step == 3
    assert tracker.checkpoint.call_args.kwargs["step"] == 3
    assert all(call.kwargs["step"] >= 3 for call in tracker.log.call_args_list)
    tracker.fail.assert_not_called()


def test_failed_save_does_not_register_or_swallow_exception(tmp_path, monkeypatch):
    import trainctl

    tracker = Mock()
    monkeypatch.setattr(trainctl, "tracker", tracker)
    monkeypatch.setenv("RANK", "0")
    cb = TrainctlCallback(name="tiny", total_steps=2)
    instance = trainer(tmp_path, cb)

    def failed_save(*args, **kwargs):
        (tmp_path / "checkpoint-1").mkdir()
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(instance, "_save_checkpoint", failed_save)
    with pytest.raises(OSError, match="checkpoint write failed"):
        instance.train()
    tracker.checkpoint.assert_not_called()
    tracker.finish.assert_not_called()
    tracker.fail.assert_not_called()


def test_wandb_callback_receives_original_metrics_and_step(monkeypatch):
    import trainctl

    tracker = Mock()
    monkeypatch.setattr(trainctl, "tracker", tracker)
    monkeypatch.setenv("RANK", "0")
    callback = TrainctlCallback(name="existing-wandb-name", total_steps=1000)
    wandb = WandbCallback()
    wandb._wandb = Mock()
    wandb._initialized = True
    state = SimpleNamespace(is_world_process_zero=True, global_step=123)
    logs = {
        "loss": 0.5,
        "samples_seen": 3936,
        "effective_epoch": 3.936,
        "train/grad_norm_lora": 0.1,
    }
    original = dict(logs)
    # HF callbacks receive the same logs; adapter must not alter W&B's values or step.
    callback.on_log(None, state, None, logs=logs)
    wandb.on_log(None, state, None, logs=logs)
    assert logs == original
    payload = wandb._wandb.log.call_args.args[0]
    assert payload["train/global_step"] == 123
    assert payload["train/loss"] == tracker.log.call_args.kwargs["metrics"]["train/loss"]
    assert payload["train/samples_seen"] == 3936
