"""CPU-only contract tests for the public tracker adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.experiment.sample_exposure import SampleExposure
from gr00t.experiment.trainctl_callback import TrainctlCallback, numeric_metrics
import pytest
import torch


def state(step=123, rank_zero=True):
    return SimpleNamespace(global_step=step, is_world_process_zero=rank_zero)


@pytest.fixture
def callback(monkeypatch):
    import trainctl

    tracker = Mock()
    monkeypatch.setattr(trainctl, "tracker", tracker)
    monkeypatch.setenv("RANK", "0")
    cb = TrainctlCallback(name="experiment-run-id", total_steps=2000)
    return cb, tracker


def test_initialization_and_no_manual_lifecycle(callback):
    cb, tracker = callback
    cb.on_train_begin(None, state(), None)
    tracker.init.assert_called_once_with(name="experiment-run-id", total_steps=2000)
    assert "train/samples_seen" in tracker.define_metrics.call_args.kwargs["definitions"]
    cb.on_train_end(None, state(), None)
    tracker.start.assert_not_called()
    tracker.finish.assert_not_called()
    tracker.fail.assert_not_called()


@pytest.mark.parametrize("rank_zero,rank", [(False, "0"), (True, "1"), (False, "1")])
def test_only_global_rank_zero(callback, monkeypatch, tmp_path, rank_zero, rank):
    cb, tracker = callback
    monkeypatch.setenv("RANK", rank)
    cb.on_train_begin(None, state(rank_zero=rank_zero), None)
    cb.on_log(None, state(rank_zero=rank_zero), None, logs={"loss": 0.5})
    cb.register_checkpoint(tmp_path, state=state(rank_zero=rank_zero))
    assert not tracker.mock_calls


def test_original_global_step_exposure_and_logs_unchanged(callback):
    cb, tracker = callback
    logs = {"loss": 0.5, "learning_rate": 1e-4, "samples_seen": 3936, "effective_epoch": 1.2}
    original = dict(logs)
    cb.on_log(None, state(), None, logs=logs)
    tracker.log.assert_called_once_with(
        step=123,
        metrics={
            "train/loss": 0.5,
            "train/learning_rate": 1e-4,
            "train/samples_seen": 3936,
            "train/effective_epoch": 1.2,
        },
    )
    assert logs == original  # Other callbacks, including W&B, receive the original mapping.
    tracker.notify.assert_not_called()


def test_metric_filtering_and_namespaces():
    assert numeric_metrics(
        {
            "loss": float("nan"),
            "missing": None,
            "bad": "value",
            "bool": True,
            "infinite": float("inf"),
            "tensor": torch.tensor(1.0),
            "grad_norm": 2.0,
            "train/grad_norm_global": 3.0,
            "loss/action": 0.2,
            "train/action_loss": 0.1,
            "grad_norm/dit": 0.4,
            "grad_norm/lora": 0.5,
            "grad_norm/sidecar": 0.6,
            "grad_norm/temporal": 0.7,
            "grad_norm/moe": 0.8,
            "system/gpu_memory_allocated": 1.1,
            "system/steps_per_second": 2.2,
            "eval_loss": 0.3,
            "train/total_loss": 0.9,
        }
    ) == {
        "train/grad_norm_global": 3.0,
        "train/action_loss": 0.1,
        "train/grad_norm_dit": 0.4,
        "train/grad_norm_lora": 0.5,
        "train/grad_norm_sidecar": 0.6,
        "train/grad_norm_temporal": 0.7,
        "train/grad_norm_moe": 0.8,
        "system/gpu_memory_allocated": 1.1,
        "system/steps_per_second": 2.2,
        "eval/loss": 0.3,
        "train/total_loss": 0.9,
    }


def test_missing_logs(callback):
    cb, tracker = callback
    cb.on_log(None, state(), None, logs=None)
    cb.on_log(None, state(), None, logs={"loss": None})
    tracker.log.assert_not_called()


def test_completed_save_and_final_deduplication(callback, tmp_path):
    cb, tracker = callback
    args = SimpleNamespace(output_dir=str(tmp_path))
    cb.on_save(args, state(), None)  # Does not announce a nonexistent checkpoint.
    tracker.checkpoint.assert_not_called()
    path = tmp_path / "checkpoint-123"
    path.mkdir()
    cb.on_log(args, state(), None, logs={"loss": 0.1, "learning_rate": 1e-4})
    cb.on_save(args, state(), None)
    cb.register_checkpoint(path / ".." / path.name, state=state())
    tracker.checkpoint.assert_called_once_with(
        path=str(path), step=123, metrics={"train/loss": 0.1, "train/learning_rate": 1e-4}
    )
    assert "eval/loss" not in tracker.checkpoint.call_args.kwargs["metrics"]
    cb.register_checkpoint(tmp_path, state=state())  # Separate final non-ZeRO artifact.
    assert tracker.checkpoint.call_count == 2


def test_checkpoint_reuses_existing_exposure(callback, tmp_path):
    cb, tracker = callback
    component = SimpleNamespace()  # Use the existing SampleExposure implementation.

    class Shards:
        dataset_path = "fixture"

        def __len__(self):
            return 1

        def get_shard_length(self, i):
            return 1000

    component.datasets, component.weights = [Shards()], [1]
    cb.sample_exposure = SampleExposure(
        component,
        SimpleNamespace(
            per_device_train_batch_size=2,
            world_size=2,
            gradient_accumulation_steps=8,
        ),
    )
    cb.register_checkpoint(tmp_path, state=state())
    metrics = tracker.checkpoint.call_args.kwargs["metrics"]
    assert metrics["train/samples_seen"] == 3936
    assert metrics["train/effective_epoch"] == 3.936
    cb.sample_exposure.enabled = False
    new = tmp_path / "checkpoint-124"
    new.mkdir()
    cb.register_checkpoint(new, state=state(124))
    assert "train/samples_seen" not in tracker.checkpoint.call_args.kwargs["metrics"]


def test_checkpoint_opt_out(callback, tmp_path):
    cb, tracker = callback
    cb.notify_checkpoints = False
    cb.register_checkpoint(tmp_path, state=state())
    tracker.checkpoint.assert_not_called()


def test_transport_errors_do_not_escape_or_leak(callback, tmp_path, caplog):
    cb, tracker = callback
    for method in ("init", "define_metrics", "log", "checkpoint"):
        getattr(tracker, method).side_effect = RuntimeError("private-webhook-credential")
    cb.on_train_begin(None, state(), None)
    cb.on_log(None, state(), None, logs={"loss": 0.5})
    cb.register_checkpoint(tmp_path, state=state())
    assert "private-webhook" not in caplog.text
    assert caplog.text.count("training will continue") == 1
