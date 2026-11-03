"""CPU contracts for nominal sample exposure and its logging path."""

from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.experiment.sample_exposure import SampleExposure
from gr00t.experiment.trainer import Gr00tTrainer
import numpy as np
import pytest
from transformers import Trainer, TrainerState
from transformers.integrations.integration_utils import WandbCallback, rewrite_logs


def _mixture(*shards, weights=None):
    components = []
    for index, lengths in enumerate(shards):
        component = object.__new__(ShardedSingleStepDataset)
        component.shard_lengths = np.array(lengths)
        component.dataset_path = f"/data/component-{index}"
        components.append(component)
    return SimpleNamespace(datasets=components, weights=weights or [1.0] * len(shards))


def _args(per_device=2, world_size=4, accumulation=4):
    return SimpleNamespace(
        per_device_train_batch_size=per_device,
        world_size=world_size,
        gradient_accumulation_steps=accumulation,
    )


def test_45_sample_dataset_multiple_gpus_and_accumulation():
    exposure = SampleExposure(_mixture([20, 25]), _args())
    assert exposure.total_training_samples == 45
    assert exposure.effective_global_batch_size == 32
    assert exposure.metrics(1000) == {"samples_seen": 32000, "effective_epoch": 32000 / 45}
    assert (
        SampleExposure(_mixture([20, 25]), _args(world_size=1)).metrics(1000)["samples_seen"]
        == 8000
    )
    assert (
        SampleExposure(_mixture([20, 25]), _args(accumulation=1)).metrics(1000)["samples_seen"]
        == 8000
    )


def test_multiple_shards_and_weighted_mixture_use_valid_timestep_total():
    exposure = SampleExposure(_mixture([20, 25], [3, 7, 10], weights=[0.9, 0.1]), _args())
    assert exposure.total_training_samples == 65
    assert exposure.metrics(10)["effective_epoch"] == 320 / 65
    inactive = SampleExposure(_mixture([20, 25], [3, 7, 10], weights=[1.0, 0.0]), _args())
    assert inactive.total_training_samples == 45


@pytest.mark.parametrize("shards", [[], [0], [-1]])
def test_invalid_shard_sizes_warn_and_omit_metrics(shards, caplog):
    exposure = SampleExposure(_mixture(shards), _args())
    assert exposure.metrics(100) == {}
    assert "Sample exposure metrics unavailable" in caplog.text


def test_missing_dataset_size_warns_and_omits_metrics(caplog):
    exposure = SampleExposure(SimpleNamespace(datasets=[object()], weights=[1]), _args())
    assert exposure.metrics(100) == {}
    assert "Sample exposure metrics unavailable" in caplog.text


def test_resume_continues_from_restored_step_and_rejects_changed_settings(tmp_path, caplog):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    original = SampleExposure(_mixture([20, 25]), _args())
    original.prepare(tmp_path, None, rank_zero=True)
    resumed = SampleExposure(_mixture([20, 25]), _args())
    resumed.prepare(tmp_path, checkpoint, rank_zero=True)
    assert resumed.metrics(1001)["samples_seen"] == 32032
    assert resumed.metrics(1001)["effective_epoch"] == 32032 / 45

    changed_batch = SampleExposure(_mixture([20, 25]), _args(per_device=3))
    changed_batch.prepare(tmp_path, checkpoint, rank_zero=True)
    assert changed_batch.metrics(1001) == {}
    assert (tmp_path / "sample_exposure_unavailable").exists()
    assert "settings changed on resume" in caplog.text

    changed_dataset = SampleExposure(_mixture([21, 24]), _args())
    changed_dataset.prepare(tmp_path, checkpoint, rank_zero=True)
    assert changed_dataset.metrics(1001) == {}


def test_legacy_resume_without_exposure_metadata_warns(tmp_path, caplog):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    exposure = SampleExposure(_mixture([45]), _args())
    exposure.prepare(tmp_path, checkpoint, rank_zero=True)
    assert exposure.metrics(1001) == {}
    assert "history is unavailable on resume" in caplog.text


def test_trainer_resume_logs_restored_progress_without_recounting(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    TrainerState(global_step=1000).save_to_json(str(checkpoint / "trainer_state.json"))
    exposure = SampleExposure(_mixture([45]), _args())
    exposure.prepare(tmp_path, None, rank_zero=True)
    recorded = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, start_time=None: recorded.append(logs))

    def one_restored_step(self, *, resume_from_checkpoint, **kwargs):
        assert resume_from_checkpoint == str(checkpoint)
        assert self.state.global_step == 1000
        self.is_in_train = True
        self.state.global_step = 1001
        self.log({"loss": 0.04})

    monkeypatch.setattr(Trainer, "train", one_restored_step)
    trainer = object.__new__(Gr00tTrainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), max_grad_norm=1.0, report_to=[])
    trainer.state = TrainerState()
    trainer.sample_exposure = SampleExposure(_mixture([45]), _args())
    trainer._validate_resumable_checkpoint = lambda path: None
    trainer.is_world_process_zero = lambda: True
    trainer.multimodal_monitor = None
    trainer._close_training_display = lambda: None
    trainer._forward_diagnostics_count = 0
    trainer._clip_observations = []
    trainer.train(resume_from_checkpoint=str(checkpoint))
    assert recorded[0]["samples_seen"] == 32032
    assert recorded[0]["effective_epoch"] == 32032 / 45


def test_trainer_log_history_and_existing_wandb_step_axis(monkeypatch):
    recorded = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, start_time=None: recorded.append(logs))
    trainer = object.__new__(Gr00tTrainer)
    trainer.args = SimpleNamespace(max_grad_norm=1.0, report_to=["wandb"])
    trainer.state = SimpleNamespace(global_step=1000, epoch=42)
    trainer.is_in_train = True
    trainer.sample_exposure = SampleExposure(_mixture([45]), _args())
    trainer._forward_diagnostics_count = 0
    trainer._clip_observations = []
    trainer.is_world_process_zero = lambda: True

    import wandb

    manual_log = Mock()
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", manual_log)
    trainer.log({"loss": 0.0421, "learning_rate": 1e-4})
    assert recorded[0]["samples_seen"] == 32000
    assert recorded[0]["effective_epoch"] == 32000 / 45
    assert trainer.state.epoch == 42
    namespaced = rewrite_logs(recorded[0])
    assert namespaced["train/samples_seen"] == 32000
    assert namespaced["train/effective_epoch"] == 32000 / 45
    manual_log.assert_not_called()  # HF owns these two keys; no duplicate direct log.
    wandb_callback = object.__new__(WandbCallback)
    wandb_callback._wandb = Mock()
    wandb_callback._wandb.run.summary = {}
    wandb_callback._initialized = True
    wandb_callback.on_log(
        trainer.args,
        SimpleNamespace(global_step=1000, is_world_process_zero=True),
        None,
        logs=recorded[0],
    )
    sent = wandb_callback._wandb.log.call_args.args[0]
    assert sent["train/global_step"] == 1000
    assert sent["train/samples_seen"] == 32000
    assert sent["train/effective_epoch"] == 32000 / 45

    trainer.args.report_to = []
    manual_log.reset_mock()
    trainer.state.global_step = 1001
    trainer.log({"loss": 0.04})
    assert recorded[-1]["samples_seen"] == 32032
    manual_log.assert_not_called()
