"""Test reporting and parent orchestration with CPU fixtures and fake processes."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

from gr00t.experiment import hydra_launch
from gr00t.experiment.optimizer_artifacts import EffectiveOptimizerCallback, describe_groups
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def config(output):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="experiment/univtac_overfit5", overrides=[f"output_dir={output}"]
        )


def test_optimizer_artifact_reports_actual_restored_lr(tmp_path):
    params = [torch.nn.Parameter(torch.zeros(2, 3))]
    groups = [{"name": "Action Head decay", "params": params, "lr": 1e-4, "weight_decay": 0.01}]
    constructed = describe_groups(groups)
    params[0].ds_numel = 99
    groups[0]["lr"] = 4e-5
    callback = EffectiveOptimizerCallback(SimpleNamespace(optimizer_group_manifest=constructed))
    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(is_world_process_zero=True, global_step=100),
        None,
        optimizer=SimpleNamespace(param_groups=groups),
    )
    artifact = json.loads((tmp_path / "effective_optimizers.json").read_text())
    assert artifact["constructed_parameter_groups"][0]["trainable_parameters"] == 6
    assert artifact["active_groups_after_state_restore"][0]["trainable_parameters"] == 99
    assert artifact["active_groups_after_state_restore"][0]["learning_rate"] == 4e-5
    assert artifact["global_step"] == 100
    assert artifact["constructed_parameter_groups"][0]["learning_rate"] == 1e-4


def test_nonzero_rank_does_not_write(tmp_path):
    callback = EffectiveOptimizerCallback(SimpleNamespace())
    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(is_world_process_zero=False),
        None,
    )
    assert not (tmp_path / "effective_optimizers.json").exists()


def dataset(path, ids):
    (path / "meta").mkdir(parents=True)
    (path / "meta/info.json").write_text("{}")
    (path / "meta/modality.json").write_text("{}")
    (path / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i}) + "\n" for i in ids)
    )


def test_subset_assertion(tmp_path):
    cfg = config(tmp_path / "run")
    path = tmp_path / "dataset"
    dataset(path, [0, 1, 2, 3, 4])
    cfg.data.datasets[0].dataset_paths = [str(path)]
    hydra_launch.validate_dataset_inputs(cfg)
    cfg.data.expected_episode_ids = [0, 1]
    with pytest.raises(ValueError, match="independent subset"):
        hydra_launch.validate_dataset_inputs(cfg)


def test_launcher_freezes_config_and_propagates_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = config(tmp_path / "run")
    path = tmp_path / "dataset"
    dataset(path, [0, 1, 2, 3, 4])
    cfg.data.datasets[0].dataset_paths = [str(path)]
    calls = []

    class FakeProcess:
        pid = 12345
        stdout = io.StringIO("fake training output\n")

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 7

        def poll(self):
            return 7

    monkeypatch.setattr(hydra_launch.subprocess, "Popen", FakeProcess)
    # Metadata is independently read-only; keep it independent of Popen stub.
    monkeypatch.setattr(
        hydra_launch.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="test-commit")
    )
    assert hydra_launch.launch(cfg) == 7
    command, options = calls[0]
    assert command[-1].endswith(".hydra/config.yaml")
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert options["start_new_session"]
    saved = OmegaConf.load(command[-1])
    assert not OmegaConf.is_interpolation(saved, "output_dir")
    assert (tmp_path / "run/logs/train.log").read_text() == "fake training output\n"
    with pytest.raises(ValueError, match="empty output_dir"):
        hydra_launch.launch(cfg)


def workflow():
    spec = importlib.util.spec_from_file_location(
        "overfit_workflow_test", ROOT / "scripts/univtac_overfit_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_refuses_existing_or_nested_dataset(tmp_path):
    module = workflow()
    with pytest.raises(ValueError, match="already exists"):
        module.prepare(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="separate"):
        module.prepare(tmp_path, tmp_path / "nested")


def test_prepare_preserves_source_and_selects_five(tmp_path, monkeypatch):
    module = workflow()
    source, target = tmp_path / "source", tmp_path / "subset"
    dataset(source, list(range(7)))
    (source / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "length": 10}) + "\n" for i in range(7))
    )
    (source / "meta/stats.json").write_text('{"source": true}')
    (source / "data/chunk-000").mkdir(parents=True)
    for i in range(7):
        (source / f"data/chunk-000/episode_{i:06d}.parquet").write_text("fixture")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    original = (source / "meta/episodes.jsonl").read_bytes()
    module.prepare(source, target)
    assert (source / "meta/episodes.jsonl").read_bytes() == original
    assert len(list((target / "data").rglob("*.parquet"))) == 5
    assert all(p.is_symlink() for p in (target / "data").rglob("*.parquet"))
    assert not (target / "meta/stats.json").exists()
    assert "--dataset-path" in calls[0]
    assert str(target) in calls[0]
    assert json.loads((target / "meta/info.json").read_text())["total_frames"] == 50
