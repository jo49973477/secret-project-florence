"""Offline composition and tracking contracts; no training or model imports."""

import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.configs.hydra_config import build_gr00t_config, preflight
from gr00t.configs.training.training_config import TrainingConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "sidecar_wrapper", ROOT / "scripts/train_univtac_sidecar.py"
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def compose_config(experiment="univtac_full", overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name=f"experiment/{experiment}", overrides=list(overrides))


@pytest.mark.parametrize(
    "stage,name",
    [
        ("A", "sidecar-point-full"),
        ("B", "sidecar-contact-warmup"),
        ("C", "sidecar-full"),
        ("overfit5", "sidecar-full-overfit5"),
    ],
)
def test_stage_identity(stage, name, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    command = wrapper.build_command(stage, ['run_id="20261029_193000"'])
    cfg = compose_config(wrapper.PRESETS[stage]["experiment"], command[4:])
    config = build_gr00t_config(cfg)
    assert cfg.name == name
    assert cfg.output_dir == f"outputs/{name}/20261029_193000"
    assert config.training.output_dir == cfg.output_dir
    assert config.training.experiment_name is None
    assert config.training.wandb_name == f"{name}-20261029_193000"
    assert config.training.wandb_project == "florence-univtac"
    assert config.training.wandb_group == "sidecar"
    custom = wrapper.build_command(stage, ["name=custom_run", "training.max_steps=500"])
    cfg = compose_config(wrapper.PRESETS[stage]["experiment"], custom[4:])
    assert cfg.name == "custom_run"
    assert build_gr00t_config(cfg).training.max_steps == 500


@pytest.mark.parametrize(
    "hardware,gpus,batch,ga",
    [
        ("dual_3090_zero3", 2, 4, 8),
        ("quad_3090_zero3", 4, 32, 1),
    ],
)
def test_local_hardware_defaults(hardware, gpus, batch, ga, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_config(overrides=[f"hardware={hardware}"])
    config = build_gr00t_config(cfg)
    summary = preflight(cfg, config)
    assert config.training.num_gpus == gpus
    assert summary["global_batch_size"] == batch
    assert summary["per_device_batch_size"] == batch // gpus
    assert summary["gradient_accumulation_steps"] == ga
    assert summary["effective_batch_size"] == batch * ga
    cfg = compose_config(
        overrides=[
            f"hardware={hardware}",
            f"training.global_batch_size={gpus}",
            f"training.gradient_accumulation_steps={32 // gpus}",
        ]
    )
    assert preflight(cfg, build_gr00t_config(cfg))["effective_batch_size"] == 32


def test_overfit_ga_and_invalid_keys(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_config("univtac_overfit5")
    assert build_gr00t_config(cfg).training.gradient_accumulation_steps == 1
    for override, message in [
        ("training.global_batch_size=3", "divisible"),
        ("training.global_batch_size=0", "positive"),
        ("training.gradient_accumulation_steps=0", ">= 1"),
        ("+hardware.unsupported=1", "unsupported keys"),
        ("+hardware.batch_size=4", "unsupported keys"),
        ("hardware.global_batch_size=oops", "hardware"),
    ]:
        with pytest.raises(ValueError, match=message):
            build_gr00t_config(compose_config(overrides=[override]))


@pytest.mark.parametrize(
    "experiment,name,group",
    [
        ("univtac_rgb", "base-full", "base"),
        ("univtac_full", "multimodal-full", "multimodal"),
        ("univtac_overfit5", "base-overfit5", "base"),
    ],
)
def test_standard_names(experiment, name, group, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_config(experiment)
    config = build_gr00t_config(cfg)
    assert cfg.name == name
    assert config.training.wandb_group == group


@pytest.mark.parametrize(
    "rank0,enabled,group",
    [
        (True, True, "sidecar"),
        (True, True, None),
        (False, True, "sidecar"),
        (True, False, "sidecar"),
    ],
)
def test_actual_wandb_init_block(rank0, enabled, group, monkeypatch):
    # Execute the actual tracking block in isolation, avoiding experiment.py's GPU/model imports.
    tree = ast.parse((ROOT / "gr00t/experiment/experiment.py").read_text())
    run = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    block = next(
        node
        for node in run.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "config.training.use_wandb"
    )
    calls = []

    @contextmanager
    def rank_context(*, label):
        calls.append(label)
        yield rank0

    wandb = SimpleNamespace(init=Mock())
    config = SimpleNamespace(
        training=TrainingConfig(
            use_wandb=enabled,
            wandb_project="florence-univtac",
            wandb_group=group,
            wandb_name="sidecar-full-20261029_193000",
        ),
        data=SimpleNamespace(mode="train"),
    )
    monkeypatch.delenv("GR00T_HYDRA_CONFIG", raising=False)
    namespace = dict(
        config=config,
        experiment_name="20261029_193000",
        wandb=wandb,
        run_or_wait_on_rank0=rank_context,
        os=os,
        OmegaConf=OmegaConf,
    )
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), "actual_wandb_block", "exec"), namespace
    )
    if enabled and rank0:
        kwargs = wandb.init.call_args.kwargs
        assert kwargs["project"] == "florence-univtac"
        assert kwargs["name"] == "sidecar-full-20261029_193000"
        assert kwargs["tags"] == ["train"]
        assert "git_commit_hash" in kwargs["config"]
        assert kwargs.get("group") == group
        assert ("group" in kwargs) == (group is not None)
    else:
        wandb.init.assert_not_called()
    assert calls == (["wandb.init"] if enabled else [])
    assert TrainingConfig().wandb_group is None
    assert TrainingConfig().wandb_name is None


def test_future_semantic_name_and_optional_group(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_config(overrides=["name=contact-moe-full", "training.wandb_group=contact-moe"])
    assert build_gr00t_config(cfg).training.wandb_group == "contact-moe"
    assert cfg.name == "contact-moe-full"


def test_resume_optional_metadata_and_batch_safety(tmp_path, monkeypatch):
    import json

    from gr00t.experiment.hydra_launch import resolve_resume, write_snapshots

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_config(overrides=[f"output_dir={tmp_path}", 'run_id="20261029_193000"'])
    config = build_gr00t_config(cfg)
    snapshot = write_snapshots(cfg, config)
    old = OmegaConf.load(snapshot)
    del old.training.wandb_group
    # Older profiles did not carry batch defaults: resolved training values remain authoritative.
    del old.hardware.global_batch_size
    del old.hardware.gradient_accumulation_steps
    OmegaConf.save(old, snapshot)
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "rng_state.pth").touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    # Match existing resume validation's native ZeRO artifact requirements.
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"test")
    (checkpoint / "scheduler.pt").write_bytes(b"test")
    zero = checkpoint / "global_step100"
    zero.mkdir()
    (zero / "mp_rank_00_model_states.pt").write_bytes(b"test")
    (zero / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"test")
    cfg.resume.enabled = True
    cfg.resume.checkpoint_path = str(checkpoint)
    cfg.run_id = "new_execution"
    cfg.training.wandb_group = "sidecar"
    resumed = build_gr00t_config(cfg)
    assert resolve_resume(cfg, resumed) == checkpoint
    assert resumed.training.wandb_name == f"{cfg.name}-20261029_193000"
    for key, value in [("global_batch_size", 8), ("gradient_accumulation_steps", 2)]:
        previous = cfg.training[key]
        cfg.training[key] = value
        with pytest.raises(ValueError, match=f"training.{key}"):
            resolve_resume(cfg, build_gr00t_config(cfg))
        cfg.training[key] = previous
    cfg.name = "different-experiment"
    with pytest.raises(ValueError, match="original experiment name"):
        resolve_resume(cfg, build_gr00t_config(cfg))
