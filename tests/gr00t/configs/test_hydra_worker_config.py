"""Config-only worker startup regression; no weights, network, GPUs or process groups."""

import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[3]


def snapshot(tmp_path, model_type="Gr00tN1d7"):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    # Deliberately contains configuration only: there are no parameter files to load.
    payload = Gr00tN1d7Config(action_horizon=24, max_action_dim=132, max_state_dim=64).to_dict()
    payload["model_type"] = model_type
    (checkpoint / "config.json").write_text(json.dumps(payload))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="experiment/univtac_overfit5",
            overrides=[
                "model=gr00t_sidecar",
                "hardware=quad_3090_zero3",
                "data=univtac_multimodal",
                "data.modalities.action.horizon=24",
                "model.checkpoint_expectations.action_horizon=24",
                "model.freeze_contact_teacher=false",
                f"model.base_model_path={checkpoint}",
                "name=worker-config-regression",
                'run_id="20261030_120000"',
            ],
        )
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    return checkpoint, path


@pytest.mark.parametrize("rank", range(4))
def test_fresh_worker_without_auto_registration(tmp_path, rank):
    checkpoint, path = snapshot(tmp_path)
    code = r"""
import importlib.util
import json
import sys
from types import ModuleType

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

# Reproduce the failing lookup before testing the config-only worker path.
assert "Gr00tN1d7" not in CONFIG_MAPPING
try:
    AutoConfig.from_pretrained(sys.argv[2], local_files_only=True)
except ValueError as exc:
    assert "Gr00tN1d7" in str(exc)
    assert isinstance(exc.__cause__ or exc.__context__, KeyError)
else:
    raise AssertionError("Fresh process unexpectedly registered model type")

def forbidden(*args, **kwargs):
    raise AssertionError("Architecture inspection attempted to load weights")
AutoModel.from_pretrained = forbidden
# Loading the heavy model implementation must not be necessary for inspection.
sys.modules["gr00t.model.gr00t_n1d7.gr00t_n1d7"] = None
runtime = ModuleType("gr00t.experiment.experiment")
seen = []
def run(config):
    assert isinstance(config.model, Gr00tN1d7Config)
    assert config.model.action_horizon == 24
    assert config.model.max_action_dim == 132
    assert config.model.max_state_dim == 64
    assert config.model.dit_type == "sidecar_conditioned_dit"
    assert config.model.freeze_contact_teacher is False
    assert config.training.num_gpus == 4
    assert config.training.learning_rate == 0.0001
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.max_grad_norm == 1.0
    assert config.training.start_from_checkpoint == sys.argv[2]
    seen.append(config)
runtime.run = run
sys.modules[runtime.__name__] = runtime
spec = importlib.util.spec_from_file_location("worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
assert sys.path[0] == sys.argv[4]
assert __import__("gr00t").__path__[0] == sys.argv[4] + "/gr00t"
worker.main(sys.argv[3])
assert len(seen) == 1
assert "Gr00tN1d7" not in CONFIG_MAPPING  # Fix does not mutate the global registry.
direct = Gr00tN1d7Config.from_pretrained(sys.argv[2], local_files_only=True)
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
registered = AutoConfig.from_pretrained(sys.argv[2], local_files_only=True)
differences = {k: (direct.to_dict().get(k), registered.to_dict().get(k))
               for k in direct.to_dict().keys() | registered.to_dict().keys()
               if direct.to_dict().get(k) != registered.to_dict().get(k)}
# AutoConfig adds source-path bookkeeping that typed config inspection does not need.
assert set(differences) <= {"_name_or_path"}, differences
assert registered.to_filtered_dict(False) == direct.to_filtered_dict(False)
print("CONFIG_ONLY_STARTUP_OK", flush=True)
"""
    env = {
        **os.environ,
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("GR00T_LIVE_CONSOLE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT / "scripts/train_hydra_worker.py"),
            str(checkpoint),
            str(path),
            str(ROOT),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONFIG_ONLY_STARTUP_OK" in result.stdout
    assert sorted(p.name for p in checkpoint.iterdir()) == ["config.json"]


def test_direct_configuration_roundtrip_and_wrong_type(tmp_path, monkeypatch):
    checkpoint, path = snapshot(tmp_path, model_type="unexpected_architecture")
    config = Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True)
    assert config.model_type == "unexpected_architecture"
    assert config.action_horizon == 24

    import importlib.util

    spec = importlib.util.spec_from_file_location("worker", ROOT / "scripts/train_hydra_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GR00T_LIVE_CONSOLE", raising=False)
    with pytest.raises(ValueError, match="Expected checkpoint model_type='Gr00tN1d7'"):
        worker.main(str(path))
