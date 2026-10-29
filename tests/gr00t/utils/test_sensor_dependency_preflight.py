"""Import-only tests: fake optional packages, no model weights or CUDA calls."""

import os
import random
import subprocess
import sys
from types import SimpleNamespace

from gr00t.utils.sensor_dependency_preflight import (
    SensorDependencyError,
    check_sensor_dependencies,
    sensor_dependencies,
)
import pytest


def model(**overrides):
    return SimpleNamespace(
        **{
            "use_point_conditioning": True,
            "point_encoder_cfg": "concerto_small",
            "point_encoder_enable_flash": False,
            "use_tactile_conditioning": False,
            "tactile_encoder_cfg": "sparsh_dino_base",
            **overrides,
        }
    )


def modules(config):
    return {dependency.module for dependency in sensor_dependencies(config)}


def fake_packages(tmp_path, monkeypatch, *, broken=None, missing=None):
    """Exercise the actual child importer deterministically on CPU-only CI."""
    sources = {
        "concerto/__init__.py": "from types import SimpleNamespace\ndef forbidden(*args, **kwargs):\n    raise AssertionError('model construction or checkpoint loading attempted')\nload = forbidden\nmodel = SimpleNamespace(PointTransformerV3=forbidden)\n",
        "spconv/__init__.py": "",
        "spconv/pytorch.py": "",
        "torch_scatter.py": "",
        "torch.py": "",
        "torchvision/__init__.py": "",
        "torchvision/models.py": "",
        "numpy.py": "",
        "PIL/__init__.py": "",
        "PIL/Image.py": "",
        "huggingface_hub.py": "",
        "safetensors/__init__.py": "",
        "safetensors/torch.py": "",
    }
    if broken:
        sources.update(broken)
    if missing:
        sources[missing] = (
            "raise ModuleNotFoundError(\"No module named 'deliberately_missing_dependency'\")\n"
        )
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def test_disabled_sensors_do_not_probe(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled sensors must not launch a probe")

    monkeypatch.setattr(subprocess, "run", forbidden)
    check_sensor_dependencies(model(use_point_conditioning=False))


@pytest.mark.parametrize("backend", ["concerto_small", "concerto-base"])
def test_concerto_requirements(backend):
    assert modules(model(point_encoder_cfg=backend)) == {
        "concerto",
        "spconv.pytorch",
        "torch_scatter",
        "torch",
    }
    assert "flash_attn" in modules(model(point_encoder_enable_flash=True))


@pytest.mark.parametrize("backend", ["pointnet2", "point-transformer"])
def test_lightweight_point_backend_needs_no_concerto(backend):
    assert modules(model(point_encoder_cfg=backend)) == {"torch"}


@pytest.mark.parametrize("backend", ["resnet18", "sparsh-dino-base"])
def test_tactile_requirements(backend):
    requirements = modules(
        model(
            use_point_conditioning=False, use_tactile_conditioning=True, tactile_encoder_cfg=backend
        )
    )
    assert {"torch", "torchvision.models", "numpy", "PIL.Image"} <= requirements
    assert not requirements & {"sparsh", "timm", "concerto", "spconv.pytorch", "torch_scatter"}
    assert ("safetensors.torch" in requirements) == (backend != "resnet18")


@pytest.mark.parametrize("point,tactile", [(True, False), (False, True), (True, True)])
def test_installed_imports_pass_without_parent_state_changes(tmp_path, monkeypatch, point, tactile):
    fake_packages(tmp_path, monkeypatch)
    import torch

    before = set(sys.modules)
    torch_rng = torch.random.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    distributed_initialized = torch.distributed.is_initialized()
    rng = random.getstate()
    env = dict(os.environ)
    check_sensor_dependencies(model(use_point_conditioning=point, use_tactile_conditioning=tactile))
    assert set(sys.modules) == before
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert torch.cuda.is_initialized() == cuda_initialized
    assert torch.distributed.is_initialized() == distributed_initialized
    assert random.getstate() == rng
    assert dict(os.environ) == env
    assert not list(tmp_path.rglob("__pycache__"))


@pytest.mark.parametrize(
    "file,module",
    [
        ("concerto/__init__.py", "concerto"),
        ("spconv/pytorch.py", "spconv.pytorch"),
        ("torch_scatter.py", "torch_scatter"),
        ("torchvision/models.py", "torchvision.models"),
        ("PIL/Image.py", "PIL.Image"),
        ("safetensors/torch.py", "safetensors.torch"),
    ],
)
def test_missing_dependency_is_actionable(tmp_path, monkeypatch, file, module):
    fake_packages(tmp_path, monkeypatch, missing=file)
    with pytest.raises(SensorDependencyError) as exc:
        check_sensor_dependencies(model(use_tactile_conditioning=True))
    assert f"import {module}" in str(exc.value)
    assert "deliberately_missing_dependency" in str(exc.value)
    assert "docs/sensor_dependencies.md" in str(exc.value)
    assert sys.executable in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        "ImportError('undefined symbol: torch_abi')",
        "OSError('libcudart.so not found')",
        "RuntimeError('operator torchvision::nms does not exist')",
    ],
)
def test_binary_import_failure_is_not_treated_as_installed(tmp_path, monkeypatch, error):
    fake_packages(tmp_path, monkeypatch, broken={"spconv/pytorch.py": f"raise {error}\n"})
    with pytest.raises(SensorDependencyError, match="spconv.pytorch"):
        check_sensor_dependencies(model())


def test_wrong_concerto_package_rejected(tmp_path, monkeypatch):
    fake_packages(tmp_path, monkeypatch, broken={"concerto/__init__.py": ""})
    with pytest.raises(SensorDependencyError, match="official Pointcept/Concerto"):
        check_sensor_dependencies(model())


@pytest.mark.parametrize("mode", ["timeout", "crash", "invalid_report", "invalid_schema"])
def test_probe_failures_fail_closed(monkeypatch, mode):
    def run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired("import probe", 60)
        return subprocess.CompletedProcess(
            [],
            -11 if mode == "crash" else 0,
            "GROOT_SENSOR_PREFLIGHT=null" if mode == "invalid_schema" else "invalid",
            "native failure",
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SensorDependencyError):
        check_sensor_dependencies(model())


def test_parent_failure_precedes_launch_or_checkpoint_access(monkeypatch):
    from gr00t.experiment import hydra_launch

    config = SimpleNamespace(model=model())
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(hydra_launch, "build_gr00t_config", lambda cfg: config)

    def missing(model_config):
        assert model_config is config.model
        raise SensorDependencyError("missing concerto")

    def forbidden(*args, **kwargs):
        raise AssertionError("reached launch/checkpoint/dataset/output before dependency check")

    monkeypatch.setattr(hydra_launch, "check_sensor_dependencies", missing)
    for name in ("validate_dataset_inputs", "resolve_resume", "write_snapshots", "worker_command"):
        monkeypatch.setattr(hydra_launch, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(SensorDependencyError, match="missing concerto"):
        hydra_launch.launch(SimpleNamespace(dry_run=False))


def test_parent_success_checks_once_before_launch(monkeypatch):
    from gr00t.experiment import hydra_launch

    config = SimpleNamespace(model=model())
    calls = []
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(hydra_launch, "build_gr00t_config", lambda cfg: config)
    monkeypatch.setattr(hydra_launch, "check_sensor_dependencies", lambda cfg: calls.append(cfg))

    def reached_dataset(cfg):
        assert calls == [config.model]
        raise LookupError("dependencies checked before proceeding")

    monkeypatch.setattr(hydra_launch, "validate_dataset_inputs", reached_dataset)
    with pytest.raises(LookupError, match="dependencies checked"):
        hydra_launch.launch(SimpleNamespace(dry_run=False))


def test_probe_environment_is_offline_and_disables_jit(monkeypatch):
    def run(command, **kwargs):
        assert command[0] == sys.executable and command[1] == "-B"
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "SPCONV_DISABLE_JIT",
            "CUMM_DISABLE_JIT",
        ):
            assert kwargs["env"][name] == "1"
            assert os.environ.get(name) != "1"
        return subprocess.CompletedProcess(command, 0, "GROOT_SENSOR_PREFLIGHT=[]\n", "")

    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "SPCONV_DISABLE_JIT",
        "CUMM_DISABLE_JIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    check_sensor_dependencies(model())
