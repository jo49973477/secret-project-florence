"""Import-only sensor dependency checks, isolated from training process state.

Keep this module standard-library-only: do not import encoders, construct models,
read checkpoints, initialize CUDA/distributed, or change the caller's RNG state.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
import subprocess
import sys


@dataclass(frozen=True)
class Dependency:
    module: str
    package: str
    callables: tuple[str, ...] = ()


def sensor_dependencies(model) -> list[Dependency]:
    """Select requirements from enabled backends, including their bootstrap path."""
    dependencies = []
    if model.use_point_conditioning:
        backend = model.point_encoder_cfg.lower().replace("-", "_")
        if backend in {"concerto_small", "concerto_base"}:
            dependencies += [
                Dependency("spconv.pytorch", "CUDA-compatible spconv (see installation guide)"),
                Dependency("torch_scatter", "torch-scatter matching the installed PyTorch/CUDA"),
                Dependency(
                    "concerto", "official Pointcept/Concerto", ("load", "model.PointTransformerV3")
                ),
            ]
            if model.point_encoder_enable_flash is True:
                dependencies.append(Dependency("flash_attn", "existing compatible FlashAttention"))
        elif backend not in {"pointnet2", "point_transformer"}:
            raise ValueError(f"Unsupported point encoder for dependency preflight: {backend!r}")
        dependencies.append(Dependency("torch", "PyTorch from the existing GR00T environment"))
    if model.use_tactile_conditioning:
        backend = model.tactile_encoder_cfg.lower().replace("-", "_")
        if backend not in {"resnet18", "sparsh_dino_base"}:
            raise ValueError(f"Unsupported tactile encoder for dependency preflight: {backend!r}")
        # tactile_encoder.py imports these for both backends. SPARSH's ViT is
        # implemented locally; neither an external sparsh package nor timm is needed.
        dependencies += [
            Dependency("torch", "PyTorch from the existing GR00T environment"),
            Dependency("torchvision.models", "torchvision compatible with the installed PyTorch"),
            Dependency("numpy", "numpy"),
            Dependency("PIL.Image", "Pillow"),
        ]
        if backend == "sparsh_dino_base":
            dependencies += [
                Dependency("huggingface_hub", "huggingface-hub"),
                Dependency("safetensors.torch", "safetensors"),
            ]
    return list(dict.fromkeys(dependencies))


# Use real imports rather than find_spec: namespace presence does not establish
# that a compiled extension can load. No constructors or loader APIs are called.
_PROBE = r"""
import importlib
import json
import sys
errors = []
for dependency in json.loads(sys.argv[1]):
    name = dependency["module"]
    try:
        module = importlib.import_module(name)
        for path in dependency["callables"]:
            value = module
            for part in path.split("."):
                value = getattr(value, part)
            if not callable(value):
                raise ImportError(f"{name}.{path} is not callable; wrong package/API")
    except Exception as exc:
        errors.append({"module": name, "package": dependency["package"],
                       "error": f"{type(exc).__name__}: {exc}"})
print("GROOT_SENSOR_PREFLIGHT=" + json.dumps(errors), flush=True)
"""


class SensorDependencyError(RuntimeError):
    """A required sensor import failed before distributed/model initialization."""


def check_sensor_dependencies(model) -> None:
    """Probe once using the worker interpreter, preserving the parent's state.

    Fail before starting workers on missing modules, incompatible native libraries,
    wrong Concerto packages, crashed probes, or timeouts. Disabled sensors spawn
    no subprocess. There are no package installs or model/checkpoint operations.
    """
    dependencies = sensor_dependencies(model)
    if not dependencies:
        return
    guidance = (
        f"\nInterpreter: {sys.executable} (Python {sys.version.split()[0]}). "
        "Install the named dependencies in this same environment; see "
        "docs/sensor_dependencies.md. Preserve the existing PyTorch/CUDA, Transformers "
        "and FlashAttention versions. A spconv-cu126 wheel is not automatically "
        "validated for PyTorch 2.9.0+cu128."
    )
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", _PROBE, json.dumps([asdict(d) for d in dependencies])],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "SPCONV_DISABLE_JIT": "1",
                "CUMM_DISABLE_JIT": "1",
            },
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise SensorDependencyError(
            f"Sensor dependency preflight could not complete: {exc}{guidance}"
        ) from exc
    if result.returncode:
        raise SensorDependencyError(
            f"Sensor dependency import probe exited with code {result.returncode}. "
            f"Checked: {', '.join(d.module for d in dependencies)}.\n"
            f"{result.stderr[-4000:]}{guidance}"
        )
    prefix = "GROOT_SENSOR_PREFLIGHT="
    reports = [
        line[len(prefix) :] for line in result.stdout.splitlines() if line.startswith(prefix)
    ]
    try:
        errors = json.loads(reports[-1])
        if not isinstance(errors, list) or any(
            not isinstance(error, dict)
            or any(not isinstance(error.get(key), str) for key in ("module", "package", "error"))
            for error in errors
        ):
            raise ValueError("Invalid dependency report schema")
    except (IndexError, ValueError) as exc:
        raise SensorDependencyError(
            f"Sensor dependency probe returned no valid report.{guidance}"
        ) from exc
    if errors:
        failures = "\n".join(
            f"- import {error['module']} ({error['package']}): {error['error']}" for error in errors
        )
        raise SensorDependencyError(
            f"Sensor dependency preflight failed before worker launch or GR00T checkpoint loading:\n"
            f"{failures}{guidance}"
        )
    logging.info(
        "Sensor dependency preflight passed: %s", ", ".join(d.module for d in dependencies)
    )
