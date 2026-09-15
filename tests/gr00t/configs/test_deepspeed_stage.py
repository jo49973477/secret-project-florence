"""CPU coverage of ZeRO selection without loading models or training dependencies."""

import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.training.training_config import TrainingConfig
import pytest
import tyro


ROOT = Path(__file__).resolve().parents[3]
REQUIRED = dict(
    base_model_path="local-model", dataset_path="local-data", embodiment_tag="NEW_EMBODIMENT"
)


def test_defaults():
    assert TrainingConfig().deepspeed_stage == 3
    assert FinetuneConfig(**REQUIRED).deepspeed_stage == 3


@pytest.mark.parametrize("stage", [2, 3])
def test_explicit_stage(stage):
    assert FinetuneConfig(**REQUIRED, deepspeed_stage=stage).deepspeed_stage == stage


@pytest.mark.parametrize("stage", [0, 1, 4])
def test_invalid_stage(stage):
    with pytest.raises(ValueError, match=f"deepspeed_stage must be 2 or 3, got {stage}"):
        FinetuneConfig(**REQUIRED, deepspeed_stage=stage)


def source_tree(relative):
    path = ROOT / relative
    return path, ast.parse(path.read_text())


def execute_nodes(path, nodes, namespace):
    # Execute the production selection/forwarding code without importing the GPU stack.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("stage", [None, 2, 3])
def test_cli_and_launcher_forwarding(stage):
    args = [arg for key, value in REQUIRED.items() for arg in ("--" + key.replace("_", "-"), value)]
    if stage is not None:
        args += ["--deepspeed-stage", str(stage)]
    ft_config = tyro.cli(FinetuneConfig, args=args)
    config = SimpleNamespace(training=TrainingConfig(deepspeed_stage=2))
    path, tree = source_tree("gr00t/experiment/launch_finetune.py")
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "config.training.deepspeed_stage" for target in node.targets)
    ]
    assert len(assignments) == 1
    execute_nodes(path, assignments, dict(config=config, ft_config=ft_config))
    assert config.training.deepspeed_stage == (3 if stage is None else stage)


@pytest.mark.parametrize("stage", [2, 3])
def test_existing_json_selection(stage):
    path, tree = source_tree("gr00t/configs/base_config.py")
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_deepspeed_config"
    )
    namespace = dict(__file__=str(path), Path=Path, json=json)
    execute_nodes(path, [method], namespace)
    actual = namespace["get_deepspeed_config"](
        SimpleNamespace(training=TrainingConfig(deepspeed_stage=stage))
    )
    expected = json.loads((ROOT / f"gr00t/configs/deepspeed/zero{stage}_config.json").read_text())
    assert actual == expected
    assert actual["zero_optimization"]["stage"] == stage


@pytest.mark.parametrize("num_gpus,use_ddp", [(1, False), (1, True), (2, False), (2, True)])
def test_deepspeed_activation_unchanged(num_gpus, use_ddp):
    path, tree = source_tree("gr00t/experiment/experiment.py")
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "config.training.num_gpus > 1 and (not config.training.use_ddp)"
    )
    sentinel = object()
    config = SimpleNamespace(
        training=TrainingConfig(num_gpus=num_gpus, use_ddp=use_ddp),
        get_deepspeed_config=lambda: sentinel,
    )
    namespace = dict(config=config)
    execute_nodes(path, [branch], namespace)
    assert namespace["deepspeed_config"] is (sentinel if num_gpus > 1 and not use_ddp else None)


def test_shell_syntax():
    scripts = subprocess.check_output(
        ["rg", "--files", "--glob", "*.sh"], cwd=ROOT, text=True
    ).splitlines()
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", script], cwd=ROOT, check=True)


@pytest.mark.parametrize("stage", [None, "2", "3"])
def test_shell_environment_forwarding(tmp_path, stage):
    # Replace the training executable so this smoke test cannot download/run a model.
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        NUM_GPUS="1",
        RESUME="0",
        RESUME_FROM_CHECKPOINT="",
        SAVE_ONLY_MODEL="0",
    )
    env.pop("DEEPSPEED_STAGE", None)
    if stage is not None:
        env["DEEPSPEED_STAGE"] = stage
    result = subprocess.run(
        [
            "bash",
            "examples/finetune.sh",
            "--base-model-path",
            "local-model",
            "--dataset-path",
            "local-data",
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--deepspeed-stage") + 1] == (stage or "3")
