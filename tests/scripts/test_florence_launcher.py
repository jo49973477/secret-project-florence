"""Shell-only tests for the Florence A-D launcher; never starts training."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "train_florence_ablations.sh"


@pytest.mark.parametrize(
    "mode,config",
    [
        ("A", "temporal_sidecar_ditlora-overfit5"),
        ("B", "moe_sidecar_ditlora-overfit5"),
        ("C", "temporal_sidecar_ditfull-overfit5"),
        ("D", "moe_sidecar_ditfull-overfit5"),
    ],
)
def test_launcher_selects_correct_config_without_training(tmp_path, mode, config):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf "<%s>\\n" "$@"\n')
    fake_uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"}
    result = subprocess.run(
        ["bash", str(LAUNCHER), mode, "dry_run=true", "training.max_steps=200"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"<experiment/{config}>" in result.stdout
    assert "<dry_run=true>" in result.stdout
    assert "<training.max_steps=200>" in result.stdout


def test_launcher_rejects_unknown_mode(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), "unknown"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Unknown or missing experiment mode" in result.stderr
