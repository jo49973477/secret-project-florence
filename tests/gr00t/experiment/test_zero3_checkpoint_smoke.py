"""Exercise ZeRO-3 checkpoint save and explicit resume with a tiny model."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.timeout(600, func_only=True)
def test_zero3_checkpoint_step_2_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("ZeRO-3 smoke test requires two visible CUDA devices")

    repo_root = Path(__file__).resolve().parents[3]
    output = tmp_path / "zero3-smoke"
    ds_config = tmp_path / "zero3.json"
    ds_config.write_text(
        json.dumps(
            {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "zero_optimization": {
                    "stage": 3,
                    "stage3_gather_16bit_weights_on_model_save": False,
                },
            }
        )
    )
    runner = Path(__file__).with_name("_run_zero3_checkpoint_smoke.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root), env.get("PYTHONPATH", "")])
    )
    env.setdefault("OMP_NUM_THREADS", "1")

    def execute(*extra):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(runner),
                str(output),
                str(ds_config),
                *extra,
            ],
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=540,
        )
        assert result.returncode == 0, result.stdout
        return result.stdout

    first_log = execute()
    checkpoint = output / "checkpoint-2"
    assert (checkpoint / "trainer_state.json").is_file()
    assert list(checkpoint.rglob("*optim_states.pt")), "missing ZeRO optimizer partitions"
    assert list(checkpoint.rglob("*model_states.pt")), "missing ZeRO model partitions"
    assert list(checkpoint.glob("rng_state*.pth")), "missing RNG state"
    checkpoint_state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert checkpoint_state["global_step"] == 2
    assert [
        row["step"] for row in checkpoint_state["log_history"] if "loss" in row
    ] == [1, 2]
    assert "checkpoint-memory" in first_log
    save_deltas = re.findall(
        r"phase=optimizer-scheduler-save\.step-2[^\n]*"
        r"sampled_peak_rss_delta_bytes=(\d+)",
        first_log,
    )
    assert len(save_deltas) == 2, "expected RSS measurements from both ranks during ZeRO save"
    assert max(map(int, save_deltas)) < 512 * 1024**2, (
        f"tiny ZeRO-3 native checkpoint had a large per-rank RSS spike: {save_deltas}"
    )
    assert "internal-zero3-skipped" in first_log

    resumed_log = execute(str(checkpoint))
    assert json.loads((output / "smoke_final_state.json").read_text())["global_step"] == 3
    assert "checkpoint-memory" in resumed_log
