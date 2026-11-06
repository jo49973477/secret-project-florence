#!/usr/bin/env python
"""Inspect notification setup offline, or explicitly send a tiny supervised smoke run.

No model, dataset, CUDA context, or Trainer is constructed. --send is required
for network notifications. Smoke state is isolated from real training runs.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr00t.experiment.trainctl_launch import check_trainctl, discord_environment, trainctl_run  # noqa: E402


def main(argv=None):
    defaults = OmegaConf.load(ROOT / "configs/trainctl/default.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--send", action="store_true", help="Actually send lifecycle/checkpoint notifications"
    )
    parser.add_argument(
        "--fail", action="store_true", help="Simulate a worker exception after checkpoint saving"
    )
    parser.add_argument(
        "--message-mode",
        default=defaults.message_mode,
        choices=["standard", "sopia", "koyori", "roboco", "neuro"],
    )
    args = parser.parse_args(argv)
    check_trainctl()
    env = dict(os.environ)
    if not args.send:
        source = Path(env.get("TRAINCTL_CONFIG", "~/.config/trainctl/config.toml")).expanduser()
        settings = tomllib.loads(source.read_text()) if source.exists() else {}
        _, status = discord_environment(env, settings)
        print(
            f"message mode: {args.message_mode}\nDiscord: {status}\nOffline inspection only; no notifications sent."
        )
        return 0
    output = Path(tempfile.mkdtemp(prefix="wah-trainctl-smoke-"))
    (output / "logs").mkdir()
    # Explicit env override isolates SQLite even if the user's TOML has a custom database.
    env["TRAINCTL_DATABASE"] = str(output / "trainctl.db")
    cfg = OmegaConf.create(
        {
            "name": "wah-notification-smoke",
            "run_id": output.name,
            "training": {"max_steps": 1, "enable_training_dashboard": False},
            "trainctl": {
                "enabled": True,
                "message_mode": args.message_mode,
                "notify_checkpoints": True,
            },
        }
    )
    worker = output / "worker.py"
    worker.write_text(
        """from pathlib import Path
from trainctl import tracker
tracker.init(total_steps=1)
tracker.log(step=1, metrics={"train/loss": 0.5})
p = Path(__file__).parent / "checkpoint-1"
p.mkdir()
(p / "smoke.txt").write_text("completed smoke artifact; not a training checkpoint")
tracker.checkpoint(p, step=1, metrics={"train/loss": 0.5})
"""
        + ("raise RuntimeError('Intentional notification smoke failure')\n" if args.fail else "")
    )
    print(f"Isolated smoke artifacts: {output}", flush=True)
    with trainctl_run(cfg, output, [sys.executable, str(worker)], env) as (command, child_env):
        return subprocess.run(command, env=child_env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
