#!/usr/bin/env python
"""Torchrun worker: read immutable resolved YAML, then call the unchanged training API."""

from pathlib import Path
import sys

from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(path: str) -> None:
    from gr00t.configs.hydra_config import build_gr00t_config
    from gr00t.experiment.hydra_launch import resolve_resume

    cfg = OmegaConf.load(path)
    config = build_gr00t_config(cfg)
    resolve_resume(cfg, config)
    # Fetch configuration only here, never in parent/preflight. setup.py still owns loading.
    from transformers import AutoConfig

    checkpoint = AutoConfig.from_pretrained(
        config.training.start_from_checkpoint,
        trust_remote_code=config.training.transformers_trust_remote_code,
        local_files_only=config.training.transformers_local_files_only,
        cache_dir=config.training.transformers_cache_dir,
        **({"revision": config.model.model_revision} if config.model.model_revision else {}),
    )
    from gr00t.configs.hydra_config import inherit_checkpoint_architecture

    inherit_checkpoint_architecture(cfg, config, checkpoint)
    config.validate()
    from gr00t.experiment.experiment import run

    run(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: train_hydra_worker.py /absolute/path/to/resolved/config.yaml")
    main(sys.argv[1])
