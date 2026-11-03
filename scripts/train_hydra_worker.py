#!/usr/bin/env python
"""Torchrun worker: read immutable resolved YAML, then call the unchanged training API."""

from pathlib import Path
import sys

from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(path: str) -> None:
    import os

    if os.environ.get("GR00T_LIVE_CONSOLE") == "1":
        from gr00t.experiment.console_logging import DetailedConsoleStream

        log_dir = Path(os.environ["GR00T_OUTPUT_DIR"]) / "logs"
        detail = (log_dir / f"rank-{os.environ.get('RANK', '0')}-full.log").open("a", buffering=1)
        sys.stdout = DetailedConsoleStream(
            sys.stdout, detail, quiet=os.environ.get("RANK", "0") != "0"
        )
        sys.stderr = DetailedConsoleStream(sys.stderr, detail)
    from gr00t.configs.hydra_config import build_gr00t_config
    from gr00t.experiment.hydra_launch import resolve_resume

    cfg = OmegaConf.load(path)
    config = build_gr00t_config(cfg)
    resolve_resume(cfg, config)
    # Fetch configuration only here, never in parent/preflight. setup.py still owns loading.
    # Do not depend on the model module's import-time AutoConfig registration.
    # Architecture inspection needs config.json only, not model/encoder imports.
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    checkpoint = Gr00tN1d7Config.from_pretrained(
        config.training.start_from_checkpoint,
        local_files_only=config.training.transformers_local_files_only,
        cache_dir=config.training.transformers_cache_dir,
        **({"revision": config.model.model_revision} if config.model.model_revision else {}),
    )
    if checkpoint.model_type != Gr00tN1d7Config.model_type:
        raise ValueError(
            f"Expected checkpoint model_type={Gr00tN1d7Config.model_type!r}, "
            f"got {checkpoint.model_type!r}"
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
