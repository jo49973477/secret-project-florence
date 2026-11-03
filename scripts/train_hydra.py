#!/usr/bin/env python
"""Compose once, validate, then launch GR00T workers. Use dry_run=true for preflight."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/univtac_full")
def main(cfg: DictConfig) -> None:
    from gr00t.experiment.hydra_launch import launch

    code = launch(cfg)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
