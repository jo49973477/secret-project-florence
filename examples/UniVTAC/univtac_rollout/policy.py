"""UniVTAC-to-GR00T deployment policy adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from .client import MinimalPolicyClient
from .core import extract_action_chunk, format_univtac_observation


def _config_field(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def validate_rgb_checkpoint_contract(configs: dict[str, Any], execution_horizon: int) -> None:
    required = {"video", "state", "action", "language"}
    missing = required.difference(configs)
    if missing:
        raise ValueError(f"GR00T checkpoint modality config is missing: {sorted(missing)}")
    unsupported = {"tactile", "pointcloud"}.intersection(configs)
    if unsupported:
        raise ValueError(
            "This RGB rollout adapter cannot satisfy checkpoint modalities "
            f"{sorted(unsupported)}. Start a server with the RGB-only UniVTAC checkpoint."
        )

    expected_keys = {
        "video": ["head", "wrist"],
        "state": ["joint"],
        "action": ["joint"],
        "language": ["annotation.human.task_description"],
    }
    for modality, expected in expected_keys.items():
        actual = _config_field(configs[modality], "modality_keys")
        if actual != expected:
            raise ValueError(f"Checkpoint {modality} keys must be {expected}; got {actual}.")

    for modality in ("video", "state", "language"):
        deltas = _config_field(configs[modality], "delta_indices")
        if deltas is None or len(deltas) != 1:
            raise ValueError(
                f"Checkpoint {modality} horizon must be 1 for the UniVTAC adapter; got {deltas}."
            )
    action_deltas = _config_field(configs["action"], "delta_indices")
    if action_deltas is None or len(action_deltas) < execution_horizon:
        raise ValueError(
            f"execution_horizon={execution_horizon} exceeds checkpoint action horizon "
            f"{0 if action_deltas is None else len(action_deltas)}."
        )


class Gr00tUniVTACPolicy:
    """Small adapter that leaves all preprocessing/action decoding on the GR00T server."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        execution_horizon: int = 1,
        timeout_ms: int = 300_000,
    ):
        if execution_horizon < 1:
            raise ValueError("execution_horizon must be positive.")
        self.execution_horizon = execution_horizon
        self.client = MinimalPolicyClient(host, port, timeout_ms=timeout_ms)
        try:
            if not self.client.ping():
                raise ConnectionError(f"GR00T server at {host}:{port} returned an invalid ping.")
            self.modality_configs = self.client.get_modality_config()
            validate_rgb_checkpoint_contract(self.modality_configs, execution_horizon)
        except Exception:
            self.client.close()
            raise

    def reset(self) -> None:
        self.client.reset()

    def predict_chunk(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        formatted = format_univtac_observation(observation, instruction)
        chunk = extract_action_chunk(self.client.get_action(formatted))
        if self.execution_horizon > len(chunk):
            raise ValueError(
                f"execution_horizon={self.execution_horizon} exceeds returned horizon {len(chunk)}."
            )
        return chunk[: self.execution_horizon], formatted

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Gr00tUniVTACPolicy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
