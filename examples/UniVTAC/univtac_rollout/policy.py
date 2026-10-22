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
        self._tactile_history = []
        self.client = MinimalPolicyClient(host, port, timeout_ms=timeout_ms)
        try:
            if not self.client.ping():
                raise ConnectionError(f"GR00T server at {host}:{port} returned an invalid ping.")
            self.modality_configs = self.client.get_modality_config()
            rgb_configs = {k: v for k, v in self.modality_configs.items() if k != "tactile"}
            validate_rgb_checkpoint_contract(rgb_configs, execution_horizon)
            if "tactile" in self.modality_configs:
                tactile = self.modality_configs["tactile"]
                if _config_field(tactile, "modality_keys") != ["rgb"] or _config_field(
                    tactile, "delta_indices"
                ) != [-1, 0]:
                    raise ValueError("Contact rollout requires tactile rgb at [-1,0]")
        except Exception:
            self.client.close()
            raise

    def reset(self) -> None:
        self._tactile_history.clear()
        self.client.reset()

    def predict_chunk(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        formatted = format_univtac_observation(observation, instruction)
        if "tactile" in self.modality_configs:
            if not self._tactile_history:
                self.observe(observation)
            # Worker calls observe after each executed action, even inside chunks.
            previous = (
                self._tactile_history[-2]
                if len(self._tactile_history) > 1
                else self._tactile_history[-1]
            )
            current = self._tactile_history[-1]
            formatted["tactile"] = {"rgb": np.stack((previous, current))[None]}
        chunk = extract_action_chunk(self.client.get_action(formatted))
        if self.execution_horizon > len(chunk):
            raise ValueError(
                f"execution_horizon={self.execution_horizon} exceeds returned horizon {len(chunk)}."
            )
        return chunk[: self.execution_horizon], formatted

    def observe(self, observation):
        """Record one post-action observation. Ambiguous physical sensors fail closed."""
        if "tactile" not in self.modality_configs:
            return
        import os

        selected = os.environ.get("GR00T_UNIVTAC_TACTILE_KEY")
        candidates = {}

        def visit(value, prefix=""):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                path = f"{prefix}/{key}" if prefix else key
                if key == "rgb":
                    candidates[path] = item
                elif isinstance(item, dict):
                    visit(item, path)

        visit(observation.get("tactile", {}), "tactile")
        if selected:
            if selected not in candidates:
                raise ValueError(
                    f"Requested tactile source {selected} unavailable: {list(candidates)}"
                )
            image = candidates[selected]
        elif len(candidates) == 1:
            image = next(iter(candidates.values()))
        else:
            raise ValueError(
                "Select the training sensor with GR00T_UNIVTAC_TACTILE_KEY; "
                f"available unmarked RGB streams: {list(candidates)}"
            )
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Tactile RGB must be uint8 HWC, matching the converter")
        self._tactile_history.append(image.copy())
        self._tactile_history = self._tactile_history[-2:]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Gr00tUniVTACPolicy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
