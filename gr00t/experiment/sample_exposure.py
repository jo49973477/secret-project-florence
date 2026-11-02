"""Nominal dataset-equivalent exposure for sharded IterableDataset training.

One optimizer step is assumed to consume a full effective global batch. Mixture
weights affect which dataset is sampled, not the corpus-size denominator: the
denominator sums valid timesteps in every positive-weight component. Repeated samples
and stochastic mixing mean effective epochs do not imply complete coverage.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_MANIFEST = "sample_exposure.json"
_UNAVAILABLE_MARKER = "sample_exposure_unavailable"


def _mark_unavailable(marker: Path) -> None:
    try:
        marker.touch()
    except OSError as exc:
        logger.warning("Could not mark sample exposure unavailable (%s).", exc)


class SampleExposure:
    def __init__(self, dataset: Any, args: Any):
        self.enabled = False
        self.total_training_samples: int | None = None
        self.effective_global_batch_size: int | None = None
        self.gradient_accumulation_steps: int | None = None
        self._manifest: dict[str, Any] | None = None
        try:
            components = []
            for component, weight in zip(dataset.datasets, dataset.weights, strict=True):
                weight = float(weight)
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("mixture weights must be finite and nonnegative")
                shard_count = len(component)
                lengths = [int(component.get_shard_length(i)) for i in range(shard_count)]
                if not lengths or any(length <= 0 for length in lengths):
                    raise ValueError("shard lengths must be positive")
                components.append(
                    {
                        "path": str(component.dataset_path),
                        "shard_lengths": lengths,
                        "weight": weight,
                    }
                )
            if not components or not any(item["weight"] > 0 for item in components):
                raise ValueError("mixture has no sampled datasets")
            per_device = int(args.per_device_train_batch_size)
            world_size = int(args.world_size)
            accumulation = int(args.gradient_accumulation_steps)
            if min(per_device, world_size, accumulation) <= 0:
                raise ValueError("batch settings must be positive")
            total = sum(sum(item["shard_lengths"]) for item in components if item["weight"] > 0)
            self.gradient_accumulation_steps = accumulation
            self.effective_global_batch_size = per_device * world_size * accumulation
            self.total_training_samples = total
            self._manifest = {
                "version": 1,
                "per_device_train_batch_size": per_device,
                "world_size": world_size,
                "gradient_accumulation_steps": accumulation,
                "components": components,
                "total_training_samples": total,
            }
            self.enabled = True
        except Exception as exc:
            # Metrics are optional; malformed dataset metadata must not stop training.
            logger.warning("Sample exposure metrics unavailable: %s", exc)

    def prepare(
        self, output_dir: str | Path, resume_checkpoint: str | Path | None, *, rank_zero: bool
    ):
        """Keep checkpoint formats intact and refuse ambiguous resumed estimates."""
        if not self.enabled:
            return
        marker = Path(output_dir) / _UNAVAILABLE_MARKER
        if resume_checkpoint is not None:
            source = Path(resume_checkpoint).parent / _MANIFEST
            try:
                previous = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Sample exposure history is unavailable on resume (%s).", exc)
                self.enabled = False
                if rank_zero:
                    _mark_unavailable(marker)
                return
            if previous != self._manifest:
                logger.warning(
                    "Sample exposure settings changed on resume; suppressing cumulative metrics "
                    "rather than applying the new batch or dataset size to historical steps."
                )
                self.enabled = False
                if rank_zero:
                    _mark_unavailable(marker)
                return
        if rank_zero:
            target = Path(output_dir) / _MANIFEST
            try:
                target.write_text(json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
                marker.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not write sample exposure metadata (%s).", exc)

    def values(self, global_step: int) -> tuple[int, float] | None:
        if not self.enabled or self.effective_global_batch_size is None:
            return None
        samples_seen = int(global_step) * self.effective_global_batch_size
        return samples_seen, samples_seen / self.total_training_samples

    def metrics(self, global_step: int) -> dict[str, int | float]:
        values = self.values(global_step)
        if values is None:
            return {}
        return {"samples_seen": values[0], "effective_epoch": values[1]}
