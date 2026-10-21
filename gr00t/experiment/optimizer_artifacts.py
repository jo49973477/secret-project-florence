"""Record actual optimizer grouping before wrapping, and restored LR at training start."""

import json
import logging
from pathlib import Path

from transformers import TrainerCallback


def describe_groups(groups: list[dict]) -> list[dict]:
    return [
        {
            "name": group.get("name", f"group_{index}"),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0)),
            "parameter_tensors": len(group["params"]),
            # ds_numel preserves full tensor sizes under ZeRO partitioning.
            "trainable_parameters": sum(getattr(p, "ds_numel", p.numel()) for p in group["params"]),
        }
        for index, group in enumerate(groups)
    ]


class EffectiveOptimizerCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
        requested = getattr(self.trainer, "optimizer_group_manifest", [])
        actual = describe_groups(optimizer.param_groups) if optimizer is not None else []
        for group in actual:
            logging.info("Active optimizer group after state restore: %s", group)
        artifact = {
            "global_step": state.global_step,
            "constructed_parameter_groups": requested,
            "active_groups_after_state_restore": actual,
            "note": "Active group counts may reflect flattened/partitioned DeepSpeed tensors; constructed groups retain original ownership. Learning rates are current values at train begin, including restored scheduler state.",
        }
        path = Path(args.output_dir) / "effective_optimizers.json"
        path.write_text(json.dumps(artifact, indent=2))
