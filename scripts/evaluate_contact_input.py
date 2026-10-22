"""Held-out masked flow-matching loss and contact ablations using the saved processor.

This is offline validation, never a training update. Episode IDs must be held out
from the training dataset. Equal seeds reuse the same flow noise in each condition.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("enabled", "off", "shuffle", "noise"),
        default=["enabled", "off"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        model_path=args.checkpoint, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device=args.device
    )
    model = policy.model.eval()
    adapter = model.action_head.contact_input_adapter
    configs = policy.modality_configs
    loader = LeRobotEpisodeLoader(args.dataset_path, configs)
    rows = []
    # One sensor permutation across the evaluation set, not independent random
    # images mislabeled as a real observed contact state.
    points = []
    for episode in args.episode_ids:
        trajectory = loader[episode]
        deltas = [delta for config in configs.values() for delta in config.delta_indices]
        start = max(0, -min(deltas))
        stop = min(len(trajectory) - max(deltas), start + args.steps)
        for step in range(start, stop):
            point = extract_step_data(trajectory, step, configs, EmbodimentTag.NEW_EMBODIMENT)
            points.append((episode, step, point))
    if not points:
        raise ValueError("No complete observation/action windows in the selected episodes")
    from copy import deepcopy

    permutation = np.random.default_rng(args.seed).permutation(len(points))
    for condition in args.conditions:
        if adapter is None and condition != "enabled":
            continue
        for index, (episode, step, original) in enumerate(points):
            point = deepcopy(original)
            if condition == "shuffle":
                point.tactile = deepcopy(points[permutation[index]][2].tactile)
            elif condition == "noise":
                rng = np.random.default_rng(args.seed + index)
                point.tactile = {
                    key: np.clip(
                        np.asarray(value, dtype=np.float32)
                        + rng.normal(0, 15, np.asarray(value).shape),
                        0,
                        255,
                    ).astype(np.uint8)
                    for key, value in point.tactile.items()
                }
            processed = policy.processor([{"type": "episode_step", "content": point}])
            batch = policy.collate_fn([processed])
            torch.manual_seed(args.seed + index)
            # Toggle the injection without changing the architecture or weights.
            handle = None
            if condition == "off":
                handle = adapter.register_forward_hook(lambda module, inputs, output: inputs[0])
            try:
                with torch.no_grad():
                    result = model(batch)
            finally:
                if handle is not None:
                    handle.remove()
            rows.append(
                {
                    "condition": condition,
                    "episode": episode,
                    "step": step,
                    "loss": float(result["loss"]),
                    "valid_action_elements": float(result["action_mask"].float().sum()),
                }
            )
    summary = {}
    for condition in args.conditions:
        selected = [row for row in rows if row["condition"] == condition]
        if selected:
            count = sum(row["valid_action_elements"] for row in selected)
            summary[condition] = sum(
                row["loss"] * row["valid_action_elements"] for row in selected
            ) / max(count, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"seed": args.seed, "masked_flow_matching_loss": summary, "samples": rows}, indent=2
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
