#!/usr/bin/env python

"""Run a deterministic one-batch action-head memorization diagnostic.

The script consumes a saved ``experiment_cfg/config.yaml`` so it reuses the
same model, dataset, modality config, and statistics path as a real run. It is
deliberately separate from production training: the fixed RNG state and fixed
batch cannot accidentally affect a normal fine-tune.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
import gr00t.model  # noqa: F401 - registers model pipelines
from gr00t.model import MODEL_REGISTRY
import torch
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--update-sample-size", type=int, default=4096)
    return parser.parse_args()


def _fixed_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _sample_parameters(parameters: list[torch.nn.Parameter], limit: int) -> list[torch.Tensor]:
    """Copy a bounded deterministic sample for update-norm diagnostics."""
    remaining = limit
    samples = []
    for parameter in parameters:
        if remaining <= 0:
            break
        flat = parameter.detach().reshape(-1)
        take = min(remaining, flat.numel())
        samples.append(flat[:take].float().clone())
        remaining -= take
    return samples


def _sampled_update_norm(
    before: list[torch.Tensor], parameters: list[torch.nn.Parameter], limit: int
) -> float:
    remaining = limit
    squared = torch.zeros((), device=parameters[0].device)
    for old, parameter in zip(before, parameters):
        take = min(remaining, parameter.numel())
        squared += (parameter.detach().reshape(-1)[:take].float() - old).square().sum()
        remaining -= take
        if remaining <= 0:
            break
    return squared.sqrt().item()


def main() -> None:
    args = _parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.update_sample_size < 1:
        raise ValueError("steps, batch-size, and update-sample-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_cfg").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config.from_pretrained(args.config)
    config.training.num_gpus = 1
    config.training.use_ddp = False
    config.training.output_dir = str(args.output_dir)
    config.training.dataloader_num_workers = 0
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.use_lora = False
    config.model.state_dropout_prob = 0.0
    config.model.action_head_dropout_override = 0.0
    config.model.vl_self_attention_dropout_override = 0.0
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = None
    config.model.image_crop_size = None
    config.model.image_target_size = None
    config.model.shortest_image_edge = 256
    config.model.crop_fraction = 1.0

    pipeline = MODEL_REGISTRY[type(config.model)](config, args.output_dir / "experiment_cfg")
    pipeline.setup()
    model = pipeline.return_model()
    train_dataset, _ = pipeline.return_dataset()
    collator = pipeline.return_collator()

    device = torch.device(args.device)
    model.to(device)
    model.requires_grad_(False)
    model.action_head.requires_grad_(True)
    model.eval()
    model.action_head.train()

    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    fixed_batch = next(iter(loader))
    inputs = fixed_batch["inputs"]
    actions = inputs["action"].float()
    action_mask = inputs["action_mask"].float()
    valid_actions = actions[action_mask.bool()]

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"name": "Action Head", "params": trainable, "lr": args.learning_rate}],
        weight_decay=0.0,
    )
    logging.info("trainable parameters: %s", f"{sum(p.numel() for p in trainable):,}")
    logging.info("action_mask.sum(): %.0f", action_mask.sum().item())
    logging.info(
        "normalized action min/max/mean/std: %.6f %.6f %.6f %.6f",
        valid_actions.min().item(),
        valid_actions.max().item(),
        valid_actions.mean().item(),
        valid_actions.std().item(),
    )
    for group in optimizer.param_groups:
        logging.info(
            "optimizer group %s: lr=%g weight_decay=%g tensors=%d parameters=%s",
            group["name"],
            group["lr"],
            group["weight_decay"],
            len(group["params"]),
            f"{sum(parameter.numel() for parameter in group['params']):,}",
        )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        before = _sample_parameters(trainable, args.update_sample_size)
        _fixed_seed(args.seed, device)
        loss = model(inputs=inputs)["loss"]
        loss.backward()
        grad_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in trainable
                if parameter.grad is not None
            )
        ).item()
        optimizer.step()
        update_norm = _sampled_update_norm(before, trainable, args.update_sample_size)
        logging.info(
            "step=%d fixed-batch loss=%.8f gradient norm=%.8f "
            "parameter update norm (first %d scalars)=%.8f",
            step,
            loss.item(),
            grad_norm,
            args.update_sample_size,
            update_norm,
        )

    if hasattr(train_dataset, "close"):
        train_dataset.close()


if __name__ == "__main__":
    main()
