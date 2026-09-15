# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for VLM/action-head optimizer parameter groups."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from gr00t.experiment.trainer import Gr00tTrainer


class _ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(2, 3))
        self.lora_B = nn.Parameter(torch.randn(3, 2))


class _ToyActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.norm = nn.LayerNorm(3)


class _ToySplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_lora=True)
        self.backbone = _ToyBackbone()
        self.action_head = _ToyActionHead()

    def forward(self, input_ids=None, labels=None):
        del input_ids, labels
        loss = sum(parameter.sum() for parameter in self.parameters() if parameter.requires_grad)
        return {"loss": loss}


def _make_trainer(tmp_path, model=None):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=7e-4,
        weight_decay=0.1,
        optim="adamw_torch",
        report_to="none",
    )
    return Gr00tTrainer(
        model=_ToySplitModel() if model is None else model,
        args=args,
        vlm_learning_rate=1e-4,
        action_head_learning_rate=1e-3,
    )


def _learning_rate_by_parameter_id(optimizer):
    return {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_split_optimizer_has_complete_unique_coverage_and_expected_lrs(tmp_path):
    trainer = _make_trainer(tmp_path)
    optimizer = trainer.create_optimizer()
    learning_rates = _learning_rate_by_parameter_id(optimizer)

    trainable_ids = {
        id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad
    }
    optimizer_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    vlm_ids = {
        id(parameter)
        for parameter in trainer.model.backbone.parameters()
        if parameter.requires_grad
    }
    action_head_ids = {
        id(parameter)
        for parameter in trainer.model.action_head.parameters()
        if parameter.requires_grad
    }
    frozen_backbone_ids = {
        id(parameter)
        for parameter in trainer.model.backbone.parameters()
        if not parameter.requires_grad
    }

    assert set(optimizer_ids) == trainable_ids
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert all(learning_rates[parameter_id] == pytest.approx(1e-4) for parameter_id in vlm_ids)
    assert all(
        learning_rates[parameter_id] == pytest.approx(1e-3)
        for parameter_id in action_head_ids
    )
    assert frozen_backbone_ids.isdisjoint(optimizer_ids)


def test_split_optimizer_preserves_hf_weight_decay_and_scheduler_lr_ratio(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.create_optimizer_and_scheduler(num_training_steps=20)

    decay_by_parameter_id = {
        id(parameter): group["weight_decay"]
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert decay_by_parameter_id[id(trainer.model.action_head.projection.weight)] == pytest.approx(
        0.1
    )
    assert decay_by_parameter_id[id(trainer.model.action_head.projection.bias)] == 0.0
    assert decay_by_parameter_id[id(trainer.model.action_head.norm.weight)] == 0.0

    base_lrs = trainer.lr_scheduler.base_lrs
    assert max(base_lrs) / min(base_lrs) == pytest.approx(10.0)
    trainer.lr_scheduler.step()
    scheduled_lrs = trainer.lr_scheduler.get_last_lr()
    assert max(scheduled_lrs) / min(scheduled_lrs) == pytest.approx(10.0)


def test_split_optimizer_allows_an_empty_vlm_group(tmp_path):
    model = _ToySplitModel()
    model.config.use_lora = False
    model.backbone.requires_grad_(False)
    trainer = _make_trainer(tmp_path, model=model)

    optimizer = trainer.create_optimizer()

    assert {group["lr"] for group in optimizer.param_groups} == {1e-3}
    assert {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    } == {
        id(parameter)
        for parameter in model.action_head.parameters()
        if parameter.requires_grad
    }


def test_split_optimizer_state_loads_with_deterministic_parameter_groups(tmp_path):
    first = _make_trainer(tmp_path / "first")
    first_optimizer = first.create_optimizer()
    loss = sum(
        parameter.square().sum()
        for parameter in first.model.parameters()
        if parameter.requires_grad
    )
    loss.backward()
    first_optimizer.step()
    state_dict = first_optimizer.state_dict()

    resumed = _make_trainer(tmp_path / "resumed")
    resumed_optimizer = resumed.create_optimizer()
    resumed_optimizer.load_state_dict(state_dict)

    assert [group["lr"] for group in resumed_optimizer.param_groups] == [
        group["lr"] for group in first_optimizer.param_groups
    ]
    assert [len(group["params"]) for group in resumed_optimizer.param_groups] == [
        len(group["params"]) for group in first_optimizer.param_groups
    ]


def test_split_optimizer_rejects_unclassified_trainable_parameters(tmp_path):
    model = _ToySplitModel()
    model.unowned = nn.Parameter(torch.ones(1))
    trainer = _make_trainer(tmp_path, model=model)

    with pytest.raises(RuntimeError, match="unclassified=.*unowned"):
        trainer.create_optimizer()


def test_split_optimizer_rejects_trainable_backbone_base_weight_in_lora_mode(tmp_path):
    model = _ToySplitModel()
    model.backbone.base.weight.requires_grad_(True)
    trainer = _make_trainer(tmp_path, model=model)

    with pytest.raises(RuntimeError, match="trainable non-LoRA backbone parameters"):
        trainer.create_optimizer()


def test_split_optimizer_rejects_duplicate_module_ownership(tmp_path):
    model = _ToySplitModel()
    model.action_head.shared_adapter = model.backbone.lora_A
    trainer = _make_trainer(tmp_path, model=model)

    with pytest.raises(RuntimeError, match="duplicate="):
        trainer.create_optimizer()
