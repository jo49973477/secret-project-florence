# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time, rank-local memory and trainability diagnostics for training startup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch


LOGGER = logging.getLogger("gr00t.first_step_memory")
# Nonzero ranks intentionally lower the root logger to WARNING. Setting this
# named logger explicitly keeps the required rank-local diagnostics visible;
# propagated records still use the root handler/format.
LOGGER.setLevel(logging.INFO)


def _read_memory_info() -> tuple[int | None, int | None, int | None, int | None]:
    """Return process RSS and host RAM use/availability from Linux procfs."""
    rss_bytes = total_bytes = available_bytes = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_bytes = int(line.split()[1]) * 1024
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_bytes = int(line.split()[1]) * 1024
            if line.startswith("MemAvailable:"):
                available_bytes = int(line.split()[1]) * 1024
    except OSError:
        pass
    used_bytes = total_bytes - available_bytes if total_bytes and available_bytes else None
    return rss_bytes, total_bytes, used_bytes, available_bytes


def log_memory_snapshot(phase: str) -> None:
    """Log GPU and host memory for this rank without collectives or synchronization."""
    rank = os.environ.get("RANK", "0")
    values: dict[str, int | None] = {
        "gpu_allocated_bytes": None,
        "gpu_reserved_bytes": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_available_bytes": None,
        "gpu_total_bytes": None,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        values.update(
            gpu_allocated_bytes=torch.cuda.memory_allocated(device),
            gpu_reserved_bytes=torch.cuda.memory_reserved(device),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            values["gpu_available_bytes"] = free_bytes
            values["gpu_total_bytes"] = total_bytes
        except (RuntimeError, AttributeError):
            pass
    rss_bytes, host_total_bytes, host_used_bytes, host_available_bytes = _read_memory_info()
    LOGGER.info(
        "TRAINING_MEMORY rank=%s phase=%s gpu_allocated_bytes=%s gpu_reserved_bytes=%s "
        "gpu_peak_allocated_bytes=%s gpu_available_bytes=%s gpu_total_bytes=%s "
        "process_rss_bytes=%s host_total_bytes=%s host_used_bytes=%s "
        "host_available_bytes=%s",
        rank,
        phase,
        values["gpu_allocated_bytes"],
        values["gpu_reserved_bytes"],
        values["gpu_peak_allocated_bytes"],
        values["gpu_available_bytes"],
        values["gpu_total_bytes"],
        rss_bytes,
        host_total_bytes,
        host_used_bytes,
        host_available_bytes,
    )


def _param_counts(parameters) -> tuple[int, int]:
    total = trainable = 0
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def log_trainable_parameter_summary(model: torch.nn.Module) -> None:
    """Report disjoint parameter ownership before ZeRO partitions parameters."""
    total, trainable = _param_counts(model.parameters())
    LOGGER.info("TRAINABLE_PARAMETERS total=%s trainable=%s", f"{total:,}", f"{trainable:,}")

    requested_categories = (
        "qwen3_vl_backbone",
        "vlm_projector",
        "base_gr00t_dit",
        "state_action_encoders_decoder",
        "concerto_backbone",
        "concerto_projection",
        "sparsh_backbone",
        "sparsh_projection",
        "multimodal_cross_attention_adapters",
        "residual_gates",
        "other",
    )
    counts = {name: [0, 0, 0] for name in requested_categories}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        category = _parameter_ownership_category(name)
        entry = counts[category]
        entry[0] += parameter.numel()
        entry[1] += parameter.numel() if parameter.requires_grad else 0
        entry[2] += 1

    if sum(entry[0] for entry in counts.values()) != total:
        raise RuntimeError("Parameter ownership diagnostics did not cover the complete model")
    if sum(entry[1] for entry in counts.values()) != trainable:
        raise RuntimeError("Trainable ownership diagnostics did not cover all trainable parameters")

    for category in requested_categories:
        category_total, category_trainable, tensor_count = counts[category]
        _log_parameter_status(category, category_total, category_trainable, tensor_count)


def _parameter_ownership_category(name: str) -> str:
    """Map each GR00T parameter name into one disjoint ownership category."""
    if name.startswith("backbone."):
        lowered = name.lower()
        if ".merger." in lowered or ".projector." in lowered:
            return "vlm_projector"
        return "qwen3_vl_backbone"
    if name.startswith(("action_head.vlln.", "action_head.vl_self_attention.")):
        return "vlm_projector"
    if name.startswith(
        (
            "action_head.state_encoder.",
            "action_head.action_encoder.",
            "action_head.action_decoder.",
            "action_head.position_embedding.",
        )
    ):
        return "state_action_encoders_decoder"
    if name.startswith("action_head.point_encoder.backbone."):
        return "concerto_backbone"
    if name.startswith("action_head.point_encoder.projection."):
        return "concerto_projection"
    if name.startswith("action_head.tactile_encoder.backbone."):
        return "sparsh_backbone"
    if name.startswith("action_head.tactile_encoder.projection."):
        return "sparsh_projection"
    if name.startswith(("action_head.model.point_gates", "action_head.model.tactile_gates")):
        return "residual_gates"
    if name.startswith(
        (
            "action_head.model.point_cross_attention.",
            "action_head.model.point_action_norms.",
            "action_head.model.point_modality_norms.",
            "action_head.model.tactile_cross_attention.",
            "action_head.model.tactile_action_norms.",
            "action_head.model.tactile_modality_norms.",
        )
    ):
        return "multimodal_cross_attention_adapters"
    if name.startswith("action_head.model."):
        return "base_gr00t_dit"
    return "other"


def _log_parameter_status(name: str, total: int, trainable: int, tensors: int) -> None:
    status = "absent" if total == 0 else "frozen" if trainable == 0 else "trainable"
    LOGGER.info(
        "PARAMETER_OWNERSHIP module=%s status=%s trainable=%s total=%s tensors=%s",
        name,
        status,
        f"{trainable:,}",
        f"{total:,}",
        tensors,
    )


def log_deepspeed_engine_state(
    optimizer,
    *,
    expected_optimizer_offload: bool,
    expected_param_offload: bool,
    expected_sub_group_size: int | None,
) -> None:
    """Log and assert the initialized ZeRO engine's effective memory settings."""
    zero_optimizer = getattr(optimizer, "optimizer", optimizer)
    optimizer_offload = bool(getattr(zero_optimizer, "offload_optimizer", False))
    param_offload = bool(getattr(zero_optimizer, "offload_param", False))
    sub_group_size = getattr(zero_optimizer, "sub_group_size", None)
    basic_optimizer = getattr(zero_optimizer, "optimizer", None)
    basic_optimizer_name = type(basic_optimizer).__name__ if basic_optimizer is not None else None
    fp32_devices = sorted(
        {
            str(getattr(parameter, "device", None))
            for parameter in getattr(zero_optimizer, "fp32_partitioned_groups_flat", [])
            if parameter is not None
        }
    )
    param_device = getattr(
        getattr(zero_optimizer, "parameter_offload", None), "offload_device", None
    )
    LOGGER.info(
        "DEEPSPEED_ENGINE rank=%s optimizer_offload=%s param_offload=%s "
        "sub_group_size=%s basic_optimizer=%s fp32_partition_devices=%s "
        "parameter_offload_device=%s",
        os.environ.get("RANK", "0"),
        optimizer_offload,
        param_offload,
        sub_group_size,
        basic_optimizer_name,
        fp32_devices,
        param_device,
    )

    mismatches = []
    if optimizer_offload != expected_optimizer_offload:
        mismatches.append(
            f"optimizer_offload expected={expected_optimizer_offload} actual={optimizer_offload}"
        )
    if param_offload != expected_param_offload:
        mismatches.append(f"param_offload expected={expected_param_offload} actual={param_offload}")
    if expected_sub_group_size is not None and sub_group_size != expected_sub_group_size:
        mismatches.append(
            f"sub_group_size expected={expected_sub_group_size} actual={sub_group_size}"
        )
    if expected_optimizer_offload and basic_optimizer_name != "DeepSpeedCPUAdam":
        mismatches.append(
            "optimizer offload requested but initialized basic optimizer is "
            f"{basic_optimizer_name!r}, not DeepSpeedCPUAdam"
        )
    if mismatches:
        raise RuntimeError("DeepSpeed runtime configuration mismatch: " + "; ".join(mismatches))
