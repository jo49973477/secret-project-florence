# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free helpers for sensor-checkpoint loading decisions."""

MULTIMODAL_ADAPTER_PREFIXES = (
    "action_head.model.point_cross_attention.",
    "action_head.model.point_action_norms.",
    "action_head.model.point_modality_norms.",
    "action_head.model.point_gates",
    "action_head.model.tactile_cross_attention.",
    "action_head.model.tactile_action_norms.",
    "action_head.model.tactile_modality_norms.",
    "action_head.model.tactile_gates",
)


def checkpoint_sensor_layout(checkpoint_config) -> tuple[bool, bool, bool]:
    is_multimodal = getattr(checkpoint_config, "dit_type", None) == "multimodal_conditioned_dit"
    return (
        is_multimodal,
        is_multimodal and getattr(checkpoint_config, "use_point_conditioning", False),
        is_multimodal and getattr(checkpoint_config, "use_tactile_conditioning", False),
    )


def unexpected_missing_keys(
    missing_keys: list[str], expected_prefixes: tuple[str, ...]
) -> list[str]:
    """Return missing keys not covered by exact newly-enabled module prefixes."""
    return [
        key
        for key in missing_keys
        if not any(
            key.startswith(prefix) if prefix.endswith(".") else key == prefix
            for prefix in expected_prefixes
        )
    ]


def has_legacy_layerwise_attention_keys(keys: list[str]) -> bool:
    """Detect checkpoint keys from the experimental per-layer MHA layout."""
    old_attention_prefixes = (
        "action_head.model.point_cross_attention.",
        "action_head.model.tactile_cross_attention.",
    )
    for key in keys:
        for prefix in old_attention_prefixes:
            suffix = key.removeprefix(prefix)
            if suffix != key and suffix.split(".", maxsplit=1)[0].isdigit():
                return True
    return False
