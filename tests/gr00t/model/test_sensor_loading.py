# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "gr00t/model/gr00t_n1d7/sensor_loading.py"
_SPEC = importlib.util.spec_from_file_location("sensor_loading_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
sensor_loading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sensor_loading)


def test_concerto_mask_token_is_accepted_only_by_sensor_prefix() -> None:
    key = "action_head.point_encoder.backbone.embedding.mask_token"

    assert sensor_loading.unexpected_missing_keys([key], ("action_head.point_encoder.",)) == []


def test_unrelated_mask_token_is_not_silently_ignored() -> None:
    key = "action_head.unrelated.mask_token"

    assert sensor_loading.unexpected_missing_keys([key], ("action_head.point_encoder.",)) == [key]


def test_shared_multimodal_adapter_missing_keys_are_explicitly_accepted() -> None:
    keys = [
        "action_head.model.point_cross_attention.attention.in_proj_weight",
        "action_head.model.point_action_norms.0.weight",
        "action_head.model.point_modality_norms.0.weight",
        "action_head.model.point_gates",
        "action_head.model.tactile_cross_attention.attention.in_proj_weight",
        "action_head.model.tactile_action_norms.0.weight",
        "action_head.model.tactile_modality_norms.0.weight",
        "action_head.model.tactile_gates",
    ]

    assert (
        sensor_loading.unexpected_missing_keys(keys, sensor_loading.MULTIMODAL_ADAPTER_PREFIXES)
        == []
    )
    assert sensor_loading.unexpected_missing_keys(
        ["action_head.model.unrelated.mask_token"], sensor_loading.MULTIMODAL_ADAPTER_PREFIXES
    ) == ["action_head.model.unrelated.mask_token"]
    assert sensor_loading.unexpected_missing_keys(
        ["action_head.model.point_gates_extra"], sensor_loading.MULTIMODAL_ADAPTER_PREFIXES
    ) == ["action_head.model.point_gates_extra"]


def test_legacy_per_layer_attention_checkpoint_is_identified() -> None:
    assert sensor_loading.has_legacy_layerwise_attention_keys(
        ["action_head.model.point_cross_attention.0.attention.in_proj_weight"]
    )
    assert sensor_loading.has_legacy_layerwise_attention_keys(
        ["action_head.model.tactile_cross_attention.31.attention.in_proj_weight"]
    )
    assert not sensor_loading.has_legacy_layerwise_attention_keys(
        ["action_head.model.point_cross_attention.attention.in_proj_weight"]
    )


def test_base_and_multimodal_checkpoint_sensor_layouts() -> None:
    base = SimpleNamespace(dit_type="alternate_vl_dit")
    multimodal = SimpleNamespace(
        dit_type="multimodal_conditioned_dit",
        use_point_conditioning=True,
        use_tactile_conditioning=True,
    )

    assert sensor_loading.checkpoint_sensor_layout(base) == (False, False, False)
    assert sensor_loading.checkpoint_sensor_layout(multimodal) == (True, True, True)
